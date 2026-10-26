"""Bundled-asset locator + subprocess runner.

Implements doc `installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md` §4.2:
PyInstaller `--onefile` unpacks bundled data files into a temp dir exposed
as `sys._MEIPASS` at runtime ("frozen" mode). When running from a plain
source checkout ("dev mode") there is no `sys._MEIPASS`, so assets are
resolved relative to this file's own directory instead.

Framework-only module: no step1-7 business logic lives here, and this
module deliberately does NOT import `privilege.py` (§9). `run_bundled_script`
instead accepts an already-prepared `env` mapping as a plain parameter; the
caller is responsible for building that environment and passing it in.

Public API:
    asset_path(*parts)                 -- locate a bundled asset.
    stage_assets(*parts, prefix=...)   -- copy a bundled asset *directory*
        out to a fresh staging dir, preserving its layout.
    run_streamed(command, renderer=, stream=, verbose=, ...)
                                       -- run a child and stream, capture and
        redact its output in one pass (doc 08 section 4.2).
    run_bundled_script(*asset_parts, args=, env=, tree=, inherit_env=,
                       cleanup=)       -- stage and execute a fragment.

Four properties this module owes its callers:

1. **Tree staging.** A bundled bash fragment that does
   `source "$SCRIPT_DIR/lib/common.sh"` only works if the whole directory
   lands together, so `stage_assets` copies a directory rather than a single
   file and `run_bundled_script(..., tree=(...))` runs the fragment from
   inside that staged tree.
2. **Environment merging.** `env` is an *overlay* on `os.environ` by default
   (`inherit_env=True`). Replacing the environment outright strips `PATH`,
   which breaks every `command -v` in the staged bash; pass
   `inherit_env=False` for the hermetic replace semantics.
3. **Transcript capture (§8.2).** The command line, any explicit environment
   overrides, and both output streams of every shelled-out fragment go to
   `logs.log` -- stdout at info level, stderr at warn level.
4. **One pass over the output (doc 08 §4.2).** `run_streamed` is the single
   place a child's output is read: it fills the `CompletedProcess` buffers
   and writes each line out as it arrives, so live output and captured
   output can never disagree and no stream is read twice.
   `run_bundled_script` runs through it rather than keeping a second
   subprocess path of its own. **Exactly one writer** touches the terminal
   for a given line -- the renderer when there is one, `logs.log` when there
   is not, nobody under `stream=False` -- because a second raw write to the
   same stream duplicates the line and tears through the live region's
   cursor arithmetic. The transcript is not the terminal, so it is outside
   that count: every line reaches it in every mode (doc 08 §7.1), via
   `logs.log` where that writes the terminal anyway and `logs.transcript`
   where nothing does. Control characters are stripped with
   `progress.sanitise` and nothing else: this module deliberately keeps no
   stripper of its own, so there is one definition of "clean" rather than
   two that agree today and drift later.

Exactly what redaction guarantees, stated precisely because the transcript
is an audited artifact:

- **By value.** Every value the child environment carries under a
  `_REDACT_KEYS` name is scrubbed wherever it appears in the command line,
  the environment dump, or either output stream, prefix or no prefix. This
  is what covers `set -x` tracing, `curl -H "Authorization: Bearer
  $NGC_API_KEY"`, and a tool echoing an argument back in an error message.
  Value scrubbing is exact-substring, so a fragment that transforms a secret
  before printing it (base64, URL-encoding, splitting it across lines)
  defeats it.
- **By key.** Any `KEY=` occurrence for a `_REDACT_KEYS` name additionally
  blanks the rest of that line, which catches a secret this process never
  held -- one the fragment read from a file or generated itself.
- **Not covered.** A secret under a key not in `_REDACT_KEYS`, and an empty
  or whitespace-only value (scrubbing those would match everywhere).

Both rules apply to streamed output before it is written anywhere a person
reads it. Showing a command live must not be how a secret becomes visible
that captured output would have hidden (doc 08 §4.2). The returned buffer is
the deliberate exception: §4.2 names the terminal and the transcript, the
buffer has never been redacted, and scrubbing it corrupts what its callers
parse -- an inherited `NGC_API_KEY=abc` turns a child's `version abc123
build` into `version <redacted>123 build`. `run_streamed(redact_capture=True)`
opts a caller in where that trade is worth making.
"""

from __future__ import annotations

import os
import pathlib
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping, Protocol, Sequence

from . import progress
from .logs import log, transcript

# Default prefix for the run-scoped staging directory created by
# stage_assets() / run_bundled_script().
STAGE_PREFIX = "mv3dt-shellout-"

# Environment variable exported by run_bundled_script() when it stages a
# tree. Its meaning is invariant: it is always the root of the staged assets
# tree -- the staged stand-in for `assets/` itself -- no matter which subtree
# was staged. A fragment therefore addresses asset subtrees as
# `$MV3DT_ASSET_ROOT/<subtree>/...` in every mode, and locates its own
# directory from `$0` rather than from this variable.
ASSET_ROOT_ENV = "MV3DT_ASSET_ROOT"

# Environment keys whose values must never reach the transcript (§8.2).
# Longest-first alternation below so `NGC_API_KEY` is matched as a whole and
# not as a bare `API_KEY` suffix.
_REDACT_KEYS = (
    "NGC_API_KEY",
    "API_KEY",
    "CAM_PASSWORD",
    "MQTT_PASSWORD",
    "WORKSTATION_TICKET_SECRET",
)
_REDACTED = "<redacted>"

# `KEY=` blanks the remainder of the line, not just the next whitespace-free
# run: a password may legitimately contain spaces, and `\S*` would have
# logged `MQTT_PASSWORD=hunter 2 three` as `MQTT_PASSWORD=<redacted> 2 three`.
# Once a secret key is named on a line, no part of that line's tail can be
# assumed safe. Callers wanting per-argument granularity redact each argument
# separately before joining, which _command_dump() below does.
_REDACT_RE = re.compile(
    r"(?<![A-Za-z0-9_])("
    + "|".join(sorted(_REDACT_KEYS, key=len, reverse=True))
    + r")=[^\n]*"
)

# Modes applied to a staged tree: executable for shell fragments, plain
# read-only-ish for everything else, traversable for directories.
_SCRIPT_MODE = 0o755
_DATA_MODE = 0o644
_DIR_MODE = 0o755


def asset_path(*parts: str) -> pathlib.Path:
    """Resolve a path under the bundled ``assets/`` directory.

    Frozen mode (running as a PyInstaller ``--onefile`` binary):
    ``sys._MEIPASS`` points at the unpacked temp dir for this run, and
    assets live under ``<_MEIPASS>/assets/...`` per the ``datas`` mapping
    in ``installer.spec`` (§4.1).

    Dev mode (running from a source checkout, no ``sys._MEIPASS``):
    assets live under ``assets/`` next to this module.

    Note: the doc's own code block spells the dev-mode line as
    ``pathlib.Path(__file__).parent / "assets" / *parts``, which is not
    valid Python (``/`` cannot be followed by an unpacked ``*parts``).
    This implements the clearly-intended equivalent using ``.joinpath()``.
    """
    base = getattr(sys, "_MEIPASS", None)
    if base:  # frozen binary
        return pathlib.Path(base, "assets", *parts)
    return pathlib.Path(__file__).parent.joinpath("assets", *parts)  # dev mode


# ---------------------------------------------------------------------------
# Redaction (§8.2)
# ---------------------------------------------------------------------------


def _redact(text: str) -> str:
    """Blank a ``KEY=...`` occurrence, and its line's tail, for a secret key."""
    return _REDACT_RE.sub(lambda m: f"{m.group(1)}={_REDACTED}", text)


def _secret_values(env: Mapping[str, str]) -> tuple[str, ...]:
    """Collect the plaintext this process is handing the fragment.

    Longest first, so a value that contains a shorter one is replaced whole
    rather than shredded around it. Empty and whitespace-only values are
    excluded: substituting those would match at every position and destroy
    the transcript without protecting anything.
    """
    values = set()
    for key in _REDACT_KEYS:
        value = env.get(key, "")
        if value.strip():
            values.add(value)
    return tuple(sorted(values, key=len, reverse=True))


def _scrub(text: str, secrets: Sequence[str]) -> str:
    """Apply value-based scrubbing, then key-based, to one line of text."""
    for secret in secrets:
        text = text.replace(secret, _REDACTED)
    return _redact(text)


def _command_dump(command: Sequence[str], secrets: Sequence[str]) -> str:
    """Render an argv for the transcript, redacting each argument alone.

    Redacting per argument rather than over the joined string keeps a
    ``KEY=`` argument from swallowing every argument that follows it.
    """
    return " ".join(_scrub(shlex.quote(part), secrets) for part in command)


def _env_dump(env: Mapping[str, str], secrets: Sequence[str]) -> str:
    """Render an environment overlay as a single sorted, redacted line.

    Each entry is redacted on its own for the same reason as ``_command_dump``:
    a secret entry must not blank the entries sorted after it.
    """
    entries = []
    for key, value in sorted(env.items()):
        if key in _REDACT_KEYS:
            entries.append(f"{key}={_REDACTED}")
        else:
            entries.append(_scrub(f"{key}={shlex.quote(value)}", secrets))
    return " ".join(entries)


# ---------------------------------------------------------------------------
# The tee runner (doc 08 §4.2)
# ---------------------------------------------------------------------------


class LineSink(Protocol):
    """What `run_streamed` needs of a live renderer: one line at a time.

    `progress.Progress` satisfies it. The runner is typed against the method
    rather than the class so it never has to know whether it is feeding a
    live region, a plain non-tty stream, or a test double.
    """

    def line(self, text: str) -> None: ...  # pragma: no cover -- structural


_STDOUT = "stdout"
_STDERR = "stderr"

# The level each stream's output is recorded under, whichever sink carries
# it. A child's stderr is not necessarily an error, but it is the half an
# operator reads first when a step fails, and warn is the level this module
# has always given it. One mapping rather than two, so a transcript line
# cannot come out labelled differently depending on who wrote it.
_LEVELS = {_STDOUT: "info", _STDERR: "warn"}

# How long to wait for a reader thread to notice a killed child before giving
# up on it. The threads are daemons, so a wedged reader can never hold the
# installer open; this only keeps the ordinary case tidy.
_REAP_TIMEOUT_S = 5.0


def _pump(stream: Any, name: str, lines: "queue.Queue") -> None:
    """Drain one pipe into `lines` until EOF, then post an end marker.

    **Why a thread per pipe.** Reading `stdout` to EOF before touching
    `stderr` is the classic two-pipe deadlock: the child blocks writing into
    a full `stderr` buffer, so it never closes `stdout`, so the parent never
    stops waiting for it. One reader per pipe means neither buffer can fill
    while we are blocked on the other. `selectors` would also work, but only
    on POSIX pipes; threads behave the same everywhere the installer is built
    and cost nothing at this scale.

    These threads only *produce*. Redaction, rendering and the transcript all
    happen on the consuming thread, so `Progress` -- which carries mutable
    cursor state and is not thread-safe -- is only ever touched from one.

    A failure here is queued rather than raised. `readline` can fail on its
    own account (a decode error under `errors="strict"`, an I/O error), and
    a thread that died on the spot would leave the consumer to see a clean
    EOF: exit status 0, empty output, and a traceback nobody attached to
    anything. The consumer re-raises what it finds, so the caller gets the
    exception `subprocess.run` would have raised rather than a plausible
    empty result.
    """
    try:
        for raw in iter(stream.readline, ""):
            lines.put((name, raw))
    except BaseException as exc:  # noqa: BLE001 -- re-raised on the consumer
        lines.put((name, exc))
    finally:
        lines.put((name, None))
        try:
            stream.close()
        except Exception:  # pragma: no cover -- already-closed pipe
            pass


def _feed(stream: Any, payload: str) -> None:
    """Write `input` to the child and close its stdin, from its own thread.

    Same reasoning as `_pump`: a child that writes more than a pipe buffer
    before reading its input would deadlock a parent that insisted on writing
    the whole payload before it started reading.
    """
    try:
        if payload:
            stream.write(payload)
        stream.flush()
    except (OSError, ValueError):  # child exited without reading its input
        pass
    finally:
        try:
            stream.close()
        except Exception:  # pragma: no cover -- already-closed pipe
            pass


def _resolve_sink(
    renderer: "LineSink | None", *, stream: bool, verbose: bool, out: Any
) -> "LineSink | None":
    """Decide where live lines go (doc 08 §4.3, §8).

    Three outcomes, one per caller intent:

    - `stream=False` -- nothing live at all. The escape hatch for a short
      probe whose output would be noise, and exactly the behaviour every call
      site had before streaming existed.
    - a supplied `renderer` -- it owns the fixed 8-line rolling window and
      its own verbose mode (doc 08 §4.3). `verbose` here would only fight it,
      so the renderer wins.
    - no renderer, `verbose=True` -- `--verbose` still means every line
      verbatim, so borrow `Progress` in its non-live mode rather than
      hand-rolling a second writer: plain lines, no escapes, per §7.
    """
    if not stream:
        return None
    if renderer is not None:
        return renderer
    if verbose:
        return progress.Progress(
            out=out if out is not None else sys.stderr,
            verbose=True,
            non_interactive=True,
        )
    return None


def run_streamed(
    command: Sequence[str],
    *,
    renderer: "LineSink | None" = None,
    out: Any = None,
    stream: bool = True,
    verbose: bool = False,
    secrets: Sequence[str] | None = None,
    label: str | None = None,
    redact_capture: bool = False,
    check: bool = False,
    timeout: float | None = None,
    input: str | None = None,
    **popen_kwargs: Any,
) -> subprocess.CompletedProcess:
    """Run `command`, streaming, capturing and recording its output in one pass.

    Doc 08 §4.2. Every line the child writes is read as it arrives, appended
    to the buffer that becomes `CompletedProcess.stdout` / `.stderr`, and
    written out once. The return value is a `CompletedProcess` populated as
    `subprocess.run(..., capture_output=True, text=True)` populates it, which
    is what lets `Context.run_root` adopt this without any of its 72 call
    sites changing.

    **Exactly one writer for the child's output.** A line is written by
    whoever owns the terminal, and by nobody else:

    - a live renderer, when one is supplied (or when `verbose` asks for the
      fallback). The renderer draws; `logs` is not called, because a second
      raw `print` to the same stream both duplicates the line and tears
      through the cursor arithmetic the live region is redrawn with.
    - `logs.log`, when nothing is streaming but suppression was not asked
      for. That is stderr plus the transcript, exactly what this module did
      before streaming existed, so `run_bundled_script` keeps the behaviour
      it has always had.
    - nobody, under `stream=False`. The escape hatch for a probe whose
      output would be noise: it is shown to no one, captured and returned.

    **The transcript is not one of those writers** (doc 08 §7.1). It is a
    file, not the terminal, so recording a line there duplicates nothing and
    disturbs no cursor arithmetic -- and a run that shows less on screen must
    not therefore hold less in the record. Where `logs.log` writes the
    terminal it appends to the transcript as it always did; on the other two
    paths `logs.transcript` appends without printing, so a line the renderer
    drew and a line `stream=False` showed nobody are both in the record an
    operator reads after a failed install.

    **Redaction comes first.** Each line is scrubbed by value and by key (see
    the module docstring) before it reaches the terminal or the transcript.
    A secret must not become visible merely because output is now shown live.
    The returned buffer is deliberately *not* scrubbed by default: §4.2 names
    the terminal and the transcript, the buffer has never been redacted, and
    scrubbing it silently corrupts what 72 call sites parse -- an inherited
    `NGC_API_KEY=abc` would turn a child's `version abc123 build` into
    `version <redacted>123 build`. Pass `redact_capture=True` for a caller
    that would rather lose fidelity than hold the plaintext.

    `secrets=None` derives the secret values from the child's own environment
    (or from this process's, when the child inherits it), so an inherited
    `NGC_API_KEY` is covered without the caller having to say so.

    `check`, `timeout` and `input` behave as they do on `subprocess.run`,
    including raising `CalledProcessError` and `TimeoutExpired` carrying the
    output collected so far. `stdout=` and `stderr=` are rejected, since the
    runner owns both pipes; `capture_output`, `text` and `universal_newlines`
    are accepted and ignored, since both are implied. Every other keyword
    goes through to `Popen`.

    **Undecodable output is replaced, not fatal** -- the one deliberate
    divergence from `subprocess.run`, which raises `UnicodeDecodeError` and
    hands back nothing at all. `apt` and `dpkg` emit non-UTF-8 bytes in
    package descriptions and maintainer names as a matter of routine, and
    killing a half-finished install over a byte in a description is the
    failure doc 08 §2 exists to prevent. Parity is unaffected for every run
    that returns: this only differs where `subprocess.run` would have raised
    instead of returning. Pass `errors="strict"` to get that behaviour back.
    """
    for owned in ("stdout", "stderr"):
        if owned in popen_kwargs:
            raise ValueError(
                f"run_streamed owns the child's {owned} pipe; pass stream=False "
                "for a command whose output should not be shown"
            )
    if input is not None and popen_kwargs.get("stdin") is not None:
        # `subprocess.run` raises ValueError for this; matching it keeps the
        # seam's error behaviour identical to what it replaces.
        raise ValueError("stdin and input arguments may not both be used")
    # Implied by the tee itself rather than rejected, so a call site that
    # already spells `capture_output=True, text=True` needs no edit.
    for implied in ("capture_output", "text", "universal_newlines"):
        popen_kwargs.pop(implied, None)
    popen_kwargs.setdefault("errors", "replace")

    if secrets is None:
        child_env = popen_kwargs.get("env")
        secrets = _secret_values(child_env if child_env is not None else os.environ)
    if label is None:
        label = os.path.basename(str(command[0])) if command else "command"

    sink = _resolve_sink(renderer, stream=stream, verbose=verbose, out=out)
    # `logs.log` writes the terminal and the transcript together, so it is
    # only safe to call when nothing else is writing the terminal, and only
    # when the caller did not ask for silence. Where it is not called, the
    # transcript-only sink takes the line instead -- see `_consume`.
    record: dict | None = None
    if sink is None and stream:
        record = {name: getattr(log, level) for name, level in _LEVELS.items()}
    stdin = subprocess.PIPE if input is not None else popen_kwargs.pop("stdin", None)

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=stdin,
        text=True,
        **popen_kwargs,
    )

    buffers: dict = {_STDOUT: [], _STDERR: []}
    lines: "queue.Queue" = queue.Queue()
    readers = [
        threading.Thread(
            target=_pump,
            args=(pipe, name, lines),
            name=f"shellout-{name}",
            daemon=True,
        )
        for pipe, name in ((proc.stdout, _STDOUT), (proc.stderr, _STDERR))
    ]
    for reader in readers:
        reader.start()
    if input is not None:
        threading.Thread(
            target=_feed, args=(proc.stdin, input), name="shellout-stdin", daemon=True
        ).start()
    elif proc.stdin is not None:
        # A caller-supplied `stdin=PIPE` with nothing to write would leave the
        # child waiting on input that is never coming; `subprocess.run` closes
        # it for the same reason.
        proc.stdin.close()

    def _captured():
        return "".join(buffers[_STDOUT]), "".join(buffers[_STDERR])

    def _consume(name: str, raw: str) -> None:
        scrubbed = _scrub(raw, secrets or ())
        # The buffer keeps the child's own text, redaction opted into rather
        # than imposed: 72 call sites parse it, and a parser is entitled to
        # exactly what `subprocess.run(capture_output=True, text=True)` would
        # have handed it -- escapes, tabs and all. Only the destinations that
        # render text get the scrubbed, sanitised form.
        buffers[name].append(scrubbed if redact_capture else raw)
        # `progress.sanitise` is the one definition of "clean" (doc 08 §7):
        # this module deliberately keeps no stripper of its own. The renderer
        # sanitises its own input too, and that second pass is a no-op on an
        # already-clean string.
        rendered = progress.sanitise(scrubbed)
        recorded = f"[{label}] {rendered}"
        if sink is not None:
            sink.line(rendered)
        if record is not None:
            # Terminal and transcript in one call, as before.
            record[name](recorded)
        else:
            # Nothing wrote the terminal that `logs.log` may write too -- the
            # renderer owns it, or `stream=False` asked for silence. Silence
            # was asked of the screen, never of the record (doc 08 §7.1).
            transcript(_LEVELS[name], recorded)

    def _drain() -> None:
        """Consume whatever the readers queued before the child was killed.

        A line the child managed to emit is evidence about why it wedged
        (doc 08 §6.1), so a timeout keeps it rather than dropping it on the
        floor with the queue.
        """
        while True:
            try:
                name, raw = lines.get_nowait()
            except queue.Empty:
                return
            if raw is None or isinstance(raw, BaseException):
                continue
            _consume(name, raw)

    deadline = None if timeout is None else time.monotonic() + timeout
    pending = len(readers)
    try:
        while pending:
            if deadline is None:
                name, raw = lines.get()
            else:
                remaining = deadline - time.monotonic()
                try:
                    if remaining <= 0:
                        raise queue.Empty
                    name, raw = lines.get(timeout=remaining)
                except queue.Empty:
                    raise subprocess.TimeoutExpired(command, timeout) from None
            if raw is None:
                pending -= 1
                continue
            if isinstance(raw, BaseException):
                # A reader died. Surface it instead of letting its stream look
                # like a clean EOF, which would report success with truncated
                # output and send a step down the wrong branch.
                raise raw
            _consume(name, raw)

        # Both pipes are at EOF by here, so the child has closed them and
        # `wait` is not where this hangs. It still gets the deadline, because
        # a child that closes its output and then sleeps is a real shape.
        returncode = proc.wait(
            timeout=None if deadline is None else max(deadline - time.monotonic(), 0.0)
        )
    except subprocess.TimeoutExpired:
        # Re-raised carrying what the child managed to say, the way
        # `subprocess.run` does: a wedged command is diagnosable only from the
        # output it produced before it wedged.
        _abandon(proc, readers)
        _drain()
        stdout, stderr = _captured()
        raise subprocess.TimeoutExpired(
            command, timeout, output=stdout, stderr=stderr
        ) from None
    except BaseException:
        # Any other non-completion -- a dead reader, a Ctrl-C, a renderer that
        # raised -- must not leave the child running behind the installer's
        # back.
        _abandon(proc, readers)
        raise

    for reader in readers:
        reader.join(timeout=_REAP_TIMEOUT_S)

    stdout, stderr = _captured()
    if check and returncode:
        raise subprocess.CalledProcessError(returncode, command, stdout, stderr)
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _abandon(proc: subprocess.Popen, readers: Sequence[threading.Thread]) -> None:
    """Kill a timed-out child and let its readers drain to EOF.

    `subprocess.run` leaves no orphan behind on a timeout and neither does
    this. Killing the child closes its pipes, which is what lets the reader
    threads finish rather than linger on a `readline` that would never
    return.
    """
    proc.kill()
    try:
        proc.wait(timeout=_REAP_TIMEOUT_S)
    except subprocess.TimeoutExpired:  # pragma: no cover -- unkillable child
        pass
    for reader in readers:
        reader.join(timeout=_REAP_TIMEOUT_S)


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------


def _apply_tree_modes(root: pathlib.Path) -> None:
    """``chmod`` a freshly staged tree: 0755 for ``*.sh``, 0644 otherwise.

    Bundled asset modes are not reliably preserved through PyInstaller's
    ``datas`` packing, so the staged copy is given its modes explicitly
    rather than inherited from whatever landed in ``sys._MEIPASS``.
    """
    root.chmod(_DIR_MODE)
    for entry in sorted(root.rglob("*")):
        if entry.is_dir():
            entry.chmod(_DIR_MODE)
        elif entry.suffix == ".sh":
            entry.chmod(_SCRIPT_MODE)
        else:
            entry.chmod(_DATA_MODE)


def stage_assets(*parts: str, prefix: str = STAGE_PREFIX) -> pathlib.Path:
    """Copy a bundled asset *directory* out to a fresh staging directory.

    Per §4.2 a fragment is never executed straight out of ``sys._MEIPASS``.
    For a fragment that ``source``s a sibling (``lib/common.sh``) or reads a
    neighbouring data file, copying the single script is not enough -- the
    whole directory has to land together, keeping its internal layout.

    ``parts`` names the directory relative to the assets root, exactly as
    ``asset_path`` takes it, and the tree is re-created at that same relative
    path under the returned staging directory. Returns the staging directory
    itself -- the thing to hand to ``shutil.rmtree`` when done -- with the
    requested tree at ``<returned>/<parts...>``.

    **Only the requested subtree is copied.** ``stage_assets("scripts")``
    produces a staging dir containing ``scripts/`` and nothing else, so a
    fragment reaching outside its own directory (``../mosquitto/mv3dt.conf``)
    gets ENOENT. Call ``stage_assets()`` with no parts to stage the whole
    ``assets/`` tree when a fragment needs a sibling subtree; that is the
    only mode in which cross-subtree paths resolve as they do inside the
    bundle.

    This function sets no environment variable. ``MV3DT_ASSET_ROOT`` is
    exported by ``run_bundled_script``, which is where a child process
    exists to receive it.
    """
    source = asset_path(*parts)
    if not source.is_dir():
        raise NotADirectoryError(f"bundled asset tree not found: {source}")

    stage_root = pathlib.Path(tempfile.mkdtemp(prefix=prefix))
    destination = stage_root.joinpath(*parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # With no `parts` the destination *is* the freshly-made staging dir, so
    # copytree has to merge into it rather than refuse an existing target.
    shutil.copytree(source, destination, dirs_exist_ok=not parts)
    _apply_tree_modes(destination)
    return stage_root


def _stage_single_file(
    *asset_parts: str, prefix: str
) -> tuple[pathlib.Path, pathlib.Path]:
    """Copy one bundled fragment out to a fresh staging dir and chmod it.

    Backward-compatible path for ``run_bundled_script`` calls that pass no
    ``tree``: the fragment stands alone and needs nothing beside it.
    """
    source = asset_path(*asset_parts)
    payload = source.read_bytes()  # read first, so a missing asset leaks no tempdir
    stage_root = pathlib.Path(tempfile.mkdtemp(prefix=prefix))
    script = stage_root / source.name
    script.write_bytes(payload)
    script.chmod(_SCRIPT_MODE)
    return stage_root, script


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_bundled_script(
    *asset_parts: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    tree: tuple[str, ...] | None = None,
    inherit_env: bool = True,
    cleanup: bool = True,
    prefix: str = STAGE_PREFIX,
    renderer: "LineSink | None" = None,
    stream: bool = True,
    verbose: bool = False,
) -> subprocess.CompletedProcess:
    """Stage a bundled bash fragment out to a temp dir and execute it.

    ``asset_parts`` names the fragment relative to the assets root.

    ``tree`` names an ancestor directory of the fragment, also relative to
    the assets root; when given, that whole directory is staged with
    ``stage_assets`` and the fragment runs from inside the staged copy, with
    ``cwd`` set to the staged copy of ``tree``. ``asset_parts`` must start
    with ``tree``. Use ``tree=()`` to stage the entire ``assets/`` tree --
    the mode a fragment needs if it reads a sibling subtree, since a partial
    stage copies only the subtree named. ``tree=None`` (the default) keeps
    the original single-file copy behaviour.

    ``MV3DT_ASSET_ROOT`` is exported to the child whenever a tree is staged,
    and always means **the root of the staged assets tree**, whichever
    subtree was staged. So ``$MV3DT_ASSET_ROOT/scripts/lib/common.sh`` is the
    correct spelling under both ``tree=("scripts",)`` and ``tree=()``; what
    differs between them is only what else exists beside ``scripts/``. A
    value for it supplied in ``env`` is ignored, with a warning -- the real
    staging path is the only correct answer, and a stale one sends the
    fragment somewhere that no longer exists.

    ``env`` is an already-prepared environment mapping (e.g. produced by
    ``privilege.py``, §9). By default (``inherit_env=True``) it is merged
    *over* ``os.environ`` rather than replacing it -- replacing outright
    strips ``PATH`` and breaks every ``command -v`` in the staged bash.
    ``inherit_env=False`` gives the hermetic replace semantics, in which
    case ``env`` (plus ``MV3DT_ASSET_ROOT`` when a tree is staged) is the
    complete child environment.

    ``cleanup=True`` (the default) removes the staging directory in a
    ``finally``. Pass ``cleanup=False`` to leave it in place for post-run
    debugging of the executed fragment.

    ``renderer``, ``stream`` and ``verbose`` are handed straight to
    ``run_streamed``, which is what actually executes the fragment. With no
    ``renderer`` -- the default, since ``run_bundled_script`` has no
    ``Context`` and so no progress handle of its own to reach for -- the
    fragment's output is written by ``logs.log`` exactly as it always was.
    Passing a ``renderer`` moves that job to it, and ``stream=False`` drops
    the fragment's output from the transcript along with the terminal, so
    reserve it for a probe whose output is noise.

    **One behaviour did change with streaming**: the fragment's output lines
    now reach the transcript interleaved, in the order the fragment actually
    produced them, where they used to be all of stdout followed by all of
    stderr. Arrival order is the more faithful record -- it shows which
    warning belongs to which step of the fragment -- and reconstructing the
    old grouping would mean buffering the whole run, which is the thing
    streaming exists to stop doing.

    The command line, the explicit environment overrides, and both output
    streams are written to the transcript per §8.2. Secrets are scrubbed by
    value and by key with the guarantees and limits spelled out in this
    module's docstring; note that transformed output (a base64-encoded key,
    say) is beyond any scrubber's reach. The ``CompletedProcess`` returned
    to the caller is unredacted, since callers parse it.
    """
    if tree is not None:
        tree_parts = tuple(tree)
        if asset_parts[: len(tree_parts)] != tree_parts:
            raise ValueError(
                f"asset_parts {asset_parts!r} is not inside tree {tree_parts!r}"
            )
        if not asset_parts[len(tree_parts) :]:
            raise ValueError(
                "asset_parts must name a fragment inside tree, not the tree itself"
            )
        stage_root = stage_assets(*tree_parts, prefix=prefix)
        script = stage_root.joinpath(*asset_parts)
        cwd: str | None = str(stage_root.joinpath(*tree_parts))
        asset_root: str | None = str(stage_root)
    else:
        stage_root, script = _stage_single_file(*asset_parts, prefix=prefix)
        cwd = None
        asset_root = None

    child_env: dict[str, str] = dict(os.environ) if inherit_env else {}
    if env:
        child_env.update(env)
    if asset_root is not None:
        # Assigned after the overlay, deliberately: the staged path is a fact
        # about this run, not a preference the caller gets to state.
        supplied = child_env.get(ASSET_ROOT_ENV)
        if supplied is not None and supplied != asset_root:
            log.warn(
                f"shellout: ignoring supplied {ASSET_ROOT_ENV}={supplied}; "
                f"using the staged tree at {asset_root}"
            )
        child_env[ASSET_ROOT_ENV] = asset_root

    command = [str(script), *(args or [])]
    secrets = _secret_values(child_env)

    try:
        log.info(f"shellout: {_command_dump(command, secrets)}")
        if env:
            log.info(f"shellout env: {_env_dump(env, secrets)}")
        result = run_streamed(
            command,
            env=child_env,
            cwd=cwd,
            renderer=renderer,
            stream=stream,
            verbose=verbose,
            secrets=secrets,
            label=script.name,
            # The default, restated because this function's documented
            # contract depends on it: the fragment's real output comes back
            # so callers can parse it. The terminal and the transcript are
            # scrubbed either way.
            redact_capture=False,
        )
        log.info(f"shellout: {script.name} exited {result.returncode}")
        return result
    finally:
        if cleanup:
            shutil.rmtree(stage_root, ignore_errors=True)
