# 08 — Install Progress and Observability (owner: DevA)

Status: shared foundation, not a step spec. It extends the logging and
reporting contract in
[`00` §8](00-FRAMEWORK-AND-BOOTSTRAP.md#8-verbose-logging--reporting-contract)
and is consumed by every step module; it does **not** restate the step
lifecycle, the state machine, or the privilege model — link back to
[`00`](00-FRAMEWORK-AND-BOOTSTRAP.md) for those. Numbered `08` rather than
`STEP-8` deliberately: it adds no step to the dispatch loop.

This doc specifies how the installer **shows what it is doing while it does
it**. Today it is silent for minutes at a time: 72 call sites pass
`capture_output=True`, so a multi-gigabyte apt transaction, a 400MB
download, and a kernel-module build all present identically as a frozen
terminal. The installer is a **single non-interactive binary an operator
runs once on a bare workstation** ([`00`
§4.1](00-FRAMEWORK-AND-BOOTSTRAP.md#41-what-builds-the-binary)), so the
terminal is the only channel it has, and silence on that channel is
indistinguishable from failure.

Every requirement here is traceable to a **real hiccup observed on the
workstation during the 0.1.2–0.1.9 install runs**, not to a general wish for
nicer output. Section 2 is that inventory.

---

## 1. Scope

**In scope:**

- A new framework module, `mv3dt_installer/progress.py`, owning all
  progress rendering.
- Streaming command output (§4) through the one seam every step already
  uses, `ctx.run_root`.
- Absolute position in the install — "Step 3 of 7, phase 2 of 5" (§3).
- Real percentages where a denominator genuinely exists, and honest
  elapsed-time reporting where it does not (§5).
- Failure and refusal messages that carry their evidence (§6).
- Behaviour off a tty, and in the transcript (§7).

**Out of scope** (see [§11](#11-out-of-scope--open-decisions)): changing any
step's install logic, the state-machine schema, the `00` §8.3 reporting
strings, or the set of steps.

### 1.1 Module identity

- Module: `installer/mv3dt_installer/progress.py`
- Framework-only, following the `waitui.py` precedent: imports `logs` and
  nothing from `steps/` or `privilege.py`.
- Consumed via `Context` (§9), never imported directly by a step module.

---

## 2. Observed failures this doc exists to fix

Each row is a real event from the workstation install, with the behaviour
that produced it.

| # | What the operator saw | Actual state | Cause |
|---|---|---|---|
| 1 | Black screen, ten minutes, no output | Installer already dead | Display manager stopped, killing the terminal; nothing left to report |
| 2 | Black screen, spinner, no progress | `.run` compiling normally | `capture_output=True` on the driver runfile |
| 3 | "Hangs" after `Invoking user:` | apt downloading several GB | `capture_output=True` on the TensorRT transaction |
| 4 | `shellout env: MV3DT_INSTALLER_CONF` | Mosquitto setup, Step 1 of 7 | No step/phase banner; internal detail surfaced as the only signal |
| 5 | "would kill this installer" on a real tty | Guard misfired | Refusal stated a verdict, not the evidence behind it |

Events 2 and 3 each caused the operator to interrupt work that was
proceeding normally. Event 3 interrupted apt mid-transaction, which required
`dpkg --configure -a` to recover. **Silence is not a cosmetic problem: it
causes operators to break working installs.**

> **The ordering constraint this implies (REQUIRED):** any output an
> operator needs in order to *not* intervene must be emitted **before** the
> disruptive action, not after it. Event 1 is the proof — after the display
> manager stops there is no process left to explain anything.

---

## 3. The progress model

Three nested levels. Only the first two are ever rendered as "N of M".

| Level | Source of truth | Denominator | Rendering |
|-------|-----------------|-------------|-----------|
| Step | `STEP_REGISTRY` order | 7, known at startup | `Step 3/7` |
| Phase | Step declares its own (§3.1) | Known when the step starts | `phase 2/5` |
| Task | Individual command | Unknown | Name + live output or elapsed |

### 3.1 Steps declare their phases (REQUIRED)

A step module gains one optional class attribute:

```python
class Step1Prerequisites:
    id = "step1_prerequisites"
    title = "Prerequisites (driver / CUDA / cuDNN / TensorRT / GStreamer)"
    order = 1
    phases = (
        "base packages",
        "CUDA repo and toolkit",
        "nouveau and distro driver cleanup",
        "NVIDIA driver runfile",
        "TensorRT and cuDNN",
        "Mosquitto broker",
    )
```

`phases` is a plain tuple of short labels, declarative and static — it is
**not** a control-flow mechanism and does not change how `run()` executes.
A step that omits it renders as a single unnamed phase, so this is
backward-compatible across all seven steps and can be adopted one step at a
time.

`run()` announces its position by calling `ctx.progress.phase(n)`; the
renderer pairs that index with the declared label.

### 3.2 The banner

Emitted by the dispatch loop when a step begins, and again on every phase
change:

```
[ 1/7 ] Prerequisites (driver / CUDA / cuDNN / TensorRT / GStreamer)

  ✓ base packages                        12s
  ✓ CUDA repo and toolkit             3m41s
  ✓ nouveau and distro cleanup            4s
  ✓ NVIDIA driver runfile             6m02s
  ▸ TensorRT and cuDNN
      ████████████░░░░░  68%   412 MB / 606 MB   14.2 MB/s   0:14
        Setting up libnvinfer10 (10.16.0.72-1+cuda13.2)
        Setting up tensorrt-dev (10.16.0.72-1+cuda13.2)
```

Two things the current output never states: which step of how many, and how
far into that step. Event 4 in §2 is exactly this gap — the operator's only
signal was an internal environment-variable name.

**Completed phases collapse to one line (LOCKED):** a tick, the phase label,
and the wall-clock duration it took. The finished phases stay on screen as a
running record of the install, so an operator can see at a glance where the
time went and which phase a later failure followed. Only the active phase
carries a bar and a log window.

### 3.3 Phase changes are logged, not just drawn

Every phase transition emits one `logs.log.info` line in addition to
redrawing the banner. The transcript therefore carries the full phase
sequence of a run even though it carries no live rendering (§7), which is
what makes a post-mortem of a failed install possible.

---

## 4. Streaming command output

### 4.1 The one seam (LOCKED)

All 72 `capture_output=True` call sites reach the subprocess through
`Context.run_root` (`app.py`). Steps therefore do **not** change: `run_root`
keeps its signature and keeps returning a `subprocess.CompletedProcess` with
`.stdout` and `.stderr` populated, and gains streaming as an internal
behaviour.

```python
def run_root(self, *args: str, stream: bool | None = None, **kwargs: Any)
        -> subprocess.CompletedProcess:
```

`stream=None` (the default) means **auto**: stream when stderr is a tty and
the caller asked for captured output. `stream=False` restores today's
behaviour for a probe whose output would be noise.

> **Unsettled, owned by [U11](#12-unit-and-wave-decomposition):** this
> tty-gating and §7's table disagree. §7 says a non-tty run still gets
> per-line output, plain; auto as written streams nothing off a tty, so
> `mv3dt-installer | tee install.log` shows the operator nothing while it
> runs. The transcript still receives everything, so no record is lost, but
> "piped to a file" is a normal way to run an installer and silence there
> repeats the failure §2 documents. Settle it in U11, which owns `app.py`
> next: either auto streams plain lines off a tty, or §7's table is narrowed
> to mean the transcript only. Do not resolve it by editing a step.

This is the decision that makes the change tractable. Rewriting 72 call
sites would be seven PRs of mechanical edits across every step module;
changing one method is one reviewable unit, and every existing test that
asserts on `.stdout` keeps passing unchanged.

### 4.2 Tee semantics (REQUIRED)

When streaming, `run_root`:

1. Runs the child with `stdout=PIPE, stderr=PIPE`.
2. Reads both streams line by line as they arrive.
3. Writes each line to the terminal, indented and dimmed, under the current
   phase banner.
4. Appends each line to the in-memory buffer that becomes
   `CompletedProcess.stdout` / `.stderr`. **This buffer is deliberately not
   scrubbed**, and that is not an oversight in the list above: §4.1 requires
   it to be byte-identical to what `subprocess.run` would have returned, and
   72 call sites parse it. Scrubbing is substring replacement against
   environment values, so a child printing `version abc123 build` with
   `NGC_API_KEY=abc` set would come back as `version <redacted>123 build` and
   silently break every one of those parsers. The buffer is in-memory and
   reaches no durable artifact on its own; the destinations that persist —
   the terminal and the transcript — are always scrubbed.
5. Appends each line to the transcript, honouring the existing redaction
   rules in [`shellout.py`](../mv3dt_installer/shellout.py).

Redaction is **not** re-implemented here. `shellout.py` already defines
value-based and key-based scrubbing for `_REDACT_KEYS`; the streaming
writer routes every line through that same scrubber before it reaches
either the terminal or the transcript. A secret must not become visible
merely because output is now shown live.

### 4.3 Output volume

Streaming apt verbatim is thousands of lines. Two controls, both in §8:

- Default verbosity keeps the **last N lines** (N = 8) in a scrolling
  region, so the terminal shows live activity without the transcript's full
  volume.
- `--verbose` streams everything verbatim.

The apt status adapter is the narrow exception to verbatim derived output:
it removes the changing `(<duration> remaining)` suffix from `dlstatus`
`Retrieving file N of M` descriptions. The percentage bar already reports
that movement, while each file transition remains a distinct line. Apt's
human `Get:` output, package-status descriptions, and errors are unchanged.

---

## 5. Percentages, honestly

A progress bar that does not correspond to real work is worse than no bar:
it teaches the operator to distrust the one signal they have. Percentages
are therefore rendered **only** where a true denominator exists.

| Operation | Denominator available? | Rendering |
|-----------|------------------------|-----------|
| File download (driver runfile, SDK tarball) | Yes — `Content-Length` | Bar + percent + rate + ETA |
| apt transaction | Yes — `APT::Status-Fd` | Bar + percent + current package |
| Docker image pull | Yes — layer progress | Bar + percent |
| Step / phase position | Yes — declared counts | `3/7`, `2/5` |
| Kernel module build (`.run`) | **No** | Spinner + elapsed + last output line |
| `dpkg --configure`, `update-initramfs` | **No** | Spinner + elapsed |

### 5.1 Downloads

The installer already writes downloads to a `.part` path
([`STEP-1` §5.2](STEP-1-PREREQUISITES.md#52-runfile-acquisition-installer-fetched-resolved)),
so percentage is `size(.part) / Content-Length` polled on an interval — no
parsing of `curl`'s own output, and it works identically for any future
download. `curl -fL` gains `-w` for the final byte count, used to confirm
the transfer completed rather than to drive the bar.

### 5.2 apt

`apt-get` reports machine-readable progress on a caller-supplied descriptor:

```bash
apt-get install -o APT::Status-Fd=<fd> ...
```

Lines are `pmstatus:<package>:<percent>:<description>`. That percent is
apt's own, covering unpack and configure as well as download, and is the
correct denominator for event 3 in §2 — the failure that caused an
interrupted transaction.

`dlstatus` descriptions may append an ETA that changes once per second. The
adapter records the stable `Retrieving file N of M` event once and uses the
reported percentage for subsequent movement; it does not alter `pmstatus`
or `pmerror` descriptions.

### 5.3 No fabricated progress (REQUIRED)

Where the table above says "No", the renderer shows elapsed time and the
most recent output line, and **never** a bar. An operation whose duration
cannot be known is reported as such:

```
        installing NVIDIA driver 595.58.03   4m12s   [ building kernel module ]
```

---

## 6. Failure and refusal messages carry their evidence

Event 5 in §2: a guard refused an operator who was doing exactly what it
asked, stating its verdict and none of its inputs. The operator's only
remaining move was an override flag.

**REQUIRED for every `USER_ACTION_REQUIRED` and `FAILED` result whose
message is the product of an inference:** the message names the inputs the
inference was drawn from.

```
the driver installer must stop the desktop session, which would kill this
installer along with it (controlling terminal: /dev/pts/1; sshd ancestor:
no; active display manager: gdm3)
```

This is already implemented for the caveat-6a guard
([`STEP-1` §5.1a](STEP-1-PREREQUISITES.md#51a-desktop-safe-systemd-handoff-resolved));
this section generalises it to every inferred refusal across steps 1-7.

### 6.1 Failure context block

On `FAILED`, the installer prints a block naming: the step and phase, the
exact command that failed, its exit code, the last 20 lines of its output,
and the transcript path. Today a failed command's output is captured into a
`CompletedProcess` that the step discards, so the operator is told a step
failed and nothing about why.

---

## 7. Behaviour off a tty (REQUIRED)

The renderer is the only thing that changes; the transcript never does.

| Context | Bars / spinners | Per-line output | Phase lines |
|---------|-----------------|-----------------|-------------|
| Interactive tty | Yes, redrawn in place | Yes, scrolling region | Yes |
| Not a tty (pipe, CI, `tee`) | No — suppressed entirely | Yes, plain | Yes |
| `--non-interactive` | No | Yes, plain | Yes |
| Transcript file | Never | Always, unredacted control chars stripped | Always |

No ANSI escape ever reaches the transcript. This is the existing rule in
[`00` §8.2](00-FRAMEWORK-AND-BOOTSTRAP.md#82-transcript-log-file) and
`logs.py` already enforces it for log lines; §4.2's writer honours it for
streamed output.

### 7.1 Live rendering must never cost the transcript (REQUIRED)

`logs._emit` writes to stderr **and** appends to the transcript from a single
call: there is no transcript-only sink. Any code that suppresses a log line
to protect the live region therefore deletes it from the auditable record as
well.

That trade is not acceptable, and it bites hardest in the one case this doc
exists for. A long download on an interactive tty draws a bar, so a periodic
plain summary would be redundant on screen and is naturally suppressed — but
suppressing it leaves the transcript with the task name, then nothing for the
duration of a multi-hundred-megabyte transfer, then the phase-done line.
Interactive runs are exactly the ones an operator performs by hand and later
asks about.

**REQUIRED:** `logs.py` grows a transcript-only write, and anything that
suppresses a line for rendering reasons uses it instead of dropping the line.
The screen may show less than the transcript. The transcript may never show
less than the screen.

> **U12 is a release blocker, and it lands before U4.** The gap opens the
> moment `run_root` streams, not at release, so U12 is resequenced ahead of
> U4 rather than left to wave 5: the loss then never exists on `main`, even
> transiently.
>
> **Why it is a blocker at all.** The tee runner
> (§4.2) must route each line to exactly one writer, or a line renders twice
> and the raw copy tears through the live region's cursor arithmetic. With
> no transcript-only sink, "exactly one writer" means a streamed line
> reaches the renderer *instead of* the transcript. That is invisible while
> nothing streams, and becomes a real loss the moment `run_root` starts
> streaming — the transcript would then hold the phase sequence and the
> reporting strings, but none of the command output an operator actually
> needs to diagnose a failed install. **No release may ship streaming
> without U12.** The two `record` call sites in the runner are where the
> sink gets used.

A related consistency rule: where a value is clamped for display (§5.1's
overshoot case renders 100 percent of the declared total), the transcript
must not mix the clamped and unclamped forms in the same run. Record the true
number, and say it is the true number.

---

## 8. Verbosity

| Flag | Command output | Bars | Use |
|------|----------------|------|-----|
| *(default)* | Last 8 lines, rolling | Yes | Normal install |
| `--verbose` | Everything, verbatim | Yes | Debugging a failure |

The rolling window is **8 lines in every phase** (LOCKED) — a fixed height
rather than one that grows for download-heavy phases, so the layout does not
shift underneath the operator as the install moves between phases.

`--verbose` is what an operator is told to re-run with when reporting a
problem, replacing today's advice to find and tail the transcript by hand.

`--quiet` is **not** part of this batch ([§11](#11-out-of-scope--open-decisions)).

---

## 9. The step-author contract

A step module interacts with progress through `Context` only:

```python
ctx.progress.phase(2)                     # advance to declared phase 2
ctx.progress.task("downloading driver")   # name the current task
ctx.progress.bytes(done, total)           # drive a real bar
```

Rules, all REQUIRED:

- A step never writes to stdout or stderr directly.
- A step never imports `progress.py`.
- `phase(n)` indexes the step's own declared `phases` tuple (§3.1); an
  out-of-range index is a programming error and raises rather than
  rendering a wrong denominator.
- Omitting all three calls is legal and yields today's behaviour plus the
  step banner — which is what keeps this adoptable one step at a time.

---

## 10. Migration

The seam in §4.1 means no step module has to change for the streaming
benefit to land. Adoption of §3.1 phases and §9 task naming is then
per-step and independent.

1. `progress.py` and the phase API, with no call sites (U1, U2).
2. The tee runner, then `run_root` routed through it — every step gains
   live output at once (U3, U4).
3. Step banner and phase rendering in the dispatch loop (U5).
4. Real bars for downloads and apt (U6, U7).
5. Per-step `phases` declarations and task naming, one PR per step group
   (U8, U9).
6. Failure context block and inferred-refusal evidence (U10).

Steps 1-2 alone resolve events 2 and 3 in §2, the two that caused
interrupted installs. **The order is deliberate: the highest-severity
failures are fixed by the first two units, before any cosmetic work.**

---

## 11. Out of scope / open decisions

Settled exclusions:

- No change to step install logic, the state-machine schema, or the `00`
  §8.3 reporting strings.
- No TUI framework, no curses full-screen mode, no third-party dependency.
  The binary is built by PyInstaller from a dependency-light tree and stays
  that way.
- No progress persisted to `state.json`. Progress is a property of a
  running process, not of installed state.

1. **Default verbosity — RESOLVED: last 8 lines, always.** A fixed-height
   rolling window under the bar in every phase, with the full stream always
   reaching the transcript (§7). Rejected: fully-verbose-by-default (buries
   the phase banner in thousands of apt lines) and bars-only (leaves nothing
   to go on when a command is misbehaving but has not yet failed).
2. **`--quiet` — RESOLVED: not in this batch.** It has no consumer; the
   systemd units in [`STEP-6`](STEP-6-REMOTE-SUPERVISION.md) run
   non-interactively and are already covered by the non-tty rule in §7.
   U11 therefore ships `--verbose` and the doc 00 §8 update only.

Open decision for the human:

3. **Docker pull progress (§5)** depends on Step 2's install method, which
   the DS 9.1 flow may route through `docker pull` or a local `.deb`. Wire
   the bar only for whichever path Step 2 actually takes.

---

## 12. Unit and wave decomposition

| Unit | Branch | Files touched | Depends on | Wave |
|------|--------|---------------|------------|------|
| U1 Progress renderer core: bars, spinner, scrolling region, tty and non-tty modes | `feat/installer-progress-core` | `mv3dt_installer/progress.py`, `tests/test_progress.py` | — | 1 |
| U2 Phase declaration API on the step interface | `feat/installer-progress-phase-api` | `mv3dt_installer/steps/__init__.py`, `tests/test_steps_protocol.py` | — | 1 |
| U3 Tee runner: stream, capture and redact in one pass | `feat/installer-progress-streaming` | `mv3dt_installer/shellout.py`, `tests/test_shellout.py` | U1 | 2 |
| U4 `Context.progress` handle and streaming `run_root` | `feat/installer-progress-context` | `mv3dt_installer/app.py`, `tests/test_app.py` | U1, U2, U3 | 3 |
| U5a Step and phase banner in the dispatch loop | `feat/installer-progress-banner` | `mv3dt_installer/app.py`, `tests/test_app.py` | U4, U12 | 5 |
| U5b The four defects in [§12.2](#122-known-defects-u5-carries) | `feat/installer-progress-defects` | `mv3dt_installer/progress.py`, `mv3dt_installer/logs.py`, their tests | U12, U5a | 6 |
| U6 Download byte-progress adapter | `feat/installer-progress-downloads` | `mv3dt_installer/progress.py`, `tests/test_progress.py` | U1, U3 | 3 |
| U7 apt `Status-Fd` percentage adapter | `feat/installer-progress-apt` | `mv3dt_installer/progress.py`, `tests/test_progress.py` | U6 | 5 |
| U8 Phases and task naming, Steps 1-3 | `feat/installer-progress-steps-1-3` | `mv3dt_installer/steps/step1_prerequisites.py`, `step2_deepstream_sdk.py`, `step3_amc_launcher.py`, their tests | U5a, U7 | 7 |
| U9 Phases and task naming, Steps 4-7 | `feat/installer-progress-steps-4-7` | `mv3dt_installer/steps/step4_calib_output_wiring.py`, `step5_per_project_exes.py`, `step6_remote_supervision.py`, `step7_webapp_integration.py`, their tests | U5a, U7 | 7 |
| U10 Failure context block and inferred-refusal evidence | `feat/installer-progress-failure-context` | `mv3dt_installer/report.py`, `tests/test_report.py` | U5a | 6 |
| U11 Verbosity flag and doc 00 section 8 update | `feat/installer-progress-verbosity` | `mv3dt_installer/app.py`, `installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md`, `tests/test_app.py` | U5a | 6 |
| U13 Plumb the parsed `--non-interactive` through to `logs.set_colour` | `feat/installer-progress-colour-plumbing` | `mv3dt_installer/app.py`, `tests/test_app.py`, `tests/test_logs.py` | U5b, U11 | 8 |
| U12 Transcript-only sink, so live rendering never costs the record | `feat/installer-progress-transcript-sink` | `mv3dt_installer/logs.py`, `tests/test_logs.py`, `mv3dt_installer/shellout.py`, `tests/test_shellout.py`, `mv3dt_installer/progress.py`, `tests/test_progress.py` | U3 | 4 |

### 12.2 Known defects U5 carries

Each was found in review of a unit that could not fix them, so they are
recorded here rather than left in a PR comment. Defects 3 and 4 make U5b a
release blocker alongside U12: U5a's banner is what makes defect 3
reachable, so U5a must not reach `main` ahead of the fix.

1. **The step banner prints twice on a tty.** `Progress.begin_step` calls
   `log.info(banner)` and then writes the same banner to its own output
   stream. When that stream is stderr, which is the default and what the
   installer uses, the operator sees it twice. Measured, not inferred. The
   one-writer rule in §4.2 applies here as much as to a streamed line: the
   transcript copy should go through the transcript-only sink U12 adds
   (§7.1), leaving exactly one write to the screen.
2. **`logs._ANSI_RE` is narrower than `progress.sanitise`.** It matches
   plain CSI and OSC but misses lone `ESC`, C1 forms (`\x9b`, `\x9d`), bare
   `BEL`, `CR`, backspace, `NUL` and `DEL`. It is safe today because both of
   U12's callers pre-sanitise, but `_emit` shares the same strip, and two
   step call sites pass raw command output straight into `log.info`
   (`step1_prerequisites.py` for `nvidia-smi`, `step2_deepstream_sdk.py` for
   an stderr tail), so those still carry C1 and `CR` into the transcript.
   `progress` imports `logs`, so `sanitise` cannot be reused without a
   cycle; widening the local class to `[\x00-\x08\x0b-\x1f\x7f-\x9f]`
   alongside the escape pattern closes it. The comment above it currently
   claims a guarantee the regex does not deliver, which is the part that
   most needs fixing: a wrong comment outlives a narrow regex.
3. **A foreign write invalidates the erase arithmetic.** The renderer
   erases its region by counting the rows it drew and moving the cursor up
   that many. Any write to stderr that does not go through the renderer
   moves the cursor without it knowing, so the count is short by exactly
   that many rows and the clear-below eats real output. The seven step
   modules alone hold 121 direct `log.info` / `log.warn` / `log.error`
   calls (7 in step 1, 45 in step 5), so on a tty install step 5 loses the
   last rows of its own output under a stale region. Found in review of
   U5a, which makes it reachable: on `main` no step is ever begun, `_drawn`
   never leaves 0, and `_erase` is inert.

   **Do not fix this by opening the region only once a step declares a
   phase.** That was the first fix proposed and it is measurably wrong:
   with no region open, `Progress.line()` appends to the window, `_draw()`
   finds `_phase_started is None`, `_frame_lines()` returns empty, and the
   terminal receives zero bytes. No step declares a phase until U8 and U9,
   so it would make every streamed line invisible on a tty across all seven
   steps, which is the failure in §2 this document exists to remove.
   Measured on the real renderer, not inferred. The fix is the other one:
   route the bypassing writes through the renderer using the `_erase` /
   write / `_draw` sandwich `Progress.line()` already uses in verbose mode.
4. **`--non-interactive` does not suppress colour.** `logs._colour_enabled()`
   is `sys.stderr.isatty()` alone and never learns about the flag, so an
   operator running `--non-interactive` on a real terminal still gets
   `\033[32m[info ]` on every line. §7 is REQUIRED and asks for no escape
   sequence at all, so this is a live violation of it. Found while
   tightening U5a's degradation test, which had been asserting no-escape
   against a stream that was never in live mode to begin with; that
   assertion is currently split around this gap and tightens to a bare
   no-escape check once it is closed.

   **U5b's argv sniff is a pre-parse default, not the answer (U13).**
   `logs` cannot import `app`, and the colour decision is needed before
   `argparse` has run, so reading `sys.argv` is the only thing available at
   that point and it is deliberately conservative. It is not authoritative:
   it duplicates parsing `argparse` owns, it has to hand-implement
   argparse's unambiguous-prefix rule (`--non`, `--non-int`), and it is
   wrong for any caller that does not pass the installer's own argv. The
   authoritative value is `logs.set_colour(enabled)`, called once from
   `app.py` with the *parsed* flag. That call cannot be written until both
   U5b (which adds `set_colour`) and U11 (which owns `app.py`) are on
   `main`, so it is split out as **U13** rather than left as a comment
   nobody owns. Until U13 lands, an exotic spelling of the flag still
   colours its output, which costs a cosmetic escape and nothing else.

---

### 12.1 Serialization points

Three files force ordering, and the waves above encode it:

- **`app.py`** appears in U4, U5 and U11, in waves 3, 4 and 5 respectively,
  so they never open concurrently. U4 adds the `Context` field and routes
  `run_root` through the tee runner, U5 uses the handle in the dispatch
  loop, U11 adds the flag that configures it — a genuine build dependency,
  not only a file conflict.

  `Context.run_root` lives in `app.py`, so the *mechanism* it calls is
  split out deliberately: U3 builds a reusable tee runner in `shellout.py`
  (which already owns the `_REDACT_KEYS` scrubbing §4.2 requires) and U4
  wires `run_root` to it. Putting the streaming implementation in U4
  alongside the `Context` change would have been one unreviewable PR
  touching the subprocess path and the progress path at once.
- **`logs.py`** is touched only by U12, but every unit that suppresses a line
  depends on the sink it adds. U12 is therefore the last progress unit to
  land, and until it does, suppressing a line for rendering reasons is a
  known gap in the record rather than a solved problem (§7.1).
- **`progress.py`** appears in U1, U6, U7 and U12 (waves 1, 3, 4, 5). The adapters
  extend the renderer's public surface, so they must land after it exists
  and after each other: U7's apt bar reuses the byte-bar primitive U6
  introduces.
- **`steps/step1_prerequisites.py`** is touched by U8 only. U4's streaming
  change deliberately lives in the `run_root` seam so no step file is
  edited for it — that is what keeps waves 2 and 5 apart.

U8 and U9 are split by step group rather than by concern because they are
the same mechanical change applied to seven files; splitting them 3/4 keeps
each PR reviewable while letting both run in parallel in wave 5. They touch
disjoint files.

---

## References

The observed-failure inventory in §2 is drawn from the workstation install
runs of `mv3dt-installer` 0.1.2 through 0.1.9 (2026-09-13), not from an
external source. The mechanisms in §5 are verified against the tools'
documented interfaces:

- [apt `APT::Status-Fd`](https://manpages.debian.org/bookworm/apt/apt.conf.5.en.html)
  — **backs §5.2**: the `pmstatus:` percentage used to drive the apt bar.
- [`curl` `-w`/`--write-out`](https://curl.se/docs/manpage.html) —
  **backs §5.1**: final byte count used to confirm a completed transfer.
- [NVIDIA driver runfile options](https://download.nvidia.com/XFree86/Linux-x86_64/)
  — **backs §5's "no denominator" row** for the kernel-module build.

Repo files referenced:

- [`00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md) — §8 is
  the logging and reporting contract this doc extends; §4.1 is the
  single-binary distribution constraint that makes the terminal the only
  output channel.
- [`STEP-1-PREREQUISITES.md`](STEP-1-PREREQUISITES.md) — §5.1a and §5.2 are
  the two sections whose observed failures drive §2 and §6.
- [`STEP-6-REMOTE-SUPERVISION.md`](STEP-6-REMOTE-SUPERVISION.md) — the
  non-interactive systemd consumer referenced by the §11 open decision on
  `--quiet`.
- [`mv3dt_installer/shellout.py`](../mv3dt_installer/shellout.py) — owns the
  redaction rules §4.2 routes streamed output through.
- [`mv3dt_installer/waitui.py`](../mv3dt_installer/waitui.py) — the
  framework-module precedent §1.1 follows, and the existing home of
  `countdown()`.
- [`mv3dt_installer/logs.py`](../mv3dt_installer/logs.py) — the transcript
  writer whose no-ANSI rule §7 preserves.
