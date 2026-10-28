"""Tests for mv3dt_installer.ngc (doc 00 §10).

Run from installer/: `python3 -m pytest tests/test_ngc.py -v`

Never touches a real home dir or /opt/mv3dt: `install_dir` and the invoking
user's `home` are always `tmp_path` locations. `store_key`'s `os.chown`
(chowning to an arbitrary uid/gid requires root, which the test runner
doesn't have and shouldn't need) is monkeypatched out via the `_no_chown`
fixture. `configure_ngc_cli` never calls `os.chown` itself -- it goes
through `privilege.run_as_user`, which most tests here monkeypatch directly
(no real `sudo` involved); one test additionally runs the real script via
`bash` with only the `sudo` wrapper skipped, to verify the actual file
content/permissions it produces.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from mv3dt_installer import ngc, privilege  # noqa: E402

_FAKE_KEY = "nvapi-fake-key-do-not-use-1234567890"


@pytest.fixture(autouse=True)
def _force_no_colour(monkeypatch):
    """Match test_logs.py / test_privilege.py's convention."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)


@pytest.fixture(autouse=True)
def _reset_transcript_state():
    from mv3dt_installer import logs

    logs._transcript_path = None
    yield
    logs._transcript_path = None


@pytest.fixture
def _no_chown(monkeypatch):
    """os.chown requires root when changing owner; the tests only care that
    it's *attempted* with the right args, not that it succeeds for real."""
    calls = []
    monkeypatch.setattr(
        ngc.os, "chown", lambda path, uid, gid: calls.append((str(path), uid, gid))
    )
    return calls


def _fake_user(home: pathlib.Path, uid: int = 1001, gid: int = 1001, name: str = "alice"):
    return privilege.InvokingUser(name=name, home=home, uid=uid, gid=gid)


# ---------------------------------------------------------------------------
# capture_key
# ---------------------------------------------------------------------------


def test_capture_key_returns_key_when_prompt_answered(monkeypatch):
    monkeypatch.setattr(ngc.getpass, "getpass", lambda prompt="": _FAKE_KEY)

    result = ngc.capture_key(non_interactive=False)

    assert result == _FAKE_KEY


def test_capture_key_reprompts_on_blank_input(monkeypatch):
    answers = iter(["", _FAKE_KEY])
    calls = []

    def _fake_getpass(prompt=""):
        calls.append(prompt)
        return next(answers)

    monkeypatch.setattr(ngc.getpass, "getpass", _fake_getpass)

    result = ngc.capture_key(non_interactive=False)

    assert result == _FAKE_KEY
    assert len(calls) == 2


def test_capture_key_reprompts_on_whitespace_only_input(monkeypatch):
    answers = iter(["   ", _FAKE_KEY])
    monkeypatch.setattr(ngc.getpass, "getpass", lambda prompt="": next(answers))

    result = ngc.capture_key(non_interactive=False)

    assert result == _FAKE_KEY


def test_capture_key_reprompts_indefinitely_until_a_key_is_given(monkeypatch):
    answers = iter(["", "", "", _FAKE_KEY])
    calls = []

    def _fake_getpass(prompt=""):
        calls.append(prompt)
        return next(answers)

    monkeypatch.setattr(ngc.getpass, "getpass", _fake_getpass)

    result = ngc.capture_key(non_interactive=False)

    assert result == _FAKE_KEY
    assert len(calls) == 4


def test_capture_key_non_interactive_never_prompts_and_dies(monkeypatch):
    def _boom(prompt=""):
        raise AssertionError("getpass.getpass must not be called in non-interactive mode")

    monkeypatch.setattr(ngc.getpass, "getpass", _boom)

    with pytest.raises(SystemExit):
        ngc.capture_key(non_interactive=True)


# ---------------------------------------------------------------------------
# store_key
# ---------------------------------------------------------------------------


def test_store_key_writes_exact_line(tmp_path, monkeypatch, _no_chown):
    monkeypatch.setattr(ngc.privilege, "resolve", lambda: _fake_user(tmp_path / "home"))

    result_path = ngc.store_key(_FAKE_KEY, tmp_path)

    expected_path = tmp_path / "secrets" / "ngc.env"
    assert result_path == expected_path
    assert expected_path.read_text(encoding="utf-8") == f"NGC_API_KEY={_FAKE_KEY}\n"


def test_store_key_sets_file_and_dir_modes(tmp_path, monkeypatch, _no_chown):
    monkeypatch.setattr(ngc.privilege, "resolve", lambda: _fake_user(tmp_path / "home"))

    result_path = ngc.store_key(_FAKE_KEY, tmp_path)
    secrets_dir = result_path.parent

    assert (result_path.stat().st_mode & 0o777) == 0o600
    assert (secrets_dir.stat().st_mode & 0o777) == 0o700


def test_store_key_chowns_to_invoking_user(tmp_path, monkeypatch, _no_chown):
    monkeypatch.setattr(ngc.privilege, "resolve", lambda: _fake_user(tmp_path / "home", uid=4242, gid=4343))

    result_path = ngc.store_key(_FAKE_KEY, tmp_path)
    secrets_dir = result_path.parent

    assert (str(result_path), 4242, 4343) in _no_chown
    assert (str(secrets_dir), 4242, 4343) in _no_chown


# ---------------------------------------------------------------------------
# load_key
# ---------------------------------------------------------------------------


def test_load_key_round_trips_stored_key(tmp_path, monkeypatch, _no_chown):
    monkeypatch.setattr(ngc.privilege, "resolve", lambda: _fake_user(tmp_path / "home"))
    ngc.store_key(_FAKE_KEY, tmp_path)

    assert ngc.load_key(tmp_path) == _FAKE_KEY


def test_load_key_returns_none_when_no_file_exists(tmp_path):
    assert ngc.load_key(tmp_path) is None


def test_load_key_returns_none_when_file_has_no_key_line(tmp_path):
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    (secrets_dir / "ngc.env").write_text("# nothing useful here\n", encoding="utf-8")

    assert ngc.load_key(tmp_path) is None


# ---------------------------------------------------------------------------
# configure_ngc_cli
#
# doc 00 §9.2 requires anything under the user's home to run via
# privilege.run_as_user(...) (equivalent to `sudo -u "$SUDO_USER" -H ...`),
# "exactly as Phases 6/10 of 00_bootstrap.sh do for ngc" -- so
# configure_ngc_cli must never write ~/.ngc/config (or chown it) from this
# process directly. These tests assert against the run_as_user call itself.
# ---------------------------------------------------------------------------


def test_configure_ngc_cli_calls_run_as_user_not_direct_chown(tmp_path, monkeypatch, _no_chown):
    """The blocking finding this fixes: configure_ngc_cli must go through
    privilege.run_as_user(...), and must never call os.chown() itself."""
    install_dir = tmp_path / "install"
    home_dir = tmp_path / "home"
    monkeypatch.setattr(ngc.privilege, "resolve", lambda: _fake_user(home_dir))
    ngc.store_key(_FAKE_KEY, install_dir)
    # store_key legitimately chowns secrets/ngc.env -- clear that record so
    # this test only inspects chown calls made by configure_ngc_cli itself.
    _no_chown.clear()

    captured = {}

    def _fake_run_as_user(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ngc.privilege, "run_as_user", _fake_run_as_user)

    result = ngc.configure_ngc_cli(install_dir)

    assert result == home_dir / ".ngc" / "config"
    assert captured["args"], "expected configure_ngc_cli to call privilege.run_as_user"
    assert not _no_chown, (
        "configure_ngc_cli must not call os.chown() directly -- the write "
        "(and its ownership) must happen entirely inside the run_as_user "
        "child process, per doc 00 §9.2"
    )


def test_configure_ngc_cli_run_as_user_shape(tmp_path, monkeypatch, _no_chown):
    install_dir = tmp_path / "install"
    home_dir = tmp_path / "home"
    monkeypatch.setattr(ngc.privilege, "resolve", lambda: _fake_user(home_dir))
    ngc.store_key(_FAKE_KEY, install_dir)

    captured = {}

    def _fake_run_as_user(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ngc.privilege, "run_as_user", _fake_run_as_user)

    ngc.configure_ngc_cli(install_dir)

    args = captured["args"]
    secrets_path = install_dir / "secrets" / "ngc.env"

    # Mirrors 00_bootstrap.sh Phase 10's `sudo -u "$SUDO_USER" -H env
    # BOOTSTRAP_ENV_FILE=... bash -s <<INNER` shape: the key travels via an
    # env-file path handed to `env`, then sourced inside the script -- never
    # as a bare CLI argument.
    assert args[0] == "env"
    assert args[1] == f"NGC_ENV_FILE={secrets_path}"
    assert args[2] == "bash"
    assert args[3] == "-lc"
    script = args[4]
    assert 'set -a; . "$NGC_ENV_FILE"; set +a' in script
    assert 'mkdir -p "$HOME/.ngc"' in script
    assert 'chmod 700 "$HOME/.ngc"' in script
    assert "<<NGCINI" in script
    assert "[CURRENT]" in script
    assert "apikey = ${NGC_API_KEY}" in script
    assert "format_type = ascii" in script
    assert 'chmod 600 "$HOME/.ngc/config"' in script

    # The raw key must never appear as a literal anywhere in the call --
    # neither in argv nor in the script text itself.
    for arg in args:
        assert _FAKE_KEY not in arg

    assert captured["kwargs"].get("capture_output") is True
    assert captured["kwargs"].get("text") is True


def test_configure_ngc_cli_returns_none_without_stored_key(tmp_path, monkeypatch):
    install_dir = tmp_path / "install-no-key"
    home_dir = tmp_path / "home-no-key"
    monkeypatch.setattr(ngc.privilege, "resolve", lambda: _fake_user(home_dir))

    def _boom(*args, **kwargs):
        raise AssertionError("run_as_user must not be called with no stored key")

    monkeypatch.setattr(ngc.privilege, "run_as_user", _boom)

    result = ngc.configure_ngc_cli(install_dir)

    assert result is None
    assert not (home_dir / ".ngc" / "config").exists()


def test_configure_ngc_cli_returns_none_when_run_as_user_fails(tmp_path, monkeypatch, _no_chown):
    install_dir = tmp_path / "install"
    home_dir = tmp_path / "home"
    monkeypatch.setattr(ngc.privilege, "resolve", lambda: _fake_user(home_dir))
    ngc.store_key(_FAKE_KEY, install_dir)

    def _fake_run_as_user(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(ngc.privilege, "run_as_user", _fake_run_as_user)

    result = ngc.configure_ngc_cli(install_dir)

    assert result is None


def test_configure_ngc_cli_end_to_end_writes_expected_config_file(tmp_path, monkeypatch, _no_chown):
    """Bypasses only the sudo(8) wrapper (not available/needed for an
    unprivileged test run) so the real script this module builds actually
    executes via bash and can be checked against real file output --
    stronger than asserting on the script string alone."""
    install_dir = tmp_path / "install"
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setattr(ngc.privilege, "resolve", lambda: _fake_user(home_dir))
    ngc.store_key(_FAKE_KEY, install_dir)

    def _run_without_sudo(*args, **kwargs):
        # privilege.run_as_user would prepend ["sudo", "-u", name, "-H"];
        # skip that (no sudo available/appropriate in a test) but keep the
        # rest of the invocation identical, with HOME pinned at the fake
        # home dir the same way `sudo -H` would pin it for real.
        env = dict(os.environ)
        env["HOME"] = str(home_dir)
        run_kwargs = dict(kwargs)
        run_kwargs["env"] = env
        return subprocess.run(args, **run_kwargs)

    monkeypatch.setattr(ngc.privilege, "run_as_user", _run_without_sudo)

    result = ngc.configure_ngc_cli(install_dir)

    config_path = home_dir / ".ngc" / "config"
    assert result == config_path
    assert config_path.read_text(encoding="utf-8") == (
        "[CURRENT]\n"
        f"apikey = {_FAKE_KEY}\n"
        "format_type = ascii\n"
    )
    assert (config_path.stat().st_mode & 0o777) == 0o600
    assert (config_path.parent.stat().st_mode & 0o777) == 0o700


# ---------------------------------------------------------------------------
# Redaction discipline -- the raw key must never reach a log.*() call.
# ---------------------------------------------------------------------------


def _log_calls(tree: ast.AST) -> list[ast.Call]:
    calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"info", "warn", "error"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "log"
        ):
            calls.append(node)
    return calls


def _referenced_names(node: ast.AST) -> set[str]:
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}


# Local variable names anywhere in ngc.py that ever hold a raw secret value.
_SECRET_VARIABLE_NAMES = {"key", "raw", "contents"}


def test_source_never_passes_raw_secret_variables_to_log_calls():
    src_path = pathlib.Path(ngc.__file__)
    tree = ast.parse(src_path.read_text(encoding="utf-8"))

    log_calls = _log_calls(tree)
    assert log_calls, "expected at least one log.*() call in ngc.py to check"

    for call in log_calls:
        leaked = _referenced_names(call) & _SECRET_VARIABLE_NAMES
        assert not leaked, (
            f"log call at line {call.lineno} references secret variable(s) "
            f"{leaked} -- use the literal 'NGC_API_KEY=<redacted>' instead"
        )


def test_source_only_uses_redacted_placeholder_literal_near_log_calls():
    src_path = pathlib.Path(ngc.__file__)
    text = src_path.read_text(encoding="utf-8")

    # The only NGC_API_KEY= literal that should ever appear as a log-visible
    # string constant is the redacted placeholder. The real "NGC_API_KEY="
    # prefix used for file writes lives in _KEY_PREFIX, not inline in a
    # log.*() f-string, and _FAKE_KEY (the test fixture's dummy value) never
    # appears in source at all.
    assert "NGC_API_KEY=<redacted>" in text
    assert _FAKE_KEY not in text


def _hide_prompts(monkeypatch):
    from mv3dt_installer import prompting

    monkeypatch.setattr(
        prompting, "prompt_unavailable_reason", lambda: "stdout is not a terminal (is the output piped?)"
    )


def _no_prompt(*_args, **_kwargs):  # pragma: no cover -- must never be called
    raise AssertionError("a hidden prompt was shown")


def test_capture_key_fails_fast_when_the_prompt_is_hidden(monkeypatch, capsys):
    _hide_prompts(monkeypatch)
    monkeypatch.setattr(ngc.getpass, "getpass", _no_prompt)
    with pytest.raises(SystemExit):
        ngc.capture_key(non_interactive=False)
    err = capsys.readouterr().err
    assert "cannot be asked for" in err and "NGC_API_KEY" in err
