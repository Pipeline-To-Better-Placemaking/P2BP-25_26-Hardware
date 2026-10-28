"""Tests for mv3dt_installer.webapp (doc 00 §14).

Run from installer/: `python3 -m pytest tests/test_webapp.py -v`
"""

from __future__ import annotations

import os
import stat
import sys
import urllib.parse
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from mv3dt_installer import webapp  # noqa: E402


def _fake_user():
    """A stand-in InvokingUser -- current process uid/gid so os.chown()
    (which store_credentials calls) succeeds without root."""
    return SimpleNamespace(name="alice", home="/home/alice", uid=os.getuid(), gid=os.getgid())


# ---------------------------------------------------------------------------
# 14.2 -- normalize_endpoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://host", "https://host"),
        ("https://host/", "https://host"),
        ("https://host/api", "https://host"),
        ("https://host/api/", "https://host"),
        ("https://host/API//", "https://host"),
    ],
)
def test_normalize_endpoint_worked_examples(raw, expected):
    assert webapp.normalize_endpoint(raw) == expected


def test_normalize_endpoint_strips_surrounding_whitespace():
    assert webapp.normalize_endpoint("  https://host/api/  ") == "https://host"


def test_normalize_endpoint_empty_raises():
    with pytest.raises(ValueError):
        webapp.normalize_endpoint("")


def test_normalize_endpoint_whitespace_only_raises():
    with pytest.raises(ValueError):
        webapp.normalize_endpoint("   ")


def test_normalize_endpoint_all_slashes_raises():
    with pytest.raises(ValueError):
        webapp.normalize_endpoint("///")


# ---------------------------------------------------------------------------
# 14.2 -- join (strict, not urljoin)
# ---------------------------------------------------------------------------


def test_join_right_strips_base_and_left_pads_path():
    assert webapp.join("https://host/", "/widgets") == "https://host/widgets"
    assert webapp.join("https://host", "widgets") == "https://host/widgets"
    assert webapp.join("https://host//", "//widgets") == "https://host/widgets"


def test_join_is_not_urljoin_semantics():
    """A base with a non-root path component is where urljoin diverges:
    urljoin drops the last path segment of the base when the base doesn't
    end in "/", which would silently corrupt the API root. join() must
    concatenate literally instead."""
    endpoint = "https://host/api/v2"
    path = "widgets"

    result = webapp.join(endpoint, path)
    wrong_urljoin_result = urllib.parse.urljoin(endpoint, path)

    assert result == "https://host/api/v2/widgets"
    assert wrong_urljoin_result == "https://host/api/widgets"
    assert result != wrong_urljoin_result


# ---------------------------------------------------------------------------
# 14.3 -- capture_credentials
# ---------------------------------------------------------------------------


def test_capture_credentials_keeps_existing_by_default(monkeypatch):
    monkeypatch.setattr(
        webapp,
        "load_credentials",
        lambda *a, **kw: webapp.Credentials(api_key="oldkey", endpoint="https://old"),
    )
    # Blank answer to both "Keep it?" prompts must mean "yes, keep".
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    def _fail_getpass(prompt=""):
        raise AssertionError("must not prompt for a new key when keeping existing")

    monkeypatch.setattr(webapp.getpass, "getpass", _fail_getpass)

    creds = webapp.capture_credentials(non_interactive=False)

    assert creds.api_key == "oldkey"
    assert creds.endpoint == "https://old"


def test_capture_credentials_replaces_when_operator_declines_keep(monkeypatch):
    monkeypatch.setattr(
        webapp,
        "load_credentials",
        lambda *a, **kw: webapp.Credentials(api_key="oldkey", endpoint="https://old"),
    )
    answers = iter(["n", "n", "https://newhost/api/"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(webapp.getpass, "getpass", lambda prompt="": "newkey")

    creds = webapp.capture_credentials(non_interactive=False)

    assert creds.api_key == "newkey"
    assert creds.endpoint == "https://newhost"


def test_capture_credentials_prompts_when_nothing_existing(monkeypatch):
    monkeypatch.setattr(webapp, "load_credentials", lambda *a, **kw: None)
    monkeypatch.setattr(webapp.getpass, "getpass", lambda prompt="": "newkey")
    monkeypatch.setattr("builtins.input", lambda prompt="": "https://newhost/api/")

    creds = webapp.capture_credentials(non_interactive=False)

    assert creds.api_key == "newkey"
    assert creds.endpoint == "https://newhost"


def test_capture_credentials_non_interactive_keeps_silently(monkeypatch):
    monkeypatch.setattr(
        webapp,
        "load_credentials",
        lambda *a, **kw: webapp.Credentials(api_key="k", endpoint="https://e"),
    )

    def _fail(prompt=""):
        raise AssertionError("non-interactive must never prompt")

    monkeypatch.setattr("builtins.input", _fail)
    monkeypatch.setattr(webapp.getpass, "getpass", _fail)

    creds = webapp.capture_credentials(non_interactive=True)

    assert creds.api_key == "k"
    assert creds.endpoint == "https://e"


def test_capture_credentials_non_interactive_missing_left_unset(monkeypatch):
    monkeypatch.setattr(webapp, "load_credentials", lambda *a, **kw: None)

    creds = webapp.capture_credentials(non_interactive=True)

    assert creds.api_key is None
    assert creds.endpoint is None


def test_capture_credentials_passes_install_dir_through_to_load_credentials(
    monkeypatch, tmp_path
):
    """The C9 bug: capture_credentials used to call a bare load_credentials()
    and silently ignore --install-dir. It must now forward its install_dir
    argument through unchanged."""
    captured = {}

    def _fake_load_credentials(install_dir=None):
        captured["install_dir"] = install_dir
        return None

    monkeypatch.setattr(webapp, "load_credentials", _fake_load_credentials)

    webapp.capture_credentials(non_interactive=True, install_dir=tmp_path / "custom")

    assert captured["install_dir"] == tmp_path / "custom"


# ---------------------------------------------------------------------------
# 14.1/14.3 -- store_credentials
# ---------------------------------------------------------------------------


def test_store_credentials_writes_exact_two_line_file(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp.privilege, "resolve", _fake_user)
    install_dir = tmp_path / "install"
    creds = webapp.Credentials(api_key="secretkey", endpoint="https://host/api/")

    path = webapp.store_credentials(creds, install_dir)

    assert path == install_dir / "secrets" / "webapp.env"
    content = path.read_text(encoding="utf-8")
    assert content == "API_KEY=secretkey\nENDPOINT=https://host\n"


def test_store_credentials_permissions(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp.privilege, "resolve", _fake_user)
    install_dir = tmp_path / "install"
    creds = webapp.Credentials(api_key="secretkey", endpoint="https://host")

    path = webapp.store_credentials(creds, install_dir)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE((install_dir / "secrets").stat().st_mode) == 0o700


def test_store_credentials_uses_os_open_with_restrictive_mode_not_open_then_chmod(
    tmp_path, monkeypatch
):
    """The file must be created with 0o600 from the very first os.open()
    call (per §14.3's "never briefly world-readable" requirement), not via
    open() followed by a separate chmod()."""
    monkeypatch.setattr(webapp.privilege, "resolve", _fake_user)
    install_dir = tmp_path / "install"
    creds = webapp.Credentials(api_key="secretkey", endpoint="https://host")

    open_calls = []
    real_os_open = os.open

    def spy_open(path, flags, mode=0o777, *args, **kwargs):
        open_calls.append((path, flags, mode))
        return real_os_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(webapp.os, "open", spy_open)

    chmod_calls = []
    real_os_chmod = os.chmod

    def spy_chmod(path, mode, *args, **kwargs):
        chmod_calls.append((str(path), mode))
        return real_os_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(webapp.os, "chmod", spy_chmod)

    path = webapp.store_credentials(creds, install_dir)

    # The secret-file-creating os.open() call used the restrictive mode.
    secret_file_open_calls = [c for c in open_calls if str(path) in str(c[0]) or c[0].endswith(".tmp")]
    assert secret_file_open_calls, "expected the secret file to be created via os.open()"
    assert all(c[2] == 0o600 for c in secret_file_open_calls)
    assert all(c[1] & os.O_CREAT for c in secret_file_open_calls)

    # No chmod() call targets the final secret file path (only the
    # directory is chmod'd separately) -- os.open() alone sets its mode.
    assert str(path) not in [c[0] for c in chmod_calls]


def test_store_credentials_requires_both_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp.privilege, "resolve", _fake_user)
    with pytest.raises(ValueError):
        webapp.store_credentials(webapp.Credentials(api_key="k", endpoint=None), tmp_path)
    with pytest.raises(ValueError):
        webapp.store_credentials(webapp.Credentials(api_key=None, endpoint="https://e"), tmp_path)


# ---------------------------------------------------------------------------
# 14.3 -- load_credentials
# ---------------------------------------------------------------------------


def test_load_credentials_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp.privilege, "resolve", _fake_user)
    install_dir = tmp_path / "install"
    creds = webapp.Credentials(api_key="abc123", endpoint="https://host")
    webapp.store_credentials(creds, install_dir)

    loaded = webapp.load_credentials(install_dir)

    assert loaded == webapp.Credentials(api_key="abc123", endpoint="https://host")


def test_load_credentials_renormalizes_hand_edited_file(tmp_path):
    install_dir = tmp_path / "install"
    secrets_dir = install_dir / "secrets"
    secrets_dir.mkdir(parents=True)
    (secrets_dir / "webapp.env").write_text(
        "API_KEY=abc123\nENDPOINT=https://host/API//\n", encoding="utf-8"
    )

    loaded = webapp.load_credentials(install_dir)

    assert loaded.api_key == "abc123"
    assert loaded.endpoint == "https://host"


def test_load_credentials_missing_file_returns_none(tmp_path):
    assert webapp.load_credentials(tmp_path / "does-not-exist") is None


def test_load_credentials_missing_value_returns_none(tmp_path):
    install_dir = tmp_path / "install"
    secrets_dir = install_dir / "secrets"
    secrets_dir.mkdir(parents=True)
    (secrets_dir / "webapp.env").write_text("API_KEY=abc123\n", encoding="utf-8")

    assert webapp.load_credentials(install_dir) is None


def test_load_credentials_defaults_to_default_install_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "DEFAULT_INSTALL_DIR", tmp_path / "install")
    secrets_dir = tmp_path / "install" / "secrets"
    secrets_dir.mkdir(parents=True)
    (secrets_dir / "webapp.env").write_text(
        "API_KEY=k\nENDPOINT=https://host\n", encoding="utf-8"
    )

    assert webapp.load_credentials() == webapp.Credentials(api_key="k", endpoint="https://host")


# ---------------------------------------------------------------------------
# 14.3 -- enabled()
# ---------------------------------------------------------------------------


def test_enabled_true_only_when_gate_on_and_credentials_load(monkeypatch):
    monkeypatch.setattr(
        webapp,
        "load_credentials",
        lambda *a, **kw: webapp.Credentials(api_key="k", endpoint="https://e"),
    )

    assert webapp.enabled("on") is True
    assert webapp.enabled("off") is False
    assert webapp.enabled("") is False
    assert webapp.enabled("ON") is False  # exact match only, per §3.4 values


def test_enabled_false_when_gate_on_but_credentials_missing(monkeypatch):
    monkeypatch.setattr(webapp, "load_credentials", lambda *a, **kw: None)

    assert webapp.enabled("on") is False


def test_enabled_passes_install_dir_through_to_load_credentials(monkeypatch, tmp_path):
    """The C9 bug, same as capture_credentials: enabled() used to call a
    bare load_credentials() and silently ignore --install-dir."""
    captured = {}

    def _fake_load_credentials(install_dir=None):
        captured["install_dir"] = install_dir
        return webapp.Credentials(api_key="k", endpoint="https://e")

    monkeypatch.setattr(webapp, "load_credentials", _fake_load_credentials)

    result = webapp.enabled("on", install_dir=tmp_path / "custom")

    assert result is True
    assert captured["install_dir"] == tmp_path / "custom"


# ---------------------------------------------------------------------------
# 14.4 -- redact_url
# ---------------------------------------------------------------------------


def test_redact_url_empty_string():
    assert webapp.redact_url("") == ""


def test_redact_url_falsy_none():
    assert webapp.redact_url(None) == ""


def test_redact_url_no_query_string():
    assert webapp.redact_url("https://host/path") == "https://host/path"


def test_redact_url_redacts_from_first_question_mark():
    url = "https://host/upload?token=abc&expires=123"
    assert webapp.redact_url(url) == "https://host/upload?<redacted>"


def test_redact_url_matches_exact_reference_implementation():
    def reference(url):
        if not url:
            return ""
        q = url.find("?")
        return url if q < 0 else url[:q] + "?<redacted>"

    cases = ["", "https://host", "https://host?a=1", "https://host/p?x=1&y=2", None]
    for case in cases:
        assert webapp.redact_url(case) == reference(case)


def _hide_prompts(monkeypatch):
    from mv3dt_installer import prompting

    monkeypatch.setattr(
        prompting, "prompt_unavailable_reason", lambda: "stdout is not a terminal (is the output piped?)"
    )


def _no_prompt(*_args, **_kwargs):  # pragma: no cover -- must never be called
    raise AssertionError("a hidden prompt was shown")


def test_capture_credentials_hidden_prompt_keeps_stored_values(monkeypatch, capsys):
    monkeypatch.setattr(
        webapp,
        "load_credentials",
        lambda *a, **kw: webapp.Credentials(api_key="k", endpoint="https://e"),
    )
    _hide_prompts(monkeypatch)
    monkeypatch.setattr("builtins.input", _no_prompt)
    monkeypatch.setattr(webapp.getpass, "getpass", _no_prompt)

    creds = webapp.capture_credentials(non_interactive=False)

    assert (creds.api_key, creds.endpoint) == ("k", "https://e")
    assert "not asking for web-app credentials" in capsys.readouterr().err
