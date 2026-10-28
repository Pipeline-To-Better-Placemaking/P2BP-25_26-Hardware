"""Tests for `mv3dt_installer.config` (doc 00 §11, §3.4).

Run from installer/: `python3 -m pytest tests/test_config.py -v`

Never touches the real `/opt/mv3dt` install dir or the real
`/var/lib/mv3dt-installer/state.json` path -- every test constructs a
`StateMachine` against a `tmp_path`-derived file, and passes a
`tmp_path`-derived `default_install_dir` so even the "installer.conf
default" precedence tier (§11.2) never consults the real `/opt/mv3dt`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mv3dt_installer import config  # noqa: E402
from mv3dt_installer.state import StateMachine  # noqa: E402


@pytest.fixture(autouse=True)
def _force_no_colour(monkeypatch):
    """Match test_logs.py / test_privilege.py's convention. `config.load()`
    now logs (the "flag overrode a persisted gate" line), so the assertions
    below must see plain, un-escaped stderr."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)


@pytest.fixture(autouse=True)
def _reset_transcript_state():
    from mv3dt_installer import logs

    logs._transcript_path = None
    yield
    logs._transcript_path = None


@pytest.fixture(autouse=True)
def _seed_tracking_gate_off(monkeypatch):
    """Most tests here script the Step 6 and 7 gate prompts only. Seeding
    the Step 8, viewer and auto-ingest gates through their env vars (tier 2)
    keeps them out of their prompt sequences; the tracking-, viewer- and
    auto-ingest-gate tests below remove them again."""
    monkeypatch.setenv(config.GATE_TRACKING_HISTORY, "off")
    monkeypatch.setenv(config.GATE_VIEWER_STREAM, "off")
    monkeypatch.setenv(config.GATE_AUTO_INGEST, "off")


def _state(tmp_path: Path) -> StateMachine:
    """A StateMachine pointed at an isolated state.json under tmp_path."""
    return StateMachine(path=tmp_path / "var" / "state.json")


def _refuse_prompt(_msg: str) -> str:
    """Stand-in for input() that fails the test if load() ever prompts."""
    raise AssertionError("load() prompted interactively when it should not have")


def _scripted_prompt(*answers: str):
    """A prompt returning `answers` in order, recording every question it
    was asked. Raises rather than blocking if load() asks more questions
    than the test scripted."""
    remaining = list(answers)
    asked: list[str] = []

    def _prompt(msg: str) -> str:
        asked.append(msg)
        if not remaining:
            raise AssertionError(f"unscripted prompt: {msg!r}")
        return remaining.pop(0)

    _prompt.asked = asked  # type: ignore[attr-defined]
    return _prompt


# ---------------------------------------------------------------------------
# Precedence order
# ---------------------------------------------------------------------------


def test_install_dir_override_wins_over_state_and_default(tmp_path):
    sm = _state(tmp_path)
    # Pre-seed state.json with a different install_dir than the override.
    state_dir = tmp_path / "from-state"
    config.load(str(state_dir), sm, non_interactive=True)

    override_dir = tmp_path / "from-override"
    cfg = config.load(str(override_dir), sm, non_interactive=True)

    assert cfg.install_dir == override_dir
    assert override_dir.is_dir()


def test_state_json_wins_over_installer_conf_default_tier(tmp_path):
    # Seed a "previous install" installer.conf at the fallback default
    # location recording a *different* dir than what state.json will hold.
    fallback_default = tmp_path / "fallback-default"
    fallback_default.mkdir()
    (fallback_default / config.CONF_FILENAME).write_text(
        "INSTALL_DIR=" + str(tmp_path / "from-conf-fallback") + "\n",
        encoding="utf-8",
    )

    sm = _state(tmp_path)
    state_dir = tmp_path / "from-state"
    # Establish state.json's install_dir = state_dir via an explicit
    # override (a separate install dir from fallback_default).
    config.load(str(state_dir), sm, non_interactive=True)

    # A subsequent no-override load must read state.json (now populated)
    # rather than falling through to the installer.conf default tier.
    cfg = config.load(
        None, sm, non_interactive=True, default_install_dir=fallback_default
    )

    assert cfg.install_dir == state_dir


def test_installer_conf_default_tier_used_when_state_missing(tmp_path):
    """When state.json has never been written, a previously-chosen dir
    recorded in installer.conf at the default location wins over the
    hardcoded default (doc 00 §11.2 edge case)."""
    fallback_default = tmp_path / "fallback-default"
    fallback_default.mkdir()
    recorded_dir = tmp_path / "previously-chosen"
    (fallback_default / config.CONF_FILENAME).write_text(
        f"INSTALL_DIR={recorded_dir}\n", encoding="utf-8"
    )

    sm = _state(tmp_path)  # state.json does not exist yet
    cfg = config.load(
        None, sm, non_interactive=True, default_install_dir=fallback_default
    )

    assert cfg.install_dir == recorded_dir


def test_default_used_when_nothing_recorded_anywhere(tmp_path):
    sm = _state(tmp_path)
    default_dir = tmp_path / "opt-mv3dt-stand-in"

    cfg = config.load(
        None, sm, non_interactive=True, default_install_dir=default_dir
    )

    assert cfg.install_dir == default_dir


# ---------------------------------------------------------------------------
# First run / re-run behaviour
# ---------------------------------------------------------------------------


def test_first_run_creates_dir_writes_conf_and_updates_state(tmp_path):
    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    cfg = config.load(str(install_dir), sm, non_interactive=True)

    assert install_dir.is_dir()
    conf_path = install_dir / config.CONF_FILENAME
    assert conf_path.is_file()
    assert cfg.conf_path == conf_path

    loaded_state = sm.load()
    assert loaded_state.install_dir == str(install_dir)


def test_rerun_reads_back_existing_conf_without_prompting(tmp_path):
    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    first = config.load(str(install_dir), sm, non_interactive=True)
    # Simulate an operator hand-editing installer.conf with a custom var
    # that config.py does not own -- it must survive the re-run untouched.
    conf_text = (install_dir / config.CONF_FILENAME).read_text(encoding="utf-8")
    conf_text += "SOME_OTHER_MODULE_VAR=custom-value\n"
    (install_dir / config.CONF_FILENAME).write_text(conf_text, encoding="utf-8")

    # Re-run with no override: must resolve via state.json (already holds
    # install_dir from the first run) and must NOT prompt.
    second = config.load(None, sm, non_interactive=False, prompt=_refuse_prompt)

    assert second.install_dir == first.install_dir == install_dir
    assert second.values["SOME_OTHER_MODULE_VAR"] == "custom-value"


def test_non_interactive_uses_default_silently(tmp_path):
    sm = _state(tmp_path)
    default_dir = tmp_path / "silent-default"

    cfg = config.load(
        None,
        sm,
        non_interactive=True,
        default_install_dir=default_dir,
        prompt=_refuse_prompt,
    )

    assert cfg.install_dir == default_dir


def test_interactive_prompts_when_nothing_else_resolved(tmp_path):
    sm = _state(tmp_path)
    default_dir = tmp_path / "prefilled-default"
    chosen_dir = tmp_path / "operator-chosen"
    prompts_seen = []

    def _fake_prompt(msg: str) -> str:
        prompts_seen.append(msg)
        # A fresh install also asks the two §3.4 gate questions through
        # this same injected prompt; only the install-dir one is under
        # test here, so the gate questions take their prefilled default.
        if msg.startswith("Install directory"):
            assert str(default_dir) in msg  # prefilled with the default
            return str(chosen_dir)
        return ""

    cfg = config.load(
        None,
        sm,
        non_interactive=False,
        default_install_dir=default_dir,
        prompt=_fake_prompt,
    )

    assert prompts_seen  # prompt was actually invoked
    assert cfg.install_dir == chosen_dir
    assert not default_dir.exists()  # never created; the chosen dir was used


def test_interactive_empty_answer_accepts_prefilled_default(tmp_path):
    sm = _state(tmp_path)
    default_dir = tmp_path / "prefilled-default"

    cfg = config.load(
        None,
        sm,
        non_interactive=False,
        default_install_dir=default_dir,
        prompt=lambda _msg: "",
    )

    assert cfg.install_dir == default_dir


# ---------------------------------------------------------------------------
# §3.4 gate keys
# ---------------------------------------------------------------------------


def test_gate_keys_default_to_off_when_absent(tmp_path):
    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    cfg = config.load(str(install_dir), sm, non_interactive=True)

    assert cfg.remote_supervision == "off"
    assert cfg.webapp_integration == "off"
    assert cfg.values[config.GATE_REMOTE_SUPERVISION] == "off"
    assert cfg.values[config.GATE_WEBAPP_INTEGRATION] == "off"

    conf_text = (install_dir / config.CONF_FILENAME).read_text(encoding="utf-8")
    assert "MV3DT_REMOTE_SUPERVISION=off" in conf_text
    assert "MV3DT_WEBAPP_INTEGRATION=off" in conf_text


def test_gate_keys_round_trip_when_present(tmp_path):
    sm = _state(tmp_path)
    install_dir = tmp_path / "install"
    install_dir.mkdir()
    (install_dir / config.CONF_FILENAME).write_text(
        "MV3DT_REMOTE_SUPERVISION=local\n"
        "MV3DT_WEBAPP_INTEGRATION=on\n",
        encoding="utf-8",
    )

    cfg = config.load(str(install_dir), sm, non_interactive=True)

    assert cfg.remote_supervision == "local"
    assert cfg.webapp_integration == "on"

    # And it round-trips again on a second load.
    cfg2 = config.load(str(install_dir), sm, non_interactive=True)
    assert cfg2.remote_supervision == "local"
    assert cfg2.webapp_integration == "on"


def test_gate_keys_seeded_from_env_var_on_first_write(tmp_path, monkeypatch):
    """Regression test (code review finding): a gate's environment variable
    must reach the persisted `installer.conf` on first write.

    `bootstrap.sh` §5.1 step 5 decides whether to capture the web-app
    credential based on `MV3DT_WEBAPP_INTEGRATION` in its own environment,
    then `exec sudo -E`'s into the built binary -- so that env var *is*
    present in the launched process's environment. Before this fix,
    `config.load()` never consulted it: `installer.conf` was always created
    with both gates hardcoded to "off", so Step 7 would stay auto-skipped
    by the dispatch loop regardless of what the operator did at bootstrap
    time, even after successfully capturing and storing credentials.
    """
    monkeypatch.setenv("MV3DT_WEBAPP_INTEGRATION", "on")
    monkeypatch.setenv("MV3DT_REMOTE_SUPERVISION", "local")

    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    cfg = config.load(str(install_dir), sm, non_interactive=True)

    assert cfg.webapp_integration == "on"
    assert cfg.remote_supervision == "local"

    conf_text = (install_dir / config.CONF_FILENAME).read_text(encoding="utf-8")
    assert "MV3DT_WEBAPP_INTEGRATION=on" in conf_text
    assert "MV3DT_REMOTE_SUPERVISION=local" in conf_text


def test_gate_keys_env_var_does_not_override_already_persisted_value(
    tmp_path, monkeypatch
):
    """A gate's env var only seeds the *first* write, mirroring the
    "capture once, then just read back" discipline already used for the
    NGC/webapp credentials (doc 00 §10, §14) -- once persisted, a later
    env var must not silently flip an already-chosen gate value out from
    under the operator."""
    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    cfg1 = config.load(str(install_dir), sm, non_interactive=True)
    assert cfg1.webapp_integration == "off"

    monkeypatch.setenv("MV3DT_WEBAPP_INTEGRATION", "on")
    cfg2 = config.load(str(install_dir), sm, non_interactive=True)
    assert cfg2.webapp_integration == "off"


# ---------------------------------------------------------------------------
# §3.4 gate seeding precedence: flag > env var > prompt > "off"
# ---------------------------------------------------------------------------


def test_gate_flag_seeds_first_write(tmp_path):
    """Tier 1 in isolation: the CLI flag with no env var and no prompt."""
    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    cfg = config.load(
        str(install_dir),
        sm,
        non_interactive=True,
        gate_overrides={
            config.GATE_REMOTE_SUPERVISION: "remote",
            config.GATE_WEBAPP_INTEGRATION: "on",
        },
    )

    assert cfg.remote_supervision == "remote"
    assert cfg.webapp_integration == "on"

    conf_text = (install_dir / config.CONF_FILENAME).read_text(encoding="utf-8")
    assert "MV3DT_REMOTE_SUPERVISION=remote" in conf_text
    assert "MV3DT_WEBAPP_INTEGRATION=on" in conf_text


def test_gate_flag_wins_over_env_var_and_prompt(tmp_path, monkeypatch):
    """Tiers 1, 2 and 3 in combination: the flag outranks both, and the
    prompt is never reached for a gate the flag already answered."""
    monkeypatch.setenv("MV3DT_WEBAPP_INTEGRATION", "on")
    monkeypatch.setenv("MV3DT_REMOTE_SUPERVISION", "local")

    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    cfg = config.load(
        str(install_dir),
        sm,
        non_interactive=False,
        prompt=_refuse_prompt,
        gate_overrides={
            config.GATE_REMOTE_SUPERVISION: "remote",
            config.GATE_WEBAPP_INTEGRATION: "off",
        },
    )

    assert cfg.remote_supervision == "remote"
    assert cfg.webapp_integration == "off"


def test_gate_env_var_wins_over_prompt(tmp_path, monkeypatch):
    """Tier 2 outranks tier 3, which is what keeps `sudo -E` working for
    anyone scripting an otherwise-interactive run."""
    monkeypatch.setenv("MV3DT_REMOTE_SUPERVISION", "local")
    monkeypatch.setenv("MV3DT_WEBAPP_INTEGRATION", "on")

    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    cfg = config.load(
        str(install_dir), sm, non_interactive=False, prompt=_refuse_prompt
    )

    assert cfg.remote_supervision == "local"
    assert cfg.webapp_integration == "on"


def test_gate_env_var_seeds_only_the_gate_it_names(tmp_path, monkeypatch):
    """Tiers 2 and 3 side by side: one gate comes from the environment,
    the other still falls through to the prompt."""
    monkeypatch.setenv("MV3DT_WEBAPP_INTEGRATION", "on")
    monkeypatch.delenv("MV3DT_REMOTE_SUPERVISION", raising=False)

    sm = _state(tmp_path)
    install_dir = tmp_path / "install"
    prompt = _scripted_prompt("remote")

    cfg = config.load(
        str(install_dir), sm, non_interactive=False, prompt=prompt
    )

    assert cfg.webapp_integration == "on"
    assert cfg.remote_supervision == "remote"
    # Exactly one question: the web-app gate never reached the prompt.
    assert len(prompt.asked) == 1
    assert prompt.asked[0].startswith("MV3DT_REMOTE_SUPERVISION")


def test_gate_prompt_lists_choices_and_prefills_off(tmp_path, monkeypatch):
    """Tier 3 in isolation, including the prompt's shape: every allowed
    value is named and `off` is shown as the prefilled default."""
    monkeypatch.delenv("MV3DT_REMOTE_SUPERVISION", raising=False)
    monkeypatch.delenv("MV3DT_WEBAPP_INTEGRATION", raising=False)

    sm = _state(tmp_path)
    install_dir = tmp_path / "install"
    prompt = _scripted_prompt("local", "on")

    cfg = config.load(
        str(install_dir), sm, non_interactive=False, prompt=prompt
    )

    assert cfg.remote_supervision == "local"
    assert cfg.webapp_integration == "on"

    remote_question, webapp_question = prompt.asked
    for value in config.GATE_CHOICES[config.GATE_REMOTE_SUPERVISION]:
        assert value in remote_question
    for value in config.GATE_CHOICES[config.GATE_WEBAPP_INTEGRATION]:
        assert value in webapp_question
    assert "[off]" in remote_question
    assert "[off]" in webapp_question


def test_gate_prompt_empty_answer_accepts_prefilled_off(tmp_path, monkeypatch):
    monkeypatch.delenv("MV3DT_REMOTE_SUPERVISION", raising=False)
    monkeypatch.delenv("MV3DT_WEBAPP_INTEGRATION", raising=False)

    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    cfg = config.load(
        str(install_dir),
        sm,
        non_interactive=False,
        prompt=_scripted_prompt("", ""),
    )

    assert cfg.remote_supervision == "off"
    assert cfg.webapp_integration == "off"


def test_gate_prompt_rejects_invalid_value_and_asks_again(
    tmp_path, monkeypatch, capsys
):
    """A typo must not be silently coerced -- it decides whether a whole
    step ever runs -- so the question is asked again."""
    monkeypatch.delenv("MV3DT_REMOTE_SUPERVISION", raising=False)
    monkeypatch.delenv("MV3DT_WEBAPP_INTEGRATION", raising=False)

    sm = _state(tmp_path)
    install_dir = tmp_path / "install"
    prompt = _scripted_prompt("yes-please", "local", "on")

    cfg = config.load(
        str(install_dir), sm, non_interactive=False, prompt=prompt
    )

    assert cfg.remote_supervision == "local"
    assert cfg.webapp_integration == "on"
    # Three questions for two gates: the rejected answer cost one re-ask.
    assert len(prompt.asked) == 3
    assert prompt.asked[0] == prompt.asked[1]
    assert "yes-please" in capsys.readouterr().err


def test_gate_prompt_answer_is_case_insensitive(tmp_path, monkeypatch):
    monkeypatch.delenv("MV3DT_REMOTE_SUPERVISION", raising=False)
    monkeypatch.delenv("MV3DT_WEBAPP_INTEGRATION", raising=False)

    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    cfg = config.load(
        str(install_dir),
        sm,
        non_interactive=False,
        prompt=_scripted_prompt("Remote", " ON "),
    )

    # Stored lowercase, so the dispatch loop's string comparisons hold.
    assert cfg.remote_supervision == "remote"
    assert cfg.webapp_integration == "on"


def test_non_interactive_never_prompts_and_yields_off(tmp_path, monkeypatch):
    """Tier 4 (doc 00 §3.4): `--non-interactive` with no flag and no env
    var leaves an unset gate at `off` without ever blocking on a human."""
    monkeypatch.delenv("MV3DT_REMOTE_SUPERVISION", raising=False)
    monkeypatch.delenv("MV3DT_WEBAPP_INTEGRATION", raising=False)

    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    cfg = config.load(
        str(install_dir), sm, non_interactive=True, prompt=_refuse_prompt
    )

    assert cfg.remote_supervision == "off"
    assert cfg.webapp_integration == "off"


# ---------------------------------------------------------------------------
# §3.4 gates: flag overriding an already-persisted value
# ---------------------------------------------------------------------------


def test_gate_flag_overrides_persisted_value_and_logs_the_change(
    tmp_path, capsys
):
    """Without this, the only way to turn a gate on after the first run
    would be to hand-edit `installer.conf`."""
    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    first = config.load(str(install_dir), sm, non_interactive=True)
    assert first.webapp_integration == "off"
    capsys.readouterr()  # discard anything the first run emitted

    second = config.load(
        str(install_dir),
        sm,
        non_interactive=True,
        gate_overrides={config.GATE_WEBAPP_INTEGRATION: "on"},
    )

    assert second.webapp_integration == "on"
    # The other gate is untouched by a flag that does not name it.
    assert second.remote_supervision == "off"

    err = capsys.readouterr().err
    assert "MV3DT_WEBAPP_INTEGRATION" in err
    assert "off -> on" in err
    assert "--webapp-integration" in err

    # And the new value is what a third, flagless run reads back.
    third = config.load(str(install_dir), sm, non_interactive=True)
    assert third.webapp_integration == "on"


def test_gate_flag_can_turn_a_persisted_gate_back_off(tmp_path):
    """`--webapp-integration off` is an explicit instruction, not the
    absence of one -- which is exactly why the flag defaults to None."""
    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    config.load(
        str(install_dir),
        sm,
        non_interactive=True,
        gate_overrides={config.GATE_WEBAPP_INTEGRATION: "on"},
    )

    cfg = config.load(
        str(install_dir),
        sm,
        non_interactive=True,
        gate_overrides={config.GATE_WEBAPP_INTEGRATION: "off"},
    )

    assert cfg.webapp_integration == "off"


def test_gate_flag_matching_persisted_value_logs_nothing(tmp_path, capsys):
    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    config.load(
        str(install_dir),
        sm,
        non_interactive=True,
        gate_overrides={config.GATE_REMOTE_SUPERVISION: "local"},
    )
    capsys.readouterr()

    cfg = config.load(
        str(install_dir),
        sm,
        non_interactive=True,
        gate_overrides={config.GATE_REMOTE_SUPERVISION: "local"},
    )

    assert cfg.remote_supervision == "local"
    assert "MV3DT_REMOTE_SUPERVISION" not in capsys.readouterr().err


def test_env_var_ignored_once_persisted_even_though_flag_is_not(
    tmp_path, monkeypatch, capsys
):
    """The two tiers behave differently on a re-run on purpose: the env
    var only ever seeds the first write, while the flag is an explicit
    per-run instruction that overwrites."""
    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    config.load(str(install_dir), sm, non_interactive=True)

    monkeypatch.setenv("MV3DT_WEBAPP_INTEGRATION", "on")
    monkeypatch.setenv("MV3DT_REMOTE_SUPERVISION", "remote")
    capsys.readouterr()

    # Env var alone: ignored, the persisted "off" is read straight back.
    from_env_only = config.load(str(install_dir), sm, non_interactive=True)
    assert from_env_only.webapp_integration == "off"
    assert from_env_only.remote_supervision == "off"

    # Same env, plus the flag: the flag lands.
    with_flag = config.load(
        str(install_dir),
        sm,
        non_interactive=True,
        gate_overrides={config.GATE_WEBAPP_INTEGRATION: "on"},
    )
    assert with_flag.webapp_integration == "on"
    # The env var still did nothing for the gate no flag named.
    assert with_flag.remote_supervision == "off"


def test_gate_prompt_not_reached_for_an_already_persisted_gate(
    tmp_path, monkeypatch
):
    """"Capture once, then just read back": a second interactive run must
    not re-ask a question the operator has already answered."""
    monkeypatch.delenv("MV3DT_REMOTE_SUPERVISION", raising=False)
    monkeypatch.delenv("MV3DT_WEBAPP_INTEGRATION", raising=False)

    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    first = config.load(
        str(install_dir),
        sm,
        non_interactive=False,
        prompt=_scripted_prompt("local", "on"),
    )
    assert (first.remote_supervision, first.webapp_integration) == ("local", "on")

    second = config.load(
        None, sm, non_interactive=False, prompt=_refuse_prompt
    )

    assert second.remote_supervision == "local"
    assert second.webapp_integration == "on"


def test_tracking_gate_defaults_off_under_non_interactive(tmp_path, monkeypatch):
    """STEP-8 section 4.1: an unattended run never opens a listener the
    operator did not ask for."""
    monkeypatch.delenv(config.GATE_TRACKING_HISTORY, raising=False)

    cfg = config.load(
        str(tmp_path / "install"), _state(tmp_path), non_interactive=True,
        prompt=_refuse_prompt,
    )

    assert cfg.tracking_history == "off"
    assert cfg.values[config.GATE_TRACKING_HISTORY] == "off"


def test_tracking_gate_flag_seeds_and_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv(config.GATE_TRACKING_HISTORY, raising=False)
    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    first = config.load(str(install_dir), sm, non_interactive=True)
    assert first.tracking_history == "off"

    second = config.load(
        None, sm, non_interactive=True,
        gate_overrides={config.GATE_TRACKING_HISTORY: "on"},
    )
    assert second.tracking_history == "on"
    conf_text = (install_dir / config.CONF_FILENAME).read_text(encoding="utf-8")
    assert "MV3DT_TRACKING_HISTORY=on" in conf_text


def test_tracking_gate_is_prompted_after_the_step_gates(tmp_path, monkeypatch):
    for key in config.GATE_KEYS:
        monkeypatch.delenv(key, raising=False)
    prompt = _scripted_prompt("", "", "on", "", "")

    cfg = config.load(
        str(tmp_path / "install"), _state(tmp_path), non_interactive=False,
        prompt=prompt,
    )

    assert cfg.tracking_history == "on"
    assert prompt.asked[2] == "MV3DT_TRACKING_HISTORY (off/on) [off]: "


def test_viewer_gate_is_prompted_after_the_step_gates(tmp_path, monkeypatch):
    for key in config.GATE_KEYS:
        monkeypatch.delenv(key, raising=False)
    prompt = _scripted_prompt("", "", "", "on", "")

    cfg = config.load(
        str(tmp_path / "install"), _state(tmp_path), non_interactive=False,
        prompt=prompt,
    )

    assert cfg.viewer_stream == "on"
    assert cfg.values[config.GATE_VIEWER_STREAM] == "on"
    assert prompt.asked[3] == "MV3DT_VIEWER_STREAM (off/on) [off]: "


def test_auto_ingest_gate_is_prompted_last_with_off_prefilled(tmp_path, monkeypatch):
    for key in config.GATE_KEYS:
        monkeypatch.delenv(key, raising=False)
    prompt = _scripted_prompt("", "", "", "", "on")

    cfg = config.load(
        str(tmp_path / "install"), _state(tmp_path), non_interactive=False,
        prompt=prompt,
    )

    assert cfg.auto_ingest == "on"
    assert cfg.values[config.GATE_AUTO_INGEST] == "on"
    assert prompt.asked[-1] == "MV3DT_AUTO_INGEST (off/on) [off]: "


def test_auto_ingest_gate_defaults_off_under_non_interactive(tmp_path, monkeypatch):
    monkeypatch.delenv(config.GATE_AUTO_INGEST, raising=False)

    cfg = config.load(
        str(tmp_path / "install"), _state(tmp_path), non_interactive=True,
        prompt=_refuse_prompt,
    )

    assert cfg.auto_ingest == "off"
    assert "MV3DT_AUTO_INGEST=off" in cfg.conf_path.read_text(encoding="utf-8")


def test_auto_ingest_gate_flag_seeds_and_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv(config.GATE_AUTO_INGEST, raising=False)
    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    first = config.load(str(install_dir), sm, non_interactive=True)
    assert first.auto_ingest == "off"

    second = config.load(
        None, sm, non_interactive=True,
        gate_overrides={config.GATE_AUTO_INGEST: "on"},
    )
    assert second.auto_ingest == "on"
    conf_text = (install_dir / config.CONF_FILENAME).read_text(encoding="utf-8")
    assert "MV3DT_AUTO_INGEST=on" in conf_text


def test_viewer_gate_defaults_off_under_non_interactive(tmp_path, monkeypatch):
    monkeypatch.delenv(config.GATE_VIEWER_STREAM, raising=False)

    cfg = config.load(
        str(tmp_path / "install"), _state(tmp_path), non_interactive=True,
        prompt=_refuse_prompt,
    )

    assert cfg.viewer_stream == "off"
    assert "MV3DT_VIEWER_STREAM=off" in cfg.conf_path.read_text(encoding="utf-8")


def test_viewer_gate_flag_seeds_and_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv(config.GATE_VIEWER_STREAM, raising=False)
    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    first = config.load(str(install_dir), sm, non_interactive=True)
    assert first.viewer_stream == "off"

    second = config.load(
        None, sm, non_interactive=True,
        gate_overrides={config.GATE_VIEWER_STREAM: "on"},
    )
    assert second.viewer_stream == "on"
    conf_text = (install_dir / config.CONF_FILENAME).read_text(encoding="utf-8")
    assert "MV3DT_VIEWER_STREAM=on" in conf_text


# ---------------------------------------------------------------------------
# installer.conf shape
# ---------------------------------------------------------------------------


def test_conf_file_is_plain_key_value_no_quoting(tmp_path):
    sm = _state(tmp_path)
    install_dir = tmp_path / "install"

    config.load(str(install_dir), sm, non_interactive=True)

    conf_text = (install_dir / config.CONF_FILENAME).read_text(encoding="utf-8")
    for line in conf_text.splitlines():
        if not line or line.startswith("#"):
            continue
        assert '"' not in line
        assert "'" not in line
        assert "=" in line


def test_write_conf_sorts_keys_and_preserves_keys_config_does_not_own(tmp_path):
    """`_write_conf` rewrites the whole file on every load, so a gate
    override must not cost the file its stable ordering or the vars other
    modules (ngc.py, webapp.py, cameras.py) park in it."""
    sm = _state(tmp_path)
    install_dir = tmp_path / "install"
    install_dir.mkdir()
    (install_dir / config.CONF_FILENAME).write_text(
        "ZZZ_LAST_ALPHABETICALLY=z\n"
        "MV3DT_WEBAPP_INTEGRATION=off\n"
        "AAA_FIRST_ALPHABETICALLY=a\n",
        encoding="utf-8",
    )

    cfg = config.load(
        str(install_dir),
        sm,
        non_interactive=True,
        gate_overrides={config.GATE_WEBAPP_INTEGRATION: "on"},
    )

    assert cfg.values["AAA_FIRST_ALPHABETICALLY"] == "a"
    assert cfg.values["ZZZ_LAST_ALPHABETICALLY"] == "z"
    assert cfg.webapp_integration == "on"

    conf_text = (install_dir / config.CONF_FILENAME).read_text(encoding="utf-8")
    keys = [
        line.partition("=")[0]
        for line in conf_text.splitlines()
        if line and not line.startswith("#")
    ]
    assert keys == sorted(keys)
    assert "AAA_FIRST_ALPHABETICALLY" in keys
    assert "ZZZ_LAST_ALPHABETICALLY" in keys
    assert "MV3DT_WEBAPP_INTEGRATION=on" in conf_text


# ---------------------------------------------------------------------------
# persist_value (doc 00 §11.2, §15.5's CAMERA_SCAN_CIDR/CAMERA_SCAN_IFACE)
# ---------------------------------------------------------------------------


def test_persist_value_writes_a_fresh_conf_file(tmp_path):
    config.persist_value(tmp_path, "CAMERA_SCAN_CIDR", "10.0.0.0/24")

    text = (tmp_path / config.CONF_FILENAME).read_text(encoding="utf-8")
    assert "CAMERA_SCAN_CIDR=10.0.0.0/24" in text


def test_persist_value_preserves_existing_keys_and_overwrites_its_own(tmp_path):
    conf_path = tmp_path / config.CONF_FILENAME
    conf_path.write_text(
        "INSTALL_DIR=" + str(tmp_path) + "\nCAMERA_SCAN_CIDR=old\n", encoding="utf-8"
    )

    config.persist_value(tmp_path, "CAMERA_SCAN_CIDR", "new")

    text = conf_path.read_text(encoding="utf-8")
    assert "CAMERA_SCAN_CIDR=new" in text
    assert "old" not in text
    assert f"INSTALL_DIR={tmp_path}" in text


def test_remove_values_drops_only_named_keys(tmp_path):
    conf_path = tmp_path / config.CONF_FILENAME
    conf_path.write_text(
        "CAM_USER=admin\nCAM_PASSWORD=secret\nCAMERA_SCAN_CIDR=10.0.0.0/24\n",
        encoding="utf-8",
    )

    config.remove_values(tmp_path, "CAM_USER", "CAM_PASSWORD")

    text = conf_path.read_text(encoding="utf-8")
    assert "CAM_USER" not in text
    assert "CAM_PASSWORD" not in text
    assert "CAMERA_SCAN_CIDR=10.0.0.0/24" in text


def test_viewer_stream_flag_maps_to_its_gate():
    from mv3dt_installer import app

    assert app.parse_args([]).viewer_stream is None
    args = app.parse_args(["--viewer-stream", "on"])
    assert app._gate_overrides_from_args(args) == {config.GATE_VIEWER_STREAM: "on"}
    # A feature gate, not a step gate: no step is skipped on it.
    assert "viewer_stream" not in app._GATE_ORDER_TO_CONF_ATTR.values()


def test_auto_ingest_flag_maps_to_its_gate():
    from mv3dt_installer import app

    assert app.parse_args([]).auto_ingest is None
    args = app.parse_args(["--auto-ingest", "on"])
    assert app._gate_overrides_from_args(args) == {config.GATE_AUTO_INGEST: "on"}
    assert "auto_ingest" not in app._GATE_ORDER_TO_CONF_ATTR.values()


def _hide_prompts(monkeypatch):
    from mv3dt_installer import prompting

    monkeypatch.setattr(
        prompting, "prompt_unavailable_reason", lambda: "stdout is not a terminal (is the output piped?)"
    )


def _no_prompt(*_args, **_kwargs):  # pragma: no cover -- must never be called
    raise AssertionError("a hidden prompt was shown")


def test_hidden_prompt_takes_every_default(tmp_path, monkeypatch, capsys):
    """With the real input() and a prompt nobody can see, load() must not
    block: the install directory and the gates take their defaults."""
    for key in config.GATE_KEYS:
        monkeypatch.delenv(key, raising=False)
    _hide_prompts(monkeypatch)
    monkeypatch.setattr("builtins.input", _no_prompt)
    default_dir = tmp_path / "default-install"

    cfg = config.load(None, _state(tmp_path), default_install_dir=default_dir)

    assert cfg.install_dir == default_dir
    for key in config.GATE_KEYS:
        assert cfg.values[key] == config.GATE_DEFAULTS[key]
    assert "not asking configuration questions" in capsys.readouterr().err
