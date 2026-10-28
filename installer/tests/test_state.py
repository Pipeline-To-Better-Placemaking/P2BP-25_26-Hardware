"""Tests for `mv3dt_installer.state` (doc 00 §6).

Run from installer/: `python3 -m pytest tests/test_state.py -v`

Never touches the real `/var/lib/mv3dt-installer/state.json` path — every
test constructs a `StateMachine` against a `tmp_path`-derived file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mv3dt_installer.state import (  # noqa: E402
    STEP_IDS,
    StateMachine,
    State,
    StepEntry,
    RebootPending,
    default_state,
    write_json_atomic,
)
from mv3dt_installer.steps import StepStatus  # noqa: E402


# ---------------------------------------------------------------------------
# write_json_atomic
# ---------------------------------------------------------------------------


def test_write_json_atomic_produces_valid_json(tmp_path):
    target = tmp_path / "sub" / "data.json"
    write_json_atomic(target, {"a": 1, "b": [1, 2, 3]})

    assert target.exists()
    with open(target, encoding="utf-8") as f:
        assert json.load(f) == {"a": 1, "b": [1, 2, 3]}

    # tmp sibling file must not be left behind after a successful replace.
    assert not target.with_suffix(target.suffix + ".tmp").exists()


def test_write_json_atomic_creates_parent_dir(tmp_path):
    target = tmp_path / "does" / "not" / "exist" / "state.json"
    write_json_atomic(target, {"ok": True})
    assert target.exists()


def test_write_json_atomic_crash_safety_missing_target(tmp_path, monkeypatch):
    """If os.replace never runs, the target must never appear as a
    zero-length or partial file -- it simply must not exist yet."""
    target = tmp_path / "state.json"

    def _boom(*_args, **_kwargs):
        raise OSError("simulated crash before replace")

    monkeypatch.setattr("mv3dt_installer.state.os.replace", _boom)

    with pytest.raises(OSError):
        write_json_atomic(target, {"new": True})

    assert not target.exists()


def test_write_json_atomic_crash_safety_preserves_prior_content(tmp_path, monkeypatch):
    """A crash during a *second* write must leave the previously-committed
    file completely untouched -- never truncated, never partial."""
    target = tmp_path / "state.json"
    write_json_atomic(target, {"prior": "content"})
    prior_bytes = target.read_bytes()

    def _boom(*_args, **_kwargs):
        raise OSError("simulated crash before replace")

    monkeypatch.setattr("mv3dt_installer.state.os.replace", _boom)

    with pytest.raises(OSError):
        write_json_atomic(target, {"new": "content"})

    # The real target is byte-for-byte the prior, fully-written content --
    # never zero-length, never partial.
    assert target.read_bytes() == prior_bytes
    assert json.loads(target.read_text(encoding="utf-8")) == {"prior": "content"}


# ---------------------------------------------------------------------------
# load() defaults / forgiving reads
# ---------------------------------------------------------------------------


def test_load_missing_file_yields_empty_default(tmp_path):
    sm = StateMachine(path=tmp_path / "nested" / "state.json")
    state = sm.load()

    assert state.install_dir == "/opt/mv3dt"
    assert state.reboot_pending is None
    assert set(state.steps.keys()) == set(STEP_IDS)
    assert all(entry.status is StepStatus.PENDING for entry in state.steps.values())


def test_load_corrupted_json_is_treated_as_absent(tmp_path):
    target = tmp_path / "state.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"schema_version": 1, "steps": {truncated', encoding="utf-8")

    sm = StateMachine(path=target)
    state = sm.load()  # must not raise

    assert state.install_dir == "/opt/mv3dt"
    assert state.reboot_pending is None


def test_load_empty_file_is_treated_as_absent(tmp_path):
    target = tmp_path / "state.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")

    sm = StateMachine(path=target)
    state = sm.load()  # must not raise
    assert state.install_dir == "/opt/mv3dt"


# ---------------------------------------------------------------------------
# status / set_status / mark_complete round trips
# ---------------------------------------------------------------------------


def test_status_defaults_to_pending_for_unknown_step(tmp_path):
    sm = StateMachine(path=tmp_path / "state.json")
    assert sm.status("step2_deepstream_sdk") is StepStatus.PENDING


def test_set_status_round_trips(tmp_path):
    path = tmp_path / "state.json"
    sm = StateMachine(path=path)

    sm.set_status("step2_deepstream_sdk", StepStatus.FAILED)
    assert sm.status("step2_deepstream_sdk") is StepStatus.FAILED

    # Persisted to disk, not just in-memory: a fresh StateMachine over the
    # same path sees it too.
    sm2 = StateMachine(path=path)
    assert sm2.status("step2_deepstream_sdk") is StepStatus.FAILED


def test_set_status_does_not_stamp_finished_utc(tmp_path):
    sm = StateMachine(path=tmp_path / "state.json")
    sm.set_status("step3_amc_launcher", StepStatus.USER_ACTION_REQUIRED)
    entry = sm.load().steps["step3_amc_launcher"]
    assert entry.status is StepStatus.USER_ACTION_REQUIRED
    assert entry.finished_utc is None


def test_mark_complete_sets_status_and_finished_utc(tmp_path):
    sm = StateMachine(path=tmp_path / "state.json")
    sm.mark_complete("step1_prerequisites")

    entry = sm.load().steps["step1_prerequisites"]
    assert entry.status is StepStatus.COMPLETE
    assert entry.finished_utc is not None
    assert entry.finished_utc.endswith("Z")


def test_mark_complete_overwrites_prior_status(tmp_path):
    sm = StateMachine(path=tmp_path / "state.json")
    sm.set_status("step4_calib_output_wiring", StepStatus.FAILED)
    sm.mark_complete("step4_calib_output_wiring")

    entry = sm.load().steps["step4_calib_output_wiring"]
    assert entry.status is StepStatus.COMPLETE
    assert entry.finished_utc is not None


# ---------------------------------------------------------------------------
# reboot_pending round trip
# ---------------------------------------------------------------------------


def test_set_and_clear_reboot_pending_round_trips(tmp_path):
    sm = StateMachine(path=tmp_path / "state.json")

    sm.set_reboot_pending("step1_prerequisites", boot_id="3f6b-c1")
    state = sm.load()
    assert state.reboot_pending is not None
    assert state.reboot_pending.requested_by == "step1_prerequisites"
    assert state.reboot_pending.boot_id_at_request == "3f6b-c1"
    assert state.reboot_pending.requested_utc

    sm.clear_reboot_pending()
    assert sm.load().reboot_pending is None


# ---------------------------------------------------------------------------
# all_complete()
# ---------------------------------------------------------------------------


def test_all_complete_false_on_fresh_state(tmp_path):
    sm = StateMachine(path=tmp_path / "state.json")
    assert sm.all_complete() is False


def test_all_complete_false_when_some_steps_pending(tmp_path):
    sm = StateMachine(path=tmp_path / "state.json")
    for step_id in STEP_IDS[:-1]:
        sm.mark_complete(step_id)
    assert sm.all_complete() is False


def test_all_complete_true_when_every_step_complete(tmp_path):
    sm = StateMachine(path=tmp_path / "state.json")
    for step_id in STEP_IDS:
        sm.mark_complete(step_id)
    assert sm.all_complete() is True


def test_mark_skipped_gate_off_leaves_the_step_pending(tmp_path):
    sm = StateMachine(path=tmp_path / "state.json")
    sm.mark_skipped_gate_off("step8_tracking_history")
    assert sm.status("step8_tracking_history") is StepStatus.PENDING
    assert sm.is_skipped_gate_off("step8_tracking_history") is True
    raw = json.loads((tmp_path / "state.json").read_text())
    assert raw["steps"]["step8_tracking_history"] == {
        "status": "PENDING",
        "skipped": "gate off",
    }


def test_mark_skipped_gate_off_demotes_an_earlier_completion(tmp_path):
    """A COMPLETE left by an older release's gate-off skip (or a real run
    before the gate was turned off) must not survive as COMPLETE, or the
    step never runs again once the gate is back on."""
    sm = StateMachine(path=tmp_path / "state.json")
    sm.mark_complete("step7_webapp_integration")
    sm.mark_skipped_gate_off("step7_webapp_integration")
    assert sm.status("step7_webapp_integration") is StepStatus.PENDING
    assert sm.load().steps["step7_webapp_integration"].finished_utc is None


def test_mark_complete_and_set_status_clear_the_skip(tmp_path):
    sm = StateMachine(path=tmp_path / "state.json")
    sm.mark_skipped_gate_off("step8_tracking_history")
    sm.mark_complete("step8_tracking_history")
    assert sm.is_skipped_gate_off("step8_tracking_history") is False
    sm.mark_skipped_gate_off("step8_tracking_history")
    sm.set_status("step8_tracking_history", StepStatus.PENDING)
    assert sm.is_skipped_gate_off("step8_tracking_history") is False


def test_all_complete_counts_gate_off_skips_as_done(tmp_path):
    sm = StateMachine(path=tmp_path / "state.json")
    for step_id in STEP_IDS[:-1]:
        sm.mark_complete(step_id)
    sm.mark_skipped_gate_off(STEP_IDS[-1])
    assert sm.all_complete() is True


# ---------------------------------------------------------------------------
# save() permissions (doc 00 §6.1)
# ---------------------------------------------------------------------------


def test_save_sets_required_permissions(tmp_path):
    path = tmp_path / "subdir" / "state.json"
    sm = StateMachine(path=path)
    sm.save(default_state())

    assert (path.stat().st_mode & 0o777) == 0o644
    assert (path.parent.stat().st_mode & 0o777) == 0o755


# ---------------------------------------------------------------------------
# State / StepEntry / RebootPending (de)serialization sanity
# ---------------------------------------------------------------------------


def test_state_to_dict_matches_schema_shape(tmp_path):
    state = State(
        install_dir="/opt/mv3dt",
        created_utc="2026-07-01T22:00:00Z",
        updated_utc="2026-07-01T22:41:12Z",
        steps={
            "step1_prerequisites": StepEntry(
                status=StepStatus.COMPLETE, finished_utc="2026-07-01T22:10:00Z"
            ),
            "step2_deepstream_sdk": StepEntry(status=StepStatus.PENDING),
        },
        reboot_pending=RebootPending(
            requested_by="step1_prerequisites",
            boot_id_at_request="3f6b-c1",
            requested_utc="2026-07-01T22:10:05Z",
        ),
    )
    d = state.to_dict()

    assert d["schema_version"] == 1
    assert d["steps"]["step1_prerequisites"] == {
        "status": "COMPLETE",
        "finished_utc": "2026-07-01T22:10:00Z",
    }
    # PENDING entries with no finished_utc omit the key entirely, matching
    # the doc 00 §6.2 example.
    assert d["steps"]["step2_deepstream_sdk"] == {"status": "PENDING"}
    assert d["reboot_pending"] == {
        "requested_by": "step1_prerequisites",
        "boot_id_at_request": "3f6b-c1",
        "requested_utc": "2026-07-01T22:10:05Z",
    }


def test_state_round_trips_through_json(tmp_path):
    path = tmp_path / "state.json"
    sm = StateMachine(path=path)
    original = default_state()
    original.install_dir = "/custom/dir"
    sm.save(original)

    reloaded = sm.load()
    assert reloaded.install_dir == "/custom/dir"
    assert reloaded.schema_version == 1
    assert set(reloaded.steps.keys()) == set(STEP_IDS)
