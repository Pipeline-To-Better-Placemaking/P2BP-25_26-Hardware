"""Tests for the installer's edit protection of rendered files."""

from __future__ import annotations

import io
import json
import os
import pathlib
import stat
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mv3dt_installer import app, logs, managed_files as mf  # noqa: E402


@pytest.fixture
def manifest(tmp_path):
    return mf.Manifest(tmp_path / "state" / "managed-files.json")


@pytest.fixture
def target(tmp_path):
    return tmp_path / "deepstream" / "config.txt"


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def _sha(text):
    return mf._sha256(text.encode("utf-8"))


def test_missing_file_is_written_and_recorded(manifest, target):
    assert mf.write_managed(target, "v1\n", manifest=manifest) is mf.Outcome.WRITTEN
    assert target.read_text() == "v1\n"
    assert manifest.get(target) == _sha("v1\n")
    assert _mode(target) == mf.DEFAULT_MODE
    on_disk = json.loads(manifest.path.read_text())
    assert on_disk == {"version": 1, "files": {str(target): _sha("v1\n")}}


def test_explicit_mode_is_applied(manifest, target):
    mf.write_managed(target, "v1\n", manifest=manifest, mode=0o600)
    assert _mode(target) == 0o600


def test_equal_content_is_unchanged_and_recorded(manifest, target):
    target.parent.mkdir(parents=True)
    target.write_text("v1\n")
    before = target.stat().st_mtime_ns
    assert mf.write_managed(target, "v1\n", manifest=manifest) is mf.Outcome.UNCHANGED
    assert target.stat().st_mtime_ns == before
    assert manifest.get(target) == _sha("v1\n")


def test_untouched_file_is_updated_in_place_keeping_its_mode(manifest, target):
    mf.write_managed(target, "v1\n", manifest=manifest)
    target.chmod(0o640)
    assert mf.write_managed(target, "v2\n", manifest=manifest) is mf.Outcome.WRITTEN
    assert target.read_text() == "v2\n"
    assert _mode(target) == 0o640
    assert manifest.get(target) == _sha("v2\n")
    assert not mf.pending_path(target).exists()


def test_operator_edit_is_kept_with_installer_new(manifest, target, capsys):
    mf.write_managed(target, "v1\n", manifest=manifest)
    target.write_text("operator\n")
    outcome = mf.write_managed(target, "v2\n", manifest=manifest, label="tracker")
    assert outcome is mf.Outcome.KEPT_MODIFIED
    assert target.read_text() == "operator\n"
    assert mf.pending_path(target).read_text() == "v2\n"
    assert manifest.get(target) == _sha("v1\n")
    err = capsys.readouterr().err
    assert "kept tracker" in err and "operator edited" in err
    assert str(mf.pending_path(target)) in err


def test_pre_manifest_differing_file_is_kept(manifest, target):
    target.parent.mkdir(parents=True)
    target.write_text("live tuning\n")
    target.chmod(0o664)
    assert mf.write_managed(target, "template\n", manifest=manifest) is (
        mf.Outcome.KEPT_MODIFIED
    )
    assert target.read_text() == "live tuning\n"
    pending = mf.pending_path(target)
    assert pending.read_text() == "template\n"
    assert _mode(pending) == 0o664
    assert manifest.load() == {str(target): None}


def test_kept_file_does_not_rewrite_identical_pending(manifest, target):
    target.parent.mkdir(parents=True)
    target.write_text("live\n")
    mf.write_managed(target, "new\n", manifest=manifest)
    pending = mf.pending_path(target)
    os.utime(pending, ns=(1, 1))
    mf.write_managed(target, "new\n", manifest=manifest)
    assert pending.stat().st_mtime_ns == 1


def test_matching_content_clears_a_stale_pending(manifest, target):
    target.parent.mkdir(parents=True)
    target.write_text("live\n")
    mf.write_managed(target, "new\n", manifest=manifest)
    assert mf.write_managed(target, "live\n", manifest=manifest) is mf.Outcome.UNCHANGED
    assert not mf.pending_path(target).exists()
    assert mf.status(manifest)[0].state is mf.FileState.CLEAN


def test_failed_write_leaves_target_and_no_temp(manifest, target, monkeypatch):
    mf.write_managed(target, "v1\n", manifest=manifest)

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(mf.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        mf.write_managed(target, "v2\n", manifest=manifest)
    assert target.read_text() == "v1\n"
    assert sorted(p.name for p in target.parent.iterdir()) == [target.name]


def test_corrupt_manifest_is_backed_up_and_treated_as_empty(manifest, target):
    manifest.path.parent.mkdir(parents=True)
    manifest.path.write_text("{not json")
    target.parent.mkdir(parents=True)
    target.write_text("old\n")
    assert mf.write_managed(target, "new\n", manifest=manifest) is (
        mf.Outcome.KEPT_MODIFIED
    )
    backup = manifest.path.with_name(manifest.path.name + mf.CORRUPT_SUFFIX)
    assert backup.read_text() == "{not json"
    assert manifest.load() == {str(target): None}


def test_wrong_shape_manifest_is_corrupt(manifest):
    manifest.path.parent.mkdir(parents=True)
    manifest.path.write_text(json.dumps({"files": {"/a": 3}}))
    assert manifest.load() == {}
    assert not manifest.path.exists()


def test_missing_manifest_is_empty(manifest):
    assert manifest.load() == {}
    assert mf.status(manifest) == []


def test_status_reports_each_state(manifest, tmp_path):
    clean = tmp_path / "clean.txt"
    modified = tmp_path / "modified.txt"
    pending = tmp_path / "pending.txt"
    missing = tmp_path / "missing.txt"
    for path in (clean, modified, missing):
        mf.write_managed(path, "v1\n", manifest=manifest)
    modified.write_text("edited\n")
    missing.unlink()
    pending.write_text("live\n")
    mf.write_managed(pending, "v1\n", manifest=manifest)
    states = {entry.path.name: entry.state for entry in mf.status(manifest)}
    assert states == {
        "clean.txt": mf.FileState.CLEAN,
        "modified.txt": mf.FileState.MODIFIED,
        "pending.txt": mf.FileState.PENDING_NEW,
        "missing.txt": mf.FileState.MISSING,
    }


def test_diff_shows_current_against_new(manifest, target):
    target.parent.mkdir(parents=True)
    target.write_text("a\nlive\n")
    mf.write_managed(target, "a\nnew\n", manifest=manifest)
    text = mf.diff(target)
    assert "-live" in text and "+new" in text
    assert str(mf.pending_path(target)) in text
    target.write_text("a\nnew\n")
    mf.pending_path(target).unlink()
    assert mf.diff(target) == ""


def test_accept_replaces_records_and_removes_pending(manifest, target):
    target.parent.mkdir(parents=True)
    target.write_text("live\n")
    target.chmod(0o600)
    mf.write_managed(target, "new\n", manifest=manifest)
    mf.accept(target, manifest=manifest)
    assert target.read_text() == "new\n"
    assert _mode(target) == 0o600
    assert manifest.get(target) == _sha("new\n")
    assert not mf.pending_path(target).exists()
    assert mf.write_managed(target, "newer\n", manifest=manifest) is mf.Outcome.WRITTEN


def test_accept_without_pending_raises(manifest, target):
    with pytest.raises(FileNotFoundError):
        mf.accept(target, manifest=manifest)


def test_accept_all_takes_every_pending(manifest, tmp_path):
    paths = [tmp_path / f"f{i}.txt" for i in range(3)]
    for path in paths[:2]:
        path.write_text("live\n")
    for path in paths:
        mf.write_managed(path, "new\n", manifest=manifest)
    assert mf.accept_all(manifest) == paths[:2]
    assert all(p.read_text() == "new\n" for p in paths)
    assert {e.state for e in mf.status(manifest)} == {mf.FileState.CLEAN}


# -- `configs` subcommand ---------------------------------------------------


class _Ctx:
    log = logs.log


def _run(argv, manifest):
    out = io.StringIO()
    rc = mf.handle_configs_subcommand(argv, _Ctx(), manifest=manifest, out=out)
    return rc, out.getvalue()


def test_configs_is_registered_with_read_only_status_and_diff():
    registration = app.SUBCOMMAND_REGISTRY["configs"]
    assert registration.handler is mf.handle_configs_subcommand
    predicate = registration.requires_root
    assert predicate(["status"]) is False
    assert predicate(["diff", "/x"]) is False
    assert predicate(["accept", "/x"]) is True
    assert predicate(["accept-all"]) is True
    assert predicate([]) is True


def test_configs_status_diff_accept_flow(manifest, target):
    rc, out = _run(["status"], manifest)
    assert rc == 0 and "no installer-managed files" in out
    target.parent.mkdir(parents=True)
    target.write_text("live\n")
    mf.write_managed(target, "new\n", manifest=manifest)
    rc, out = _run(["status"], manifest)
    assert rc == 0 and f"pending-new  {target}" in out
    rc, out = _run(["diff"], manifest)
    assert rc == 0 and "+new" in out
    rc, out = _run(["diff", str(target)], manifest)
    assert rc == 0 and "-live" in out
    assert _run(["accept", str(target)], manifest)[0] == 0
    assert target.read_text() == "new\n"
    assert _run(["diff", str(target)], manifest)[0] == 1
    assert _run(["accept", str(target)], manifest)[0] == 1
    assert _run(["accept"], manifest)[0] == 2


def test_configs_accept_all(manifest, target):
    target.parent.mkdir(parents=True)
    target.write_text("live\n")
    mf.write_managed(target, "new\n", manifest=manifest)
    assert _run(["accept-all", "--non-interactive"], manifest)[0] == 0
    assert target.read_text() == "new\n"
