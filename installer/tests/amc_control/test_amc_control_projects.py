"""The AMC web project store: folder rule, bindings and safe file access.

Run from installer/: `python3 -m pytest tests/amc_control/test_amc_control_projects.py -v`

Everything happens under pytest's tmp_path; no AMC and no web app.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mv3dt_amc_control import projects as P

NOW = 1794879242


class _Clock:
    def __init__(self, t: float = NOW):
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def store(tmp_path: Path, clock: _Clock) -> P.ProjectStore:
    return P.ProjectStore(tmp_path / "amc_projects", clock=clock)


# --- folder rule -------------------------------------------------------------

def test_folder_regex_is_the_spec_rule():
    assert P.FOLDER_RE.pattern == r"^[a-z0-9](?:[a-z0-9-]{0,46}[a-z0-9])?$"


@pytest.mark.parametrize("name", [
    "a", "7", "ab", "a-b", "a--b", "kevin-living-room", "x" * 48, "a" + "-" * 46 + "b",
])
def test_folder_accepted(name):
    assert P.validate_folder(name) == name


@pytest.mark.parametrize("name", [
    "", "x" * 49, "-a", "a-", "-", "Abc", "a_b", "a b", "a.b", "a/b", "..", "é",
    "a\n", None, 5,
])
def test_folder_rejected(name):
    with pytest.raises(P.InvalidFolder):
        P.validate_folder(name)


def test_invalid_folder_is_a_value_error():
    assert issubclass(P.InvalidFolder, ValueError)


@pytest.mark.parametrize("title,folder", [
    ("Kevin Living Room", "kevin-living-room"),
    ("  Lab #3 -- North_Wing!  ", "lab-3-north-wing"),
    ("ALLCAPS", "allcaps"),
    ("Café 2026", "caf-2026"),
    ("!!!", ""),
    ("", ""),
    ("a" * 60, "a" * 48),
    ("a" * 47 + " b", "a" * 47),  # cut leaves a trailing hyphen, trimmed
])
def test_suggest_folder(title, folder):
    got = P.suggest_folder(title)
    assert got == folder
    if got:
        P.validate_folder(got)


def test_default_projects_root():
    assert P.default_projects_root({}) == Path("/opt/mv3dt/amc_projects")
    assert P.default_projects_root({"MV3DT_INSTALL_DIR": "/srv/m"}) == Path("/srv/m/amc_projects")
    assert P.default_projects_root(
        {"MV3DT_INSTALL_DIR": "/srv/m", "AMC_PROJECTS_DIR": "/data/p"}) == Path("/data/p")


# --- bindings ----------------------------------------------------------------

def test_create_lays_out_folder_and_binding(store):
    b = store.create("pid-1", "kevin-living-room", "Kevin Living Room")
    root = store.folder_path("kevin-living-room")
    assert root == store.root / "kevin-living-room"
    for sub in P.SUBDIRS:
        assert (root / sub).is_dir()
    assert b.state == P.STATE_PROVISIONING
    assert (b.created_at, b.updated_at, b.last_run_at, b.calib_runs) == (NOW, NOW, None, 0)
    on_disk = json.loads((root / "project.json").read_text())
    assert on_disk == {
        "schema_version": 1, "ProjectId": "pid-1", "Folder": "kevin-living-room",
        "Title": "Kevin Living Room", "AmcProjectId": None, "AmcProjectName": None,
        "State": "provisioning", "Error": "", "CreatedAt": NOW, "UpdatedAt": NOW,
        "LastRunAt": None, "CalibRuns": 0,
    }


def test_to_wire_drops_schema_version_and_adds_path(store):
    b = store.create("pid-1", "lab", "Lab")
    wire = b.to_wire(store.folder_path("lab"))
    assert "schema_version" not in wire
    assert wire["Path"] == str(store.root / "lab")
    assert wire["ProjectId"] == "pid-1"


def test_lookups_and_list(store):
    store.create("p2", "zeta", "Z")
    store.create("p1", "alpha", "A")
    assert store.get_by_pid("p1").folder == "alpha"
    assert store.get_by_folder("zeta").project_id == "p2"
    assert store.get_by_pid("nope") is None
    assert store.get_by_folder("nope") is None
    assert store.get_by_folder("../zeta") is None
    assert [b.folder for b in store.list()] == ["alpha", "zeta"]


def test_list_on_missing_root_is_empty(tmp_path):
    assert P.ProjectStore(tmp_path / "absent").list() == []


def test_create_validates_folder(store):
    with pytest.raises(P.InvalidFolder):
        store.create("p1", "Bad_Name", "x")
    assert not store.root.exists() or not any(store.root.iterdir())


def test_folder_taken_by_another_pid(store):
    store.create("p1", "lab", "Lab")
    with pytest.raises(P.FolderTaken):
        store.create("p2", "lab", "Lab")


def test_pid_already_bound_elsewhere(store):
    store.create("p1", "lab", "Lab")
    with pytest.raises(P.ProjectAlreadyBound):
        store.create("p1", "other", "Lab")
    assert not (store.root / "other").exists()


def test_create_same_pid_and_folder_returns_existing_unchanged(store, clock):
    first = store.create("p1", "lab", "Lab")
    store.update("lab", state=P.STATE_FAILED, error="boom")
    clock.t += 100
    again = store.create("p1", "lab", "New Title")
    assert again.state == P.STATE_FAILED
    assert again.title == "Lab"
    assert again.updated_at == first.updated_at


def test_update_and_save_bump_updated_at(store, clock):
    store.create("p1", "lab", "Lab")
    clock.t += 30
    b = store.update("lab", state=P.STATE_READY, amc_project_id="20261117_1", amc_project_name="lab")
    assert (b.state, b.updated_at, b.created_at) == ("ready", NOW + 30, NOW)
    assert store.get_by_pid("p1") == b
    clock.t += 5
    saved = store.save(store.get_by_folder("lab"))
    assert saved.updated_at == NOW + 35
    assert store.get_by_folder("lab").updated_at == NOW + 35


def test_update_rejects_bad_changes(store):
    store.create("p1", "lab", "Lab")
    with pytest.raises(TypeError):
        store.update("lab", nonsense=1)
    with pytest.raises(ValueError):
        store.update("lab", folder="other")
    with pytest.raises(ValueError):
        store.update("lab", project_id="p2")
    with pytest.raises(ValueError):
        store.update("lab", state="bogus")
    with pytest.raises(P.ProjectNotFound):
        store.update("missing", state="ready")
    assert store.get_by_folder("lab").state == P.STATE_PROVISIONING


@pytest.mark.parametrize("content", [
    "not json",
    "[]",
    json.dumps({"schema_version": 2}),
    "MISSING_KEY",
    "WRONG_FOLDER",
    "BAD_STATE",
])
def test_malformed_binding_is_invisible(store, content):
    good = store.create("p1", "lab", "Lab").to_json()
    if content == "MISSING_KEY":
        del good["State"]
        content = json.dumps(good)
    elif content == "WRONG_FOLDER":
        good["Folder"] = "elsewhere"
        content = json.dumps(good)
    elif content == "BAD_STATE":
        good["State"] = "weird"
        content = json.dumps(good)
    (store.root / "lab" / "project.json").write_text(content)
    assert store.get_by_folder("lab") is None
    assert store.get_by_pid("p1") is None
    assert store.list() == []
    # The folder is free again: a malformed binding does not hold it.
    assert store.create("p2", "lab", "Lab").project_id == "p2"


def test_missing_binding_and_stray_entries_ignored(store):
    store.create("p1", "lab", "Lab")
    (store.root / "empty-dir").mkdir()
    (store.root / "Not_A_Folder").mkdir()
    (store.root / "file.txt").write_text("x")
    os.symlink(store.root / "lab", store.root / "alias")
    assert [b.folder for b in store.list()] == ["lab"]
    assert store.get_by_folder("empty-dir") is None
    assert store.get_by_folder("alias") is None


def test_atomic_write_leaves_no_temp_files(store):
    store.create("p1", "lab", "Lab")
    for i in range(5):
        store.update("lab", calib_runs=i)
    names = sorted(p.name for p in (store.root / "lab").iterdir())
    assert names == sorted(["project.json", *P.SUBDIRS])


def test_failed_write_keeps_old_binding_and_no_temp(store, monkeypatch):
    store.create("p1", "lab", "Lab")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(P.os, "replace", boom)
    with pytest.raises(OSError):
        store.update("lab", state=P.STATE_READY)
    monkeypatch.undo()
    assert store.get_by_folder("lab").state == P.STATE_PROVISIONING
    assert not [p for p in (store.root / "lab").iterdir() if p.name.endswith(".tmp")]


# --- files -------------------------------------------------------------------

@pytest.fixture
def files_store(store, tmp_path):
    store.create("p1", "lab", "Lab")
    base = store.folder_path("lab")
    (base / "footage" / "cam_00.mp4").write_bytes(b"0" * 10)
    (base / "footage" / "cam_01.mp4").write_bytes(b"1" * 3)
    (base / "calibration" / "20261117T101500Z").mkdir()
    (base / "calibration" / "20261117T101500Z" / "result.zip").write_bytes(b"zz")
    (base / "notes.txt").write_text("hi")
    os.utime(base / "notes.txt", (NOW + 0.7, NOW + 0.7))
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    os.symlink(outside, base / "escape.txt")
    os.symlink(tmp_path, base / "escape-dir")
    os.symlink(base / "notes.txt", base / "footage" / "inside-link.txt")
    os.symlink(base / "project.json", base / "binding-link")
    return store


def test_list_files(files_store):
    entries = files_store.list_files("lab")
    assert [e["Path"] for e in entries] == [
        "calibration/20261117T101500Z/result.zip",
        "footage/cam_00.mp4",
        "footage/cam_01.mp4",
        "notes.txt",
    ]
    by_path = {e["Path"]: e for e in entries}
    assert by_path["footage/cam_00.mp4"]["Size"] == 10
    assert by_path["notes.txt"]["ModifiedAt"] == NOW
    assert all(isinstance(e["ModifiedAt"], int) for e in entries)
    assert set(entries[0]) == {"Path", "Size", "ModifiedAt"}


def test_list_files_keeps_nested_project_json(files_store):
    base = files_store.folder_path("lab")
    (base / "runs" / "project.json").write_text("{}")
    assert "runs/project.json" in [e["Path"] for e in files_store.list_files("lab")]


def test_resolve_file_allows_files_inside(files_store):
    base = files_store.folder_path("lab").resolve()
    assert files_store.resolve_file("lab", "footage/cam_00.mp4") == base / "footage" / "cam_00.mp4"
    assert files_store.resolve_file("lab", "./notes.txt") == base / "notes.txt"
    # A link that stays inside the folder resolves to its target.
    assert files_store.resolve_file("lab", "footage/inside-link.txt") == base / "notes.txt"


@pytest.mark.parametrize("rel", [
    "",
    "../lab/notes.txt",
    "footage/../notes.txt",
    "footage/../../secret.txt",
    "/etc/passwd",
    "project.json",
    "./project.json",
    "binding-link",
    "escape.txt",
    "escape-dir/secret.txt",
    "footage",
    "calibration/20261117T101500Z",
    "missing.txt",
    "footage\\cam_00.mp4",
    "notes.txt\x00",
])
def test_resolve_file_refuses(files_store, rel):
    with pytest.raises(FileNotFoundError):
        files_store.resolve_file("lab", rel)


def test_resolve_file_absolute_inside_path_refused(files_store):
    absolute = str(files_store.folder_path("lab") / "notes.txt")
    with pytest.raises(FileNotFoundError):
        files_store.resolve_file("lab", absolute)


def test_resolve_file_bad_folder(files_store):
    with pytest.raises(FileNotFoundError):
        files_store.resolve_file("../lab", "notes.txt")
    with pytest.raises(FileNotFoundError):
        files_store.resolve_file("absent", "notes.txt")
