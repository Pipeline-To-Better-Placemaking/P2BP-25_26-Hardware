"""Tests for mv3dt_installer.systemd (plan unit U5).

Run from installer/: `python3 -m pytest tests/test_systemd.py -v`

No test in this file invokes a real systemctl or writes anywhere near
/etc/systemd/system: every lifecycle call goes through a recording fake
runner, and every unit directory is a `tmp_path`.
"""

from __future__ import annotations

import pathlib
import shlex
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mv3dt_installer import logs, shellout, systemd  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_transcript_state():
    """Every test starts with no transcript open, and leaves none dangling."""
    logs._transcript_path = None
    yield
    logs._transcript_path = None


@pytest.fixture(autouse=True)
def _force_no_colour(monkeypatch):
    """Default all tests to a non-tty stderr so plain text is asserted on."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)


@pytest.fixture(autouse=True)
def _no_real_unit_dir(monkeypatch, tmp_path):
    """Belt and braces: point the module constant at a scratch dir so a bug
    in the code under test can never reach /etc/systemd/system."""
    monkeypatch.setattr(systemd, "UNIT_DIR", tmp_path / "unsafe-default")


class RecordingRunner:
    """Stand-in for `subprocess.run` that records argv instead of forking."""

    def __init__(self, returncode: int = 0, stdout: str = ""):
        self.calls: list[list[str]] = []
        self.kwargs: list[dict] = []
        self.returncode = returncode
        self.stdout = stdout

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        self.kwargs.append(kwargs)
        return subprocess.CompletedProcess(list(argv), self.returncode, self.stdout)


# ---------------------------------------------------------------------------
# render_unit
# ---------------------------------------------------------------------------


def test_render_unit_substitutes_every_marker_in_the_real_template():
    """Round-trip against the committed .path template: markers in, values
    out, and no marker syntax survives."""
    rendered = systemd.render_unit(
        ("systemd", "mv3dt-ingest.path.in"),
        {
            "PROJECT": "Warehouse A",
            "SLUG": "warehouse-a",
            "EXPORT_DIR": "/home/op/amc/exports",
        },
    )

    assert "PathChanged=/home/op/amc/exports" in rendered
    assert "PathModified=/home/op/amc/exports" in rendered
    assert "Unit=mv3dt-ingest-warehouse-a.service" in rendered
    assert "WantedBy=multi-user.target" in rendered
    assert "Description=" in rendered
    assert "Warehouse A" in rendered
    assert not _has_marker(rendered)


def test_render_unit_service_template_has_the_expected_execstart():
    rendered = systemd.render_unit(
        ("systemd", "mv3dt-ingest.service.in"),
        {
            "PROJECT": "warehouse",
            "SLUG": "warehouse",
            "EXPORT_DIR": "/exports",
            "INSTALL_DIR": "/opt/mv3dt",
            "INSTALLER_BIN": "/opt/mv3dt/bin/mv3dt-installer",
        },
    )

    assert "Type=oneshot" in rendered
    assert "ConditionPathIsDirectory=/exports" in rendered
    assert _argv(rendered) == [
        "/opt/mv3dt/bin/mv3dt-installer",
        "ingest",
        "--project",
        "warehouse",
        "--non-interactive",
        "--install-dir",
        "/opt/mv3dt",
    ]


def test_render_unit_raises_when_a_marker_survives(tmp_path, monkeypatch):
    """A half-rendered unit file is worse than a missing one."""
    template = tmp_path / "partial.in"
    template.write_text("[Path]\nPathChanged=@EXPORT_DIR@\nUnit=@SLUG@.service\n")
    monkeypatch.setattr(shellout, "asset_path", lambda *parts: template)

    with pytest.raises(ValueError) as excinfo:
        systemd.render_unit(("systemd", "partial.in"), {"EXPORT_DIR": "/exports"})

    message = str(excinfo.value)
    assert "@SLUG@" in message
    assert "@EXPORT_DIR@" not in message


def test_render_unit_ignores_an_at_sign_inside_a_substituted_value(
    tmp_path, monkeypatch
):
    """A value containing an at-sign is not mistaken for a live marker."""
    template = tmp_path / "mail.in"
    template.write_text("[Unit]\nDescription=@OWNER@\n")
    monkeypatch.setattr(shellout, "asset_path", lambda *parts: template)

    rendered = systemd.render_unit(("systemd", "mail.in"), {"OWNER": "a@b.example"})

    assert rendered == "[Unit]\nDescription=a@b.example\n"


def _has_marker(text: str) -> bool:
    return systemd._MARKER_RE.search(text) is not None


def _argv(unit_text: str) -> list[str]:
    """The argument vector systemd would build from a unit's ExecStart=.

    `shlex.split` stands in for systemd's own splitter: the two agree on the
    subset used here (whitespace separation plus double-quoted words), and this
    keeps the assertion about the resulting argv rather than about a string.
    """
    exec_start = systemd._directive(unit_text, "ExecStart")
    assert exec_start is not None, "unit declares no ExecStart="
    return shlex.split(exec_start)


# ---------------------------------------------------------------------------
# render_ingest_units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", ["warehouse-a", "warehouse.a"])
def test_render_ingest_units_produces_both_files(slug):
    """Slugs containing a hyphen and a dot both render cleanly -- neither is
    an instance name, so neither needs systemd path escaping."""
    units = systemd.render_ingest_units(
        project=slug,
        slug=slug,
        export_dir="/home/op/amc/exports",
        install_dir="/opt/mv3dt",
        installer_bin="/opt/mv3dt/bin/mv3dt-installer",
    )

    assert set(units) == {
        f"mv3dt-ingest-{slug}.path",
        f"mv3dt-ingest-{slug}.service",
    }
    path_unit = units[f"mv3dt-ingest-{slug}.path"]
    service_unit = units[f"mv3dt-ingest-{slug}.service"]

    assert f"Unit=mv3dt-ingest-{slug}.service" in path_unit
    assert "PathChanged=/home/op/amc/exports" in path_unit
    assert "[Install]" in path_unit and "WantedBy=multi-user.target" in path_unit

    assert "Type=oneshot" in service_unit
    assert _argv(service_unit)[2:4] == ["--project", slug]
    # The oneshot service is owned by the .path unit's lifecycle: it carries
    # no [Install] section, so it can never be enabled by accident.
    assert "[Install]" not in service_unit

    for text in (path_unit, service_unit):
        assert not _has_marker(text)


def test_render_ingest_units_survives_a_project_name_containing_spaces():
    """Regression: systemd word-splits ExecStart=, and PROJECT_NAME is
    free-form operator text from the AMC GUI (STEP-5 section 3.1, worked
    example "North Lobby #2"). Unquoted, `--project North Lobby #2` would
    deliver `--project North` plus stray positional arguments and every
    trigger would fail."""
    units = systemd.render_ingest_units(
        project="North Lobby #2",
        slug="north-lobby-2",
        export_dir="/home/op/auto-magic-calib/projects/North Lobby #2/exports",
        install_dir="/opt/mv3dt",
        installer_bin="/opt/mv3dt/bin/mv3dt-installer",
    )
    service_unit = units["mv3dt-ingest-north-lobby-2.service"]

    assert _argv(service_unit) == [
        "/opt/mv3dt/bin/mv3dt-installer",
        "ingest",
        "--project",
        "North Lobby #2",
        "--non-interactive",
        "--install-dir",
        "/opt/mv3dt",
    ]

    # The path-valued settings carry the spaced directory unquoted and whole:
    # systemd does not word-split these, so quoting them would make the quotes
    # part of the path.
    path_unit = units["mv3dt-ingest-north-lobby-2.path"]
    watched = "/home/op/auto-magic-calib/projects/North Lobby #2/exports"
    assert f"PathChanged={watched}\n" in path_unit
    assert f"PathModified={watched}\n" in path_unit
    assert f"ConditionPathIsDirectory={watched}\n" in service_unit


def test_render_ingest_units_escapes_a_literal_percent_in_operator_text():
    """Every directive these markers land in expands systemd specifiers, so a
    `%` in free-form project text must be doubled or systemd reads it as one
    and refuses to load the unit."""
    units = systemd.render_ingest_units(
        project="50% Zone",
        slug="50-zone",
        export_dir="/exports/50% Zone",
        install_dir="/opt/mv3dt",
        installer_bin="/opt/mv3dt/bin/mv3dt-installer",
    )
    service_unit = units["mv3dt-ingest-50-zone.service"]

    assert "--project \"50%% Zone\"" in service_unit
    assert "Description=Ingest MV3DT calibration exports for project 50%% Zone" in service_unit
    assert "ConditionPathIsDirectory=/exports/50%% Zone\n" in service_unit
    assert "PathChanged=/exports/50%% Zone\n" in units["mv3dt-ingest-50-zone.path"]
    # No bare `%` survives anywhere systemd would expand one.
    assert "%" not in service_unit.replace("%%", "").replace("%i", "")


def test_enable_now_unescapes_a_percent_in_the_execstart_binary(
    tmp_path, monkeypatch
):
    """The existence check must compare against the real path, not the escaped
    form written into the unit."""
    unit_dir = tmp_path / "units"
    binary = tmp_path / "50% dir" / "mv3dt-installer"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    units = systemd.render_ingest_units(
        project="demo",
        slug="demo",
        export_dir=str(tmp_path / "exports"),
        install_dir=str(tmp_path),
        installer_bin=str(binary),
    )
    runner = RecordingRunner()
    for name, content in units.items():
        systemd.install_unit(name, content, unit_dir=unit_dir, runner=runner)
    monkeypatch.setattr(systemd, "UNIT_DIR", unit_dir)

    runner = RecordingRunner()
    systemd.enable_now("mv3dt-ingest-demo.path", runner=runner)
    assert runner.calls == [
        ["systemctl", "enable", "--now", "mv3dt-ingest-demo.path"]
    ]


def test_render_ingest_units_survives_spaces_in_the_paths_it_execs():
    """The install dir and the binary path are word-split by systemd too."""
    units = systemd.render_ingest_units(
        project="demo",
        slug="demo",
        export_dir="/exports",
        install_dir="/opt/mv3dt lab",
        installer_bin="/opt/mv3dt lab/bin/mv3dt-installer",
    )

    assert _argv(units["mv3dt-ingest-demo.service"]) == [
        "/opt/mv3dt lab/bin/mv3dt-installer",
        "ingest",
        "--project",
        "demo",
        "--non-interactive",
        "--install-dir",
        "/opt/mv3dt lab",
    ]


@pytest.mark.parametrize(
    "bad",
    ['North "Lobby" 2', "back\\slash", "two\nlines", "bell\x07"],
)
def test_render_ingest_units_rejects_values_quoting_cannot_rescue(bad):
    """Quoting handles whitespace; it cannot handle a value that closes the
    quote or ends the directive. Refuse rather than emit a unit that fails on
    every trigger."""
    with pytest.raises(ValueError) as excinfo:
        systemd.render_ingest_units(
            project=bad,
            slug="demo",
            export_dir="/exports",
            install_dir="/opt/mv3dt",
            installer_bin="/opt/mv3dt/bin/mv3dt-installer",
        )
    assert "project" in str(excinfo.value)


def test_render_ingest_units_rejects_an_unquotable_install_dir():
    with pytest.raises(ValueError) as excinfo:
        systemd.render_ingest_units(
            project="demo",
            slug="demo",
            export_dir="/exports",
            install_dir='/opt/"mv3dt"',
            installer_bin="/opt/mv3dt/bin/mv3dt-installer",
        )
    assert "install_dir" in str(excinfo.value)


def test_render_ingest_units_rejects_an_unquotable_export_dir():
    """A newline in `export_dir` would inject an extra directive line into the
    rendered `.path`/`.service` units (e.g. a bogus `ExecStartPre=`) rather
    than being carried as literal path text -- refuse it the same way
    `project`, `install_dir`, and `installer_bin` are refused."""
    with pytest.raises(ValueError) as excinfo:
        systemd.render_ingest_units(
            project="demo",
            slug="demo",
            export_dir="/exports/demo\nExecStartPre=/bin/rm -rf /",
            install_dir="/opt/mv3dt",
            installer_bin="/opt/mv3dt/bin/mv3dt-installer",
        )
    assert "export_dir" in str(excinfo.value)


def test_render_ingest_units_accepts_path_objects():
    units = systemd.render_ingest_units(
        project="p",
        slug="p",
        export_dir=pathlib.Path("/exports"),
        install_dir=pathlib.Path("/opt/mv3dt"),
        installer_bin=pathlib.Path("/opt/mv3dt/bin/mv3dt-installer"),
    )
    assert "PathChanged=/exports" in units["mv3dt-ingest-p.path"]


# ---------------------------------------------------------------------------
# install_unit
# ---------------------------------------------------------------------------


def test_install_unit_writes_then_detects_no_change_then_rewrites_on_drift(
    tmp_path,
):
    runner = RecordingRunner()
    name = "mv3dt-ingest-demo.path"
    content = "[Unit]\nDescription=demo\n"

    # 1. First install -> file created, changed=True.
    assert systemd.install_unit(
        name, content, unit_dir=tmp_path, runner=runner
    ) is True
    target = tmp_path / name
    assert target.read_text() == content
    assert target.stat().st_mode & 0o777 == 0o644

    # 2. Same content again -> no write, changed=False.
    before = target.stat().st_mtime_ns
    assert systemd.install_unit(
        name, content, unit_dir=tmp_path, runner=runner
    ) is False
    assert target.stat().st_mtime_ns == before

    # 3. Someone edits the unit by hand -> rewritten, changed=True.
    target.write_text("[Unit]\nDescription=hand-edited\n")
    assert systemd.install_unit(
        name, content, unit_dir=tmp_path, runner=runner
    ) is True
    assert target.read_text() == content

    # install_unit is pure file I/O: it must never shell out.
    assert runner.calls == []


def test_install_unit_creates_a_missing_unit_dir(tmp_path):
    unit_dir = tmp_path / "etc" / "systemd" / "system"
    assert systemd.install_unit(
        "a.path", "[Unit]\n", unit_dir=unit_dir, runner=RecordingRunner()
    ) is True
    assert (unit_dir / "a.path").is_file()


def test_install_unit_logs_distinguishable_lines(tmp_path, capsys):
    runner = RecordingRunner()
    systemd.install_unit("a.path", "x\n", unit_dir=tmp_path, runner=runner)
    systemd.install_unit("a.path", "x\n", unit_dir=tmp_path, runner=runner)

    err = capsys.readouterr().err
    assert "wrote unit a.path" in err
    assert "already up to date" in err


# ---------------------------------------------------------------------------
# daemon_reload / is_active / is_enabled -> exact argv
# ---------------------------------------------------------------------------


def test_daemon_reload_argv():
    runner = RecordingRunner()
    systemd.daemon_reload(runner=runner)
    assert runner.calls == [["systemctl", "daemon-reload"]]


def test_is_active_argv_and_true_on_zero_exit():
    runner = RecordingRunner(returncode=0)
    assert systemd.is_active("mv3dt-ingest-demo.path", runner=runner) is True
    assert runner.calls == [
        ["systemctl", "is-active", "--quiet", "mv3dt-ingest-demo.path"]
    ]


def test_is_active_false_on_nonzero_exit():
    runner = RecordingRunner(returncode=3)
    assert systemd.is_active("mv3dt-ingest-demo.path", runner=runner) is False


def test_is_enabled_argv_and_true_when_enabled():
    runner = RecordingRunner(returncode=0, stdout="enabled\n")
    assert systemd.is_enabled("mv3dt-ingest-demo.timer", runner=runner) is True
    assert runner.calls == [["systemctl", "is-enabled", "mv3dt-ingest-demo.timer"]]
    assert runner.kwargs[0]["capture_output"] is True
    assert runner.kwargs[0]["text"] is True


def test_is_enabled_true_when_enabled_runtime():
    runner = RecordingRunner(returncode=0, stdout="enabled-runtime\n")
    assert systemd.is_enabled("x.service", runner=runner) is True


@pytest.mark.parametrize("state", ["static", "indirect", "alias", "generated"])
def test_is_enabled_false_for_zero_exit_states_that_are_not_enabled(state):
    """systemctl exits 0 for these too; the v0.4.10 Step 4 verify failure was
    a `static` oneshot service read as enabled."""
    runner = RecordingRunner(returncode=0, stdout=f"{state}\n")
    assert systemd.is_enabled("x.service", runner=runner) is False


def test_is_enabled_false_on_nonzero_exit():
    runner = RecordingRunner(returncode=1, stdout="disabled\n")
    assert systemd.is_enabled("mv3dt-ingest-demo.path", runner=runner) is False


def test_is_enabled_false_on_zero_exit_with_no_output():
    runner = RecordingRunner(returncode=0)
    assert systemd.is_enabled("x.service", runner=runner) is False


def test_status_helpers_treat_an_unusable_runner_result_as_failure():
    """A runner that returns nothing must not read as 'active'."""

    def dud(argv, **kwargs):
        return None

    assert systemd.is_active("x.path", runner=dud) is False
    assert systemd.is_enabled("x.path", runner=dud) is False


# ---------------------------------------------------------------------------
# enable_now
# ---------------------------------------------------------------------------


def _install_pair(unit_dir: pathlib.Path, slug: str, installer_bin: pathlib.Path):
    units = systemd.render_ingest_units(
        project=slug,
        slug=slug,
        export_dir=str(unit_dir / "exports"),
        install_dir=str(unit_dir / "opt"),
        installer_bin=str(installer_bin),
    )
    runner = RecordingRunner()
    for name, content in units.items():
        systemd.install_unit(name, content, unit_dir=unit_dir, runner=runner)
    return units


def test_enable_now_enables_only_the_path_unit_when_the_binary_exists(
    tmp_path, monkeypatch
):
    unit_dir = tmp_path / "units"
    installer_bin = tmp_path / "bin" / "mv3dt-installer"
    installer_bin.parent.mkdir(parents=True)
    installer_bin.write_text("#!/bin/sh\n")
    _install_pair(unit_dir, "demo", installer_bin)
    monkeypatch.setattr(systemd, "UNIT_DIR", unit_dir)

    runner = RecordingRunner()
    systemd.enable_now("mv3dt-ingest-demo.path", runner=runner)

    assert runner.calls == [
        ["systemctl", "enable", "--now", "mv3dt-ingest-demo.path"]
    ]


def test_enable_now_refuses_when_the_execstart_binary_is_missing(
    tmp_path, monkeypatch, capsys
):
    """STEP-3 has not built <install_dir>/bin/mv3dt-installer yet, so enabling
    the path unit today would fail on every trigger."""
    unit_dir = tmp_path / "units"
    installer_bin = tmp_path / "opt" / "bin" / "mv3dt-installer"  # never created
    _install_pair(unit_dir, "demo", installer_bin)
    monkeypatch.setattr(systemd, "UNIT_DIR", unit_dir)

    runner = RecordingRunner()
    systemd.enable_now("mv3dt-ingest-demo.path", runner=runner)

    assert runner.calls == []
    err = capsys.readouterr().err
    assert "refusing to enable mv3dt-ingest-demo.path" in err
    assert str(installer_bin) in err


def test_enable_now_refuses_a_service_unit_with_a_missing_binary_directly(
    tmp_path, monkeypatch
):
    """The check does not depend on the .path -> .service hop."""
    unit_dir = tmp_path / "units"
    installer_bin = tmp_path / "nope" / "mv3dt-installer"
    _install_pair(unit_dir, "demo", installer_bin)
    monkeypatch.setattr(systemd, "UNIT_DIR", unit_dir)

    runner = RecordingRunner()
    systemd.enable_now("mv3dt-ingest-demo.service", runner=runner)
    assert runner.calls == []


def test_enable_now_proceeds_when_the_unit_is_unreadable(tmp_path, monkeypatch):
    """'Cannot tell' is not 'is broken': an unknown unit is still enabled, and
    systemctl gets to report the real error."""
    monkeypatch.setattr(systemd, "UNIT_DIR", tmp_path)

    runner = RecordingRunner()
    systemd.enable_now("some-other.service", runner=runner)

    assert runner.calls == [["systemctl", "enable", "--now", "some-other.service"]]


def test_enable_now_strips_systemd_execstart_prefix_characters(
    tmp_path, monkeypatch
):
    unit_dir = tmp_path / "units"
    unit_dir.mkdir()
    binary = tmp_path / "tool"
    binary.write_text("#!/bin/sh\n")
    (unit_dir / "prefixed.service").write_text(
        f"[Service]\nType=oneshot\nExecStart=-{binary} --flag\n"
    )
    monkeypatch.setattr(systemd, "UNIT_DIR", unit_dir)

    runner = RecordingRunner()
    systemd.enable_now("prefixed.service", runner=runner)
    assert runner.calls == [["systemctl", "enable", "--now", "prefixed.service"]]


def test_exec_start_lookup_does_not_follow_more_than_one_unit_hop(
    tmp_path, monkeypatch
):
    """A Unit= cycle must terminate rather than recurse forever."""
    unit_dir = tmp_path / "units"
    unit_dir.mkdir()
    (unit_dir / "a.path").write_text("[Path]\nUnit=b.path\n")
    (unit_dir / "b.path").write_text("[Path]\nUnit=a.path\n")
    monkeypatch.setattr(systemd, "UNIT_DIR", unit_dir)

    assert systemd._exec_start_binary("a.path") is None


# ---------------------------------------------------------------------------
# Bundled templates are actually present in the asset tree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["mv3dt-ingest.path.in", "mv3dt-ingest.service.in"]
)
def test_templates_are_bundled_under_assets_systemd(name):
    assert shellout.asset_path("systemd", name).is_file()


# ---------------------------------------------------------------------------
# Every bundled template renders to a unit systemd itself accepts as valid.
#
# render_unit()'s own marker-survival check (test_render_unit_raises_when_a_
# marker_survives, above) only catches an unsubstituted @MARKER@ -- it says
# nothing about whether the *substituted* text is a well-formed unit file.
# A directive misplaced between [Unit]/[Service]/[Install] (e.g.
# StartLimitIntervalSec landing in [Service], where systemd silently drops
# it instead of erroring) renders cleanly and passes every other test here,
# but is still wrong. `systemd-analyze verify` is the only thing that
# actually parses the file the way systemd will.
# ---------------------------------------------------------------------------

#: Realistic substitutions for every marker `render_unit` requires per
#: template. Values don't need to resolve to anything real on disk --
#: `systemd-analyze verify` only complains about the ExecStart *binary*
#: being unexecutable, which every case here still triggers (there is no
#: real /opt/mv3dt on a CI runner or dev box); that specific complaint is
#: filtered out below since it's a "not installed yet" fact, not a unit
#: syntax/section problem.
_TEMPLATE_CASES = {
    "mv3dt-driver-handoff.service.in": {
        "WORKER": "/var/lib/mv3dt-installer/driver-handoff/install-driver.sh",
        "RUNFILE": "/opt/mv3dt/downloads/nvidia/NVIDIA.run",
        "STATUS_FILE": "/var/lib/mv3dt-installer/driver-handoff/status",
        "LOG_FILE": "/var/lib/mv3dt-installer/driver-handoff/driver-install.log",
        "DRIVER_VERSION": "595.58.03",
        "BOOT_ID_FILE": "/proc/sys/kernel/random/boot_id",
    },
    "mv3dt-agent.service.in": {"USER": "mv3dt"},
    "mv3dt-pipeline@.service.in": {"USER": "mv3dt"},
    "mv3dt-reporter.service.in": {
        "USER": "mv3dt",
        "INSTALL_DIR": "/opt/mv3dt",
        "INSTALLER_BIN": "/opt/mv3dt/bin/mv3dt-installer",
    },
    "mv3dt-uploader.service.in": {
        "USER": "mv3dt",
        "INSTALL_DIR": "/opt/mv3dt",
        "INSTALLER_BIN": "/opt/mv3dt/bin/mv3dt-installer",
    },
    "mv3dt-ingest.path.in": {
        "EXPORT_DIR": "/home/mv3dt/exports",
        "PROJECT": "test-lab-01",
        "SLUG": "test-lab-01",
    },
    "mv3dt-ingest.service.in": {
        "EXPORT_DIR": "/home/mv3dt/exports",
        "PROJECT": "test-lab-01",
        "SLUG": "test-lab-01",
        "INSTALL_DIR": "/opt/mv3dt",
        "INSTALLER_BIN": "/opt/mv3dt/bin/mv3dt-installer",
    },
}


@pytest.mark.parametrize("name", sorted(_TEMPLATE_CASES))
def test_rendered_template_passes_systemd_analyze_verify(name, tmp_path):
    systemd_analyze = shutil.which("systemd-analyze")
    if systemd_analyze is None:
        pytest.skip("systemd-analyze not on PATH (non-systemd dev environment)")

    rendered = systemd.render_unit(("systemd", name), _TEMPLATE_CASES[name])

    # `mv3dt-pipeline@.service.in` is a genuine systemd template unit --
    # `systemd-analyze verify` needs an instantiated `@<instance>.service`
    # filename (not the bare `@.service` template name) to load it at all.
    unit_name = name.removesuffix(".in")
    if unit_name.endswith("@.service"):
        unit_name = unit_name.replace("@.service", "@test-instance.service")
    unit_path = tmp_path / unit_name
    unit_path.write_text(rendered, encoding="utf-8")

    result = subprocess.run(
        [systemd_analyze, "verify", "--recursive-errors=no", str(unit_path)],
        capture_output=True,
        text=True,
    )
    problems = [
        line
        for line in (result.stdout + result.stderr).splitlines()
        # The only expected complaint on a box without a real
        # /opt/mv3dt/bin/mv3dt-installer: not a unit-syntax/section problem.
        if "is not executable" not in line
    ]
    assert not problems, (
        f"systemd-analyze verify flagged {name}:\n" + "\n".join(problems)
    )
