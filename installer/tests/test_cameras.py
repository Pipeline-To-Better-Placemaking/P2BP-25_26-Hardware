"""Tests for mv3dt_installer.cameras (doc 00 §15).

Run from installer/: `python3 -m pytest tests/test_cameras.py -v`

No test opens a socket or spawns a real arp-scan/ffmpeg/ffprobe/ping --
every subprocess goes through an injected fake `runner`.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mv3dt_installer import cameras  # noqa: E402

_SEED_PATH = pathlib.Path(__file__).resolve().parents[2] / "laptop" / "config" / "cameras.yml"


def _cp(argv, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# normalize_mac / matches_oui
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["d0:3b:f4:01:52:79", "D0-3B-F4-01-52-79", "d03b.f401.5279", "D0:3b:F4:01:52:79"],
)
def test_normalize_mac_accepts_colon_dash_and_cisco_dot_forms(raw):
    assert cameras.normalize_mac(raw) == "d0:3b:f4:01:52:79"


def test_normalize_mac_rejects_garbage():
    with pytest.raises(ValueError):
        cameras.normalize_mac("not-a-mac")


def test_matches_oui_true_for_fleet_prefix():
    assert cameras.matches_oui("d0:3b:f4:01:52:79") is True


def test_matches_oui_false_for_other_vendor():
    assert cameras.matches_oui("aa:bb:cc:01:52:79") is False


def test_matches_oui_false_for_malformed_mac_not_an_error():
    assert cameras.matches_oui("garbage") is False


# ---------------------------------------------------------------------------
# parse_inventory / render_inventory
# ---------------------------------------------------------------------------


def test_render_then_parse_round_trips():
    original = [
        cameras.Camera(
            id="c1", mac="d0:3b:f4:01:52:79", ip="169.254.1.2", position="top-left"
        ),
        cameras.Camera(
            id="c2",
            mac="d0:3b:f4:01:52:91",
            ip="169.254.1.3",
            position="",
            enabled=False,
            stream_ok=True,
        ),
    ]

    text = cameras.render_inventory(original, header="# a header")
    parsed = cameras.parse_inventory(text)

    assert parsed == original


def test_parse_inventory_against_the_real_committed_seed_file():
    text = _SEED_PATH.read_text(encoding="utf-8")
    parsed = cameras.parse_inventory(text)

    assert len(parsed) == 8
    ids = [cam.id for cam in parsed]
    assert ids == [f"c{i}" for i in range(1, 9)]
    # The seed predates MAC tracking entirely -- doc 00 §15.1.
    assert all(cam.mac == "" for cam in parsed)
    assert parsed[0].ip == "169.254.9.14"
    assert parsed[0].position == "top-right"
    assert parsed[3].enabled is False  # c4
    assert all(cam.rtsp_path == "/Streaming/Channels/101" for cam in parsed)


def test_parse_inventory_on_malformed_text_returns_empty():
    assert cameras.parse_inventory("not yaml at all\njust text") == []


def test_parse_inventory_ignores_comments_and_blank_lines():
    text = (
        "# header comment\n"
        "\n"
        "cameras:\n"
        "  - id: c1\n"
        "    # a comment inside the block\n"
        "    ip: 1.2.3.4\n"
        '    position: "top-left"\n'
    )
    parsed = cameras.parse_inventory(text)
    assert len(parsed) == 1
    assert parsed[0].ip == "1.2.3.4"


# ---------------------------------------------------------------------------
# candidate_interfaces
# ---------------------------------------------------------------------------


def _sysfs_iface(net_dir, name, *, carrier="1", flags="0x1003"):
    iface = net_dir / name
    iface.mkdir()
    if carrier is not None:
        (iface / "carrier").write_text(carrier + "\n")
    (iface / "flags").write_text(flags + "\n")


def test_candidate_interfaces_drops_excluded_names_dead_links_and_noarp(tmp_path):
    for name in ("eth0", "lo", "docker0", "veth1234", "br-abc", "virbr0"):
        _sysfs_iface(tmp_path, name)
    _sysfs_iface(tmp_path, "eth1", carrier="0")           # no link
    _sysfs_iface(tmp_path, "eth2", carrier=None)          # admin down: carrier unreadable
    _sysfs_iface(tmp_path, "tailscale0", flags="0x10d1")  # POINTOPOINT | NOARP
    _sysfs_iface(tmp_path, "wlan0")

    result = cameras.candidate_interfaces(net_dir=tmp_path)

    assert result == ["eth0", "wlan0"]


def test_candidate_interfaces_keeps_link_up_interface_without_ipv4(tmp_path):
    # The PoE port on a link-local camera LAN often has no address at all;
    # it must reach discover() so it can be reported, not vanish.
    _sysfs_iface(tmp_path, "enp8s0")

    def runner(argv, **kwargs):
        return _cp(argv, 0, stdout="")

    assert cameras.candidate_interfaces(runner=runner, net_dir=tmp_path) == ["enp8s0"]


def test_candidate_interfaces_missing_net_dir_returns_empty(tmp_path):
    assert cameras.candidate_interfaces(net_dir=tmp_path / "does-not-exist") == []


# ---------------------------------------------------------------------------
# discover
# ---------------------------------------------------------------------------

_ARP_SCAN_OUTPUT = (
    "Interface: eth0, type: EN10MB, MAC: 00:00:00:00:00:00, IPv4: 169.254.1.5\n"
    "Starting arp-scan\n"
    "169.254.1.10\td0:3b:f4:01:52:79\tUnknown\n"
    "169.254.1.11\taa:bb:cc:dd:ee:ff\tUnknown\n"
)


def test_discover_via_arp_scan_filters_by_oui_and_records_unmatched():
    def runner(argv, **kwargs):
        if argv[0] == "arp-scan":
            return _cp(argv, 0, stdout=_ARP_SCAN_OUTPUT)
        if argv[:5] == ["ip", "-4", "-o", "addr", "show"]:
            return _cp(argv, 0, stdout="inet 169.254.1.5/16")
        return _cp(argv, 1)

    result = cameras.discover(interfaces=["eth0"], runner=runner)

    assert result.tool == "arp-scan"
    assert len(result.cameras) == 1
    assert result.cameras[0].mac == "d0:3b:f4:01:52:79"
    assert result.cameras[0].ip == "169.254.1.10"
    assert result.unmatched == ["aa:bb:cc:dd:ee:ff"]


def _addr_runner(addresses, calls, *, arp_stdout=""):
    """Fake runner: `addresses` maps iface -> `ip -4 -o addr` stdout."""

    def runner(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "arp-scan":
            return _cp(argv, 0, stdout=arp_stdout)
        if argv[:5] == ["ip", "-4", "-o", "addr", "show"]:
            return _cp(argv, 0, stdout=addresses.get(argv[-1], ""))
        return _cp(argv, 1)

    return runner


def test_discover_sweeps_only_the_cidr_with_the_in_range_source_address():
    calls = []
    runner = _addr_runner(
        {
            "enp8s0": "2: enp8s0    inet 169.254.3.134/16 brd 169.254.255.255",
            # A campus network: --localnet here would be 16 million hosts.
            "wlp9s0": "3: wlp9s0    inet 10.10.217.57/8 brd 10.255.255.255",
        },
        calls,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cameras, "candidate_interfaces", lambda runner: ["enp8s0", "wlp9s0"])
        result = cameras.discover(runner=runner)

    arp_calls = [c for c in calls if c[0] == "arp-scan"]
    assert arp_calls == [
        ["arp-scan", "--interface", "enp8s0", "--arpspa", "169.254.3.134", "169.254.0.0/16"]
    ]
    assert not any("--localnet" in c for c in calls)
    assert result.interfaces == ["enp8s0"]
    assert result.skipped == ["wlp9s0"]


def test_discover_picks_the_in_range_address_when_an_interface_has_several():
    calls = []
    runner = _addr_runner(
        {"eth0": "inet 192.168.1.20/24 brd 192.168.1.255\ninet 169.254.9.9/16 brd 169.254.255.255"},
        calls,
    )

    cameras.discover(interfaces=["eth0"], runner=runner)

    arp = [c for c in calls if c[0] == "arp-scan"]
    assert arp == [["arp-scan", "--interface", "eth0", "--arpspa", "169.254.9.9", "169.254.0.0/16"]]


def test_discover_sweeps_an_explicit_interface_even_outside_the_cidr():
    calls = []
    runner = _addr_runner({"eth0": "inet 10.0.0.5/24"}, calls)

    result = cameras.discover(interfaces=["eth0"], runner=runner)

    arp = [c for c in calls if c[0] == "arp-scan"]
    assert arp == [["arp-scan", "--interface", "eth0", "169.254.0.0/16"]]
    assert result.skipped == []


def test_discover_reports_an_unaddressed_interface_and_runs_no_scan(capsys):
    calls = []
    runner = _addr_runner({}, calls)

    result = cameras.discover(interfaces=["enp8s0"], runner=runner)

    assert not any(c[0] == "arp-scan" for c in calls)
    assert result.tool == "none"
    assert result.unaddressed == ["enp8s0"]
    assert result.cameras == [] and result.interfaces == []
    err = capsys.readouterr().err
    assert "enp8s0 has a link but no IPv4 address" in err
    assert "ipv4.method link-local" in err


def test_discover_with_an_invalid_cidr_warns_and_runs_no_scan(capsys):
    calls = []
    runner = _addr_runner({"eth0": "inet 169.254.1.5/16"}, calls)

    result = cameras.discover(interfaces=["eth0"], cidr="169.254.0.0/33", runner=runner)

    assert result.tool == "none"
    assert result.cameras == [] and calls == []
    assert "not a valid CIDR" in capsys.readouterr().err


def test_discover_deduplicates_unmatched_hosts_by_mac():
    calls = []
    proxy_arp = "".join(
        f"169.254.0.{n}\taa:bb:cc:dd:ee:ff\tUnknown\n" for n in range(1, 50)
    )
    runner = _addr_runner({"eth0": "inet 169.254.1.5/16"}, calls, arp_stdout=proxy_arp)

    result = cameras.discover(interfaces=["eth0"], runner=runner)

    assert result.unmatched == ["aa:bb:cc:dd:ee:ff"]


def test_discover_falls_back_to_ip_neigh_when_arp_scan_is_absent():
    def runner(argv, **kwargs):
        if argv[0] == "arp-scan":
            raise FileNotFoundError("arp-scan not found")
        if argv[:5] == ["ip", "-4", "-o", "addr", "show"]:
            return _cp(argv, 0, stdout="inet 169.254.1.5/16")
        if argv[0] == "ping":
            return _cp(argv, 0)
        if argv[:3] == ["ip", "-4", "neigh"]:
            return _cp(
                argv,
                0,
                stdout="169.254.1.10 dev eth0 lladdr d0:3b:f4:01:52:79 REACHABLE\n",
            )
        return _cp(argv, 1)

    result = cameras.discover(interfaces=["eth0"], prime_ips=["169.254.1.10"], runner=runner)

    assert result.tool == "ip-neigh"
    assert len(result.cameras) == 1
    assert result.cameras[0].mac == "d0:3b:f4:01:52:79"


def test_discover_ip_neigh_pings_every_prime_ip_first():
    pinged = []

    def runner(argv, **kwargs):
        if argv[0] == "arp-scan":
            raise FileNotFoundError()
        if argv[0] == "ping":
            pinged.append(argv[-1])
            return _cp(argv, 0)
        if argv[:5] == ["ip", "-4", "-o", "addr", "show"]:
            return _cp(argv, 0, stdout="inet 169.254.1.5/16")
        return _cp(argv, 0, stdout="")

    cameras.discover(prime_ips=["1.2.3.4", "1.2.3.5"], interfaces=["eth0"], runner=runner)

    assert pinged == ["1.2.3.4", "1.2.3.5"]


# ---------------------------------------------------------------------------
# probe_rtsp / grab_still
# ---------------------------------------------------------------------------


def test_probe_rtsp_true_on_success_and_password_never_reaches_a_log_call(capsys):
    cam = cameras.Camera(id="c1", mac="d0:3b:f4:01:52:79", ip="1.2.3.4", position="top-left")

    def runner(argv, **kwargs):
        assert "topsecret" in argv[-1]  # only ever in the argv, not logged
        return _cp(argv, 0)

    assert cameras.probe_rtsp(cam, user="admin", password="topsecret", runner=runner) is True
    assert "topsecret" not in capsys.readouterr().err


def test_probe_rtsp_false_on_failure():
    cam = cameras.Camera(id="c1", mac="d0:3b:f4:01:52:79", ip="1.2.3.4", position="top-left")
    assert (
        cameras.probe_rtsp(cam, user="a", password="b", runner=lambda argv, **kw: _cp(argv, 1))
        is False
    )


def test_grab_still_returns_path_when_ffmpeg_succeeds_and_file_exists(tmp_path):
    cam = cameras.Camera(id="c1", mac="d0:3b:f4:01:52:79", ip="1.2.3.4", position="top-left")
    dest = tmp_path / "cameras" / "still-d03bf4015279.jpg"

    def runner(argv, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake jpeg")
        return _cp(argv, 0)

    result = cameras.grab_still(cam, user="admin", password="pw", dest=dest, runner=runner)

    assert result == dest
    assert dest.is_file()


def test_grab_still_returns_none_when_ffmpeg_fails(tmp_path):
    cam = cameras.Camera(id="c1", mac="d0:3b:f4:01:52:79", ip="1.2.3.4", position="top-left")
    dest = tmp_path / "cameras" / "still.jpg"

    result = cameras.grab_still(
        cam, user="a", password="b", dest=dest, runner=lambda argv, **kw: _cp(argv, 1)
    )

    assert result is None


def _times_out(argv, **kwargs):
    raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))


def test_probe_rtsp_is_bounded_and_false_on_timeout():
    cam = cameras.Camera(id="c1", mac="d0:3b:f4:01:52:79", ip="1.2.3.4", position="")
    seen = {}

    def runner(argv, **kwargs):
        seen.update(kwargs)
        return _times_out(argv, **kwargs)

    assert cameras.probe_rtsp(cam, user="a", password="b", runner=runner) is False
    assert seen["timeout"] > 5


def test_grab_still_is_bounded_and_none_on_timeout(tmp_path):
    cam = cameras.Camera(id="c1", mac="d0:3b:f4:01:52:79", ip="1.2.3.4", position="")
    seen = {}

    def runner(argv, **kwargs):
        seen["argv"] = argv
        seen.update(kwargs)
        return _times_out(argv, **kwargs)

    result = cameras.grab_still(
        cam, user="a", password="b", dest=tmp_path / "still.jpg", runner=runner
    )

    assert result is None
    assert seen["timeout"] > 5
    # The RTSP socket timeout is an input option, so it must precede -i.
    argv = seen["argv"]
    assert argv.index("-timeout") < argv.index("-i")


def test_refresh_skips_the_still_for_a_camera_whose_probe_failed(tmp_path):
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "arp-scan":
            return _cp(argv, 0, stdout=_ARP_SCAN_OUTPUT)
        if argv[:5] == ["ip", "-4", "-o", "addr", "show"]:
            return _cp(argv, 0, stdout="inet 169.254.1.5/16")
        if argv[0] == "ffprobe":
            return _cp(argv, 1)
        return _cp(argv, 0)

    result = cameras.refresh(
        tmp_path / "install",
        cam_user="admin",
        cam_password="pw",
        interfaces=["eth0"],
        non_interactive=True,
        runner=runner,
    )

    assert result.cameras[0].stream_ok is False
    assert not any(c[0] == "ffmpeg" for c in calls)


def test_refresh_records_unaddressed_and_skipped_interfaces_in_scan_json(tmp_path):
    import json

    def runner(argv, **kwargs):
        return _cp(argv, 0, stdout="")

    cameras.refresh(
        tmp_path / "install", interfaces=["enp8s0"], non_interactive=True, runner=runner
    )

    record = json.loads((tmp_path / "install" / "cameras.scan.json").read_text())
    assert record["tool"] == "none"
    assert record["unaddressed"] == ["enp8s0"]
    assert record["skipped"] == []


# ---------------------------------------------------------------------------
# bind_positions
# ---------------------------------------------------------------------------


def _cam(mac, position="", id=""):
    return cameras.Camera(id=id, mac=mac, ip="1.2.3.4", position=position)


def test_bind_positions_skips_cameras_that_already_have_a_position():
    cams = [_cam("d0:3b:f4:00:00:01", position="top-left", id="c1")]

    def _boom(prompt=""):
        raise AssertionError("must not prompt for an already-labeled camera")

    result = cameras.bind_positions(cams, non_interactive=False, prompt=_boom)

    assert result == cams


def test_bind_positions_non_interactive_assigns_mac_sorted_ids_and_leaves_position_blank():
    cams = [_cam("d0:3b:f4:00:00:02"), _cam("d0:3b:f4:00:00:01")]

    result = cameras.bind_positions(cams, non_interactive=True)

    by_mac = {cam.mac: cam for cam in result}
    assert by_mac["d0:3b:f4:00:00:01"].id == "c1"
    assert by_mac["d0:3b:f4:00:00:02"].id == "c2"
    assert all(cam.position == "" for cam in result)


def test_bind_positions_interactive_prompts_once_per_unlabeled_camera():
    cams = [_cam("d0:3b:f4:00:00:01"), _cam("d0:3b:f4:00:00:02")]
    answers = iter(["top-left", "bottom-right"])
    prompts = []

    def prompt(text):
        prompts.append(text)
        return next(answers)

    result = cameras.bind_positions(cams, non_interactive=False, prompt=prompt)

    assert len(prompts) == 2
    positions = {cam.mac: cam.position for cam in result}
    assert positions == {
        "d0:3b:f4:00:00:01": "top-left",
        "d0:3b:f4:00:00:02": "bottom-right",
    }


def test_bind_positions_non_interactive_new_id_does_not_collide_with_already_bound_ids():
    already_bound = [
        cameras.Camera(id=f"c{i}", mac=f"d0:3b:f4:00:00:0{i}", ip="1.2.3.4", position="top-left")
        for i in range(1, 9)
    ]
    new_camera = _cam("d0:3b:f4:00:00:99")

    result = cameras.bind_positions(already_bound + [new_camera], non_interactive=True)

    ids = [cam.id for cam in result]
    assert len(ids) == len(set(ids)), f"duplicate id assigned: {ids}"
    new_result = next(cam for cam in result if cam.mac == "d0:3b:f4:00:00:99")
    assert new_result.id not in {f"c{i}" for i in range(1, 9)}


def test_bind_positions_interactive_logs_still_path_when_stills_dir_given(tmp_path, capsys):
    cams = [_cam("d0:3b:f4:00:00:01")]

    cameras.bind_positions(
        cams, non_interactive=False, prompt=lambda p: "top-left", stills_dir=tmp_path
    )

    assert "still-d03bf4000001.jpg" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def test_merge_refreshes_ip_and_preserves_id_position_enabled():
    previous = [
        cameras.Camera(
            id="c1",
            mac="d0:3b:f4:00:00:01",
            ip="old-ip",
            position="top-left",
            enabled=False,
            stream_ok=True,
        )
    ]
    discovered = [cameras.Camera(id="", mac="d0:3b:f4:00:00:01", ip="new-ip", position="")]

    merged = cameras.merge(previous, discovered)

    assert len(merged) == 1
    assert merged[0].id == "c1"
    assert merged[0].ip == "new-ip"
    assert merged[0].position == "top-left"
    assert merged[0].enabled is False


def test_merge_retains_and_flags_a_camera_missing_from_this_scan():
    previous = [
        cameras.Camera(
            id="c1", mac="d0:3b:f4:00:00:01", ip="1.2.3.4", position="top-left", stream_ok=True
        )
    ]

    merged = cameras.merge(previous, [])

    assert len(merged) == 1
    assert merged[0].ip == "1.2.3.4"  # retained, not deleted
    assert merged[0].stream_ok is None  # flagged: not probed this scan


def test_merge_appends_a_newly_discovered_camera():
    previous = [cameras.Camera(id="c1", mac="d0:3b:f4:00:00:01", ip="1.2.3.4", position="top-left")]
    discovered = [cameras.Camera(id="", mac="d0:3b:f4:00:00:99", ip="9.9.9.9", position="")]

    merged = cameras.merge(previous, discovered)

    macs = {cam.mac for cam in merged}
    assert macs == {"d0:3b:f4:00:00:01", "d0:3b:f4:00:00:99"}


# ---------------------------------------------------------------------------
# refresh (end to end, fake runner throughout)
# ---------------------------------------------------------------------------


def test_refresh_first_run_writes_inventory_and_scan_json(tmp_path):
    install_dir = tmp_path / "install"

    def runner(argv, **kwargs):
        if argv[0] == "arp-scan":
            return _cp(argv, 0, stdout=_ARP_SCAN_OUTPUT)
        if argv[:5] == ["ip", "-4", "-o", "addr", "show"]:
            return _cp(argv, 0, stdout="inet 169.254.1.5/16")
        if argv[0] in ("ffprobe", "ffmpeg"):
            return _cp(argv, 0)
        return _cp(argv, 1)

    result = cameras.refresh(
        install_dir,
        seed_header="# seed header",
        cam_user="admin",
        cam_password="pw",
        interfaces=["eth0"],
        non_interactive=True,
        runner=runner,
    )

    assert len(result.cameras) == 1
    inventory_path = install_dir / "cameras.yml"
    scan_json_path = install_dir / "cameras.scan.json"
    assert inventory_path.is_file()
    assert scan_json_path.is_file()

    reparsed = cameras.parse_inventory(inventory_path.read_text(encoding="utf-8"))
    assert reparsed[0].mac == "d0:3b:f4:01:52:79"
    assert reparsed[0].id == "c1"  # non-interactive MAC-sorted assignment


def test_refresh_second_run_merges_with_the_previous_inventory(tmp_path):
    install_dir = tmp_path / "install"
    install_dir.mkdir()
    (install_dir / "cameras.yml").write_text(
        cameras.render_inventory(
            [
                cameras.Camera(
                    id="c1",
                    mac="d0:3b:f4:01:52:79",
                    ip="old-ip",
                    position="top-left",
                )
            ],
            header="# generated",
        ),
        encoding="utf-8",
    )

    def runner(argv, **kwargs):
        if argv[0] == "arp-scan":
            return _cp(argv, 0, stdout=_ARP_SCAN_OUTPUT)
        if argv[:5] == ["ip", "-4", "-o", "addr", "show"]:
            return _cp(argv, 0, stdout="inet 169.254.1.5/16")
        return _cp(argv, 1)

    result = cameras.refresh(install_dir, interfaces=["eth0"], non_interactive=True, runner=runner)

    assert len(result.cameras) == 1
    assert result.cameras[0].id == "c1"
    assert result.cameras[0].position == "top-left"
    assert result.cameras[0].ip == "169.254.1.10"  # refreshed


def test_refresh_does_not_duplicate_its_own_banner_across_repeated_scans(tmp_path):
    install_dir = tmp_path / "install"

    def runner(argv, **kwargs):
        if argv[0] == "arp-scan":
            return _cp(argv, 0, stdout=_ARP_SCAN_OUTPUT)
        if argv[:5] == ["ip", "-4", "-o", "addr", "show"]:
            return _cp(argv, 0, stdout="inet 169.254.1.5/16")
        return _cp(argv, 1)

    cameras.refresh(
        install_dir, seed_header="# seed header", interfaces=["eth0"],
        non_interactive=True, runner=runner,
    )
    cameras.refresh(
        install_dir, seed_header="# seed header", interfaces=["eth0"],
        non_interactive=True, runner=runner,
    )

    text = (install_dir / "cameras.yml").read_text(encoding="utf-8")
    assert text.count("# Generated by mv3dt-installer") == 1


def test_refresh_missing_camera_is_retained_across_a_scan(tmp_path):
    install_dir = tmp_path / "install"
    install_dir.mkdir()
    (install_dir / "cameras.yml").write_text(
        cameras.render_inventory(
            [
                cameras.Camera(
                    id="c1", mac="d0:3b:f4:01:52:79", ip="169.254.1.10", position="top-left"
                )
            ],
            header="# generated",
        ),
        encoding="utf-8",
    )

    def runner(argv, **kwargs):
        # Nothing found this time -- the camera is powered off.
        if argv[0] == "arp-scan":
            return _cp(argv, 0, stdout="")
        if argv[:5] == ["ip", "-4", "-o", "addr", "show"]:
            return _cp(argv, 0, stdout="inet 169.254.1.5/16")
        return _cp(argv, 1)

    result = cameras.refresh(install_dir, interfaces=["eth0"], non_interactive=True, runner=runner)

    assert len(result.cameras) == 1
    assert result.cameras[0].id == "c1"
    assert result.cameras[0].ip == "169.254.1.10"
