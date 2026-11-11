"""Step 1 — Prerequisites (driver / CUDA / cuDNN / TensorRT / GStreamer).

Implements `installer/plan/STEP-1-PREREQUISITES.md` against the framework
contracts in `installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md` sections 7
(reboot), 8 (logging/reporting), 9 (privilege/USER-ACTION), and 12
(step-module interface). Installs and verifies every DeepStream 9.1 dGPU
prerequisite on a brand-new Ubuntu 24.04 workstation before the DeepStream
SDK itself is pulled (Step 2's job).

Two-launch structure (STEP-1 section 5)
----------------------------------------
The NVIDIA driver `.run` install requires a reboot mid-step, so `run()`
tracks its own internal stage via an idempotent probe -- `nvidia-smi`
reporting a driver version -- rather than writing `state.json` itself
(only the framework does that, doc 00 section 12.2):

* **Launch A** (driver not yet loaded): base deps, the DS 9.1 section 4.1
  apt prerequisites, the CUDA repo + toolkit, nouveau/distro-driver
  cleanup, the Secure Boot gate, and the driver `.run`. A desktop launch
  hands display teardown, the runfile, and the success reboot to a persistent
  systemd worker before this process exits; TTY/SSH launches retain the
  synchronous path. Ends in `USER_ACTION_REQUIRED` for the handoff/reboot,
  nouveau/distro-driver cleanup, Secure Boot gate, or staging gap; `FAILED`
  only if scheduling or the synchronous `.run` exits non-zero.
* **Launch B** (driver loaded): TensorRT + cuDNN via apt, a GStreamer pin
  confirmation, and the Mosquitto broker via the bundled
  `10_setup_mosquitto.sh` (section 3.2). Ends in `COMPLETE`, letting the
  framework call `verify()`.

Reboot handling deliberately does not use `ctx.reboot.request()` /
`StepStatus.REBOOT_REQUIRED` for either of Launch A's two reboot points
(the driver `.run` and the nouveau/distro-driver cleanup). The merged
`reboot.reconcile()` (`mv3dt_installer/reboot.py`) marks the *requesting*
step `COMPLETE` in `state.json` the instant it confirms a reboot happened,
and `app._dispatch()` skips a `COMPLETE` step without re-running its
lifecycle -- so a step with real work left to do post-reboot (this one:
TensorRT/cuDNN/mosquitto/verify) cannot safely ask the framework to
auto-complete it on reboot. Both reboot points instead return
`StepResult(status=USER_ACTION_REQUIRED, ...)` with the same reboot
instructions and `sudo reboot` action; `state.json` never records a
pending reboot for this step, and the step stays non-`COMPLETE` across the
reboot. On the next launch, `_driver_loaded(ctx)` -- the same idempotent
probe used to route `run()` to Launch A vs. B -- is what detects the
reboot actually happened and lets Launch A's own idempotent cleanup
probes (nouveau no longer loaded / no distro packages left to purge) fall
through to whatever comes next.

All subprocess work goes through `ctx.run_root` (never a bare
`subprocess.run`), which is the seam tests inject a fake `Context` through.
System paths this module writes directly (`/etc/profile.d/cuda.sh`,
`/etc/modprobe.d/blacklist-nouveau.conf`, the driver-handoff state, and
`/etc/mosquitto/conf.d/mv3dt.conf`) are named by module-level constants,
following the same pattern
`systemd.py`'s `UNIT_DIR` uses, so tests can monkeypatch them to a `tmp_path`
without ever touching the real filesystem.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import shutil
from typing import TYPE_CHECKING, Sequence

from mv3dt_installer import progress_exec, shellout, systemd as systemd_mod, waitui
from mv3dt_installer.steps import StepResult, StepStatus, UserAction, register

if TYPE_CHECKING:  # pragma: no cover
    from mv3dt_installer.app import Context

__all__ = ["Step1Prerequisites"]

# ---------------------------------------------------------------------------
# STEP-1 section 2 -- the DS 9.1 dGPU prerequisite pins (EQUALITY)
# ---------------------------------------------------------------------------

DRIVER_VERSION = "595.58.03"
CUDA_VERSION = "13.2"
CUDNN_VERSION = "9.20.0.48"
CUDNN_APT_VERSION = f"{CUDNN_VERSION}-1"
TENSORRT_VERSION = "10.16.0.72-1+cuda13.2"
GSTREAMER_VERSION = "1.24.2"
CUDA_TOOLKIT_PACKAGE = "cuda-toolkit-13-2"

# STEP-1 section 2.1 -- every libnvinfer* package pinned to one version.
TENSORRT_PACKAGES: tuple[str, ...] = (
    "libnvinfer-dev",
    "libnvinfer-dispatch-dev",
    "libnvinfer-dispatch10",
    "libnvinfer-headers-dev",
    "libnvinfer-headers-plugin-dev",
    "libnvinfer-safe-headers-dev",
    "libnvinfer-lean-dev",
    "libnvinfer-lean10",
    "libnvinfer-plugin-dev",
    "libnvinfer-plugin10",
    "libnvinfer-vc-plugin-dev",
    "libnvinfer-vc-plugin10",
    "libnvinfer10",
    "libnvonnxparsers-dev",
    "libnvonnxparsers10",
    "tensorrt-dev",
    "libnvinfer-headers-python-plugin-dev",
    "libnvinfer-win-builder-resource10",
)

# CUDA 13 uses concrete, versioned cuDNN packages on Ubuntu 24.04. Pin the
# complete meta-package dependency chain so apt cannot resolve a newer cuDNN,
# and pin/query the concrete runtime package used by DeepStream.
CUDNN_PACKAGES: tuple[str, ...] = (
    "cudnn9-cuda-13",
    "cudnn9-cuda-13-2",
    "libcudnn9-static-cuda-13",
    "libcudnn9-dev-cuda-13",
    "libcudnn9-headers-cuda-13",
    "libcudnn9-cuda-13",
)
CUDNN_QUERY_PACKAGE = "libcudnn9-cuda-13"

# STEP-1 section 4, caveats 1-2: kernel headers + minimal-24.04 tooling,
# installed before the DS 9.1 apt prerequisites and the CUDA repo.
BASE_KERNEL_PACKAGES: tuple[str, ...] = ("build-essential", "dkms")
BASE_TOOLING_PACKAGES: tuple[str, ...] = (
    "software-properties-common",
    "ca-certificates",
    "gnupg",
    "curl",
)

# STEP-1 section 3 -- the DS 9.1 section 4.1 apt prerequisite package list,
# installed in a single transaction. Includes both the DS 9.1-authoritative
# packages (section 3.1) and the four installer-subsystem additions
# (mosquitto, mosquitto-clients, arp-scan, ffmpeg, section 3.1's table).
APT_PREREQ_PACKAGES: tuple[str, ...] = (
    "libssl3",
    "libssl-dev",
    "libcurl4-openssl-dev",
    "libgles2-mesa-dev",
    "libgstreamer1.0-0",
    "gstreamer1.0-tools",
    "gstreamer1.0-plugins-good",
    "gstreamer1.0-plugins-bad",
    "gstreamer1.0-plugins-ugly",
    "gstreamer1.0-libav",
    "libgstreamer-plugins-base1.0-dev",
    "libgstrtspserver-1.0-0",
    "libjansson4",
    "libyaml-cpp-dev",
    "libjsoncpp-dev",
    "protobuf-compiler",
    "libmosquitto1",
    "gcc",
    "make",
    "git",
    "python3",
    "mosquitto",
    "mosquitto-clients",
    "arp-scan",
    "ffmpeg",
)

# STEP-1 section 5.1 -- the driver .run is not bundled (too large at ~400MB
# to ship in every GitHub Release binary). It is fetched automatically from
# NVIDIA's public runfile mirror; the operator-staged path remains the
# fallback when the download is unavailable (air-gapped host, proxy, 404 on
# a withdrawn build).
DRIVER_RUN_FILENAME = f"NVIDIA-Linux-x86_64-{DRIVER_VERSION}.run"

# The version-stamped public runfile URL. Derived from DRIVER_VERSION rather
# than hardcoded so the section 2 pin stays the single source of truth: a pin
# bump cannot leave this pointing at the old build.
DRIVER_DOWNLOAD_URL = (
    "https://us.download.nvidia.com/XFree86/Linux-x86_64/"
    f"{DRIVER_VERSION}/{DRIVER_RUN_FILENAME}"
)

# Optional integrity pin for the runfile. NVIDIA publishes no stable checksum
# manifest alongside the runfile mirror, so this is None until a human pins
# the value they verified out of band; when set, a mismatch is fatal and the
# download is discarded. `_verify_driver_run` always enforces the embedded
# version check below regardless, which is what actually guarantees DS 9.1
# compatibility (section 2's equality pin).
DRIVER_RUN_SHA256: "str | None" = None

# A .run runfile is a self-extracting shell archive whose plain-text header
# carries the build it was cut from, e.g.
#   #  NVIDIA Accelerated Graphics Driver for Linux-x86_64 595.58.03
# Reading that is how a downloaded-or-staged file is proven to be the pinned
# build before it is ever executed -- a mirror redirect, a resumed partial
# fetch, or an operator staging last year's driver under the right filename
# all fail this check.
DRIVER_HEADER_READ_BYTES = 8192
_DRIVER_HEADER_VERSION_RE = re.compile(rb"Linux-x86_64[\s-]+(\d+\.\d+(?:\.\d+)?)")

# STEP-1 section 4, caveat 7 / section 6.2 -- CUDA on PATH for new shells.
CUDA_HOME = f"/usr/local/cuda-{CUDA_VERSION}"
CUDA_NVCC_PATH = f"{CUDA_HOME}/bin/nvcc"
CUDA_PROFILE_PATH = pathlib.Path("/etc/profile.d/cuda.sh")

# STEP-1 section 4, caveat 4 -- nouveau must be out of the way before the
# .run installer runs.
NOUVEAU_BLACKLIST_PATH = pathlib.Path("/etc/modprobe.d/blacklist-nouveau.conf")

# STEP-1 section 5.1a -- a desktop terminal cannot survive GDM stopping, so
# the runfile is handed to a root-owned oneshot service. Every artifact the
# service needs is copied out of PyInstaller's temporary extraction directory
# before it starts.
DRIVER_HANDOFF_ROOT = pathlib.Path("/var/lib/mv3dt-installer/driver-handoff")
DRIVER_HANDOFF_WORKER_PATH = DRIVER_HANDOFF_ROOT / "install-driver.sh"
DRIVER_HANDOFF_RUNFILE_PATH = DRIVER_HANDOFF_ROOT / DRIVER_RUN_FILENAME
DRIVER_HANDOFF_STATUS_PATH = DRIVER_HANDOFF_ROOT / "status"
DRIVER_HANDOFF_LOG_PATH = DRIVER_HANDOFF_ROOT / "driver-install.log"
DRIVER_HANDOFF_UNIT_NAME = "mv3dt-driver-handoff.service"
DRIVER_HANDOFF_UNIT_DIR = pathlib.Path("/etc/systemd/system")
DRIVER_HANDOFF_BOOT_ID_PATH = pathlib.Path("/proc/sys/kernel/random/boot_id")

# STEP-1 section 3.2 -- the Mosquitto broker drop-in this step's Python
# wraps the bundled script's install with change detection for.
MOSQUITTO_CONF_DIR = pathlib.Path("/etc/mosquitto/conf.d")
MOSQUITTO_CONF_NAME = "mv3dt.conf"

_NVCC_RELEASE_RE = re.compile(r"release\s+(\d+\.\d+)")
_GSTREAMER_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+)")


# ---------------------------------------------------------------------------
# Small probes, all routed through ctx.run_root (the test seam)
# ---------------------------------------------------------------------------


def _dpkg_version(ctx: "Context", package: str) -> str | None:
    """`dpkg-query -W -f='${Version}' <package>`, or None if not installed."""
    result = ctx.run_root(
        "dpkg-query",
        "-W",
        "-f=${Version}",
        package,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def _cudnn_installed_version(ctx: "Context") -> str | None:
    raw = _dpkg_version(ctx, CUDNN_QUERY_PACKAGE)
    if raw == CUDNN_APT_VERSION:
        return CUDNN_VERSION
    return raw


def _driver_version(ctx: "Context") -> str:
    result = ctx.run_root(
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _driver_loaded(ctx: "Context") -> bool:
    """The idempotent A/B stage probe (STEP-1 section 6.3 / 7.1): driver
    present and `nvidia-smi` loads."""
    return bool(_driver_version(ctx))


def _nvcc_release(ctx: "Context") -> str:
    result = ctx.run_root(
        CUDA_NVCC_PATH, "--version", capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return ""
    match = _NVCC_RELEASE_RE.search(result.stdout)
    return match.group(1) if match else ""


def _gstreamer_version(ctx: "Context") -> str:
    result = ctx.run_root(
        "gst-inspect-1.0", "--version", capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return ""
    match = _GSTREAMER_VERSION_RE.search(result.stdout)
    return match.group(1) if match else ""


def _os_version(ctx: "Context") -> str:
    result = ctx.run_root(
        "lsb_release", "-rs", capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _arch(ctx: "Context") -> str:
    result = ctx.run_root("uname", "-m", capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _kernel_release(ctx: "Context") -> str:
    result = ctx.run_root("uname", "-r", capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _gpu_present(ctx: "Context") -> bool:
    result = ctx.run_root(
        "bash",
        "-c",
        "lspci | grep -qi nvidia",
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _secure_boot_enabled(ctx: "Context") -> bool:
    """STEP-1 section 4, caveat 5. `mokutil` absent (or a non-UEFI system)
    is treated as "not enabled" -- there is nothing to gate on."""
    result = ctx.run_root(
        "mokutil", "--sb-state", capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return False
    return "enabled" in result.stdout.lower()


_VIRTUAL_CONSOLE_RE = re.compile(r"^/dev/tty\d+$")

# Display-manager units, in the order they are tried. `gdm3` leads because
# that is what Ubuntu 24.04 -- the only supported target (section 1) --
# actually ships; looking only for `gdm`/`lightdm` found nothing on every
# real workstation, which made `_stop_display_manager` report failure for a
# desktop it had simply failed to name.
DISPLAY_MANAGER_UNITS: tuple[str, ...] = ("gdm3", "gdm", "lightdm", "sddm")


# Escape hatch for the caveat-6a guard. The guard infers the session from
# the environment, and an inference that is wrong must never be the thing
# that makes the install impossible -- set this to run anyway.
ALLOW_DISPLAY_STOP_ENV = "MV3DT_ALLOW_DISPLAY_STOP"


def _controlling_tty() -> str | None:
    """The process's controlling terminal, or None when it has none.

    `/dev/ttyN` is a virtual console, which survives the display manager
    being stopped. `/dev/pts/N` is a pseudo-terminal -- a terminal emulator
    window (dies with the X session) or an SSH session (does not).

    All three standard descriptors are tried, not just stdin: a redirected
    or closed stdin (`< /dev/null`, a pipe, a wrapper that closes it) makes
    `ttyname(0)` raise even on a perfectly good virtual console, and
    treating that as "no console" refuses an operator who is doing exactly
    what the guard asked of them.
    """
    for fd in (0, 1, 2):
        try:
            name = os.ttyname(fd)
        except (OSError, AttributeError):
            continue
        if name:
            return name
    # Last resort: the kernel's own record of the controlling terminal.
    # /proc/self/stat field 7 is tty_nr; major 4 is the virtual consoles,
    # so minor N means /dev/ttyN.
    try:
        stat = pathlib.Path("/proc/self/stat").read_text()
        fields = stat[stat.rfind(")") + 2 :].split()
        tty_nr = int(fields[4])
    except (OSError, IndexError, ValueError):
        return None
    if tty_nr <= 0:
        return None
    major = (tty_nr >> 8) & 0xFF
    minor = (tty_nr & 0xFF) | ((tty_nr >> 20) << 8)
    return f"/dev/tty{minor}" if major == 4 else None


def _has_sshd_ancestor(pid: "int | None" = None) -> bool:
    """Whether this process descends from sshd.

    Walks /proc rather than reading SSH_CONNECTION/SSH_TTY, because sudo's
    default env_reset strips those before a step ever sees them -- an SSH
    operator would otherwise be refused for a hazard that cannot touch
    them: stopping the display manager does not disturb an SSH session.
    """
    pid = os.getpid() if pid is None else pid
    seen = set()
    while pid and pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            return False
        # comm is parenthesised and may itself contain spaces/parens.
        open_paren, close_paren = stat.find("("), stat.rfind(")")
        if open_paren < 0 or close_paren < 0:
            return False
        comm = stat[open_paren + 1 : close_paren]
        if comm == "sshd":
            return True
        rest = stat[close_paren + 2 :].split()
        try:
            pid = int(rest[1])  # ppid
        except (IndexError, ValueError):
            return False
    return False


def _active_display_manager(ctx: "Context") -> str | None:
    """The name of the running display-manager unit, or None if none is."""
    for unit in DISPLAY_MANAGER_UNITS:
        if (
            ctx.run_root(
                "systemctl",
                "is-active",
                "--quiet",
                unit,
                check=False,
                capture_output=True,
                text=True,
            ).returncode
            == 0
        ):
            return unit
    return None


def _session_hazard(ctx: "Context") -> str | None:
    """STEP-1 section 4, caveat 6a: whether stopping the display manager
    would take this installer process down with it.

    `service gdm stop` tears down the X session, killing every terminal
    emulator inside it -- including the one the installer is running in, if
    it was launched from the desktop. The installer then dies on SIGHUP
    part-way through, most often before the driver `.run` starts at all,
    leaving a black screen, no `/var/log/nvidia-installer.log`, and no
    indication that anything went wrong. A virtual console and an SSH
    session are both safe; a desktop terminal emulator is not.
    """
    if os.environ.get(ALLOW_DISPLAY_STOP_ENV):
        return None
    tty = _controlling_tty()
    if tty is not None and _VIRTUAL_CONSOLE_RE.match(tty):
        return None
    if _has_sshd_ancestor():
        return None
    unit = _active_display_manager(ctx)
    if unit is None:
        return None
    # Report the evidence, not just the verdict. A refusal that says only
    # "this would kill the installer" is undebuggable from the console it
    # just refused: the operator can see they are on a tty, the installer
    # disagrees, and neither can see why.
    return (
        f"controlling terminal: {tty or 'none detected'}; "
        f"sshd ancestor: no; active display manager: {unit}"
    )


def _stop_display_manager(ctx: "Context") -> bool:
    """STEP-1 section 4, caveat 6: `service gdm stop` (fallback `lightdm`),
    then `pkill -9 Xorg`. Returns whether a display manager was actually
    stopped (or none was running to begin with).

    Counts down first. This is the only point in the install where the
    operator's screen goes black, and it is indistinguishable from a crash
    or a spontaneous reboot unless it is announced: the desktop dies, the
    driver installer keeps running with its output captured, and nothing
    on screen says the machine is still working. Announcing it is also what
    stops an operator power-cycling mid-`.run`.
    """
    waitui.countdown(
        description=(
            "Stopping the desktop session to install the NVIDIA driver -- "
            "the screen will go black; do NOT power off"
        ),
        non_interactive=bool(getattr(ctx, "non_interactive", False)),
    )
    # Stop whichever display manager is actually running. "None running"
    # is success, not failure -- there is nothing to stop, which is exactly
    # the case when the operator has already done as they were asked and
    # launched from a console.
    stopped_any = False
    for unit in DISPLAY_MANAGER_UNITS:
        if (
            ctx.run_root(
                "systemctl",
                "is-active",
                "--quiet",
                unit,
                check=False,
                capture_output=True,
                text=True,
            ).returncode
            != 0
        ):
            continue
        result = ctx.run_root(
            "systemctl", "stop", unit, check=False, capture_output=True, text=True
        )
        if result.returncode != 0:
            return False
        stopped_any = True

    ctx.run_root("pkill", "-9", "Xorg", check=False, capture_output=True, text=True)
    ctx.log.info(
        f"display manager stopped: {stopped_any and 'yes' or 'none was running'}"
    )
    return True


def _start_display_manager(ctx: "Context", unit: str) -> bool:
    """Best-effort recovery for a synchronous runfile failure."""
    result = ctx.run_root(
        "systemctl",
        "start",
        unit,
        check=False,
        capture_output=True,
        text=True,
    )
    restored = result.returncode == 0
    ctx.log.info(f"display manager restore: {unit}: {'ok' if restored else 'failed'}")
    return restored


def _nouveau_loaded(ctx: "Context") -> bool:
    result = ctx.run_root(
        "bash",
        "-c",
        "lsmod | grep -q '^nouveau'",
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _driver_run_path(ctx: "Context") -> pathlib.Path:
    return pathlib.Path(ctx.install_dir) / "downloads" / "nvidia" / DRIVER_RUN_FILENAME


def _driver_handoff_state() -> str | None:
    """Return the persistent worker state, or None before the first handoff."""
    try:
        return DRIVER_HANDOFF_STATUS_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _current_boot_id() -> str | None:
    try:
        return DRIVER_HANDOFF_BOOT_ID_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _driver_handoff_result() -> StepResult | None:
    """Block duplicate work while a prior desktop handoff is actionable."""
    state = _driver_handoff_state()
    if state in {"scheduled", "running"}:
        return StepResult(
            status=StepStatus.USER_ACTION_REQUIRED,
            message="the NVIDIA driver handoff is still running",
            user_actions=[
                UserAction(
                    text=(
                        "Do not start another installer. The desktop will close while "
                        "the driver installs, and the workstation will reboot "
                        "automatically when it succeeds."
                    ),
                    command=f"sudo journalctl -fu {DRIVER_HANDOFF_UNIT_NAME}",
                    path=str(DRIVER_HANDOFF_LOG_PATH),
                )
            ],
        )
    if state is not None and state.startswith("succeeded"):
        _, _, handoff_boot_id = state.partition(":")
        current_boot_id = _current_boot_id()
        if handoff_boot_id and current_boot_id and handoff_boot_id != current_boot_id:
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message=(
                    "the NVIDIA driver handoff succeeded, but the driver did not "
                    "load after reboot"
                ),
                user_actions=[
                    UserAction(
                        text=(
                            "Check Secure Boot or MOK enrollment and inspect the "
                            "persistent driver log. Remove the handoff marker only "
                            "after correcting the cause, then run the installer again."
                        ),
                        command=f"sudo rm -f {DRIVER_HANDOFF_STATUS_PATH}",
                        path=str(DRIVER_HANDOFF_LOG_PATH),
                    )
                ],
            )
        return StepResult(
            status=StepStatus.USER_ACTION_REQUIRED,
            message="the NVIDIA driver handoff succeeded and is waiting for reboot",
            user_actions=_driver_reboot_actions(),
        )
    if state is not None and state.startswith("failed:"):
        return StepResult(
            status=StepStatus.USER_ACTION_REQUIRED,
            message=f"the NVIDIA driver background install {state}",
            user_actions=[
                UserAction(
                    text=(
                        "The worker attempted to restore the desktop. Inspect the "
                        "driver log, remove the failure marker after correcting the "
                        "cause, then run the installer again."
                    ),
                    command=f"sudo rm -f {DRIVER_HANDOFF_STATUS_PATH}",
                    path=str(DRIVER_HANDOFF_LOG_PATH),
                )
            ],
        )
    return None


def _schedule_driver_handoff(ctx: "Context", run_path: pathlib.Path) -> str | None:
    """Stage and start the desktop-safe systemd worker.

    Returns None after systemd accepts the asynchronous job, otherwise a short
    operator-facing reason. The service owns display teardown, the runfile,
    failure recovery, and the success reboot after this process exits.
    """
    DRIVER_HANDOFF_ROOT.mkdir(parents=True, exist_ok=True)
    DRIVER_HANDOFF_ROOT.chmod(0o700)

    source = ctx.asset_path("systemd", "mv3dt-driver-handoff.sh")
    shutil.copyfile(source, DRIVER_HANDOFF_WORKER_PATH)
    DRIVER_HANDOFF_WORKER_PATH.chmod(0o700)
    shutil.copyfile(run_path, DRIVER_HANDOFF_RUNFILE_PATH)
    DRIVER_HANDOFF_RUNFILE_PATH.chmod(0o700)

    staged_reason = _verify_driver_run(DRIVER_HANDOFF_RUNFILE_PATH)
    if staged_reason is not None:
        DRIVER_HANDOFF_RUNFILE_PATH.unlink(missing_ok=True)
        return f"could not stage the NVIDIA driver handoff: {staged_reason}"

    rendered = systemd_mod.render_unit(
        ("systemd", "mv3dt-driver-handoff.service.in"),
        {
            "WORKER": str(DRIVER_HANDOFF_WORKER_PATH),
            "RUNFILE": str(DRIVER_HANDOFF_RUNFILE_PATH),
            "STATUS_FILE": str(DRIVER_HANDOFF_STATUS_PATH),
            "LOG_FILE": str(DRIVER_HANDOFF_LOG_PATH),
            "DRIVER_VERSION": DRIVER_VERSION,
            "BOOT_ID_FILE": str(DRIVER_HANDOFF_BOOT_ID_PATH),
        },
    )
    changed = systemd_mod.install_unit(
        DRIVER_HANDOFF_UNIT_NAME,
        rendered,
        unit_dir=DRIVER_HANDOFF_UNIT_DIR,
    )
    runner = lambda argv, **kwargs: ctx.run_root(*argv, **kwargs)
    if changed:
        systemd_mod.daemon_reload(runner=runner)

    DRIVER_HANDOFF_STATUS_PATH.write_text("scheduled\n", encoding="utf-8")
    DRIVER_HANDOFF_STATUS_PATH.chmod(0o600)
    ctx.run_root(
        "systemctl",
        "reset-failed",
        DRIVER_HANDOFF_UNIT_NAME,
        check=False,
        capture_output=True,
        text=True,
    )
    result = ctx.run_root(
        "systemctl",
        "start",
        "--no-block",
        DRIVER_HANDOFF_UNIT_NAME,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        DRIVER_HANDOFF_STATUS_PATH.write_text(
            f"failed:systemd-start:{result.returncode}\n", encoding="utf-8"
        )
        return f"systemd could not start the driver handoff (exit {result.returncode})"
    return None


def _driver_run_embedded_version(path: pathlib.Path) -> str | None:
    """The driver build stamped in the runfile's self-extracting header, or
    None if the file carries no recognisable NVIDIA runfile header."""
    try:
        with path.open("rb") as handle:
            header = handle.read(DRIVER_HEADER_READ_BYTES)
    except OSError:
        return None
    match = _DRIVER_HEADER_VERSION_RE.search(header)
    return match.group(1).decode("ascii") if match else None


def _verify_driver_run(path: pathlib.Path) -> str | None:
    """STEP-1 section 2 equality pin, applied to the runfile itself.

    Returns None when `path` is the pinned DRIVER_VERSION build, or a short
    operator-facing reason string when it is not. Checked before the file is
    ever made executable, so a wrong or truncated download is never run.
    """
    embedded = _driver_run_embedded_version(path)
    if embedded is None:
        return (
            "file is not a recognisable NVIDIA runfile (no driver version in "
            "its header); it is most likely truncated or an HTML error page"
        )
    if embedded != DRIVER_VERSION:
        return (
            f"file contains driver {embedded}, but DeepStream 9.1 requires "
            f"exactly {DRIVER_VERSION}"
        )
    if DRIVER_RUN_SHA256 is not None:
        actual = _sha256_file(path)
        if actual != DRIVER_RUN_SHA256:
            return (
                f"SHA256 mismatch: expected {DRIVER_RUN_SHA256}, got "
                f"{actual or 'unreadable'}"
            )
    return None


def _download_driver_run(ctx: "Context", dest: pathlib.Path) -> str | None:
    """Fetch the pinned driver runfile to `dest` (STEP-1 section 5.1).

    Downloads to a sibling `.part` file and only renames into place once
    `_verify_driver_run` passes, so an interrupted or wrong-build fetch can
    never leave something at `dest` that a later launch would mistake for a
    good staged file. Returns None on success, or a reason string on failure
    -- callers fall back to the operator-staging user action rather than
    treating a failed download as fatal.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    partial.unlink(missing_ok=True)

    ctx.log.info(
        f"Downloading NVIDIA driver {DRIVER_VERSION} from {DRIVER_DOWNLOAD_URL}"
    )
    # curl, not wget: BASE_TOOLING_PACKAGES guarantees curl is installed and
    # says nothing about wget, which a minimal or server image need not ship.
    # -f so an HTTP error is a non-zero exit rather than an error page saved
    # under the runfile's name, -L to follow the mirror's redirect.
    result = progress_exec.download(
        ctx,
        partial,
        DRIVER_DOWNLOAD_URL,
        lambda: ctx.run_root(
            "curl",
            "-fL",
            "--retry",
            "3",
            "--connect-timeout",
            "30",
            "-o",
            str(partial),
            DRIVER_DOWNLOAD_URL,
            check=False,
            capture_output=True,
            text=True,
        ),
        task=f"NVIDIA driver {DRIVER_VERSION}",
    )
    if result.returncode != 0 or not partial.is_file():
        partial.unlink(missing_ok=True)
        return f"download failed (curl exit {result.returncode})"

    reason = _verify_driver_run(partial)
    if reason is not None:
        partial.unlink(missing_ok=True)
        return f"downloaded {reason}"

    partial.replace(dest)
    ctx.report_installed(DRIVER_RUN_FILENAME, DRIVER_VERSION)
    return None


def _mosquitto_dst_path() -> pathlib.Path:
    return MOSQUITTO_CONF_DIR / MOSQUITTO_CONF_NAME


def _sha256_file(path: "pathlib.Path | str") -> str | None:
    try:
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _mosquitto_conf_matches_bundled(ctx: "Context") -> bool:
    dst_hash = _sha256_file(_mosquitto_dst_path())
    bundled_hash = _sha256_file(ctx.asset_path("mosquitto", "mv3dt.conf"))
    return dst_hash is not None and dst_hash == bundled_hash


# ---------------------------------------------------------------------------
# apt install + report_installed/report_already_installed wrapping
# ---------------------------------------------------------------------------


def _apt_install_reported(
    ctx: "Context",
    query_packages: Sequence[str],
    *,
    apt_args: Sequence[str] | None = None,
    reported_version: str | None = None,
    allow_downgrades: bool = False,
) -> StepResult | None:
    """Install `query_packages` (or `apt_args`, if the apt invocation needs
    version-pinned `pkg=version` arguments) in one apt transaction, then
    report each with the doc 00 section 8.3 exact strings, based on a
    before/after `dpkg-query` presence probe (STEP-1 section 2.2 / 7.2)."""
    before = {pkg: _dpkg_version(ctx, pkg) for pkg in query_packages}
    argv = list(apt_args) if apt_args is not None else list(query_packages)
    requested = {
        package: version
        for arg in argv
        if "=" in arg
        for package, version in [arg.split("=", 1)]
    }
    install_options = ["-y", "--no-install-recommends"]
    if allow_downgrades:
        install_options.append("--allow-downgrades")
    result = progress_exec.apt(
        ctx,
        "install",
        *install_options,
        *argv,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return StepResult(
            status=StepStatus.FAILED,
            message=(
                f"apt install failed for {', '.join(query_packages)} "
                f"(exit {result.returncode})"
            ),
        )
    after = {pkg: _dpkg_version(ctx, pkg) for pkg in query_packages}
    for pkg, version in after.items():
        if version is None:
            return StepResult(
                status=StepStatus.FAILED,
                message=f"apt reported success but {pkg} is not installed",
            )
        if pkg in requested and version != requested[pkg]:
            return StepResult(
                status=StepStatus.FAILED,
                message=(
                    f"apt installed {pkg} version {version}; "
                    f"expected {requested[pkg]}"
                ),
            )
    for pkg, version in after.items():
        display_version = reported_version or version
        if before[pkg] == version:
            ctx.report_already_installed(pkg, display_version)
        else:
            ctx.report_installed(pkg, display_version)
    return None


def _install_cuda_toolkit(ctx: "Context") -> StepResult | None:
    """STEP-1 section 5 step 4: CUDA repo + keyring, then
    `cuda-toolkit-13-2`. The reported version is the CUDA_VERSION pin
    (`13.2`), not the package's raw apt version suffix -- matching the
    exact example string in STEP-1 section 2.2
    ("installed cuda-toolkit-13-2 version 13.2")."""
    before = _dpkg_version(ctx, CUDA_TOOLKIT_PACKAGE)
    keyring = ctx.run_root(
        "bash",
        "-c",
        # curl for the same reason as the driver download: curl is in
        # BASE_TOOLING_PACKAGES, wget is not guaranteed on a minimal image.
        # -f turns an HTTP error into a non-zero exit, so `&&` stops rather
        # than handing dpkg an error page saved as a .deb.
        "curl -fsSL -o /tmp/cuda-keyring.deb "
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb "
        "&& dpkg -i /tmp/cuda-keyring.deb",
        check=False,
        capture_output=True,
        text=True,
    )
    if keyring.returncode != 0:
        return StepResult(
            status=StepStatus.FAILED,
            message=f"CUDA repository keyring install failed (exit {keyring.returncode})",
        )
    update = progress_exec.apt(
        ctx, "update", check=False, capture_output=True, text=True
    )
    if update.returncode != 0:
        return StepResult(
            status=StepStatus.FAILED,
            message=f"apt update failed after CUDA keyring install (exit {update.returncode})",
        )
    install = progress_exec.apt(
        ctx,
        "install",
        "-y",
        "--no-install-recommends",
        CUDA_TOOLKIT_PACKAGE,
        check=False,
        capture_output=True,
        text=True,
    )
    if install.returncode != 0:
        return StepResult(
            status=StepStatus.FAILED,
            message=f"apt install failed for {CUDA_TOOLKIT_PACKAGE} (exit {install.returncode})",
        )
    after = _dpkg_version(ctx, CUDA_TOOLKIT_PACKAGE)
    if after is None:
        return StepResult(
            status=StepStatus.FAILED,
            message=f"apt reported success but {CUDA_TOOLKIT_PACKAGE} is not installed",
        )
    if before == after:
        ctx.report_already_installed(CUDA_TOOLKIT_PACKAGE, CUDA_VERSION)
    else:
        ctx.report_installed(CUDA_TOOLKIT_PACKAGE, CUDA_VERSION)
    return None


def _write_cuda_profile() -> None:
    """STEP-1 section 4, caveat 7."""
    CUDA_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CUDA_PROFILE_PATH.write_text(
        "# Managed by mv3dt-installer (installer/plan/STEP-1-PREREQUISITES.md section 4).\n"
        f'export PATH="{CUDA_HOME}/bin:$PATH"\n'
        f'export LD_LIBRARY_PATH="{CUDA_HOME}/lib64:$LD_LIBRARY_PATH"\n',
        encoding="utf-8",
    )


def _write_nouveau_blacklist() -> None:
    """STEP-1 section 4, caveat 4."""
    NOUVEAU_BLACKLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOUVEAU_BLACKLIST_PATH.write_text(
        "# Managed by mv3dt-installer (installer/plan/STEP-1-PREREQUISITES.md section 4).\n"
        "blacklist nouveau\n"
        "options nouveau modeset=0\n",
        encoding="utf-8",
    )


def _purge_distro_nvidia_packages(ctx: "Context") -> tuple[bool, StepResult | None]:
    """STEP-1 section 4, caveat 3. Returns purge state and any failure.

    Only packages dpkg reports as actually *present* are purged. `dpkg-query
    -W` lists every package name dpkg knows about, which on a stock Ubuntu
    24.04 desktop includes `nvidia-common`, `nvidia-prime`,
    `libnvidia-encode1` and friends at `unknown ok not-installed` -- names
    referenced as dependencies by other packages but never installed. Keying
    the return value on "the name list is non-empty" made those four look
    like a purge on every launch, so this returned True forever and section 5
    step 5 demanded a reboot the reboot could never satisfy. The current
    status field (third word of `${Status}`) is what distinguishes a package
    that is really there from one dpkg has merely heard of.
    """
    result = ctx.run_root(
        "bash",
        "-c",
        "dpkg-query -W -f='${Status}|${Package}\\n' 'nvidia-*' 'libnvidia-*' "
        "2>/dev/null",
        capture_output=True,
        text=True,
        check=False,
    )
    packages: list[str] = []
    for line in result.stdout.splitlines():
        status, _, package = line.partition("|")
        package = package.strip()
        # `${Status}` is "<want> <flag> <current>"; only the current status
        # says whether the package exists on disk. "not-installed" is the
        # name-known-but-absent case above; "config-files" is a genuine
        # leftover that purging does clear, so it stays in scope.
        current = status.split()[-1] if status.split() else ""
        if package and current not in ("", "not-installed"):
            packages.append(package)
    if not packages:
        return False, None
    purge = progress_exec.apt(
        ctx, "purge", "-y", *packages, check=False, capture_output=True, text=True
    )
    if purge.returncode != 0:
        return False, StepResult(
            status=StepStatus.FAILED,
            message=f"apt purge failed for distro NVIDIA packages (exit {purge.returncode})",
        )
    autoremove = progress_exec.apt(
        ctx, "autoremove", "-y", check=False, capture_output=True, text=True
    )
    if autoremove.returncode != 0:
        return False, StepResult(
            status=StepStatus.FAILED,
            message=f"apt autoremove failed after NVIDIA purge (exit {autoremove.returncode})",
        )
    return True, None


def _clean_nouveau_and_distro_driver(
    ctx: "Context",
) -> tuple[bool, StepResult | None]:
    """STEP-1 section 4, caveats 3-4 / section 5 step 5. Returns whether a
    reboot is now required (nouveau was loaded, or a distro package was
    purged)."""
    nouveau_loaded = _nouveau_loaded(ctx)
    _write_nouveau_blacklist()
    if nouveau_loaded:
        initramfs = ctx.run_root(
            "update-initramfs", "-u", check=False, capture_output=True, text=True
        )
        if initramfs.returncode != 0:
            return False, StepResult(
                status=StepStatus.FAILED,
                message=f"update-initramfs failed (exit {initramfs.returncode})",
            )

    purged, failure = _purge_distro_nvidia_packages(ctx)
    return nouveau_loaded or purged, failure


def _record_gpu_info(ctx: "Context") -> None:
    """STEP-1 section 8: informational, not a gate."""
    result = ctx.run_root(
        "nvidia-smi",
        "--query-gpu=name,compute_cap",
        "--format=csv,noheader",
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        ctx.log.info(f"GPU: {result.stdout.strip()}")


def _verify_mosquitto_active(ctx: "Context") -> bool:
    """STEP-1 section 3.2 / section 7.3: broker active + drop-in matches the
    bundled asset. Uses `ctx.log`/`verify_pinned`-style wording rather than
    `verify_pinned` itself, since this is a pass/fail check, not a pinned
    version (section 3.2's explicit call-out)."""
    active = (
        ctx.run_root(
            "systemctl",
            "is-active",
            "--quiet",
            "mosquitto",
            check=False,
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )
    conf_ok = _mosquitto_conf_matches_bundled(ctx)
    if active and conf_ok:
        ctx.log.info("Version OK: mosquitto broker == active, mv3dt.conf == bundled")
        return True
    ctx.log.info(
        "Version check failed: mosquitto broker — "
        f"expected active=True, conf_matches_bundled=True, "
        f"got active={active}, conf_matches_bundled={conf_ok}"
    )
    return False


# ---------------------------------------------------------------------------
# Reboot user-action blocks (STEP-1 section 6.2)
# ---------------------------------------------------------------------------


def _driver_reboot_actions() -> list[UserAction]:
    return [
        UserAction(
            text=(
                f"The NVIDIA driver {DRIVER_VERSION} kernel module is installed but not "
                "yet loaded. Reboot so nvidia.ko loads before TensorRT/cuDNN install."
            ),
        ),
        UserAction(
            text=(
                "CUDA 13.2 has been added to PATH/LD_LIBRARY_PATH for new login shells."
            ),
            path=str(CUDA_PROFILE_PATH),
        ),
        UserAction(command="sudo reboot", text="Reboot the workstation."),
    ]


def _cleanup_reboot_actions() -> list[UserAction]:
    return [
        UserAction(
            text=(
                "nouveau was unloaded and/or a conflicting distro NVIDIA package was "
                "purged. Reboot before the NVIDIA driver .run installer runs, since it "
                "cannot build against a live nouveau or a partially-removed driver."
            ),
        ),
        UserAction(command="sudo reboot", text="Reboot the workstation."),
    ]


# ---------------------------------------------------------------------------
# The Step
# ---------------------------------------------------------------------------


class Step1Prerequisites:
    """STEP-1-PREREQUISITES.md section 1: module identity."""

    id = "step1_prerequisites"
    title = "Prerequisites (driver / CUDA / cuDNN / TensorRT / GStreamer)"
    order = 1

    # Doc 08 §3.1. The numbering spans both launches deliberately: phases 1
    # to 4 belong to launch A (pre-driver-reboot) and 5 to 6 to launch B, so
    # an operator who reboots mid-step sees "phase 5/6" resume rather than
    # a second step restarting at 1. Labels match doc 08 §3.2's mock-up.
    phases = (
        "base packages",
        "CUDA repo and toolkit",
        "nouveau and distro cleanup",
        "NVIDIA driver runfile",
        "TensorRT and cuDNN",
        "Mosquitto broker",
    )

    # -- preflight (section 7.1) -------------------------------------------

    def preflight(self, ctx: "Context") -> StepResult:
        os_version = _os_version(ctx)
        arch = _arch(ctx)
        if os_version != "24.04" or arch != "x86_64":
            return StepResult(
                status=StepStatus.FAILED,
                message=(
                    f"unsupported platform: Ubuntu {os_version or '?'} / {arch or '?'} "
                    "(Step 1 requires Ubuntu 24.04 / x86_64; see "
                    "installer/plan/DEEPSTREAM-SETUP.md sections 2-3)"
                ),
            )
        if not _gpu_present(ctx):
            return StepResult(
                status=StepStatus.FAILED,
                message="no NVIDIA GPU detected via lspci",
            )
        return StepResult(status=StepStatus.COMPLETE)

    # -- run (section 7.2) ---------------------------------------------------

    def run(self, ctx: "Context") -> StepResult:
        if not _driver_loaded(ctx):
            return self._run_launch_a(ctx)
        return self._run_launch_b(ctx)

    def _run_launch_a(self, ctx: "Context") -> StepResult:
        handoff_result = _driver_handoff_result()
        if handoff_result is not None:
            return handoff_result

        kernel = _kernel_release(ctx)
        base_kernel_packages = [
            *BASE_KERNEL_PACKAGES,
            f"linux-headers-{kernel}" if kernel else "linux-headers-generic",
        ]
        ctx.progress.phase(1)
        ctx.progress.task("kernel headers and build tools")
        failure = _apt_install_reported(ctx, base_kernel_packages)
        if failure is not None:
            return failure
        ctx.progress.task("base tooling")
        failure = _apt_install_reported(ctx, list(BASE_TOOLING_PACKAGES))
        if failure is not None:
            return failure

        ctx.progress.task("GStreamer and step prerequisites")
        failure = _apt_install_reported(ctx, list(APT_PREREQ_PACKAGES))
        if failure is not None:
            return failure

        ctx.progress.phase(2)
        ctx.progress.task(f"CUDA {CUDA_VERSION} toolkit")
        failure = _install_cuda_toolkit(ctx)
        if failure is not None:
            return failure
        _write_cuda_profile()

        ctx.progress.phase(3)
        ctx.progress.task("nouveau and distro NVIDIA packages")
        reboot_required, failure = _clean_nouveau_and_distro_driver(ctx)
        if failure is not None:
            return failure
        if reboot_required:
            # USER_ACTION_REQUIRED, not ctx.reboot.request() -- see the
            # "Reboot handling" note in this class's module docstring. The
            # merged reboot.reconcile() marks the *requesting* step COMPLETE
            # the moment it confirms the reboot, and _dispatch() then skips
            # a COMPLETE step without re-running its lifecycle -- so a
            # REBOOT_REQUIRED here would let the framework auto-complete
            # Step 1 before the driver .run (let alone TensorRT/cuDNN) ever
            # runs. Rendering the same instructions as USER_ACTION_REQUIRED
            # instead leaves the step PENDING; on the next launch `run()`
            # re-enters _run_launch_a(), whose own idempotent probes (no
            # nouveau loaded, no distro packages left to purge) see the
            # cleanup already done and fall through to the Secure Boot
            # check / driver .run.
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message=(
                    "nouveau and/or a distro NVIDIA package were cleaned up; a reboot "
                    "is required before the driver .run installer can run"
                ),
                user_actions=_cleanup_reboot_actions(),
            )

        if _secure_boot_enabled(ctx):
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message=(
                    "Secure Boot is enabled; the unsigned NVIDIA driver module cannot "
                    "load"
                ),
                user_actions=[
                    UserAction(
                        text=(
                            "Disable Secure Boot in the BIOS, or complete MOK Manager "
                            "enrollment on the next boot."
                        ),
                    ),
                ],
            )

        ctx.progress.phase(4)
        ctx.progress.task(f"NVIDIA driver {DRIVER_VERSION}")
        run_path = _driver_run_path(ctx)

        # STEP-1 section 5.1: fetch the pinned runfile ourselves. An already
        # staged file is verified rather than re-downloaded, so an operator
        # who pre-staged the driver (or a resumed launch after the download
        # already landed) does not pay for it twice.
        if run_path.is_file():
            reason = _verify_driver_run(run_path)
            if reason is not None:
                ctx.log.warn(
                    f"Staged driver runfile rejected: {reason}; re-downloading"
                )
                run_path.unlink(missing_ok=True)
                reason = _download_driver_run(ctx, run_path)
        else:
            reason = _download_driver_run(ctx, run_path)

        if reason is not None:
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message=(
                    f"could not obtain NVIDIA driver {DRIVER_VERSION} "
                    f"automatically: {reason}"
                ),
                user_actions=[
                    UserAction(
                        text=(
                            f"Download driver {DRIVER_VERSION} for Linux x86_64 from "
                            "NVIDIA's driver download page and place it at the exact "
                            "path shown below, then re-run the installer."
                        ),
                        command="https://www.nvidia.com/en-us/drivers/",
                        path=str(run_path),
                    ),
                ],
            )

        # A desktop terminal cannot survive GDM stopping. Hand the disruptive
        # work to systemd before touching the display manager; TTY and SSH
        # sessions retain the synchronous path and its live output.
        hazard = _session_hazard(ctx)
        if hazard is not None:
            reason = _schedule_driver_handoff(ctx, run_path)
            if reason is not None:
                return StepResult(status=StepStatus.FAILED, message=reason)
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message=(
                    "the NVIDIA driver installation was handed to systemd; the "
                    "desktop will close shortly and the workstation will reboot "
                    f"automatically when installation succeeds ({hazard})"
                ),
                user_actions=[
                    UserAction(
                        text=(
                            "Save any other open work now. Do not power off after the "
                            "desktop closes. After the automatic reboot, log in and "
                            "run mv3dt-installer again to continue with TensorRT and "
                            "cuDNN."
                        ),
                        command=f"sudo journalctl -fu {DRIVER_HANDOFF_UNIT_NAME}",
                        path=str(DRIVER_HANDOFF_LOG_PATH),
                    ),
                ],
            )

        active_display_manager = _active_display_manager(ctx)
        if not _stop_display_manager(ctx):
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message=(
                    "could not stop the desktop session (gdm/lightdm) to run the "
                    "driver installer"
                ),
                user_actions=[
                    UserAction(
                        text=(
                            "Switch to a TTY (Ctrl+Alt+F3), log in, and re-run the "
                            "installer from there."
                        ),
                    ),
                ],
            )

        run_path.chmod(0o755)
        # --silent is documented drift from the DS 9.1 page's verbatim
        # command (STEP-1 section 5.1). The runfile is interactive by
        # default: it can stop on ncurses questions about DKMS
        # registration, 32-bit compatibility libraries, or an existing
        # driver. run_root captures output, so such a question is never
        # drawn on any screen -- the install simply blocks forever with
        # nothing to show why, which is indistinguishable from a slow
        # kernel-module build. --silent (which implies --no-questions and
        # accepts the licence) makes that class of hang impossible.
        result = ctx.run_root(
            str(run_path),
            "--silent",
            "--no-cc-version-check",
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            if active_display_manager is None:
                restored = True
                restore_state = "not needed"
            else:
                restored = _start_display_manager(ctx, active_display_manager)
                restore_state = "succeeded" if restored else "failed"
            actions = []
            if not restored:
                actions.append(
                    UserAction(
                        text="Restart the graphical login or reboot the workstation.",
                        command=f"sudo systemctl start {active_display_manager}",
                    )
                )
            return StepResult(
                status=StepStatus.FAILED,
                message=(
                    f"NVIDIA driver installer exited {result.returncode}; "
                    f"desktop restore {restore_state}"
                ),
                user_actions=actions,
            )
        ctx.report_installed("nvidia-driver", DRIVER_VERSION)

        # USER_ACTION_REQUIRED, not ctx.reboot.request() -- see the reboot
        # cleanup branch above for the full "why". On the next launch,
        # `run()`'s `_driver_loaded(ctx)` probe (nvidia-smi reporting a
        # driver version) is what actually detects the reboot happened and
        # routes into `_run_launch_b`; the framework's boot-id auto-complete
        # is never asked to do that job.
        return StepResult(
            status=StepStatus.USER_ACTION_REQUIRED,
            message=(
                "the NVIDIA driver kernel module is installed but not yet loaded; a "
                "reboot is required before TensorRT/cuDNN install"
            ),
            user_actions=_driver_reboot_actions(),
        )

    def _run_launch_b(self, ctx: "Context") -> StepResult:
        ctx.progress.phase(5)
        ctx.progress.task(f"CUDA {CUDA_VERSION} toolkit")
        if _nvcc_release(ctx) != CUDA_VERSION:
            failure = _install_cuda_toolkit(ctx)
            if failure is not None:
                return failure
        _write_cuda_profile()
        if _nvcc_release(ctx) != CUDA_VERSION:
            return StepResult(
                status=StepStatus.FAILED,
                message=f"CUDA {CUDA_VERSION} did not verify after toolkit install",
            )

        ctx.progress.task(f"TensorRT {TENSORRT_VERSION}")
        apt_args = [f"{pkg}={TENSORRT_VERSION}" for pkg in TENSORRT_PACKAGES]
        failure = _apt_install_reported(ctx, TENSORRT_PACKAGES, apt_args=apt_args)
        if failure is not None:
            return failure

        ctx.progress.task(f"cuDNN {CUDNN_VERSION}")
        cudnn_args = [f"{package}={CUDNN_APT_VERSION}" for package in CUDNN_PACKAGES]
        failure = _apt_install_reported(
            ctx,
            CUDNN_PACKAGES,
            apt_args=cudnn_args,
            reported_version=CUDNN_VERSION,
            allow_downgrades=True,
        )
        if failure is not None:
            return failure

        # No separate GStreamer report here: `gstreamer1.0-tools` is already
        # reported (installed/already-installed) by Launch A's
        # `APT_PREREQ_PACKAGES` transaction; `verify()`'s `verify_pinned`
        # call is what confirms the `1.24.2` pin.

        mosquitto_failure = self._run_mosquitto(ctx)
        if mosquitto_failure is not None:
            return mosquitto_failure

        return StepResult(status=StepStatus.COMPLETE)

    def _run_mosquitto(self, ctx: "Context") -> StepResult | None:
        """STEP-1 section 3.2. Runs the bundled script and wraps it with the
        before/after diff-detection reporting the script itself does not
        do. Returns a `StepResult` only on failure; `None` means "continue".
        """
        ctx.progress.phase(6)
        ctx.progress.task("Mosquitto broker and mv3dt.conf")
        before_version = _dpkg_version(ctx, "mosquitto")
        dst_path = _mosquitto_dst_path()
        before_hash = _sha256_file(dst_path)
        bundled_hash = _sha256_file(ctx.asset_path("mosquitto", "mv3dt.conf"))

        result = shellout.run_bundled_script(
            "scripts",
            "10_setup_mosquitto.sh",
            args=["--non-interactive"],
            env={
                "MV3DT_INSTALLER_CONF": str(
                    pathlib.Path(ctx.install_dir) / "installer.conf"
                )
            },
            tree=(),
        )

        if result.returncode != 0:
            return StepResult(
                status=StepStatus.FAILED,
                message=f"mosquitto setup script exited {result.returncode}",
            )

        after_version = _dpkg_version(ctx, "mosquitto") or "unknown"
        if before_version is None:
            ctx.report_installed("mosquitto", after_version)
        else:
            ctx.report_already_installed("mosquitto", after_version)

        after_hash = _sha256_file(dst_path)
        conf_label = (after_hash or "unknown")[:12]
        if before_hash != bundled_hash:
            ctx.report_installed("mv3dt.conf", conf_label)
        else:
            ctx.report_already_installed("mv3dt.conf", conf_label)

        return None

    # -- verify (section 7.3) -------------------------------------------------

    def verify(self, ctx: "Context") -> StepResult:
        checks = [
            ctx.verify_pinned("NVIDIA driver", _driver_version(ctx), DRIVER_VERSION),
            ctx.verify_pinned("CUDA (nvcc release)", _nvcc_release(ctx), CUDA_VERSION),
            ctx.verify_pinned(
                f"cuDNN ({CUDNN_QUERY_PACKAGE})",
                _cudnn_installed_version(ctx) or "",
                CUDNN_VERSION,
            ),
            ctx.verify_pinned(
                "TensorRT (libnvinfer10)",
                _dpkg_version(ctx, "libnvinfer10") or "",
                TENSORRT_VERSION,
            ),
            ctx.verify_pinned("GStreamer", _gstreamer_version(ctx), GSTREAMER_VERSION),
            _verify_mosquitto_active(ctx),
        ]

        _record_gpu_info(ctx)

        if all(checks):
            return StepResult(status=StepStatus.COMPLETE)

        return StepResult(
            status=StepStatus.USER_ACTION_REQUIRED,
            message=(
                "one or more DeepStream 9.1 prerequisite pins did not verify; see the "
                "transcript above for which"
            ),
            user_actions=[
                UserAction(
                    text=(
                        "Resolve the version mismatch reported above, then re-run the "
                        "installer."
                    ),
                ),
            ],
        )

    # -- report (section 7.4) --------------------------------------------------

    def report(self, ctx: "Context") -> None:
        ctx.log.info(
            "Step 1 (Prerequisites) summary: "
            f"driver {DRIVER_VERSION}, CUDA {CUDA_VERSION}, cuDNN {CUDNN_VERSION}, "
            f"TensorRT {TENSORRT_VERSION}, GStreamer {GSTREAMER_VERSION}; "
            f"CUDA profile at {CUDA_PROFILE_PATH}; "
            f"nouveau blacklist at {NOUVEAU_BLACKLIST_PATH}; "
            f"mosquitto broker active with drop-in at {_mosquitto_dst_path()}."
        )


register(Step1Prerequisites())
