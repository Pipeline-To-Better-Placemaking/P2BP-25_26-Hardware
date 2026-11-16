"""Start / stop / status control of NVIDIA AutoMagicCalib (AMC).

AMC is two Docker containers (UI and REST API) that installer Step 3 wraps in
`<install_dir>/bin/amc`. This module only runs that wrapper with fixed
arguments; nothing from an HTTP request reaches a command line. State is not
kept here: `status()` asks AMC's readiness endpoints each time, so a restart
of the service loses nothing.

Environment (rendered into the unit by installer Step 9):
  MV3DT_INSTALL_DIR  install root, default /opt/mv3dt (installer.conf, bin/amc)
  AMC_STATE_DIR      where the launcher log and automation progress are kept,
                     default <install_dir>/amc_control/state (owned by the
                     service user; <install_dir>/run is root's)
  AMC_BIN            override the wrapper path
  AMC_USE_SUDO       "0" runs the wrapper without `sudo -n` (tests). `bin/amc`
                     runs installer subcommands, which require root, so Step 9
                     writes a sudoers rule for exactly the two commands below.
  AMC_PUBLIC_HOST    the address the user's browser uses to reach AMC, passed
                     to `amc --host-ip` (documented as an IP). Step 9 sets the
                     workstation's Tailscale IPv4; default `tailscale ip -4`.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

_HOST_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9.\-]*[A-Za-z0-9])?$|^[0-9A-Fa-f:]+$")


def _http_get(url: str, timeout: float = 2.0) -> tuple[int | None, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (localhost only)
            return r.status, r.read(65536)
    except urllib.error.HTTPError as e:
        return e.code, b""
    except (urllib.error.URLError, OSError, ValueError):
        return None, b""


def _read_conf(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip().strip("'\"")
    return out


def _int(v: str | None) -> int | None:
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


class AmcController:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._kind: str | None = None  # "start" | "stop"
        self._old: list[subprocess.Popen] = []
        self._error = ""
        self._ts_cache: tuple[float, str | None] = (0.0, None)

    @property
    def install_dir(self) -> Path:
        return Path(os.environ.get("MV3DT_INSTALL_DIR", "/opt/mv3dt"))

    @property
    def amc_bin(self) -> str:
        return os.environ.get("AMC_BIN") or str(self.install_dir / "bin" / "amc")

    @property
    def state_dir(self) -> Path:
        return Path(os.environ.get("AMC_STATE_DIR") or self.install_dir / "amc_control" / "state")

    @property
    def log_path(self) -> Path:
        return self.state_dir / "amc_service.log"

    def _prefix(self) -> list[str]:
        # `sudo -n` fails fast instead of hanging on a password prompt.
        use_sudo = os.environ.get("AMC_USE_SUDO", "1") != "0"
        return (["sudo", "-n"] if use_sudo else []) + [self.amc_bin]

    def ports(self) -> tuple[int | None, int | None]:
        conf = _read_conf(self.install_dir / "installer.conf")
        return _int(conf.get("AUTO_MAGIC_CALIB_MS_PORT")), _int(conf.get("AUTO_MAGIC_CALIB_UI_PORT"))

    def public_host(self) -> str | None:
        override = os.environ.get("AMC_PUBLIC_HOST", "").strip()
        host = override or self._tailscale_host()
        return host if host and _HOST_RE.match(host) else None

    def _tailscale_host(self) -> str | None:
        at, cached = self._ts_cache
        if time.time() - at < 60:
            return cached
        found: str | None = None
        try:
            out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                found = next((line.strip() for line in out.stdout.splitlines() if line.strip()), None)
        except (OSError, subprocess.SubprocessError):
            pass
        self._ts_cache = (time.time(), found)
        return found

    def ready(self) -> bool:
        """AMC is ready when the API answers /v1/ready with code 0 AND the UI answers 200."""
        ms, ui = self.ports()
        if not ms or not ui:
            return False
        status, body = _http_get(f"http://127.0.0.1:{ms}/v1/ready")
        if status != 200:
            return False
        try:
            if json.loads(body).get("code") != 0:
                return False
        except (ValueError, AttributeError):
            return False
        ui_status, _ = _http_get(f"http://127.0.0.1:{ui}/")
        return ui_status == 200

    def _launch(self, kind: str, args: list[str]) -> None:
        cmd = self._prefix() + args
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "ab") as lf:
                lf.write(f"\n--- {time.strftime('%F %T')} {kind} ---\n".encode())
                lf.flush()
                self._proc = subprocess.Popen(
                    cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True
                )
            self._kind = kind
        except OSError as e:
            self._proc, self._kind = None, None
            self._error = f"could not run amc: {e}"

    def _log_tail(self, n: int = 600) -> str:
        try:
            with open(self.log_path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                fh.seek(max(0, fh.tell() - n))
                return fh.read().decode("utf-8", "replace").strip()
        except OSError:
            return ""

    def _reap(self) -> None:
        self._old = [p for p in self._old if p.poll() is None]

    def start(self) -> None:
        with self._lock:
            self._reap()
            self._error = ""
            if self.ready():
                return
            if self._proc and self._proc.poll() is None and self._kind == "start":
                return
            host = self.public_host()
            if not host:
                self._error = "Tailscale address unavailable (is tailscale up? or set AMC_PUBLIC_HOST)"
                return
            self._launch("start", ["--keep-up", "--no-open", "--non-interactive", "--host-ip", host])

    def stop(self) -> None:
        with self._lock:
            self._reap()
            self._error = ""
            if self._proc and self._proc.poll() is None:
                if self._kind == "stop":
                    return
                # `--down` makes the in-flight start fail on its own; its exit
                # code is irrelevant, so move it aside instead of reporting it.
                self._old.append(self._proc)
                self._proc = None
            self._launch("stop", ["--down"])

    def status(self) -> dict:
        with self._lock:
            self._reap()
            state = self._current()
            ui_url = ""
            if state == "ready":
                host, (_, ui) = self.public_host(), self.ports()
                if host and ui:
                    ui_url = f"http://{host}:{ui}"
            return {"State": state, "UiUrl": ui_url, "Error": self._error, "UpdatedAt": int(time.time())}

    def _current(self) -> str:
        if self._proc is not None:
            rc = self._proc.poll()
            if rc is None:
                return "starting" if self._kind == "start" else "stopping"
            if rc != 0:
                self._error = f"amc {self._kind} exited {rc}: {self._log_tail()}"
            self._proc = None
        if self.ready():
            self._error = ""
            return "ready"
        return "error" if self._error else "down"
