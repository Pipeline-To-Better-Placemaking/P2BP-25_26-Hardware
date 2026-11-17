"""`python -m mv3dt_amc_control serve | openapi | check-config | projects`."""

from __future__ import annotations

import argparse
import sys

from .settings import Settings, is_loopback


def _serve(settings: Settings) -> int:
    missing = [
        name
        for name, value in (
            ("WORKSTATION_TICKET_SECRET", settings.ticket_secret),
            ("ALLOWED_ORIGINS", ",".join(settings.allowed_origins)),
        )
        if not value
    ]
    if missing:
        print(f"refusing to serve: {', '.join(missing)} not set", file=sys.stderr)
        return 2
    if any("*" in o for o in settings.allowed_origins):
        print("refusing to serve: ALLOWED_ORIGINS must list origins, not *", file=sys.stderr)
        return 2
    try:
        host, port = settings.bind_address()
    except ValueError as exc:
        print(f"refusing to serve: {exc}", file=sys.stderr)
        return 2
    if not is_loopback(host) and not settings.allow_non_local_bind:
        print(
            f"refusing to bind {host}: only loopback is allowed (set ALLOW_NON_LOCAL_BIND=1 to override)",
            file=sys.stderr,
        )
        return 2

    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(settings), host=host, port=port, log_level="info", proxy_headers=False)
    return 0


def _openapi(settings: Settings) -> int:
    """Print AMC's own API description (served by AMC at /openapi.yaml)."""
    from .amc_api import AmcApi, AmcApiError, default_base_url
    from .controller import AmcController

    try:
        ms_port, _ = AmcController().ports()
        print(AmcApi(default_base_url(ms_port)).openapi())
    except AmcApiError as exc:
        print(f"could not read AMC's API description: {exc}", file=sys.stderr)
        return 1
    return 0


def _check_config(settings: Settings) -> int:
    """Validate the automation settings (base file merged with the override)."""
    from .automation_config import ConfigError, load_config

    try:
        cfg = load_config(str(settings.automation_config) if settings.automation_config else None)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"ok: project.mode={cfg.project.mode} rectification.mode={cfg.rectification.mode} "
          f"detector={cfg.calibration.detector_type}")
    return 0


def _projects(settings: Settings) -> int:
    """One row per web project folder (AMC-WEB-PROJECTS.md section 7)."""
    from datetime import datetime, timezone

    from .projects import ProjectStore, default_projects_root

    store = ProjectStore(default_projects_root())
    bindings = store.list()
    if not bindings:
        print(f"no project folders in {store.root}")
        return 0

    def when(ts: int | None) -> str:
        if ts is None:
            return "-"
        return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    header = ("FOLDER", "TITLE", "WEB PROJECT", "AMC PROJECT", "STATE", "LAST RUN", "RUNS", "PATH")
    rows = [header] + [
        (b.folder, b.title or "-", b.project_id, b.amc_project_id or "-", b.state,
         when(b.last_run_at), str(b.calib_runs), str(store.root / b.folder))
        for b in bindings
    ]
    widths = [max(len(r[i]) for r in rows) for i in range(len(header) - 1)]
    for r in rows:
        print("  ".join([c.ljust(w) for c, w in zip(r, widths)] + [r[-1]]))
    for b in bindings:
        if b.state == "failed" and b.error:
            print(f"{b.folder}: failed: {b.error}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mv3dt_amc_control")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="run the AMC control API (what the systemd unit runs)")
    sub.add_parser("openapi", help="print AMC's /openapi.yaml (AMC must be running)")
    sub.add_parser("check-config", help="validate the automation settings")
    sub.add_parser("projects", help="list the web project folders on this workstation")
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    return {"serve": _serve, "openapi": _openapi, "check-config": _check_config,
            "projects": _projects}[args.command](settings)
