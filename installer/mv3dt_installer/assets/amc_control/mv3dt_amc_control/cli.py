"""`python -m mv3dt_amc_control serve | openapi | check-config`."""

from __future__ import annotations

import argparse
import sys

from .settings import Settings, is_loopback


def _serve(settings: Settings) -> int:
    missing = [
        name
        for name, value in (
            ("WORKSTATION_TICKET_SECRET", settings.ticket_secret),
            ("TRACKING_PROJECT_ID", settings.project_id),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mv3dt_amc_control")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="run the AMC control API (what the systemd unit runs)")
    sub.add_parser("openapi", help="print AMC's /openapi.yaml (AMC must be running)")
    sub.add_parser("check-config", help="validate the automation settings")
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    return {"serve": _serve, "openapi": _openapi, "check-config": _check_config}[args.command](settings)
