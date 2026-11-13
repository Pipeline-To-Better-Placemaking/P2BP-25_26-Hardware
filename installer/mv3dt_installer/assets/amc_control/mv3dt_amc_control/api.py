"""The AMC control API: start/stop/status of AMC and the automated workflow.

Routes (all but /health need a web app ticket with scope "amc"):

    GET  /health                  open; for the installer's verify and monitoring
    GET  /amc/status              AMC's state and, when ready, the address to open
    POST /amc/start               launch AMC (`bin/amc --keep-up ...`)
    POST /amc/stop                shut AMC down (`bin/amc --down`)
    GET  /amc/automation/status   the current or last automated run
    POST /amc/automation/start    start a run from a spec signed by the web app
    POST /amc/automation/continue after the manual steps, or to retry
    POST /amc/automation/cancel   stop the run (AMC keeps calibrating if it started)
    GET  /amc/automation/config   the automation settings in effect

The browser calls this service directly over the tailnet, so CORS allows
only ALLOWED_ORIGINS, and Chrome's Private Network Access preflight is
answered for those origins (a public https page calling a private address).
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .auth import ticket_dependency
from .automation import Busy, CalibrationRunner, NothingToResume
from .automation_config import ConfigError, load_config
from .calibration_spec import SpecError, verify_spec
from .controller import AmcController
from .models import AmcStatus, AutomationJob, AutomationStart
from .settings import Settings


def create_app(
    settings: Settings,
    controller: AmcController | None = None,
    runner: CalibrationRunner | None = None,
) -> FastAPI:
    if any("*" in o for o in settings.allowed_origins):
        raise ValueError("ALLOWED_ORIGINS must list origins, not a wildcard")
    controller = controller or AmcController()
    if runner is None:
        override = str(settings.automation_config) if settings.automation_config else None
        runner = CalibrationRunner(controller, config_loader=lambda: load_config(override))

    app = FastAPI(title="mv3dt-amc-control", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.controller = controller
    app.state.runner = runner

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
        allow_credentials=False,
        # Chrome's Private Network Access preflight: the web app is a public
        # https page and this service is a private (tailnet) address. Without
        # this the preflight is refused and the browser never sends the request.
        allow_private_network=True,
    )

    ticket = ticket_dependency(settings.ticket_secret, settings.project_id, settings.ticket_issuer)

    def same_project(claims: dict) -> None:
        job = runner.status()
        if job and job["ProjectId"] != claims.get("pid"):
            raise HTTPException(403, "The current run belongs to a different project")

    # -- open ---------------------------------------------------------------

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "ticket_configured": bool(settings.ticket_secret)}

    # -- AMC on/off ---------------------------------------------------------

    @app.get("/amc/status", response_model=AmcStatus)
    def amc_status(claims: dict = Depends(ticket)):
        return controller.status()

    @app.post("/amc/start", response_model=AmcStatus, status_code=202)
    def amc_start(claims: dict = Depends(ticket)):
        controller.start()
        return controller.status()

    @app.post("/amc/stop", response_model=AmcStatus, status_code=202)
    def amc_stop(claims: dict = Depends(ticket)):
        if runner.uses_amc():
            raise HTTPException(409, "A calibration run is using AMC; cancel the run first")
        controller.stop()
        return controller.status()

    # -- automation ---------------------------------------------------------

    @app.get("/amc/automation/status", response_model=AutomationJob | None)
    def automation_status(claims: dict = Depends(ticket)):
        job = runner.status()
        if job and job["ProjectId"] != claims.get("pid"):
            return None  # never show another project's run
        return job

    @app.post("/amc/automation/start", response_model=AutomationJob, status_code=202)
    def automation_start(body: AutomationStart, claims: dict = Depends(ticket)):
        try:
            spec = verify_spec(body.Spec, settings.ticket_secret, claims["pid"], settings.ticket_issuer)
            return runner.start(spec)
        except SpecError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Busy as exc:
            raise HTTPException(409, str(exc)) from exc
        except ConfigError as exc:
            raise HTTPException(500, str(exc)) from exc

    @app.post("/amc/automation/continue", response_model=AutomationJob, status_code=202)
    def automation_continue(claims: dict = Depends(ticket)):
        same_project(claims)
        try:
            return runner.resume()
        except NothingToResume as exc:
            raise HTTPException(409, str(exc)) from exc
        except ConfigError as exc:
            raise HTTPException(500, str(exc)) from exc

    @app.post("/amc/automation/cancel", response_model=AutomationJob, status_code=202)
    def automation_cancel(claims: dict = Depends(ticket)):
        same_project(claims)
        try:
            return runner.cancel()
        except NothingToResume as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/amc/automation/config")
    def automation_config(claims: dict = Depends(ticket)):
        try:
            return runner.config().model_dump(mode="json", by_alias=True)
        except ConfigError as exc:
            raise HTTPException(500, str(exc)) from exc

    return app
