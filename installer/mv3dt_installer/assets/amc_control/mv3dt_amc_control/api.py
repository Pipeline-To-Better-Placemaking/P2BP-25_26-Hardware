"""The AMC control API: start/stop/status of AMC and the automated workflow.

Routes (all but /health need a web app ticket for any web project; scope
"amc" unless noted):

    GET  /health                  open; for the installer's verify and monitoring
    GET  /amc/status              AMC's state and, when ready, the address to open
    POST /amc/start               launch AMC (`mv3dt-installer amc --keep-up ...`)
    POST /amc/stop                shut AMC down (`mv3dt-installer amc --down`)
    GET  /amc/automation/status   the current or last automated run
    POST /amc/automation/start    start a run from a spec signed by the web app
    POST /amc/automation/continue after the manual steps, or to retry
    POST /amc/automation/cancel   stop the run (AMC keeps calibrating if it started)
    GET  /amc/automation/config   the automation settings in effect
    POST /projects                set up the ticket's project on the workstation (scope "amc-provision")
    GET  /projects/current        the ticket's project folder binding (scope "amc" or "amc-provision")
    GET  /projects/current/files  list the folder
    GET  /projects/current/files/{path}  download one file (ticket also accepted as ?ticket=)

The project is always the ticket's `pid`. While a project is being set up,
POST /projects, POST /amc/stop and POST /amc/automation/start answer 409
(AMC-WEB-PROJECTS.md sections 5 and 8).

The browser calls this service directly over the tailnet, so CORS allows
only ALLOWED_ORIGINS, and Chrome's Private Network Access preflight is
answered for those origins (a public https page calling a private address).
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import __version__
from .auth import PROVISION_SCOPE, SCOPE, TicketError, ticket_dependency, verify_ticket
from .automation import Busy, CalibrationRunner, NothingToResume
from .automation_config import ConfigError, load_config
from .calibration_spec import SpecError, verify_spec
from .controller import AmcController
from .models import AmcStatus, AutomationJob, AutomationStart, ProjectBinding, ProjectFile, ProjectProvision
from .projects import Binding, ProjectStore, default_projects_root
from .provisioning import Provisioner, ProvisionError
from .settings import Settings


def create_app(
    settings: Settings,
    controller: AmcController | None = None,
    runner: CalibrationRunner | None = None,
    store: ProjectStore | None = None,
    provisioner: Provisioner | None = None,
) -> FastAPI:
    if any("*" in o for o in settings.allowed_origins):
        raise ValueError("ALLOWED_ORIGINS must list origins, not a wildcard")
    controller = controller or AmcController()
    override = str(settings.automation_config) if settings.automation_config else None
    if runner is None:
        runner = CalibrationRunner(controller, config_loader=lambda: load_config(override))
    if provisioner is None:
        store = store or ProjectStore(default_projects_root())
        provisioner = Provisioner(store, controller, runner_busy=runner.uses_amc,
                                  config_loader=lambda: load_config(override))
    store = provisioner.store
    provisioner.recover()  # no setup job survives a restart

    app = FastAPI(title="mv3dt-amc-control", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.controller = controller
    app.state.runner = runner
    app.state.store = store
    app.state.provisioner = provisioner

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

    ticket = ticket_dependency(settings.ticket_secret, settings.ticket_issuer, SCOPE)
    provision_ticket = ticket_dependency(settings.ticket_secret, settings.ticket_issuer, PROVISION_SCOPE)
    any_ticket = ticket_dependency(settings.ticket_secret, settings.ticket_issuer, {SCOPE, PROVISION_SCOPE})

    def not_while_provisioning() -> None:
        if provisioner.busy():
            raise HTTPException(409, "A project is being set up on the workstation; try again when it finishes")

    def wire(binding: Binding) -> dict:
        return binding.to_wire(store.root / binding.folder)

    def project_of(claims: dict) -> Binding:
        binding = store.get_by_pid(claims["pid"])
        if binding is None:
            raise HTTPException(404, "This project has no workstation folder")
        return binding

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
        not_while_provisioning()
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
        not_while_provisioning()
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

    # -- project folders ----------------------------------------------------

    @app.post("/projects", response_model=ProjectBinding, status_code=202)
    def projects_provision(body: ProjectProvision, response: Response, claims: dict = Depends(provision_ticket)):
        try:
            spec = verify_spec(body.Spec, settings.ticket_secret, claims["pid"], settings.ticket_issuer)
            binding, started = provisioner.provision(spec, claims["pid"])
        except SpecError as exc:
            raise HTTPException(400, str(exc)) from exc
        except ProvisionError as exc:
            raise HTTPException(exc.status, exc.message) from exc
        if not started:
            response.status_code = 200
        return wire(binding)

    @app.get("/projects/current", response_model=ProjectBinding)
    def projects_current(claims: dict = Depends(any_ticket)):
        return wire(project_of(claims))

    @app.get("/projects/current/files", response_model=list[ProjectFile])
    def projects_files(claims: dict = Depends(ticket)):
        try:
            return store.list_files(project_of(claims).folder)
        except FileNotFoundError:
            raise HTTPException(404, "This project has no workstation folder") from None

    @app.get("/projects/current/files/{path:path}")
    def projects_file(path: str, request: Request):
        # A plain download link cannot send a header, so the ticket may also
        # come as ?ticket= (AMC-WEB-PROJECTS.md section 8, a LOCKED trade-off).
        if not settings.ticket_secret:
            raise HTTPException(503, "WORKSTATION_TICKET_SECRET is not configured")
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        token = token.strip() if scheme.lower() == "bearer" else ""
        token = token or request.query_params.get("ticket", "").strip()
        if not token:
            raise HTTPException(401, "Missing bearer ticket")
        try:
            claims = verify_ticket(token, settings.ticket_secret, settings.ticket_issuer, SCOPE)
        except TicketError as exc:
            raise HTTPException(exc.status, exc.message) from None
        binding = project_of(claims)
        try:
            target = store.resolve_file(binding.folder, path)
        except FileNotFoundError:
            raise HTTPException(404, "No such file in this project") from None
        return FileResponse(target, media_type="application/octet-stream", filename=target.name)

    return app
