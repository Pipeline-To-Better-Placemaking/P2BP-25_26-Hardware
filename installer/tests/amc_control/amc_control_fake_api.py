"""A stand-in for AMC's REST API, built from the endpoints NVIDIA documents in the
amc-run-video-calibration skill. It records every call so tests can check what
the runner sent. The /v1/rectification and /v1/linear_media routes follow AMC
3.3.0's own API description; /v1/test_rectify is invented for the generic
rectification.mode api, whose requests are configured, not hard-coded.

Like AMC 3.3.0, a calibration shows in both project_state and amc_state:
calibrate sets both RUNNING, and when it ends project_state goes back to READY
and amc_state becomes COMPLETED (or ERROR).

Like AMC 3.3.0, the fake keeps one current configuration for all projects
(GET /v1/config), which POST /v1/config/{id} replaces whole, and serves the
factory values at GET /v1/config/defaults.

Like AMC 3.3.0, the export is built in a separate step: mv3dt_result answers
HTTP 400 until POST /v1/postprocess/{id} has run, which get_project_info shows
as postprocess_state READY, RUNNING, then COMPLETED with amc_export_ready true.
Whether a calibration clears an earlier export is not verified on AMC 3.3.0, so by
default it keeps it (calibration_clears_export turns clearing on).
"""
from __future__ import annotations

import io
import json
import zipfile
import copy
from dataclasses import dataclass, field

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response


# A cut-down AMC ConfigParams; layout_px_per_m is AMC 3.3.0's factory value.
DEFAULT_CONFIG = {"layout_px_per_m": 24.22, "detector": "resnet", "bundle_adjustment": {"iterations": 5}}


@dataclass
class FakeAmc:
    ready: bool = True
    projects: dict[str, dict] = field(default_factory=dict)
    calls: list[tuple[str, dict]] = field(default_factory=list)
    running_polls: int = 2          # get_project_info calls that report RUNNING after calibrate
    fail_calibration: bool = False
    # The amc_state a calibration ends in when not "" (e.g. READY for one stopped in AMC).
    calibration_end: str = ""
    # get_project_info reads right after calibrate that still report the states from before
    # it, and the project_state calibrate replies with (None: RUNNING, or the earlier state
    # while reads are stale; "": no project_state in the reply).
    calib_stale_reads: int = 0
    calib_reply: str | None = None
    # calibrate posts answered with HTTP 500 before one is accepted; a refused post changes nothing.
    calibrate_refusals: int = 0
    rect_polls: int = 1
    ui_alignment_done: bool = False  # set by a test to stand in for the user's UI work
    with_gt: bool = False
    next_id: int = 1
    delete_status: int | None = None  # set by a test to make delete_project answer this status
    # AMC's rectification: state reads that report RUNNING before a phase ends, the
    # phase that ends in ERROR instead (estimate, generate or linear), and the cameras.
    rect_running_polls: int = 1
    rect_fail: str = ""
    # State reads right after a start, commit or linear stage that still report the state
    # from before it, and the rectification_state those requests reply with (None: RUNNING,
    # or the earlier state while reads are stale; "": no state in the reply).
    rect_stale_reads: int = 0
    rect_reply: str | None = None
    # AMC's Layout Post-processing: state reads that report RUNNING before it ends, whether
    # it ends in ERROR, reads right after the post that still report the state from before
    # it, posts answered with HTTP 500 before one is accepted, and whether a calibration
    # clears an earlier export (by default it keeps it, which is not ruled out on AMC 3.3.0).
    postprocess_running_polls: int = 2
    postprocess_fail: bool = False
    postprocess_stale_reads: int = 0
    postprocess_refusals: int = 0
    calibration_clears_export: bool = False
    rect_cameras: list[dict] = field(default_factory=lambda: [
        {"camera_id": 0, "committed": None, "auto_estimate": {"model": "simple_radial", "k1": -0.12, "k2": None}},
        {"camera_id": 1, "committed": None, "auto_estimate": {"model": "simple_divisional", "k1": -0.3, "k2": 0.01}},
    ])
    # AMC's one current configuration, starting at the factory values.
    config: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG))

    def add_project(self, pid: str, name: str = "Existing") -> None:
        # An existing project was calibrated before: READY with amc_state COMPLETED.
        self.projects[pid] = {"name": name, "state": "READY", "amc_state": "COMPLETED", "videos": [], "layout": None,
                              "alignment": None, "config": None, "focal": [], "polls": 0, "rect": 0,
                              "rect_state": "NOT_STARTED", "rect_phase": "", "rect_n": 0,
                              "pp_state": "COMPLETED", "export_ready": True, "pp_n": 0}

    def app(self) -> FastAPI:
        app = FastAPI()
        fake = self

        def project(pid: str) -> dict:
            if pid not in fake.projects:
                raise HTTPException(404, f"no project {pid}")
            return fake.projects[pid]

        def log(_call: str, **kw) -> None:
            fake.calls.append((_call, kw))

        @app.get("/v1/ready")
        def ready():
            return {"code": 0} if fake.ready else Response(status_code=503)

        @app.post("/v1/create_project")
        def create_project(project_name: str = Form(...)):
            pid = f"2026{fake.next_id:04d}"
            fake.next_id += 1
            fake.add_project(pid, project_name)
            fake.projects[pid]["state"] = fake.projects[pid]["amc_state"] = "INIT"
            fake.projects[pid]["pp_state"], fake.projects[pid]["export_ready"] = "INIT", False
            log("create_project", name=project_name)
            return {"project_id": pid}

        @app.get("/v1/get_project_info/{pid}")
        def get_project_info(pid: str):
            p = project(pid)
            stale_left, stale_states = p.get("calib_stale", (0, ("", "")))
            if stale_left:
                p["calib_stale"] = (stale_left - 1, stale_states)
                project_state, amc_state = stale_states
            else:
                if "RUNNING" in (p["state"], p["amc_state"]):
                    p["polls"] += 1
                    if p["polls"] > fake.running_polls:
                        p["state"] = "READY"
                        p["amc_state"] = fake.calibration_end or ("ERROR" if fake.fail_calibration else "COMPLETED")
                project_state, amc_state = p["state"], p["amc_state"]
            pp_left, pp_before = p.get("pp_stale", (0, ("", False)))
            if pp_left:
                p["pp_stale"] = (pp_left - 1, pp_before)
                pp_state, export_ready = pp_before
            else:
                if p["pp_state"] == "RUNNING":
                    p["pp_n"] += 1
                    if p["pp_n"] > fake.postprocess_running_polls:
                        p["pp_state"] = "ERROR" if fake.postprocess_fail else "COMPLETED"
                        p["export_ready"] = not fake.postprocess_fail
                pp_state, export_ready = p["pp_state"], p["export_ready"]
            log("get_project_info", project_state=project_state, amc_state=amc_state,
                postprocess_state=pp_state, amc_export_ready=export_ready)
            return {"code": 0, "project_info": {
                "project_state": project_state, "amc_state": amc_state, "project_name": p["name"],
                "postprocess_state": pp_state, "amc_export_ready": export_ready, "vggt_export_ready": False,
                "vggt_state": "INIT", "amc_results_available": amc_state == "COMPLETED",
                "video_files_count": len(p["videos"]), "layout_file": p["layout"] is not None,
                "alignment_file": p["alignment"] is not None or fake.ui_alignment_done,
                "rectification_state": p["rect_state"]}}

        @app.post("/v1/upload_video_files/{pid}")
        async def upload_video_files(pid: str, files: list[UploadFile] = File(...)):
            p = project(pid)
            p["videos"] = [(f.filename, len(await f.read())) for f in files]
            log("upload_video_files", names=[f.filename for f in files])
            return {"code": 0}

        @app.get("/v1/config")
        def get_config():
            log("get_config")
            return {"code": 0, "config_params": copy.deepcopy(fake.config)}

        @app.get("/v1/config/defaults")
        def get_default_config():
            log("get_default_config")
            return {"code": 0, "config_params": copy.deepcopy(DEFAULT_CONFIG)}

        @app.post("/v1/config/{pid}")
        async def config(pid: str, request: Request):
            body = await request.json()
            project(pid)["config"] = body
            fake.config = copy.deepcopy(body)   # AMC keeps one current config, replaced whole
            log("config", settings=body)
            return {"code": 0}

        @app.post("/v1/upload_layout/{pid}")
        async def upload_layout(pid: str, layout_file: UploadFile = File(...)):
            data = await layout_file.read()
            project(pid)["layout"] = data
            log("upload_layout", filename=layout_file.filename, png=data.startswith(b"\x89PNG"))
            return {"code": 0}

        @app.post("/v1/upload_alignment/{pid}")
        async def upload_alignment(pid: str, coord_space: str, alignment_file: UploadFile = File(...)):
            project(pid)["alignment"] = json.loads(await alignment_file.read())
            log("upload_alignment", coord_space=coord_space)
            return {"code": 0}

        @app.post("/v1/upload_focal_length/{pid}")
        def upload_focal_length(pid: str, focal_length: list[str] = Form(...)):
            project(pid)["focal"] = [float(f) for f in focal_length]
            log("upload_focal_length", values=project(pid)["focal"])
            return {"code": 0}

        @app.post("/v1/verify_project/{pid}")
        def verify_project(pid: str):
            p = project(pid)
            ok = p["videos"] and (p["layout"] or fake.ui_alignment_done) and (p["alignment"] or fake.ui_alignment_done)
            log("verify_project", ready=bool(ok))
            return {"project_state": "READY" if ok else "INIT"}

        @app.post("/v1/calibrate/{pid}")
        async def calibrate(pid: str, request: Request):
            body = await request.json()
            p = project(pid)
            if fake.calibrate_refusals:
                fake.calibrate_refusals -= 1
                log("calibrate_refused", detector_type=body.get("detector_type"))
                raise HTTPException(500, "calibration could not start")
            before = (p["state"], p["amc_state"])
            p["calib_stale"] = (fake.calib_stale_reads, before)
            p["state"], p["amc_state"], p["polls"] = "RUNNING", "RUNNING", 0
            if fake.calibration_clears_export:
                p["pp_state"], p["export_ready"] = "READY", False
            log("calibrate", detector_type=body.get("detector_type"))
            reply = fake.calib_reply
            if reply is None:
                reply = before[0] if fake.calib_stale_reads else "RUNNING"
            return {"code": 0, **({"project_state": reply} if reply else {})}

        @app.post("/v1/postprocess/{pid}")
        def postprocess(pid: str):
            p = project(pid)
            if fake.postprocess_refusals:
                fake.postprocess_refusals -= 1
                log("postprocess_refused", pid=pid)
                raise HTTPException(500, "post-processing could not start")
            p["pp_stale"] = (fake.postprocess_stale_reads, (p["pp_state"], p["export_ready"]))
            p["pp_state"], p["export_ready"], p["pp_n"] = "RUNNING", False, 0
            log("postprocess", pid=pid)
            # AMC 3.3.0 answers with a CalibrationResponse: no postprocess_state.
            return {"code": 0, "message": "post-processing started", "project_state": p["state"],
                    "calibration_pid": None}

        @app.post("/v1/stop_calibration/{pid}")
        def stop_calibration(pid: str):
            p = project(pid)
            if "RUNNING" in (p["state"], p["amc_state"]):
                p["state"], p["amc_state"] = "INIT", "INIT"
            log("stop_calibration", pid=pid)
            return {"code": 0}

        @app.delete("/v1/delete_project/{pid}")
        def delete_project(pid: str):
            log("delete_project", pid=pid)
            if fake.delete_status is not None:
                raise HTTPException(fake.delete_status, "refused by the test")
            project(pid)
            del fake.projects[pid]
            return {"code": 0}

        @app.get("/v1/amc/calibrate/{pid}/log", response_class=PlainTextResponse)
        def calibrate_log(pid: str):
            return "line 1\nline 2\nnot enough tracklets\n"

        @app.get("/v1/result/{pid}/evaluation_statistics")
        def evaluation_statistics(pid: str):
            if not fake.with_gt:
                raise HTTPException(404, "no ground truth")
            return {"Average L2 distance(m)": 0.4, "Average reprojection error 0(px)": 2.1}

        @app.get("/v1/result/{pid}/mv3dt_result")
        def mv3dt_result(pid: str, result_type: str):
            if not project(pid)["export_ready"]:
                log("mv3dt_result_refused", result_type=result_type)
                raise HTTPException(400, "AMC export artifacts are not available. Run post-processing after "
                                         "calibration completes.")
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("transforms.yml", f"project: {pid}\n")
            log("mv3dt_result", result_type=result_type)
            return Response(buf.getvalue(), media_type="application/zip")

        def rect_phase(p: dict, phase: str, message: str) -> dict:
            before = p["rect_state"]
            p["rect_stale"] = (fake.rect_stale_reads, before)
            p["rect_state"], p["rect_phase"], p["rect_n"] = "RUNNING", phase, 0
            reply = fake.rect_reply
            if reply is None:
                reply = before if fake.rect_stale_reads else "RUNNING"
            return {"code": 0, "message": message, **({"rectification_state": reply} if reply else {})}

        @app.get("/v1/rectification/{pid}")
        def rectification_state(pid: str):
            p = project(pid)
            stale_left, stale_state = p.get("rect_stale", (0, ""))
            if stale_left:
                p["rect_stale"] = (stale_left - 1, stale_state)
                log("rectification_state", state=stale_state)
                return {"code": 0, "message": "", "rectification_state": stale_state, "rectification_pid": 4242}
            if p["rect_state"] == "RUNNING":
                p["rect_n"] += 1
                if p["rect_n"] > fake.rect_running_polls:
                    if fake.rect_fail == p["rect_phase"]:
                        p["rect_state"] = "ERROR"
                    else:
                        p["rect_state"] = "READY_FOR_REVIEW" if p["rect_phase"] == "estimate" else "COMPLETED"
            log("rectification_state", state=p["rect_state"])
            return {"code": 0, "message": "", "rectification_state": p["rect_state"], "rectification_pid": 4242}

        @app.post("/v1/rectification/{pid}")
        def start_rectification(pid: str):
            p = project(pid)
            reply = rect_phase(p, "estimate", "started")
            log("start_rectification", pid=pid)
            return reply

        @app.get("/v1/rectification/{pid}/cameras")
        def rectification_cameras(pid: str):
            project(pid)
            cams = [{"video_name": f"cam_{c['camera_id']:02d}.mp4", "width": 1920, "height": 1080, "cx": 960,
                     "cy": 540, "k1_min": -1, "k1_max": 1, "k2_min": -1, "k2_max": 1, "grid_lines": [], **c}
                    for c in fake.rect_cameras]
            return {"code": 0, "message": "", "cameras": cams}

        @app.post("/v1/rectification/{pid}/manual")
        async def commit_rectification(pid: str, request: Request):
            p = project(pid)
            body = await request.json()
            reply = rect_phase(p, "generate", "generating")
            log("commit_rectification", body=body)
            return reply

        @app.post("/v1/linear_media/{pid}")
        def stage_linear_media(pid: str):
            reply = rect_phase(project(pid), "linear", "staged")
            log("stage_linear_media", pid=pid)
            return reply

        @app.get("/v1/rectification/{pid}/log", response_class=PlainTextResponse)
        def rectification_log(pid: str):
            return "\n".join(f"rect line {i}" for i in range(1, 15)) + "\nno grid lines found\n"

        # Test-only rectification routes, reached through the configurable request() path.
        @app.post("/v1/test_rectify/{pid}")
        async def rectify(pid: str, request: Request):
            project(pid)["rect"] = 0
            log("rectify", body=await request.json())
            return {"code": 0}

        @app.get("/v1/test_rectify/{pid}/status")
        def rectify_status(pid: str):
            p = project(pid)
            p["rect"] += 1
            return {"rectification": {"state": "COMPLETED" if p["rect"] > fake.rect_polls else "RUNNING"}}

        return app
