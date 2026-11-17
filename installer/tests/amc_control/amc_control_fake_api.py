"""A stand-in for AMC's REST API, built from the endpoints NVIDIA documents in the
amc-run-video-calibration skill. It records every call so tests can check what
the runner sent. The /v1/rectification and /v1/linear_media routes follow AMC
3.3.0's own API description; /v1/test_rectify is invented for the generic
rectification.mode api, whose requests are configured, not hard-coded.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response


@dataclass
class FakeAmc:
    ready: bool = True
    projects: dict[str, dict] = field(default_factory=dict)
    calls: list[tuple[str, dict]] = field(default_factory=list)
    running_polls: int = 2          # get_project_info calls that report RUNNING after calibrate
    fail_calibration: bool = False
    rect_polls: int = 1
    ui_alignment_done: bool = False  # set by a test to stand in for the user's UI work
    with_gt: bool = False
    next_id: int = 1
    delete_status: int | None = None  # set by a test to make delete_project answer this status
    # AMC's rectification: state reads that report RUNNING before a phase ends, the
    # phase that ends in ERROR instead (estimate, generate or linear), and the cameras.
    rect_running_polls: int = 1
    rect_fail: str = ""
    rect_cameras: list[dict] = field(default_factory=lambda: [
        {"camera_id": 0, "committed": None, "auto_estimate": {"model": "simple_radial", "k1": -0.12, "k2": None}},
        {"camera_id": 1, "committed": None, "auto_estimate": {"model": "simple_divisional", "k1": -0.3, "k2": 0.01}},
    ])

    def add_project(self, pid: str, name: str = "Existing") -> None:
        self.projects[pid] = {"name": name, "state": "COMPLETED", "videos": [], "layout": None,
                              "alignment": None, "config": None, "focal": [], "polls": 0, "rect": 0,
                              "rect_state": "NOT_STARTED", "rect_phase": "", "rect_n": 0}

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
            fake.projects[pid]["state"] = "INIT"
            log("create_project", name=project_name)
            return {"project_id": pid}

        @app.get("/v1/get_project_info/{pid}")
        def get_project_info(pid: str):
            p = project(pid)
            if p["state"] == "RUNNING":
                p["polls"] += 1
                if p["polls"] > fake.running_polls:
                    p["state"] = "ERROR" if fake.fail_calibration else "COMPLETED"
            return {"code": 0, "project_info": {"project_state": p["state"], "project_name": p["name"],
                                                "vggt_state": "INIT", "video_files_count": len(p["videos"])}}

        @app.post("/v1/upload_video_files/{pid}")
        async def upload_video_files(pid: str, files: list[UploadFile] = File(...)):
            p = project(pid)
            p["videos"] = [(f.filename, len(await f.read())) for f in files]
            log("upload_video_files", names=[f.filename for f in files])
            return {"code": 0}

        @app.post("/v1/config/{pid}")
        async def config(pid: str, request: Request):
            project(pid)["config"] = await request.json()
            log("config", settings=project(pid)["config"])
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
            p["state"], p["polls"] = "RUNNING", 0
            log("calibrate", detector_type=body.get("detector_type"))
            return {"code": 0}

        @app.post("/v1/stop_calibration/{pid}")
        def stop_calibration(pid: str):
            p = project(pid)
            if p["state"] == "RUNNING":
                p["state"] = "INIT"
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
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("transforms.yml", f"project: {pid}\n")
            log("mv3dt_result", result_type=result_type)
            return Response(buf.getvalue(), media_type="application/zip")

        def rect_phase(p: dict, phase: str) -> None:
            p["rect_state"], p["rect_phase"], p["rect_n"] = "RUNNING", phase, 0

        @app.get("/v1/rectification/{pid}")
        def rectification_state(pid: str):
            p = project(pid)
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
            rect_phase(project(pid), "estimate")
            log("start_rectification", pid=pid)
            return {"code": 0, "message": "started"}

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
            rect_phase(p, "generate")
            log("commit_rectification", body=body)
            return {"code": 0, "message": "generating"}

        @app.post("/v1/linear_media/{pid}")
        def stage_linear_media(pid: str):
            rect_phase(project(pid), "linear")
            log("stage_linear_media", pid=pid)
            return {"code": 0, "message": "staged"}

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
