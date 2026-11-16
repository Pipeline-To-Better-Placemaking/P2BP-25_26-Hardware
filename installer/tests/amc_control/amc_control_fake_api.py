"""A stand-in for AMC's REST API, built from the endpoints NVIDIA documents in the
amc-run-video-calibration skill. It records every call so tests can check what
the runner sent. The rectification routes are invented for tests: the real ones
come from AMC's /openapi.yaml and are configured, not hard-coded.
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

    def add_project(self, pid: str, name: str = "Existing") -> None:
        self.projects[pid] = {"name": name, "state": "COMPLETED", "videos": [], "layout": None,
                              "alignment": None, "config": None, "focal": [], "polls": 0, "rect": 0}

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
                                                "vggt_state": "INIT"}}

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
