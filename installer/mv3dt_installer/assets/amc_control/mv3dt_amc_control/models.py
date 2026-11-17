"""Response and request bodies of the AMC control API (field names match the web app)."""

from typing import Annotated

from pydantic import BaseModel, Field

RECORD_MIN_SECONDS = 30
RECORD_MAX_SECONDS = 1800


class AmcStatus(BaseModel):
    State: str = Field(description="down | starting | ready | stopping | error")
    UiUrl: str = Field(default="", description="Where the user's browser opens AMC; empty unless ready")
    Error: str = ""
    UpdatedAt: int


class AutomationStep(BaseModel):
    Key: str
    Number: int = Field(description="AMC 3.3.0 workflow step number")
    Title: str
    State: str = Field(description="pending | running | done | skipped | waiting | failed")
    Message: str = Field(default="", description="What happened, or for a waiting step what the user has to do")
    Done: str = Field(default="", description="What the automation already finished in this step")
    NeedsUser: bool = False


class AutomationJob(BaseModel):
    JobId: str
    ProjectId: str
    ProjectTitle: str
    State: str = Field(description="running | waiting_for_user | completed | failed | cancelled | interrupted")
    Message: str = ""
    AmcProjectId: str = ""
    AmcProjectName: str = ""
    Steps: list[AutomationStep]
    UserActions: list[str] = Field(default_factory=list, description="What the user still has to do in AMC")
    Result: dict = Field(default_factory=dict)
    StartedBy: str
    StartedAt: int
    UpdatedAt: int


class RecordRequest(BaseModel):
    Seconds: Annotated[int, Field(strict=True, ge=RECORD_MIN_SECONDS, le=RECORD_MAX_SECONDS,
                                  description="Length of the recording, 30 to 1800 seconds")]


class AutomationStart(BaseModel):
    Spec: str = Field(description="Signed calibration spec from POST /api/workstation/project/{id}/calibration-spec")
    Record: RecordRequest | None = Field(
        default=None,
        description="Record new footage from every camera in step 2 (project.mode web_project only); "
                    "absent uploads the footage already in the folder")


class ProjectProvision(BaseModel):
    Spec: str = Field(description="Signed calibration spec with a folder, from POST /api/workstation/project/{id}/provision")


class ProjectBinding(BaseModel):
    """A web project's workstation folder: project.json without schema_version, plus Path."""

    ProjectId: str
    Folder: str
    Title: str
    AmcProjectId: str | None = None
    AmcProjectName: str | None = None
    State: str = Field(description="provisioning | ready | failed")
    Error: str = ""
    CreatedAt: int
    UpdatedAt: int
    LastRunAt: int | None = None
    CalibRuns: int = 0
    Path: str = Field(description="The folder's absolute path on the workstation")


class ProjectFile(BaseModel):
    Path: str = Field(description="Relative to the project folder, '/' separated")
    Size: int
    ModifiedAt: int
    Downloadable: bool = Field(description="False for a video file, which stays on the workstation")


class ProjectDeleted(BaseModel):
    """What DELETE /projects/current removed (AMC-PROJECT-LIFECYCLE.md section 3)."""

    Folder: str
    AmcProjectId: str | None = Field(default=None, description="The binding's AMC project, if it had one")
    AmcProjectDeleted: bool = Field(description="True when AMC deleted the project or no longer had it")
    Leftover: str | None = Field(
        default=None, description="The renamed folder that could not be fully removed; null when it is gone")
