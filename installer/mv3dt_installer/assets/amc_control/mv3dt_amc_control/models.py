"""Response and request bodies of the AMC control API (field names match the web app)."""

from pydantic import BaseModel, Field


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


class AutomationStart(BaseModel):
    Spec: str = Field(description="Signed calibration spec from POST /api/workstation/project/{id}/calibration-spec")
