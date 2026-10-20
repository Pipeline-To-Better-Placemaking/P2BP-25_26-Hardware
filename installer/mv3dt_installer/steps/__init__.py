"""Step-module interface and registry.

Implements doc `00-FRAMEWORK-AND-BOOTSTRAP.md` §12.1 (the `Step` protocol)
and §12.2 (`StepStatus`, `StepResult`, and the `UserAction` type referenced
from §9.3). This module is framework-only: it defines the contract that
`stepN_*.py` modules (step1_prerequisites.py … step7_webapp_integration.py)
implement and register against. No step business logic lives here.

The optional `phases` declaration and its accessors implement doc
`08-PROGRESS-AND-OBSERVABILITY.md` §3.1 and §9. They are declarative only:
nothing here influences how a step's `run()` executes.

`Context` (§12.3) is built later in `app.py`, which depends on nearly every
other module in this package and is therefore integrated last. To avoid a
circular/premature import, this module uses `from __future__ import
annotations` (PEP 563) so `Context` can appear as an unevaluated type hint
in the `Step` protocol's method signatures without being imported or
stubbed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

__all__ = [
    "StepStatus",
    "UserAction",
    "StepResult",
    "Step",
    "STEP_REGISTRY",
    "register",
    "validate_phases",
    "step_phases",
    "phase_count",
    "phase_label",
]


class StepStatus(Enum):
    """Status recorded by the state machine for a step's effective result.

    Doc 00 §12.2.
    """

    PENDING = "PENDING"
    COMPLETE = "COMPLETE"
    REBOOT_REQUIRED = "REBOOT_REQUIRED"
    USER_ACTION_REQUIRED = "USER_ACTION_REQUIRED"
    FAILED = "FAILED"


@dataclass
class UserAction:
    """One instruction line in a USER-ACTION block.

    Doc 00 §9.3. `command` (if set) is rendered verbatim and is meant to be
    copy-pasteable; `path` (if set) names the file the operator should edit.
    """

    text: str
    command: str | None = None
    path: str | None = None


@dataclass
class StepResult:
    """Result returned by each of a step's four lifecycle methods.

    Doc 00 §12.2. The dispatch loop (§3.2) takes the first non-`COMPLETE`
    result across `preflight -> run -> verify` as the step's effective
    result, else `COMPLETE`. Steps never write `state.json` directly — they
    only return a `StepResult`; the framework owns all persistence.
    """

    status: StepStatus
    message: str = ""
    user_actions: list[UserAction] = field(default_factory=list)


class Step(Protocol):
    """The contract every `stepN_*.py` module implements (doc 00 §12.1).

    `ctx` is typed as the bare name `Context` (doc 00 §12.3), which is
    intentionally never imported here — see the module docstring.
    """

    id: str  # e.g. "step2_deepstream_sdk"  (matches state.json key)
    title: str  # human title for logs/USER-ACTION blocks
    order: int  # 1..5

    # Optional, doc 08 §3.1: a tuple of short phase labels the progress
    # renderer pairs with `ctx.progress.phase(n)`. Deliberately absent from
    # this Protocol body so a step that has not adopted phases yet is still
    # a structurally valid `Step` — that is what lets the seven steps
    # migrate one at a time. Read it through `step_phases()` below rather
    # than with a bare `getattr`, so the omitted case stays uniform.

    def preflight(self, ctx: Context) -> StepResult: ...

    def run(self, ctx: Context) -> StepResult: ...

    def verify(self, ctx: Context) -> StepResult: ...

    def report(self, ctx: Context) -> None: ...


# Ordered (by `.order`) registry of step implementations. Populated at
# import time by each `stepN_*.py` module via `register()`; `__main__.py`
# imports all seven for that side effect, so a process that imports only
# this package sees it empty. Iterating `STEP_REGISTRY` always yields steps
# in ascending `.order`.
STEP_REGISTRY: list[Step] = []


def register(step: Step) -> None:
    """Register a `Step` implementation into `STEP_REGISTRY`.

    Call this once at module scope in each `stepN_*.py` module, e.g.:

        from mv3dt_installer.steps import register

        class Step2DeepStreamSdk:
            id = "step2_deepstream_sdk"
            title = "DeepStream SDK"
            order = 2
            ...

        register(Step2DeepStreamSdk())

    `STEP_REGISTRY` is kept sorted by `.order` after every call, so callers
    never need to sort it themselves.

    A `phases` declaration (doc 08 §3.1), if present, is validated here: a
    malformed one is an authoring mistake, and surfacing it at import time
    beats discovering it mid-install when the renderer asks for a label
    that isn't there.
    """

    validate_phases(step)
    STEP_REGISTRY.append(step)
    STEP_REGISTRY.sort(key=lambda s: s.order)


def _step_name(step: Step) -> str:
    return repr(getattr(step, "id", step))


def validate_phases(step: Step) -> None:
    """Reject a malformed `phases` declaration on `step` (doc 08 §3.1).

    No attribute at all is valid and means "one unnamed phase". Anything
    else must be a tuple or list of non-empty, non-blank strings. A bare
    string is rejected explicitly: it is itself a sequence of strings and
    would otherwise validate as one phase per character.

    Raises `TypeError` for a wrong shape, `ValueError` for a right-shaped
    but empty declaration.
    """

    phases = getattr(step, "phases", None)
    if phases is None:
        return

    if isinstance(phases, (str, bytes)) or not isinstance(phases, (tuple, list)):
        raise TypeError(
            f"step {_step_name(step)}: phases must be a tuple of strings, "
            f"got {type(phases).__name__}"
        )
    if not phases:
        raise ValueError(
            f"step {_step_name(step)}: phases must not be empty; omit the "
            "attribute entirely for a single unnamed phase"
        )
    for index, label in enumerate(phases, start=1):
        if not isinstance(label, str):
            raise TypeError(
                f"step {_step_name(step)}: phase {index} must be a string, "
                f"got {type(label).__name__}"
            )
        if not label.strip():
            raise ValueError(f"step {_step_name(step)}: phase {index} label is empty")


def step_phases(step: Step) -> tuple[str, ...]:
    """Return `step`'s declared phase labels, or `()` if it declared none."""

    return tuple(getattr(step, "phases", ()) or ())


def phase_count(step: Step) -> int:
    """Phase denominator for `step` — the `M` in a rendered `phase n/M`.

    A step that declares no phases still has one phase; it just has no
    label for it, so the count is 1 rather than 0.
    """

    return len(step_phases(step)) or 1


def phase_label(step: Step, index: int) -> str | None:
    """Label for 1-based phase `index` of `step`, or `None` if unnamed.

    Doc 08 §9 requires an out-of-range index to raise rather than render a
    wrong denominator, so the bounds check lives here, beside the
    declaration it checks against, rather than in the renderer.
    """

    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError(
            f"step {_step_name(step)}: phase index must be an int, "
            f"got {type(index).__name__}"
        )
    count = phase_count(step)
    if index < 1 or index > count:
        raise IndexError(
            f"step {_step_name(step)}: phase index {index} out of range 1..{count}"
        )
    phases = step_phases(step)
    return phases[index - 1] if phases else None
