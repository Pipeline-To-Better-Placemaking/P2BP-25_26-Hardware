# AMC rectification and scale — automatic rectification, a checked scale step (owner: DevD)

Status: design spec, **not implemented**. It changes steps 3, 4 and the
resume behaviour of the automated run in
[`STEP-9` section 6](STEP-9-AMC-CONTROL.md#6-automation-settings) and
[`AMC-WEB-PROJECTS` section 6](AMC-WEB-PROJECTS.md#6-locking-calibration-to-the-project);
it does **not** restate them — it links back to them.

Three problems showed up in an automated run on the workstation
(2026-11-17, project `automation-test`):

- **Rectification is manual.** `rectification.mode` defaults to `ui`, so
  step 4 always waits for the user, and the `api` mode was never configured
  because NVIDIA documents rectification only through the UI.
- **A step can read "done" when it is not.** After Continue, a step left to
  the user is marked done with "finished in AMC" without asking AMC. Step 4
  read done while AMC's rectification had not completed.
- **The floor plan scale is skipped.** With no scale from the web app's floor
  plan, step 3 notes "no floor plan scale available" and goes on, so
  calibration runs with AMC's default `layout_px_per_m`.

This spec makes **rectification automatic** through AMC's own REST API,
**checks every step left to the user against AMC** before calling it done,
and makes the scale a **manual step the run waits for** when it has no
value. Where the scale is set was the operator's decision (2026-11-17): in
AMC's Parameters step, whose two-point tool measures pixels per metre on the
floor plan.

All AMC calls below were read from AMC 3.3.0's `GET /v1/openapi.yaml` on the
workstation and from the requests AMC's own UI bundle makes.

---

## 1. Scope and identity

| Repository | Area | Owner of |
|---|---|---|
| `P2BP-25_26-Hardware` | `installer/mv3dt_installer/assets/amc_control/mv3dt_amc_control/` | steps 3 and 4, resume checks, the remembered scale |

No web app change: the Calibration page already shows each step's state and
message.

Out of scope is [section 7](#7-out-of-scope--open-decisions).

---

## 2. AMC's rectification API

| Call | Purpose |
|---|---|
| `POST /v1/rectification/{id}` | start auto-estimation of each camera's distortion |
| `GET /v1/rectification/{id}` | `{rectification_state}`: `NOT_STARTED`, `RUNNING`, `READY_FOR_REVIEW`, `COMPLETED`, `ERROR` |
| `GET /v1/rectification/{id}/cameras` | per camera: `camera_id`, `committed`, `auto_estimate`, each `{model, k1, k2}` or `null` |
| `POST /v1/rectification/{id}/manual` | `{"cameras": {"<camera_id>": {"model", "k1", "k2"}}}`: generate the rectified videos |
| `POST /v1/linear_media/{id}` | stage videos that are already linear, unchanged |
| `GET /v1/rectification/{id}/log` | `rectification.log`, for failure messages |

AMC's UI does auto rectification as: start, poll every 2 s until not
`RUNNING`, and at `READY_FOR_REVIEW` its "Generate Rectified Videos" button
posts `/manual` with each camera's `committed ?? auto_estimate`. A model
outside `simple_radial`, `radial`, `simple_divisional`, or a missing
estimate, becomes `simple_divisional`; a missing `k2` becomes `0`. The run
then reaches `COMPLETED`.

---

## 3. Step 4: automatic rectification (LOCKED)

`rectification.mode` gains `auto` and `linear`, and **`auto` becomes the
default**. `ui`, `api` and `skip` keep their meaning.

`auto`, after `amc_changed` is saved
([`AMC-PROJECT-LIFECYCLE` section 5](AMC-PROJECT-LIFECYCLE.md#5-cancel-resets-the-run-locked)):

1. **State**: `GET /v1/rectification/{id}`. `COMPLETED` is done at once.
   `READY_FOR_REVIEW` goes to step 4. `RUNNING` goes to step 3.
2. **Start**: `POST /v1/rectification/{id}`.
3. **Estimate**: poll every `rectification.api.poll_seconds` until the state
   leaves `RUNNING`, within `rectification.api.timeout_seconds`. The step's
   message shows the state.
4. **Commit**: at `READY_FOR_REVIEW`, read the cameras and post `/manual`
   with each camera's `committed ?? auto_estimate`, mapped as in
   [section 2](#2-amcs-rectification-api).
5. **Generate**: poll the same way until `COMPLETED`, which is done; the
   message names each camera's model.

`ERROR` at any point, or a timeout, fails the step with the last 10 lines of
`GET /v1/rectification/{id}/log`. Continue runs the step again from item 1.

`linear` posts `/v1/linear_media/{id}` and polls until `COMPLETED`, for
footage that is already rectified.

`AmcApi` gains `rectification_state`, `start_rectification`,
`rectification_cameras`, `commit_rectification`, `stage_linear_media` and
`rectification_log`.

---

## 4. Step 3: the scale (LOCKED)

AMC 3.3.0 keeps **one** current configuration, not one per project:
`GET /v1/config` returns `{config_params}` with no project id, and no file
under AMC's `projects/` holds `layout_px_per_m`. AMC's UI saves the scale by
reading `GET /v1/config`, changing `layout_px_per_m`, and posting the whole
object to `POST /v1/config/{id}`. The factory value from
`GET /v1/config/defaults` is `24.22`.

The run therefore keeps the scale itself:

1. **Known scale**: the first of: the web app's floor plan
   (`parameters.scale.source: database`), `fixed_px_per_m` (`fixed`), or
   the scale remembered for this project ([section 4.1](#41-remembered-scale)).
   The run posts it the way AMC's UI does (current config with
   `layout_px_per_m` replaced), with or without a `settings_file`, and
   saves it on the job as `layout_px_per_m`.
2. **No scale**: the step is `waiting`: "Set the floor plan scale in AMC's
   Parameters step: measure two points on the floor plan, or enter its
   pixels per metre, then Save and press Continue".
3. **On Continue**: read `GET /v1/config`. A value that is positive and not
   the factory default is saved on the job and remembered for the project;
   otherwise the step stays `waiting` with the value AMC reported.
4. **Before calibrating**: step 6 posts the job's `layout_px_per_m` to the
   project again, because AMC's single configuration may have changed or
   been reset since step 3.

> **Accepted limit (flagged):** a floor plan whose true scale is exactly
> `24.22` px/m cannot be confirmed from AMC; enter it as `24.2201`, or give
> the floor plan that scale in the web app.

`parameters.scale.source: none` keeps today's behaviour, for operators who
manage the scale outside the run.

### 4.1 Remembered scale

The folder binding (`project.json`,
[`AMC-WEB-PROJECTS` section 4](AMC-WEB-PROJECTS.md#4-workstation-folder-layout))
gains `layout_px_per_m`, `null` by default and on older files. It is
written when a value is confirmed in item 3, so later runs of the same
project do not pause for it. The web app's floor plan scale, when present,
still wins.

---

## 5. Checked resume (REQUIRED)

A step left to the user is done after Continue only when AMC confirms it.
Otherwise it stays `waiting`, with a message saying what AMC reports:

| Step | Confirmed when |
|---|---|
| 2 Video Configuration | `get_project_info`: `video_files_count` at least 1 and `layout_file` true |
| 3 Parameters | [section 4](#4-step-3-the-scale-locked) item 3 |
| 4 Rectification (`ui`) | `rectification_state` is `COMPLETED` |
| 5 Manual Alignment | `get_project_info`: `alignment_file` true |

`skip` modes and steps turned off in the config are not checked. A step that
stays `waiting` after Continue does not stop the other steps from being
checked; the run pauses again with every unfinished step listed.

---

## 6. Not verified on the workstation

- Auto rectification end to end on real footage: how long the estimate takes
  and whether `committed ?? auto_estimate` is always present at
  `READY_FOR_REVIEW`.
- That `POST /v1/config/{id}` before `calibrate` is what calibration uses.

---

## 7. Out of scope / open decisions

- Reviewing the rectification previews; the auto estimate is committed as
  AMC's UI would commit it unchanged.
- Setting the scale from the web app (a decision for later; the run already
  takes the floor plan's scale when the web app has one).
- Manual rectification through the API.

---

## 8. Decomposition

| Unit | Branch | Files touched | Depends on | Wave |
|---|---|---|---|---|
| R0 This spec | `feat/installer-amc-rectify-scale-plan` | this doc, `STEP-9-AMC-CONTROL.md` sections 6 and 10 | none | 1 |
| R1 Automatic rectification | `feat/installer-amc-auto-rectification` | `mv3dt_amc_control/automation.py`, `automation_config.py`, `amc_api.py`, `amc_automation.example.yaml`, `tests/amc_control/test_amc_control_automation.py`, `tests/amc_control/amc_control_fake_api.py` | R0 | 2 |
| R2 Scale step and checked resume | `feat/installer-amc-scale-and-resume-checks` | `mv3dt_amc_control/automation.py`, `projects.py`, `amc_api.py`, `tests/amc_control/test_amc_control_automation.py`, `tests/amc_control/test_amc_control_projects.py`, `tests/amc_control/amc_control_fake_api.py` | R1 | 3 |

Paths: `mv3dt_amc_control/` is under
`installer/mv3dt_installer/assets/amc_control/`; `tests/` under `installer/`;
the docs under `installer/plan/`.

Serialization points: R1 and R2 both change `automation.py`, `amc_api.py`,
the automation tests and the fake AMC, so R2 is stacked on R1 rather than
built beside it. R2's resume check for step 4 reads the state call R1 adds.

---

## References

AMC 3.3.0's REST API is from its own OpenAPI document,
`GET /v1/openapi.yaml` on the workstation, and the request sequence from the
UI bundle the AMC UI container serves
(`vss-auto-calibration-ui:3.3.0`, `/assets/index-*.js`).

- [NVIDIA DeepStream](https://github.com/NVIDIA/DeepStream) — upstream of
  AMC (`tools/auto-magic-calib`), whose README documents rectification only
  through the UI.

Repo files referenced:

- [`STEP-9-AMC-CONTROL.md`](STEP-9-AMC-CONTROL.md) — the automated run's
  steps and settings.
- [`AMC-WEB-PROJECTS.md`](AMC-WEB-PROJECTS.md) — `web_project` runs and the
  folder binding.
- [`AMC-PROJECT-LIFECYCLE.md`](AMC-PROJECT-LIFECYCLE.md) — `amc_changed` and
  the cancel undo the new AMC calls join.
- `installer/mv3dt_installer/assets/amc_control/mv3dt_amc_control/automation.py`
  — `_parameters`, `_rectification`, `_finished_in_amc`, `_execute`.
- `.../mv3dt_amc_control/automation_config.py` — `RectificationSection`,
  `ScaleSection`.
- `.../mv3dt_amc_control/projects.py` — `Binding`.
- `.../mv3dt_amc_control/amc_api.py` — the AMC client the new calls join.
