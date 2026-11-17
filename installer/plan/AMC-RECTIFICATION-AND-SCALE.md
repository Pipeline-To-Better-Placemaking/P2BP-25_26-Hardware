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
default**. `ui`, `api` and `skip` keep their meaning. `auto` and `linear`
always run the flow below, also on Continue; they never take the
"finished in AMC" shortcut.

### 3.1 Before starting

- **Videos first**: when step 2 is `waiting`, or `get_project_info`
  reports `video_files_count` of 0, the step is `waiting` with
  `needs_user=False`: "Rectification runs once the videos are in AMC". A
  `web_project` run's reused AMC project can still hold the previous run's
  videos while step 2 waits for new footage, so the count alone is not
  enough. AMC's own UI disables Start when there are no videos.
- **Configuration**: AMC's UI posts the configuration just before starting;
  the run already posted it in step 3
  ([section 4](#4-step-3-the-scale-locked)), so it posts nothing extra.

### 3.2 The flow

A saved job field, `rectification_started`, is set before the run's first
`POST /v1/rectification/{id}`. Until it is set, the run never trusts AMC's
state: an AMC project a `web_project` run reuses may be `COMPLETED` from
footage this run has just replaced.

1. **State**: `GET /v1/rectification/{id}`.
   - Not yet `rectification_started`: wait (polling) while another
     rectification is `RUNNING`, then go to item 2 whatever the state.
   - Started: `RUNNING` goes to item 3 or 5 (whichever phase it was in,
     saved as `rectification_phase`); `READY_FOR_REVIEW` goes to item 4;
     `COMPLETED` is done; `NOT_STARTED` or `ERROR` goes to item 2, so
     Continue after a failure starts over.
2. **Start**: `_changing_amc()`, save `rectification_started` and
   `rectification_phase: estimate`, then `POST /v1/rectification/{id}`.
3. **Estimate**: poll every `rectification.api.poll_seconds` until the state
   leaves `RUNNING`, within its own `rectification.api.timeout_seconds`. The
   step message shows the state.
4. **Commit**: read the cameras and post `/manual` with each camera's
   `committed ?? auto_estimate`, mapped as in
   [section 2](#2-amcs-rectification-api), after saving
   `rectification_phase: generate`. A camera with no estimate is committed
   with `k1 = k2 = 0`, as AMC's UI would, and the step message names it.
5. **Generate**: poll the same way, with a fresh timeout, until
   `COMPLETED`, which is done; the message names each camera's model.

`ERROR`, or a timeout, fails the step with the last 10 lines of
`GET /v1/rectification/{id}/log`. Cancel is honoured between polls, as in
step 6.

`linear` uses the same field and polling: `POST /v1/linear_media/{id}`
instead of items 2 to 4, then polls until `COMPLETED`.

### 3.3 Cancel

The cancel undo
([`AMC-PROJECT-LIFECYCLE` section 5.2](AMC-PROJECT-LIFECYCLE.md#52-undo))
has no call to stop a rectification. Before deleting the AMC project it
polls `GET /v1/rectification/{id}` while the state is `RUNNING`, up to
`rectification.api.timeout_seconds`, then deletes it either way.

`AmcApi` gains `rectification_state`, `start_rectification`,
`rectification_cameras`, `commit_rectification`, `stage_linear_media` and
`rectification_log`.

---

## 4. Step 3: the scale (LOCKED)

AMC 3.3.0 keeps **one** current configuration, not one per project:
`GET /v1/config` returns `{config_params}` with no project id, and no file
under AMC's `projects/` holds `layout_px_per_m`. AMC's UI saves the scale by
reading `GET /v1/config`, changing `layout_px_per_m`, and posting the whole
object to `POST /v1/config/{id}`. `ConfigParams` allows no extra keys, so a
post is always a whole object. The factory value from
`GET /v1/config/defaults` is `24.22`.

### 4.1 Where the scale comes from

`parameters.scale.source` picks the first source:

| `source` | Scale |
|---|---|
| `database` | the web app's floor plan scale (`px_per_m`); otherwise, in `web_project` mode, the remembered scale ([section 4.3](#43-remembered-scale)) |
| `fixed` | `fixed_px_per_m` |
| `none` | none: the run neither posts nor asks for a scale, and step 6 does not re-post one (today's behaviour for operators who manage it themselves) |

### 4.2 Posting and waiting

Every config post is built in this order, each layer replacing the keys it
names: `GET /v1/config`'s `config_params`, then `settings_file`, then
`settings_overrides`, then `settings_key` (default `layout_px_per_m`) set to
the scale when one is known. A config is posted even without a
`settings_file`. The two existing "Enter X px/m on AMC's Parameters step"
waits are removed: a known scale is always posted.

1. **Known scale**: post it and save it on the job as `layout_px_per_m`.
2. **No scale**: post the config with `settings_key` set to the factory
   default, so a value left by another project cannot pass item 3, then wait:
   "Set the floor plan scale in AMC's Parameters step: measure two points on
   the floor plan, or enter its pixels per metre, then Save and press
   Continue".
3. **On Continue**: read `GET /v1/config`. A value that is positive and not
   the factory default is saved on the job and remembered
   ([section 4.3](#43-remembered-scale)); otherwise the step stays `waiting`
   with the value AMC reported.
4. **Before calibrating**: when the job has a scale, step 6 reads
   `GET /v1/config` first. A positive, non-default value different from the
   job's is a correction made during the pause: it replaces the job's value
   and is remembered. The run then posts the job's value, and only then calls
   `verify_project` and `calibrate`.

> **Accepted limit (flagged):** a floor plan whose true scale is exactly
> `24.22` px/m cannot be confirmed from AMC; enter it as `24.2201`, or give
> the floor plan that scale in the web app.

### 4.3 Remembered scale

The folder binding (`project.json`,
[`AMC-WEB-PROJECTS` section 4](AMC-WEB-PROJECTS.md#4-workstation-folder-layout))
gains two optional keys, `LayoutPxPerM` and `LayoutSha256`. They are written
together when a scale is confirmed (items 3 and 4): the scale and the
SHA-256 of the `floorplan/layout.png` this run uploaded. A remembered scale
is used only when the run's layout has the same hash, so a new floor plan
asks again.

- **Hash**: step 2 saves the SHA-256 of the layout it uploads on the job as
  `layout_sha256`; steps 3 and 6 use that saved value.
- **No uploaded layout** (the user uploads it in AMC, or `layout.enabled` is
  off): the job has no hash, so no remembered scale is used and none is
  written; the scale is still confirmed for this run.

- **Schema**: `schema_version` stays 1. Both keys are read as optional
  (`null` when absent), so older files still load.
- **Wire**: they appear in the binding `GET /projects/current` returns; the
  web app ignores keys it does not know, so it needs no change.
- **Cancel**: the undo keeps the remembered scale, like the floor plan
  itself: it describes the floor plan, not the run.

---

## 5. Checked resume (REQUIRED)

A step left to the user is done after Continue only when AMC confirms it.
Otherwise it stays `waiting`, with a message saying what AMC reports:

| Step | Confirmed when |
|---|---|
| 2 Video Configuration | `get_project_info`: `video_files_count` at least 1 and `layout_file` true |
| 3 Parameters | [section 4.2](#42-posting-and-waiting) item 3 |
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
- Manual rectification through the API.

Open decisions:

1. **Setting the scale from the web app — RESOLVED** by the operator
   (2026-11-17): the scale is set in AMC's Parameters step; the web app's
   floor plan scale is still used when present.

---

## 8. Decomposition

| Unit | Branch | Files touched | Depends on | Wave |
|---|---|---|---|---|
| R0 This spec | `feat/installer-amc-rectify-scale-plan` | this doc, `STEP-9-AMC-CONTROL.md` sections 6 and 10 | none | 1 |
| R1 Automatic rectification | `feat/installer-amc-auto-rectification` | `mv3dt_amc_control/automation.py` (including its module docstring and the cancel undo's wait, section 3.3), `automation_config.py`, `amc_api.py`, `amc_automation.example.yaml`, `tests/amc_control/test_amc_control_automation.py`, `tests/amc_control/amc_control_fake_api.py` | R0 | 2 |
| R2 Scale step and checked resume | `feat/installer-amc-scale-and-resume-checks` | `mv3dt_amc_control/automation.py` (including its module docstring), `automation_config.py`, `projects.py`, `amc_api.py`, `amc_automation.example.yaml`, `tests/amc_control/test_amc_control_automation.py`, `tests/amc_control/test_amc_control_projects.py`, `tests/amc_control/amc_control_fake_api.py` | R1 | 3 |

Paths: `mv3dt_amc_control/` is under
`installer/mv3dt_installer/assets/amc_control/`; `tests/` under `installer/`;
the docs under `installer/plan/`.

Serialization points: R1 and R2 both change `automation.py`,
`automation_config.py`, `amc_api.py`, `amc_automation.example.yaml`, the
automation tests and the fake AMC, so R2 is stacked on R1 rather than
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
