# AMC web projects — one AMC project and folder per web app project (owner: DevD)

Status: design spec, **not implemented**. It extends
[`STEP-9`](STEP-9-AMC-CONTROL.md) (the AMC control service) and the web app's
workstation routes; it does **not** restate STEP-9's install, sudo rule or
ticket basics — it links back to them. Wire shapes in
[section 8](#8-workstation-api-contract-required) and
[section 11](#11-web-app-changes) are the contract both repositories build to.

Today the web app can start and stop AMC and the service can automate a run,
but only for the single web project named by `TRACKING_PROJECT_ID`, and each
run either recalibrates the installer's own AMC project or makes a throwaway
one ([`STEP-9` section 6](STEP-9-AMC-CONTROL.md#6-automation-settings)). This
spec makes **project creation in the web app** set up a matching AMC project
on the workstation, ready for calibration, inside **a folder of its own named
by the user**, and **locks every later calibration of that web project to
that AMC project and folder**.

---

## 1. Scope and identity

| Repository | Area | Owner of |
|---|---|---|
| `P2BP-25_26-Hardware` | `installer/mv3dt_installer/assets/amc_control/mv3dt_amc_control/` | the project store, provisioning, locking and the workstation routes |
| `P2BP-25_26-Hardware` | `installer/mv3dt_installer/steps/step9_amc_control.py` | the projects root, the home-directory link, the seeded mode |
| `P2BP-25_26-Web` | `BetterPlacemaking.SERVER.EXPRESS/src/` | the `workstation_folder` column, the provision route, the spec's `folder` claim |
| `P2BP-25_26-Web` | `BetterPlacemaking.CLIENT.REACT/src/pages/projects/` | the folder field at creation, the Calibration page |

In scope: everything in [section 2](#2-end-to-end-flow). Out of scope, with
reasons, is [section 12](#12-out-of-scope--open-decisions).

---

## 2. End-to-end flow

1. **Create**: the user creates a project in the web app and fills in
   **Workstation folder** (prefilled from the title, [section 3](#3-folder-names-locked)).
   The web app stores it in `projects.workstation_folder`.
2. **Provision**: the browser asks the web app for a provision ticket and spec
   ([section 11.2](#112-server-routes)) and sends both to the workstation's
   `POST /projects` ([section 8](#8-workstation-api-contract-required)).
3. **Ready**: the service creates `<projects root>/<folder>/`
   ([section 4](#4-workstation-folder-layout)), starts AMC if needed, creates
   an AMC project named after the folder, saves the floor plan if the project
   has one, and marks the binding `ready` ([section 5](#5-provisioning)).
4. **Calibrate**: from the Calibration page the user starts automated runs.
   Every run for that web project uses its bound AMC project and folder and
   nothing else ([section 6](#6-locking-calibration-to-the-project)).
5. **Retrieve**: footage, floor plan, results and run records stay in that
   folder, browsable from the web app and from the workstation
   ([section 7](#7-retrieval)).

---

## 3. Folder names (LOCKED)

The folder name is user input, used verbatim as the directory name and the
AMC project name, so both sides validate it with one rule:

```
^[a-z0-9](?:[a-z0-9-]{0,46}[a-z0-9])?$
```

1 to 48 characters, lowercase letters, digits and inner hyphens. The web form
prefills it by lowercasing the title, replacing each run of other characters
with one hyphen, trimming hyphens and cutting to 48. **REQUIRED:** a folder
name is unique per workstation and per web database, and **immutable once
set** — the web app refuses to change a non-empty value, and the workstation
refuses to bind a folder already bound to another web project.

---

## 4. Workstation folder layout

The projects root is `AMC_PROJECTS_DIR`, default `<install_dir>/amc_projects`,
owned by the service user, mode 750. It is deliberately not Step 5's
`<install_dir>/projects/`, which is root-owned and keyed by installer project
names ([`STEP-5` section 4.1](STEP-5-PER-PROJECT-EXES.md#41-path)).

```
<install_dir>/amc_projects/<folder>/
  project.json            the binding (below)
  footage/                cam_00.mp4, cam_01.mp4, ... for the next run
  floorplan/layout.png    the floor plan as last uploaded to AMC
  alignment/alignment_data.json   a saved manual alignment, re-used by runs
  calibration/<UTC>/      one directory per completed run: result ZIP and statistics
  runs/<job id>.json      the final record of every automated run
```

`project.json` (written atomically: temp file, `fsync`, `os.replace`):

```json
{
  "schema_version": 1,
  "ProjectId": "web project id",
  "Folder": "kevin-living-room",
  "Title": "Kevin Living Room",
  "AmcProjectId": "20261117_101500_1234",
  "AmcProjectName": "kevin-living-room",
  "State": "ready",
  "Error": "",
  "CreatedAt": 1794879242,
  "UpdatedAt": 1794879242,
  "LastRunAt": null,
  "CalibRuns": 0
}
```

`State` is one of `provisioning`, `ready`, `failed`. Times are Unix seconds,
matching `AmcStatus.UpdatedAt`. A missing or malformed `project.json` makes
that directory invisible to lookups, never an exception.

---

## 5. Provisioning

`POST /projects` with a provision ticket and spec
([section 9](#9-tickets-and-specs)):

1. **Validate**: the spec's `folder` matches [section 3](#3-folder-names-locked)
   and its `pid` equals the ticket's.
2. **Idempotency**: if this `pid` is already bound to this folder and the
   state is `ready` or `provisioning`, return `200` with the binding. If it is
   `failed`, provision again. If the `pid` is bound to another folder, or the
   folder to another `pid`, return `409`.
3. **Busy**: refuse with `409` while an automated run is active or waiting,
   since provisioning may start AMC.
4. **Create** the directory tree from [section 4](#4-workstation-folder-layout),
   write `project.json` with `State: provisioning`, and return `202`.
5. **In the background**: start AMC through the existing controller if it is
   not up, waiting up to `amc.ready_timeout_seconds`; `create_project(folder)`;
   if the spec carries a floor plan, download it under `layout.allowed_hosts`
   into `floorplan/layout.png`. Then set `AmcProjectId`, `AmcProjectName` and
   `State: ready`. Any failure sets `State: failed` and `Error`, and keeps the
   directory.

"Ready for calibration" means the AMC project exists and the folder is laid
out; footage and the AMC workflow come later ([section 6](#6-locking-calibration-to-the-project)).

**REQUIRED — one AMC user at a time.** While a provisioning job runs, a
second `POST /projects`, `POST /amc/stop` and `POST /amc/automation/start`
for any project are refused with `409`; while an automated run is active or
waiting, `POST /projects` is refused (step 3). H3 owns this: the provisioner
exposes `busy()` and the API routes check it before calling the runner or
the controller.

**REQUIRED — restart recovery.** When the service starts, every binding still
in `provisioning` becomes `failed` with `Error` "interrupted by a service
restart; set it up again", so step 2's retry path applies.

---

## 6. Locking calibration to the project

**LOCKED.** A new `project.mode` value, `web_project`, binds each automated
run to the calling web project. The seeded `amc_automation.yaml` uses it on
new installs ([section 10](#10-installer-step-9-changes)); the existing modes
keep their meaning for workstations that still set them.

| AMC step | `web_project` behaviour |
|---|---|
| 1 Project Setup | look up the binding for the spec's `pid`; none or not `ready` is `409` "set this project up on the workstation first". `get_project_info(AmcProjectId)` must succeed, or the run fails naming the folder: a deleted AMC project is **never** silently recreated |
| 2 Video Configuration | videos from `<folder>/footage/` (with `videos.pattern` and `videos.order`); none is a `waiting` step telling the user to copy footage there or upload it in AMC's UI. The floor plan is copied to `floorplan/layout.png` before upload |
| 5 Manual Alignment | `alignment.file: auto` uses `<folder>/alignment/alignment_data.json` when it exists, otherwise AMC's own saved copy as today |
| 7 Results | the result ZIP and statistics always go to `<folder>/calibration/<UTC>/` |
| end of run | `runs/<job id>.json` written; on success `LastRunAt` and `CalibRuns` updated and AMC's saved alignment file, if any, copied to `<folder>/alignment/alignment_data.json` |

> **Amended:** cancelling a run now replaces the AMC project with an empty
> one on purpose and records the new id
> ([`AMC-PROJECT-LIFECYCLE` section 5](AMC-PROJECT-LIFECYCLE.md#5-cancel-resets-the-run-locked)).

A spec whose `folder` differs from the binding's is `400`. AMC stays one shared
instance: the existing one-run-at-a-time rule is unchanged. The runner builds
its own `ProjectStore` from `projects.default_projects_root()` (which reads
`AMC_PROJECTS_DIR`) unless one is passed to its constructor, so H4 needs no
change to `api.py` or `settings.py`.

---

## 7. Retrieval

- **Web app**: `GET /projects/current/files` lists the folder and
  `GET /projects/current/files/{path}` downloads one file
  ([section 8](#8-workstation-api-contract-required)); the Calibration page
  shows both.
- **Workstation**: Step 9 links `~<service user>/mv3dt-projects` to the
  projects root, and
  `cd <install_dir>/amc_control && venv/bin/python3 -m mv3dt_amc_control projects`
  prints one row per folder: folder, title, web project id, AMC project id,
  state, last run, calibration runs, path.

**REQUIRED — path safety.** A requested path is resolved against the folder
with symlinks followed and must stay inside it; `..`, absolute paths and
`project.json` itself are refused with `404`. Listings skip symlinks.

---

## 8. Workstation API contract (REQUIRED)

All new routes need a bearer ticket ([section 9](#9-tickets-and-specs)); the
project is always the ticket's `pid`, never a request field.

| Route | Ticket scope | Body | Response |
|---|---|---|---|
| `POST /projects` | `amc-provision` | `{"Spec": "<jwt>"}` | `202` or `200` binding; `400`, `409` |
| `GET /projects/current` | `amc` or `amc-provision` | none | binding, or `404` when the project has no folder |
| `GET /projects/current/files` | `amc` | none | `[{"Path": "footage/cam_00.mp4", "Size": 123, "ModifiedAt": 1794879242}]` |
| `GET /projects/current/files/{path}` | `amc` | none | the file, `application/octet-stream` with `Content-Disposition: attachment` |

The binding on the wire is `project.json` without `schema_version`, plus
`"Path"`, the folder's absolute path on the workstation. A `failed` binding is
returned as such (`State`, `Error`); runs for it are `409`. Errors keep
FastAPI's `{"detail": "..."}`.

- **Listing**: recursive, regular files only, sorted by `Path`; directories,
  symlinks and the top-level `project.json` are left out. Every route other
  than `GET /projects/current` answers `404` for a project with no folder.
- **`{path}`**: a listing's `Path` with each segment percent-encoded and `/`
  kept literal. A missing, unsafe or excluded path is `404`.
- **Large files**: the download route also accepts the ticket as
  `?ticket=<amc ticket>`, so the browser can follow a plain link and stream
  multi-GB footage to disk instead of buffering it. **LOCKED trade-off:** the
  ticket then appears in the service's access log; it expires within
  `WORKSTATION_AMC_TICKET_MINUTES` and the journal is readable only by
  administrators. CORS and Private Network Access handling are STEP-9's
([`STEP-9` section 7](STEP-9-AMC-CONTROL.md#7-api-and-ticket-contract)).

> **Amended:** video files are listed with `Downloadable: false` and their
> download is `403`, and `DELETE /projects/current` (`amc-delete` scope)
> deletes the project
> ([`AMC-PROJECT-LIFECYCLE` sections 3 and 4](AMC-PROJECT-LIFECYCLE.md#4-no-video-downloads-locked)).

---

## 9. Tickets and specs

- **Any web project**: the AMC service no longer pins `TRACKING_PROJECT_ID`.
  Tickets are still HS256 with `aud: "p2bp-workstation"` and the shared
  secret ([`STEP-8` section 3.2](STEP-8-TRACKING-HISTORY.md#32-tickets-required)),
  so only the web app can mint them, and it checks the user's project
  permission first. `serve` stops requiring the project id; tracking is
  unchanged.
- **Scopes**: `amc` (STEP-9's routes and the read routes above) and the new
  `amc-provision` (`POST /projects`). Each route accepts exactly the scopes
  listed in [section 8](#8-workstation-api-contract-required): one, except
  `GET /projects/current`, which takes either so the creating page can watch
  provisioning finish. A wrong scope is `403`.
- **Accepted limitation (flagged)**: without the pin, any user with
  `Project.Vision.Read` on any project can start and stop the one shared AMC,
  and once it is up, AMC's own unauthenticated UI shows every AMC project
  ([`STEP-9` section 10](STEP-9-AMC-CONTROL.md#10-not-verified-on-the-workstation)).
  The mitigations here are the busy rules in [section 5](#5-provisioning) and
  STEP-9's refusal of `/amc/stop` during a run; per-project isolation inside
  AMC is not possible.
- **Spec**: the `amc-calibration-spec` claim `spec` gains `folder`
  (string or `null`). Provisioning requires it; runs check it against the
  binding when present.

---

## 10. Installer Step 9 changes

| Change | Detail |
|---|---|
| Projects root | create `<install_dir>/amc_projects`, owned by the service user, mode 750; never delete it on upgrade |
| Link | create `~<service user>/mv3dt-projects` pointing at it only if nothing exists at that path |
| Unit | `service_environment()` adds `AMC_PROJECTS_DIR=<install_dir>/amc_projects` to the rendered unit; `pending_update()` already re-runs on a unit change |
| Seeded settings | new `amc_automation.yaml` files set `project.mode: web_project`; an existing file is never rewritten, so a workstation seeded with `create_new` is switched by hand |
| `report` | prints the projects root and the link |

---

## 11. Web app changes

### 11.1 Data

- `projects.workstation_folder` (`text`, nullable, unique), generated with
  `npm run db:generate` into `supabase/migrations/`. API field
  `WorkstationFolder` on the project DTO.
- `POST /api/projects` accepts it (validated with
  [section 3](#3-folder-names-locked)'s rule, `409` on a duplicate);
  `PUT /api/projects/:id` may set it only while it is empty, otherwise `409`.

### 11.2 Server routes

| Route | Permission | Response |
|---|---|---|
| `POST /api/workstation/project/:projectId/provision` | `Project.Update` | `{Ticket, Spec, ExpiresAt, WorkstationUrl}`; ticket scope `amc-provision`, spec with `folder`, both 15 minutes; `400` when the project has no folder; `503` until configured |
| `POST /api/workstation/project/:projectId/calibration-spec` | `Project.Vision.Read` (unchanged) | the spec now includes `folder` |

### 11.3 Client

- `ProjectForm.tsx`: a **Workstation folder** field, prefilled from the title
  while untouched, validated live, read-only once the project has one.
- `ProjectsList.tsx`: after a create that set a folder, call provisioning and
  report the result; a failure leaves the project created.
- `calibration/amcApi.ts`: `provisionProject`, `getWorkstationProject`,
  `listProjectFiles`, `projectFileUrl` (a download link carrying
  `?ticket=`, [section 8](#8-workstation-api-contract-required)) and the
  automation calls `startAutomation`,
  `getAutomation`, `continueAutomation`, `cancelAutomation`.
- `calibration/Calibration.tsx`: replaces the local-only session with the
  workstation panel: binding state and a **Set up on workstation** retry;
  AMC status, start, stop and **Open AMC**; automated run start, step list,
  continue and cancel; the folder's file list with downloads.

---

## 12. Out of scope / open decisions

Settled exclusions:

- Deleting a project in either place. **Reversed** by
  [`AMC-PROJECT-LIFECYCLE` section 3](AMC-PROJECT-LIFECYCLE.md#3-deleting-a-project-required):
  deleting a web project now deletes its folder and AMC project.
- Applying the migration to the production database
  (`npm run db:migrate`) is an operator action, done once the web PRs merge.

Open decisions:

1. **Recording footage from the web app — RESOLVED** by
   [`AMC-FOOTAGE-RECORDING.md`](AMC-FOOTAGE-RECORDING.md): an automated run
   can record into `<folder>/footage/` through one fixed sudo command.
2. **Feeding a web project's calibration into DeepStream (flagged).** Step 4
   ingests only `installer.conf`'s `AMC_PROJECT_ID`
   ([`STEP-4` section 7](STEP-4-CALIB-OUTPUT-WIRING.md#7-later-recalibration));
   results stay in the folder until that is designed.
3. **Per-project access to the shared AMC (flagged).** See the accepted
   limitation in [section 9](#9-tickets-and-specs).

---

## 13. Decomposition

| Unit | Branch | Files touched | Depends on | Wave |
|---|---|---|---|---|
| P0 This spec | `feat/installer-amc-web-projects-plan` | this doc, a pointer in `STEP-9-AMC-CONTROL.md` | none | 1 |
| W1 Folder column | `feat/workstation-folder-column` | `src/db/schema/core.ts`, `supabase/migrations/`, project service, `project.routes.ts`, their tests | none | 1 |
| H1 Project store | `feat/installer-amc-project-store` | `mv3dt_amc_control/projects.py`, `tests/amc_control/test_amc_control_projects.py` | P0 | 2 |
| H2 Ticket scopes and any project | `feat/installer-amc-ticket-scopes` | `mv3dt_amc_control/auth.py`, `settings.py`, `cli.py` (required settings only), `calibration_spec.py`, `api.py` (ticket dependency only), `tests/amc_control/test_amc_control_api.py` | P0 | 2 |
| W2 Provision route and spec folder | `feat/workstation-provision-route` | `workstation.routes.ts`, `calibrationSpecService.ts`, ticket service, their tests | W1 | 2 |
| H3 Provisioning and file routes | `feat/installer-amc-provisioning` | `mv3dt_amc_control/provisioning.py`, `api.py`, `models.py`, `cli.py` (the `projects` command), `tests/amc_control/test_amc_control_provisioning.py` | H1, H2 | 3 |
| H4 Locked automation | `feat/installer-amc-locked-runs` | `mv3dt_amc_control/automation.py`, `automation_config.py`, `amc_automation.example.yaml`, `tests/amc_control/test_amc_control_automation.py` | H1, H2 | 3 |
| W3 Folder at creation and client API | `feat/project-workstation-folder-ui` | `ProjectForm.tsx`, `ProjectsList.tsx`, `projectTypes.ts`, `calibration/amcApi.ts` | W2 | 3 |
| H5 Step 9 projects root | `feat/installer-step9-projects-root` | `steps/step9_amc_control.py`, `tests/test_step9_amc_control.py`, `STEP-9-AMC-CONTROL.md` | H4 | 4 |
| W4 Calibration page | `feat/calibration-workstation-panel` | `calibration/Calibration.tsx` | W3 | 4 |

Hardware paths: `mv3dt_amc_control/...` is under
`installer/mv3dt_installer/assets/amc_control/`, `steps/` and `assets/` under
`installer/mv3dt_installer/`, `tests/` under `installer/`, and the two docs
under `installer/plan/`. Web paths are relative to `BetterPlacemaking.SERVER.EXPRESS/` or
`BetterPlacemaking.CLIENT.REACT/src/pages/projects/`. W units live in
`P2BP-25_26-Web` and stack only on each other.

Serialization points: `api.py` and `cli.py` are touched by H2 (the ticket
dependency, the required settings) and H3 (the routes, the `projects`
command), so H3 waits for H2. H3 and H4 both read H1's store but
touch disjoint files. H5 seeds `project.mode: web_project`, a value only H4's
settings schema accepts, so H5 stacks on H4 and merges after it; H5 also
edits `STEP-9-AMC-CONTROL.md` after P0's pointer. W3 owns every `amcApi.ts`
change so W4 touches one file. The two repositories meet only at the wire
contract in sections 8, 9 and 11.2, so W units never wait on H units.

---

## References

AMC behaviour is NVIDIA's AutoMagicCalib REST API as wrapped by
`mv3dt_amc_control/amc_api.py`; web facts come from `P2BP-25_26-Web` `main`
at `fdbb742`.

- [RFC 7519](https://datatracker.ietf.org/doc/html/rfc7519) — the JWT claims
  (`aud`, `exp`, `pid`, `scope`) behind [section 9](#9-tickets-and-specs).
- [Drizzle Kit generate](https://orm.drizzle.team/docs/drizzle-kit-generate)
  — how W1's migration is produced.

Repo files referenced:

- [`STEP-9-AMC-CONTROL.md`](STEP-9-AMC-CONTROL.md) — the service this extends.
- [`STEP-8-TRACKING-HISTORY.md`](STEP-8-TRACKING-HISTORY.md) — the ticket
  contract.
- [`STEP-5-PER-PROJECT-EXES.md`](STEP-5-PER-PROJECT-EXES.md) — the installer's
  own project registry, kept separate.
- [`STEP-4-CALIB-OUTPUT-WIRING.md`](STEP-4-CALIB-OUTPUT-WIRING.md) — ingest,
  which stays single-project.
- `installer/mv3dt_installer/assets/amc_control/mv3dt_amc_control/` — the
  service package H1 to H4 change.
- `installer/mv3dt_installer/steps/step9_amc_control.py` — the step H5
  changes.
