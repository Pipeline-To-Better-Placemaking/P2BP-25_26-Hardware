# AMC project lifecycle — delete, cancel reset, no video downloads, UI caching (owner: DevD)

Status: design spec, **not implemented**. It amends
[`AMC-WEB-PROJECTS.md`](AMC-WEB-PROJECTS.md) (one AMC project and folder per
web project), [`AMC-FOOTAGE-RECORDING.md`](AMC-FOOTAGE-RECORDING.md) (footage
recorded from step 2 of a run) and
[`STEP-3-AMC-LAUNCHER.md`](STEP-3-AMC-LAUNCHER.md) (the AMC Compose stack);
it does **not** restate them — it links back to them.

Four gaps showed up in the first real automated run on the workstation
(2026-11-17):

- Deleting a project in the web app removes only its database row. The
  workstation folder and the AMC project stay, and the folder name can never
  be reused, because the workstation still binds it to the deleted project.
- Cancelling a run stops it but undoes nothing: footage the run recorded,
  videos and the layout uploaded to AMC, and any work inside AMC all stay.
- Any file in a project folder can be downloaded through the workstation,
  calibration footage included.
- AMC's UI container serves `env-config.js`, which holds the API address
  baked in at container start, with `Cache-Control: public, immutable` and a
  one-year expiry. A browser that loaded AMC once keeps the old address
  forever, so after AMC moved to the Tailscale address "Open AMC" waited on
  an unreachable host and never listed projects.

This spec makes **deleting a web project delete its folder and AMC project**,
makes **cancel return the AMC project to an empty `INIT` state** and remove
footage the run recorded, keeps **video files on the workstation**, and
makes the **AMC UI's runtime config uncacheable**.

---

## 1. Scope and identity

| Repository | Area | Owner of |
|---|---|---|
| `P2BP-25_26-Hardware` | `installer/mv3dt_installer/steps/step3_amc_launcher.py` | the UI cache override ([section 2](#2-amc-ui-runtime-config-caching-locked)) |
| `P2BP-25_26-Hardware` | `installer/mv3dt_installer/assets/amc_control/mv3dt_amc_control/` | project delete, the video download rule, cancel reset |
| `P2BP-25_26-Web` | `BetterPlacemaking.SERVER.EXPRESS/src/` | the delete ticket |
| `P2BP-25_26-Web` | `BetterPlacemaking.CLIENT.REACT/src/` | the delete flow, download links, the cancel confirmation |

Out of scope is [section 8](#8-out-of-scope--open-decisions).

---

## 2. AMC UI runtime config caching (LOCKED)

The pinned image `nvcr.io/nvidia/vss-core/vss-auto-calibration-ui:3.3.0`
serves `/usr/share/nginx/html` from `/etc/nginx/conf.d/default.conf`, whose
`location ~* \.(js|css|...)$` block sets `expires 1y` and
`Cache-Control: public, immutable` on every script, `env-config.js`
included. Its entrypoint rewrites `__API_URL_PLACEHOLDER__` in
`env-config.js` from `API_URL` at container start.

### 2.1 Files

Step 3 writes two more files next to `compose/.env`, in the same place and
the same atomic, content-idempotent way
([`STEP-3` section 4.2](STEP-3-AMC-LAUNCHER.md#42-pinned-compose-environment)),
and hands both to the invoking user with the same `chown` as `.env`:

1. **`compose/mv3dt-ui-nginx.conf`**: the image's `default.conf`
   unchanged (the `client_max_body_size 4g` and `client_body_timeout 1800s`
   upload limits, the SPA `try_files`, the static-asset caching, gzip) plus:
   - `location = /env-config.js` with `expires off` and
     `add_header Cache-Control "no-store" always`; an exact-match location
     wins over the regex asset block;
   - in `location /`, `add_header Cache-Control "no-cache" always`,
     `sub_filter_once on` and
     `sub_filter '/env-config.js"' '/env-config.js?v=<hash>"'`, where
     `<hash>` is the first 12 hex digits of the SHA-256 of the API URL the
     UI is given. A browser holding the old immutable copy then asks for a
     new URL, so the fix also reaches browsers that cached the wrong
     address before it shipped.
2. **`compose/compose.override.yml`**: bind-mounts that file read-only over
   `/etc/nginx/conf.d/default.conf` in the `auto-magic-calib-ui` service.
   Docker Compose reads `compose.override.yml` automatically because Step 3
   runs Compose from `compose/` without `-f`.

The API URL in the hash is the one the UI compose file resolves:
`AUTO_MAGIC_CALIB_MS_API_URL` when set, otherwise
`http://<HOST_IP>:<AUTO_MAGIC_CALIB_MS_PORT>/v1`.

### 2.2 Applying a change (REQUIRED)

`launch_amc` skips `compose up` when the stack is already running, and
`up -d` does not recreate a container whose bind-mounted file only changed
content. So when either file, or `compose/.env`, changed and the stack is
running, Step 3 runs
`docker compose up -d --force-recreate --no-deps auto-magic-calib-ui`.
`.env` is included because a changed `HOST_IP` changes the UI's `API_URL`,
the fault this section exists for. A stack that is not running comes up
with the new files as usual.

### 2.3 Checkout check (REQUIRED)

`clone_amc` refuses an AMC checkout with local changes outside
`_expected_amc_worktree_change`, which today allows only `compose/.env`.
That function allows the two new files as untracked paths, or every AMC
launch after the first would fail.

### 2.4 Verified on the workstation

Checked on 2026-11-17 against the pinned UI image and Docker Compose
`2.40.3`:

- `docker compose config` with the override merges the bind mount into the
  `auto-magic-calib-ui` service that `compose.yml` pulls in through
  `include`.
- A probe container with the section 2.1 config served `/` and
  `/index.html` with `Cache-Control: no-cache` and the rewritten
  `<script src="/env-config.js?v=...">`, gzip on; `env-config.js` with
  `Cache-Control: no-store`; and the app bundle still with
  `public, immutable`. The image has no `gzip_static`, so `sub_filter`
  sees uncompressed HTML.

> **Pinned image (REQUIRED):** the override replaces the image's whole
> `default.conf`, so it is tied to the 3.3.0 UI image. A test asserts that
> the rendered file still carries the image's upload limits and SPA
> fallback; a change to the pinned UI image must re-copy `default.conf`.

---

## 3. Deleting a project (REQUIRED)

### 3.1 Wire contract

| Piece | Contract |
|---|---|
| Ticket scope | `amc-delete`, same audience, issuer and `pid` claim as the other workstation tickets ([`AMC-WEB-PROJECTS` section 9](AMC-WEB-PROJECTS.md#9-tickets-and-specs)) |
| Web route | `POST /api/workstation/project/:projectId/delete-ticket`, permission `Project.Delete`, 5-minute ticket; `200 {Ticket, ExpiresAt, WorkstationUrl}`, `503` when workstation control is not configured, `404` for an unknown project |
| Workstation route | `DELETE /projects/current`, `amc-delete` scope only |
| CORS | the service's `allow_methods` gains `DELETE`, and `expose_headers` gains `Retry-After` so the browser can read it |

Workstation responses:

| Status | When | Body |
|---|---|---|
| `200` | deleted | `{Folder, AmcProjectId, AmcProjectDeleted, Leftover}` |
| `404` | the ticket's project has no folder | `{detail}` |
| `409` | this project has a run that is not `completed` or `cancelled`, or a provisioning job runs | `{detail}` |
| `502` | AMC refused the delete (anything but `404`) | `{detail}` |
| `503` | AMC is not ready; the service has asked it to start | `{detail}` (includes the controller's last start error, if any), `Retry-After: 15` |

### 3.2 Workstation behaviour

Inside the provisioner's `run_exclusive`, so no provisioning job or run can
start meanwhile. Nothing inside it waits for AMC to start:

1. **Find**: the binding for the ticket's `pid`; `404` without one.
2. **Refuse**: `409` "Cancel this project's run first" when the runner's
   job belongs to this `pid` and its state is not `completed` or
   `cancelled`. A paused, failed or interrupted run could otherwise be
   cancelled after its folder is gone.
3. **AMC**: when the binding has an `amc_project_id` and AMC is not ready,
   call `controller.start()` (it returns at once) and answer `503` with
   `Retry-After: 15`; nothing is removed. When AMC is ready,
   `DELETE /v1/delete_project/{id}`. AMC's `404` counts as deleted. Any
   other failure is `502` and **nothing is removed**, so the user can retry.
   AMC is left running.
4. **Folder**: rename `<root>/<folder>` to
   `<root>/.deleting-<folder>-<unix seconds>` (one `rename(2)`, so the
   folder name is free at once), then remove the renamed tree without
   following symlinks. A removal error is logged and returned as `Leftover`
   (the renamed path); it is never a failed request, because the binding is
   already gone. A dot-prefixed name fails the folder regex, so a leftover
   never reads as a project.

A `completed` or `cancelled` job of a deleted project stays as the runner's
last job; `GET /amc/automation/status` shows a job only to its own `pid`,
and neither state can be continued or cancelled.

`AmcApi` gains `delete_project(id)` and `stop_calibration(id)`.

### 3.3 Web flow

1. **Confirm**: for a project with a `WorkstationFolder`, the confirmation
   says the workstation folder (footage, floor plan, calibrations) and its
   AMC project are deleted too.
2. **Workstation first**: request the delete ticket, then
   `DELETE /projects/current`. On `503` with `Retry-After`, show "Starting
   AMC on the workstation" and retry after that many seconds, for up to
   three minutes. `200` and `404` go on to step 3.
3. **Database**: `DELETE /api/project/:id` as today.
4. **On any other outcome** (unreachable or a CORS failure from an older
   workstation, `401`, `403`, `409`, `500`, `502`, `503` past three
   minutes): show the reason and offer **Delete from the web app only**,
   which leaves the folder and AMC project on the workstation, or cancel.

A project without a `WorkstationFolder`, or with workstation control not
configured on the web server, is deleted as today.

---

## 4. No video downloads (LOCKED)

Calibration footage stays on the workstation. This narrows the file
download of
[`AMC-WEB-PROJECTS` section 8](AMC-WEB-PROJECTS.md#8-workstation-api-contract-required).

- **Video file**: any path under `footage/`, and any path whose suffix,
  case-insensitive, is one of `.mp4 .mkv .mov .avi .ts .h264 .h265`.
- **List**: `GET /projects/current/files` still lists video files (the
  Calibration page uses them to tell whether footage exists), with a new
  `Downloadable: false`; every other file has `Downloadable: true`.
- **Download**: `GET /projects/current/files/{path}` refuses a video file
  with `403` "Video files stay on the workstation". The rule is applied
  twice: to the requested path, normalized as a POSIX path, before any
  filesystem access; and to the resolved target's path relative to the
  folder, so a symlink to a clip is refused as well.
- **Client**: no download link for a file with `Downloadable: false`.

---

## 5. Cancel resets the run (LOCKED)

For a `web_project` run (the job has a `folder`), cancel undoes the run
instead of only stopping it. Other modes are unchanged.

### 5.1 What the job records

Three job fields are saved, each before the change it records:

| Field | Set |
|---|---|
| `amc_changed` | `true` before the run's first call that changes the AMC project (an upload, a config post, verify, calibrate) |
| `cancelling` | `true` when the undo starts, until the job finishes `cancelled` |
| `undone` | the undo actions finished so far, each appended and saved as it completes: `amc_deleted`, `amc_created`, `amc_bound`, `footage_cleared`, `footage_restored`, `results_removed` |
| `reset_amc_project_id` | the new AMC project's id, saved as soon as `create_project` returns and before the binding is written |
| `recorded_files` | the `files` list from the recorder's final line, with `footage_recorded` |
| `calibration_dir` | the `calibration/<UTC>` folder name, before step 7 creates it |

### 5.2 Undo

`cancel()` sets the job to `running` with the message "Cancelling: undoing
this run" and returns at once. The undo runs on the runner's own thread:
after the run thread stops for a running job, or on a new thread for a
paused, failed or interrupted one. No lock is held while it waits on AMC.

1. **AMC calibration**: when `amc_changed`, `stop_calibration` on the
   job's AMC project, ignoring errors. Step state cannot say whether step 6
   started, because cancel resets the running step to `pending`.
2. **AMC project**: when `amc_changed`, make AMC ready the way provisioning
   does (start it and wait up to `amc.ready_timeout_seconds`), delete the
   job's AMC project and create a new one named after the folder, then
   store the new id in the binding. This is the one reset that is certain
   to leave nothing uploaded: videos, layout, alignment, config, annotations
   and results all go, and the project is `INIT`. If AMC does not become
   ready, or the delete fails, the AMC project is left as it is. If the
   delete succeeds but the create fails, the binding becomes `failed` with
   `amc_project_id` cleared, so setting the project up again from the web
   app creates a fresh AMC project
   ([`AMC-WEB-PROJECTS` section 5](AMC-WEB-PROJECTS.md#5-provisioning)).
3. **Footage**: when `footage_recorded`, delete exactly `recorded_files` and
   the recorder's `.mv3dt-footage.json` from `footage/`, move everything in
   `footage/previous/` back into `footage/`, and remove `previous/` when it
   is empty. Footage that was there before the run started is kept. A
   recording cancelled while it runs is already restored by the recorder
   ([`AMC-FOOTAGE-RECORDING` section 5](AMC-FOOTAGE-RECORDING.md#5-the-record-footage-command)).
4. **Results**: when `calibration_dir` is set, remove that folder.
5. **Finish**: `cancelled`, with a message saying what was undone, or what
   could not be undone and why (for example "Cancelled, but the AMC project
   could not be reset: ...").

A job loaded with `cancelling` set, after a service restart, restarts the
undo instead of becoming `interrupted`, and `resume()` refuses it: a
half-undone run is never continued against the new, empty AMC project.

**Restart-safe (REQUIRED):** a restarted undo skips every action already in
`undone`. Re-running a finished action is not safe in general: recorded
clips and the older clips in `previous/` share names
(`<camera id>-<position>.mp4`), so deleting `recorded_files` again after
`previous/` was moved back would delete the older footage. So:

- the AMC reset records `amc_deleted` after the delete, saves
  `reset_amc_project_id` and records `amc_created` once `create_project`
  returns, and records `amc_bound` after the binding is written; a restart
  after `amc_created` reuses the saved id and never creates a second
  project;
- the footage undo records `footage_cleared` after deleting
  `recorded_files` and the manifest, before moving anything back from
  `previous/`, and `footage_restored` after the move; the move itself
  tolerates files already moved;
- within one action, a missing file or an AMC `404` counts as done.

Kept: the floor plan and alignment files in the folder (project inputs,
not run outputs), earlier `calibration/<UTC>` results, and the run's
`runs/<job id>.json` record.

The Calibration page's cancel confirmation says what is undone: "Footage
recorded by this run is deleted and the AMC project is reset to empty.
Earlier footage and calibrations are kept."

---

## 6. Operational change on the workstation

`installer.conf` on the workstation has `HOST_IP=192.168.4.106`, the home
network address. The AMC control service always passes `--host-ip` with
the Tailscale address, but an AMC start from Step 3 by hand uses
`installer.conf`, which is how the UI first got the wrong address. Set
`HOST_IP=100.95.170.92` there when this ships; it is configuration, not
code.

---

## 7. Not verified on the workstation

- That `DELETE /v1/delete_project/{id}` removes the project's files under
  AMC's `PROJECT_DIR`, not only its entry in the list.
- How AMC responds to `delete_project` while a calibration it started is
  still running; cancel calls `stop_calibration` first for that reason.

---

## 8. Out of scope / open decisions

- Undoing a completed run, or deleting individual calibrations.
- Cleaning up folders and AMC projects left behind by projects deleted
  before this ships; that is a one-time manual step.
- Recovering `.deleting-*` leftovers automatically.

---

## 9. Documented drift

| Earlier contract | Change |
|---|---|
| [`AMC-WEB-PROJECTS` section 12](AMC-WEB-PROJECTS.md#12-out-of-scope--open-decisions): deleting a project is a settled exclusion | reversed by [section 3](#3-deleting-a-project-required) |
| [`AMC-WEB-PROJECTS` section 6](AMC-WEB-PROJECTS.md#6-locking-calibration-to-the-project): a deleted AMC project is never silently recreated | still true for a run; cancel replaces the AMC project on purpose and records the new id ([section 5](#5-cancel-resets-the-run-locked)) |
| [`AMC-WEB-PROJECTS` section 8](AMC-WEB-PROJECTS.md#8-workstation-api-contract-required): any file can be downloaded | video files cannot ([section 4](#4-no-video-downloads-locked)) |
| [`STEP-3` section 4.2](STEP-3-AMC-LAUNCHER.md#42-pinned-compose-environment): `.env` is the only generated Compose file | two more files ([section 2](#2-amc-ui-runtime-config-caching-locked)) |

L0 adds a pointer to this doc at each of those places.

---

## 10. Decomposition

| Unit | Branch | Files touched | Depends on | Wave |
|---|---|---|---|---|
| L0 This spec | `feat/installer-amc-lifecycle-plan` | this doc, `AMC-WEB-PROJECTS.md` sections 6, 8 and 12, `STEP-3-AMC-LAUNCHER.md` section 4.2 | none | 1 |
| L1 AMC UI cache override | `feat/installer-amc-ui-no-cache` | `steps/step3_amc_launcher.py`, `tests/test_step3_amc_launcher.py` | L0 | 2 |
| L2 Project delete and video downloads | `feat/installer-amc-project-delete` | `mv3dt_amc_control/auth.py`, `api.py`, `provisioning.py`, `projects.py`, `amc_api.py`, `models.py`, `tests/amc_control/` | L0 | 2 |
| L3 Cancel reset | `feat/installer-amc-cancel-reset` | `mv3dt_amc_control/automation.py`, `tests/amc_control/test_amc_control_automation.py` | L2 | 3 |
| W1 Delete ticket | `feat/workstation-delete-ticket` | `services/workstationTicketService.ts`, `routes/workstation.routes.ts`, their tests | none | 1 |
| W2 Delete flow, download links, cancel text | `feat/project-delete-workstation-cleanup` | `pages/projects/useProjectEditor.ts`, `calibration/amcApi.ts`, `calibration/Calibration.tsx` | W1 | 2 |

Paths: `steps/` is under `installer/mv3dt_installer/`; `mv3dt_amc_control/`
under `installer/mv3dt_installer/assets/amc_control/`; `tests/` under
`installer/`; web server paths under `BetterPlacemaking.SERVER.EXPRESS/src/`
and client paths under `BetterPlacemaking.CLIENT.REACT/src/`.

Serialization points: L3 is stacked on L2 because it uses the
`delete_project` and `stop_calibration` calls L2 adds to `amc_api.py`. W2
is stacked on W1 because it calls the delete ticket route. L1 shares no file
with L2 or L3. L2 and W2 should ship together: without L2 the browser's
`DELETE` fails CORS preflight, and the client falls back to offering the
web-only delete.

---

## References

Behaviour of the AMC microservice is from its own OpenAPI document,
`GET /v1/openapi.yaml` on the workstation's AMC 3.3.0 (`delete_project`,
`stop_calibration`, `create_project`).

- [nginx `ngx_http_sub_module`](https://nginx.org/en/docs/http/ngx_http_sub_module.html)
  — the `sub_filter` cache-busting rewrite in
  [section 2](#2-amc-ui-runtime-config-caching-locked).
- [nginx `location`](https://nginx.org/en/docs/http/ngx_http_core_module.html#location)
  — an exact-match location takes precedence over a regex one.
- [Compose merge](https://docs.docker.com/compose/how-tos/multiple-compose-files/merge/)
  — `compose.override.yml` is read automatically.
- [RFC 8246](https://www.rfc-editor.org/rfc/rfc8246) — `immutable`, why a
  reload never revalidated `env-config.js`.

Repo files referenced:

- [`AMC-WEB-PROJECTS.md`](AMC-WEB-PROJECTS.md) — folders, bindings, tickets
  and the file download this narrows.
- [`AMC-FOOTAGE-RECORDING.md`](AMC-FOOTAGE-RECORDING.md) — the recording and
  `previous/` set-aside cancel undoes.
- [`STEP-3-AMC-LAUNCHER.md`](STEP-3-AMC-LAUNCHER.md) — the Compose stack and
  `compose/.env`.
- `installer/mv3dt_installer/steps/step3_amc_launcher.py` — `render_env`,
  `write_env_atomic`, `launch_amc`, `clone_amc`,
  `_expected_amc_worktree_change`.
- `installer/mv3dt_installer/assets/amc_control/mv3dt_amc_control/api.py` —
  the routes and CORS settings.
- `.../mv3dt_amc_control/provisioning.py` — `run_exclusive`, `_ensure_amc`.
- `.../mv3dt_amc_control/projects.py` — `ProjectStore`, `resolve_file`,
  `list_files`.
- `.../mv3dt_amc_control/automation.py` — `cancel`, `_finish`,
  `_close_web_project_run`, the recording and results steps.
- `.../mv3dt_amc_control/amc_api.py` — the AMC client the new calls join.
- `installer/mv3dt_installer/record_footage.py` — the recorder's final
  line and `previous/` handling.
