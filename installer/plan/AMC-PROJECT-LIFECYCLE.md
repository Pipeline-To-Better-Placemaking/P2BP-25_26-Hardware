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
`env-config.js` from `API_URL` at container start. The image's nginx is
built `--with-http_sub_module` (checked on the workstation).

Step 3 writes two more files next to `compose/.env`, in the same place and
the same atomic, content-idempotent way
([`STEP-3` section 4.2](STEP-3-AMC-LAUNCHER.md#42-pinned-compose-environment)):

1. **`compose/mv3dt-ui-nginx.conf`**: the image's `default.conf`
   unchanged (the `client_max_body_size 4g` and `client_body_timeout 1800s`
   upload limits, the SPA `try_files`, the static-asset caching, gzip) plus:
   - `location = /env-config.js` with `Cache-Control: no-store`, so the
     regex asset block never matches it;
   - in `location /`, `Cache-Control: no-cache` on the HTML and
     `sub_filter '/env-config.js"' '/env-config.js?v=<hash>"'`, where
     `<hash>` is the first 12 hex digits of the SHA-256 of the API URL the
     UI is given. A browser holding the old immutable copy then asks for a
     new URL, so the fix also reaches browsers that cached the wrong
     address before it shipped.
2. **`compose/compose.override.yml`**: bind-mounts that file read-only over
   `/etc/nginx/conf.d/default.conf` in the `auto-magic-calib-ui` service.
   Docker Compose merges `compose.override.yml` automatically because
   Step 3 runs Compose from `compose/` without `-f`.

The API URL in the hash is the one the UI compose file resolves:
`AUTO_MAGIC_CALIB_MS_API_URL` when set, otherwise
`http://<HOST_IP>:<AUTO_MAGIC_CALIB_MS_PORT>/v1`. A change to either file
makes Step 3 recreate the UI container, as a changed `.env` already does.

> **Pinned image (REQUIRED):** the override replaces the image's whole
> `default.conf`, so it is tied to the 3.3.0 UI image. A test asserts that
> the rendered file still carries the image's upload limits and SPA
> fallback; a change to the pinned UI image must re-copy `default.conf`.

If Step 3 checks the AMC checkout for local changes, the two new files must
be excluded from that check, as `compose/.env` is.

---

## 3. Deleting a project (REQUIRED)

### 3.1 Wire contract

| Piece | Contract |
|---|---|
| Ticket scope | `amc-delete`, same audience, issuer and `pid` claim as the other workstation tickets ([`AMC-WEB-PROJECTS` section 9](AMC-WEB-PROJECTS.md#9-tickets-and-specs)) |
| Web route | `POST /api/workstation/project/:projectId/delete-ticket`, permission `Project.Delete`, 5-minute ticket; `200 {Ticket, ExpiresAt, WorkstationUrl}`, `503` when workstation control is not configured, `404` for an unknown project |
| Workstation route | `DELETE /projects/current`, `amc-delete` scope only |

Workstation responses:

| Status | When | Body |
|---|---|---|
| `200` | deleted | `{Folder, AmcProjectId, AmcProjectDeleted, Leftover}` |
| `404` | the ticket's project has no folder | `{detail}` |
| `409` | this project's run is running or waiting for the user, or a provisioning job runs | `{detail}` |
| `502` | AMC refused the delete (anything but `404`) | `{detail}` |
| `503` | AMC could not be started | `{detail}` |

### 3.2 Workstation behaviour

Inside the provisioner's `run_exclusive`, so no provisioning job or run can
start meanwhile:

1. **Find**: the binding for the ticket's `pid`; `404` without one.
2. **Refuse**: `409` when the runner's job belongs to this `pid` and
   `uses_amc()` is true ("Cancel this project's run first").
3. **AMC**: when the binding has an `amc_project_id`, make AMC ready the
   way provisioning does (start it and wait, `503` on timeout), then
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

`AmcApi` gains `delete_project(id)` and `stop_calibration(id)`.

### 3.3 Web flow

1. **Confirm**: for a project with a `WorkstationFolder`, the confirmation
   says the workstation folder (footage, floor plan, calibrations) and its
   AMC project are deleted too.
2. **Workstation first**: request the delete ticket, then
   `DELETE /projects/current`. `200` and `404` go on to step 3.
3. **Database**: `DELETE /api/project/:id` as today.
4. **On failure** (unreachable, `409`, `502`, `503`): show the reason and
   offer **Delete from the web app only**, which leaves the folder and AMC
   project on the workstation, or cancel.

A project without a `WorkstationFolder`, or with workstation control not
configured, is deleted as today.

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
  with `403` "Video files stay on the workstation", checked on the
  requested path before any filesystem access.
- **Client**: no download link for a file with `Downloadable: false`.

---

## 5. Cancel resets the run (LOCKED)

For a `web_project` run (the job has a `folder`), cancel undoes the run
instead of only stopping it. Other modes are unchanged.

Once nothing is executing (immediately for a paused, failed or interrupted
run; after the run thread stops for a running one), the runner, with the
job still `running` and the message "Cancelling: undoing this run":

1. **AMC calibration**: when step 6 started, `stop_calibration` on the
   job's AMC project, ignoring errors.
2. **AMC project**: when any step after `project_setup` ran, delete the
   job's AMC project and create a new one named after the folder, then
   store the new id in the binding (`amc_project_id`). This is the one
   reset that is certain to leave nothing uploaded: videos, layout,
   alignment, config, annotations and results all go, and the project is
   `INIT`.
3. **Footage**: when this run recorded footage (`footage_recorded`),
   delete the clips and manifest the recording wrote in `footage/`, move
   everything in `footage/previous/` back into `footage/`, and remove
   `previous/` when it is empty. Footage that was there before the run
   started is kept. A recording cancelled while it runs is already
   restored by the recorder
   ([`AMC-FOOTAGE-RECORDING` section 5](AMC-FOOTAGE-RECORDING.md#5-the-record-footage-command)).
4. **Finish**: `cancelled`, with a message saying what was undone, or what
   could not be undone and why (for example "Cancelled, but the AMC project
   could not be reset: ...").

Kept: the floor plan and alignment files in the folder (project inputs,
not run outputs) and earlier `calibration/<UTC>` results.

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

## 9. Decomposition

| Unit | Branch | Files touched | Depends on | Wave |
|---|---|---|---|---|
| L0 This spec | `feat/installer-amc-lifecycle-plan` | this doc | none | 1 |
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
with L2 or L3. L2 and W2 must ship together: without L2, the workstation
answers `DELETE /projects/current` with `405`, and the client offers the
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
  `write_env_atomic`, `launch_amc`.
