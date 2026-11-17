# STEP 9 — AMC control service: start, stop and automate AutoMagicCalib (owner: DevD)

Status: step spec, **implemented and unit-tested, never run on the
workstation** (dated evidence in
[`DEVELOPMENT-STATUS`](DEVELOPMENT-STATUS.md#current-workstation-state-last-verified-2026-11-16)).
It specifies the `step9_amc_control` module and the bundled
`mv3dt_amc_control` service package it installs. It builds on the contracts in
[`00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md) and follows
[`STEP-8`](STEP-8-TRACKING-HISTORY.md)'s structure wherever the two overlap;
it does **not** restate those, it links back to them. The web app's routes
and environment are owned by the web app repository;
[section 9](#9-web-app-setup) mirrors them.

The web app's Calibration page lets a permitted user start AutoMagicCalib
(AMC) on the workstation, see when it is ready, open it in a new browser tab
over the tailnet, and run AMC's workflow automatically up to the manual
alignment. **Step 9 installs the service that does this on the workstation.**
AMC is not streamed or proxied: the service only controls it, and the user's
browser loads AMC's own UI directly.

> **Extended by [`AMC-WEB-PROJECTS.md`](AMC-WEB-PROJECTS.md):** one AMC
> project and workstation folder per web app project, created from the web
> app and locked for every later run. Where the two differ, that spec wins
> once implemented.

---

## 1. Module identity and scope

Module: `installer/mv3dt_installer/steps/step9_amc_control.py`, step id
`step9_amc_control`, order 9. Package:
`installer/mv3dt_installer/assets/amc_control/mv3dt_amc_control/`. Unit
template: `assets/systemd/mv3dt-amc-control.service.in`. Tests:
`installer/tests/test_step9_amc_control.py` and `installer/tests/amc_control/`.

In scope: installing and running the service, its sudo rule, its tailnet
exposure and its settings file. Out of scope: AMC itself (Step 3), ingesting
calibration results (Step 4), and the web app's routes and page, which
the web app repository owns ([section 9](#9-web-app-setup)).

---

## 2. What it installs

```
<install_dir>/amc_control/
  mv3dt_amc_control/        the package, replaced on upgrade
  requirements.txt
  venv/
  amc_automation.yaml       operator-managed settings, seeded once (section 6)
  state/                    launcher log and automation progress, mode 750
<install_dir>/amc_projects/ one folder per web project, mode 750, never deleted
~<service user>/mv3dt-projects -> <install_dir>/amc_projects
/etc/systemd/system/mv3dt-amc-control.service
/etc/sudoers.d/mv3dt-amc-control
```

Everything under `amc_control/` is owned by the invoking user, who is also
the service user. Nothing is written to `secrets/`: the service reads Step 8's
`secrets/tracking.env`.

The **projects root** `<install_dir>/amc_projects/` is created on every run
if missing, owned by the service user, mode 750, and is never deleted or
emptied on upgrade or re-run; a symlink or a non-directory at that path fails
the step rather than being replaced. Its layout is
[`AMC-WEB-PROJECTS` section 4](AMC-WEB-PROJECTS.md#4-workstation-folder-layout).
The link `~<service user>/mv3dt-projects` is created, owned by the user, only
when nothing exists at that path: an existing file, directory or link there
is left alone and `run()` logs a warning naming it.

---

## 3. Gate and prerequisites

Gated by `MV3DT_AMC_CONTROL` in `installer.conf` with the CLI flag
`--amc-control`, default `off`. Preflight requires:

| Requirement | Why |
|---|---|
| Python 3.10+ with `venv` | the service venv |
| `<install_dir>/bin/mv3dt-installer` (Step 3) | the service runs its `amc` subcommand; the sudo rule names it |
| `bin/mv3dt-installer` and every directory above it owned by root, not group- or other-writable | otherwise whoever can replace the binary gets passwordless root through the rule (section 5); preflight and verify fail naming the offending path |
| `MV3DT_TRACKING_HISTORY=on` and a complete `secrets/tracking.env` (Step 8) | the web app signs AMC and tracking tickets with one `WORKSTATION_TICKET_SECRET`; `ALLOWED_ORIGINS` applies to both services. `TRACKING_PROJECT_ID` is still required because Step 8 needs it, but it no longer pins AMC ([section 7](#7-api-and-ticket-contract)) |
| `visudo` | the sudo rule is checked before it is installed |
| A Tailscale IPv4 (`tailscale ip -4`) | it is the address remote browsers use for AMC and is part of the sudo rule |

Network exposure follows Step 8's rules: the service binds `127.0.0.1:8091`
and is published only with `tailscale serve --bg --https=8444`, never Funnel.
Tracking keeps 8090 and 8443.

---

## 4. Package, venv and unit

The package and its pinned `requirements.txt` are copied and the venv built
exactly as in Step 8. The FastAPI stack uses the same versions as
`mv3dt_tracking`; `httpx` (AMC's REST API) and `Pillow` (floor plan
conversion) are added. `python-multipart` is test-only
(`requirements-dev.txt`).

The unit runs `venv/bin/python3 -m mv3dt_amc_control serve` as the invoking
user, loads `installer.conf` and `secrets/tracking.env` as environment files,
and sets `MV3DT_INSTALL_DIR`, `AMC_STATE_DIR`, `AMC_AUTOMATION_CONFIG`,
`AMC_PROJECTS_DIR` (`<install_dir>/amc_projects`, the projects root),
`AMC_PUBLIC_HOST` (the Tailscale IPv4) and `BIND`. A unit written by an
earlier release, without `AMC_PROJECTS_DIR`, differs from the rendered one,
so `pending_update()` re-runs the step. `NoNewPrivileges` must stay
unset because the service uses `sudo`: it starts and stops AMC with
`sudo -n <install_dir>/bin/mv3dt-installer amc ...`, never through the
`bin/amc` wrapper (section 5).

`serve` refuses to start without the ticket secret and the allowed origins,
with a wildcard origin, or on a non-loopback bind. It no longer requires the
project id
([`AMC-WEB-PROJECTS` section 9](AMC-WEB-PROJECTS.md#9-tickets-and-specs)).

---

## 5. The sudo rule

**LOCKED.** `mv3dt-installer amc` and `mv3dt-installer record-footage`
require root; the service does not run as root.
`/etc/sudoers.d/mv3dt-amc-control` (mode 440) allows the service user
exactly three command lines and nothing else:

```
<user> ALL=(root) NOPASSWD: <install_dir>/bin/mv3dt-installer amc --keep-up --no-open --non-interactive --host-ip <tailscale IPv4>, <install_dir>/bin/mv3dt-installer amc --down, <install_dir>/bin/mv3dt-installer record-footage --pending
```

`record-footage --pending` validates the request file the service wrote,
drops root privileges to the service user and records every camera into the
web project's `footage/` directory
([`AMC-FOOTAGE-RECORDING` section 5](AMC-FOOTAGE-RECORDING.md#5-the-record-footage-command)).

The rule names the root-owned installer binary that Step 3 stages, not
Step 3's `bin/amc` wrapper. The wrapper is only
`exec "<install_dir>/bin/mv3dt-installer" amc "$@"`, but Step 3 chowns it to
the invoking user, who is also the service user, so a rule naming the
wrapper would let that user rewrite it and run anything as root. Naming the
binary is safe only while nobody but root can replace it, so preflight,
`run()` and verify check the binary and every directory from it up to `/`
(along the path as written and its resolved path): each must be owned by
root and, unless it is a symlink, not writable by group or other. Any other
owner or mode fails the step with the offending path in the message, and no
rule is written.

The service runs these argument lists verbatim and never builds a command
from request data. The candidate rule is written under a dotted name (which
sudo ignores), checked with `visudo -cf`, then renamed into place. A user
name or install path that cannot be expressed safely in sudoers fails
preflight. When the Tailscale address changes, `pending_update()` re-runs the
step and rewrites the rule and the unit. A rule on disk that differs from the
one this release renders (for example the earlier rule naming `bin/amc`, or
one without the `record-footage` line) is an update too, and is replaced the
same way.

Note: `--host-ip` applies to that launch only; it does not change `HOST_IP`
in `installer.conf`
([`STEP-3` section 4.1](STEP-3-AMC-LAUNCHER.md#41-persistent-installer-configuration)).
An AMC started by hand with plain `bin/amc` therefore uses the LAN address,
and its UI fails for remote browsers. The service does not restart an AMC
that is already up, so stop it (`bin/amc --down`) and start it from the web
app, or set `HOST_IP` in `installer.conf` to the Tailscale IPv4 if manual
launches should be reachable remotely too.

---

## 6. Automation settings

Every setting, with documentation, is in the packaged
`mv3dt_amc_control/amc_automation.example.yaml`. The operator's overrides go
in `<install_dir>/amc_control/amc_automation.yaml`, which is merged on top and
never replaced by the installer. Unknown keys and invalid values are rejected.
Check edits with:

```
cd <install_dir>/amc_control && venv/bin/python3 -m mv3dt_amc_control check-config
```

The seeded file sets `project.mode: web_project`: every automated run is
locked to the calling web project's own AMC project and folder under the
projects root
([`AMC-WEB-PROJECTS` section 6](AMC-WEB-PROJECTS.md#6-locking-calibration-to-the-project)),
so it never touches the installer's live calibration that
[`STEP-4`](STEP-4-CALIB-OUTPUT-WIRING.md#7-later-recalibration)'s ingest timer
would install. `create_new` and `reuse_installer_project` keep their meaning.
**The file is never rewritten**, so a workstation seeded by an earlier release
with `create_new` is switched by hand: set `project.mode: web_project` in
`amc_automation.yaml` and run `check-config`. `layout.allowed_hosts` must list
the web app's storage host before floor plans can be downloaded.

The automated workflow, against AMC 3.3.0's seven steps:

| AMC step | Automation |
|---|---|
| 1 Project Setup | starts AMC if needed; reuses `AMC_PROJECT_ID` or creates a project |
| 2 Video Configuration | uploads the recorded footage as `cam_00.mp4`, …; uploads the project's floor plan as `layout.png` |
| 3 Parameters | posts the floor plan scale (waits for it in AMC when unknown); uploads focal lengths |
| 4 Rectification | automatic through AMC's API (default), or done in the UI |
| 5 Manual Alignment | re-uploads a saved `alignment_data.json` if present; otherwise the user |
| 6 Execute | `verify_project` must be `READY`; `calibrate`; wait for `COMPLETED` |
| 7 Results | evaluation statistics; optional result ZIP; Step 4 ingests the installer's project |

Steps 1 to 5 all run first; if any needs the user the run pauses once before
step 6 with a to-do list and continues on request. A step left to the user counts as done only when AMC confirms it
([`AMC-RECTIFICATION-AND-SCALE`](AMC-RECTIFICATION-AND-SCALE.md#5-checked-resume-required)).

---

## 7. API and ticket contract

| Route | Purpose |
|---|---|
| `GET /health` | open; version and whether the secret is configured |
| `GET /amc/status`, `POST /amc/start`, `POST /amc/stop` | AMC's state, launch, shutdown |
| `GET /amc/automation/status`, `POST .../start`, `.../continue`, `.../cancel` | the automated run |
| `GET /amc/automation/config` | settings in effect |
| `POST /projects`, `GET /projects/current`, `.../files`, `.../files/{path}` | web project provisioning and folder retrieval: [`AMC-WEB-PROJECTS` section 8](AMC-WEB-PROJECTS.md#8-workstation-api-contract-required) |

Tickets are Step 8's ([`STEP-8` section 3.2](STEP-8-TRACKING-HISTORY.md#32-tickets-required)) with
`scope: "amc"` (or `amc-provision` for `POST /projects`): HS256,
`aud: "p2bp-workstation"`, `exp`, `sub`, `pid`, and `iss` checked when
`TICKET_ISSUER` is set. A wrong scope is 403, so a tracking ticket cannot
control AMC. **The service accepts tickets for any web project**: it no
longer compares `pid` with `TRACKING_PROJECT_ID`, and the web app checks the
user's permission on the project before signing
([`AMC-WEB-PROJECTS` section 9](AMC-WEB-PROJECTS.md#9-tickets-and-specs)). `POST /amc/automation/start` also takes a
spec signed the same way with `scope: "amc-calibration-spec"`, carrying the
project title and a short-lived link to its floor plan.

CORS allows only `ALLOWED_ORIGINS`, and Chrome's Private Network Access
preflight is answered for those origins. `/amc/stop` is refused while a run is
executing or waiting for the user.

---

## 8. Run, verify, update

`run` phases: package and venv; settings and state; sudo rule; unit; tailnet
exposure (a failed `tailscale serve` is a warning with the manual command).
`verify` requires the unit enabled and active, the projects root a
directory (not a symlink) owned by the service user, the sudo rule present with
mode 440 and matching the current Tailscale address, and `GET /health`
answering with the secret configured. `pending_update()` re-runs a completed
step when the bundled package, requirements, unit or sudo rule differ.

---

## 9. Web app setup

**LOCKED — the web app repository is the source of truth for route names,
authentication, environment names and wire shapes.** This section mirrors
web app PR
[`P2BP-25_26-Web#34`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Web/pull/34)
(branch `feat/amc-workstation-control`, head `740c300`, 2026-11-16), the
Express API under `BetterPlacemaking.SERVER.EXPRESS/src/`. Where the two
disagree, the web app wins and this section and `step9_amc_control.py` change
to match, as for Step 7
([`STEP-7` section G](STEP-7-WEBAPP-INTEGRATION.md#g-route-table-reconciled-against-the-express-backend)).

### 9.1 Environment

`report` prints the URL to set and where the secret comes from, plus the
projects root and the link to it. In the web
app's environment:

| Variable | Value |
|---|---|
| `WORKSTATION_TICKET_SECRET` | already set for tracking: the same value as `WORKSTATION_TICKET_SECRET` in `<install_dir>/secrets/tracking.env` ([`STEP-8` section 4.3](STEP-8-TRACKING-HISTORY.md#43-trackingenv)) |
| `WORKSTATION_AMC_URL` | `https://<tailnet name>:8444` |
| `WORKSTATION_AMC_TICKET_MINUTES` | optional, default `25`; independent of tracking's `WORKSTATION_TICKET_MINUTES` (default `5`) |

**REQUIRED — do not use `WORKSTATION_BASE_URL`.** On the web app that name
configures its unrelated `RemoteWorkstation` seam. Step 9 as first merged
(PR #67) printed it in `report`; the AMC control URL is
`WORKSTATION_AMC_URL`. Both web routes answer `503` until the secret and
`WORKSTATION_AMC_URL` are both set.

### 9.2 Routes

Both routes require a user session and the `Project.Vision.Read` permission
on the project, and sign with `WORKSTATION_TICKET_SECRET`, HS256,
`aud: "p2bp-workstation"` and `iss` set to the web app's `JWT_ISSUER`.

| Route | Signs | Lifetime | Response |
|---|---|---|---|
| `POST /api/workstation/project/:projectId/ticket` | ticket, `scope: "amc"` | `WORKSTATION_AMC_TICKET_MINUTES` | `Ticket`, `ExpiresAt`, `WorkstationUrl` |
| `POST /api/workstation/project/:projectId/calibration-spec` | spec, `scope: "amc-calibration-spec"`, carrying the project title and a floor plan link | 15 min, the floor plan link too | `Spec`, `ExpiresAt`, `Summary` |

The browser sends the ticket as `Authorization: Bearer <ticket>` to the
workstation routes in [section 7](#7-api-and-ticket-contract), and passes the
spec as the `Spec` field of `POST /amc/automation/start`. The spec route takes
an optional `FloorplanId`; without one it uses the project's most recently
updated floor plan.

### 9.3 Gaps

1. **The web PR is not merged.** Until it is, no deployed web app issues AMC
   tickets or specs, and nothing in this step can be exercised from the web
   app. Re-check this section against the merged routes.
2. **No web UI for automation.** The web client's `amcApi.ts` calls only
   `/amc/status`, `/amc/start` and `/amc/stop`, and no page uses it yet.
   Nothing calls `/amc/automation/*` or the calibration-spec route, so the
   automated workflow in [section 6](#6-automation-settings) is reachable
   only by calling the service directly.

---

## 10. Not verified on the workstation

- Everything against the real AMC 3.3.0; the tests use a fake AMC built from
  NVIDIA's documented REST API (`skills/amc-run-video-calibration` in
  NVIDIA/DeepStream).
- Whether uploading videos to an existing AMC project replaces or appends.
- Automatic rectification end to end; its API calls were read from AMC's
  OpenAPI document and UI bundle
  ([`AMC-RECTIFICATION-AND-SCALE` section 2](AMC-RECTIFICATION-AND-SCALE.md#2-amcs-rectification-api)).
- AMC's UI and API ports listen on every interface with no login
  ([`LIVE-BASELINE-AND-EDIT-PROTECTION.md`](LIVE-BASELINE-AND-EDIT-PROTECTION.md)),
  so tailnet ACLs decide who can open AMC once it is running.

---

## 11. Decomposition

Step 9 is implemented; these are the units as they shipped.

| Unit | Branch | Files touched | Depends on | Wave |
|---|---|---|---|---|
| A1 Step 9 (PR #67) | `amc-auto` | `steps/step9_amc_control.py`, `assets/amc_control/`, `assets/systemd/mv3dt-amc-control.service.in`, `__main__.py`, `app.py`, `config.py`, `state.py`, `installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md`, this doc, `tests/test_step9_amc_control.py`, `tests/amc_control/`, `tests/test_app.py`, `tests/test_config.py`, `tests/test_main.py` | none | 1 |
| A2 Sudo rule targets the installer binary (PR #69) | `feat/installer-amc-sudo-root-target` | `steps/step9_amc_control.py`, `assets/amc_control/mv3dt_amc_control/api.py`, `controller.py`, `assets/systemd/mv3dt-amc-control.service.in`, this doc sections 3 to 5, `tests/test_step9_amc_control.py`, `tests/amc_control/test_amc_control_controller.py` | A1 | 2 |
| A3 CI installs service test requirements (PR #68) | `feat/installer-ci-service-deps` | `.github/workflows/installer-tests.yml`, `tests/amc_control/test_amc_control_api.py` | A1 | 2 |
| A4 Doc follow-ups | `feat/installer-step9-doc-fixes` | this doc, `DEVELOPMENT-STATUS.md`, `installer/README.md`, `CLAUDE.md`, the `report` line in `steps/step9_amc_control.py` and its test | A2 | 3 |
| H1 to H5 AMC web projects (PRs #73, #74, #76, #75 and #77) | `feat/installer-amc-project-store` to `feat/installer-step9-projects-root` | the units in [`AMC-WEB-PROJECTS` section 13](AMC-WEB-PROJECTS.md#13-decomposition); H5 is `steps/step9_amc_control.py`, `tests/test_step9_amc_control.py` and this doc | A4 | 4 |

Package paths are relative to `installer/mv3dt_installer/`; test paths are
relative to `installer/`. A4 is stacked on A2 because both edit this doc. The
H units' own dependencies and waves are in
[`AMC-WEB-PROJECTS` section 13](AMC-WEB-PROJECTS.md#13-decomposition).

---

## References

AMC facts come from NVIDIA's AutoMagicCalib REST API as documented in the
`skills/amc-run-video-calibration` skill in NVIDIA/DeepStream; the web app
contract comes from the web app repository.

- [`P2BP-25_26-Web#34`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Web/pull/34)
  — **backs [section 9](#9-web-app-setup)**: routes, environment names,
  ticket and spec scopes and lifetimes.
- [Tailscale serve](https://tailscale.com/kb/1312/serve) — tailnet-only
  HTTPS on `8444` ([section 3](#3-gate-and-prerequisites)).
- [sudoers(5)](https://www.sudo.ws/docs/man/sudoers.man/) — command-line
  matching in the rule ([section 5](#5-the-sudo-rule)).
- [RFC 7519](https://datatracker.ietf.org/doc/html/rfc7519) — JWT claims
  (`aud`, `exp`, `iss`) used by tickets and specs.

Repo files referenced:

- [`00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md) — the
  step contract and the gate mechanism
  ([`00` section 3.4](00-FRAMEWORK-AND-BOOTSTRAP.md#34-opt-in-step-gates)).
- [`STEP-3-AMC-LAUNCHER.md`](STEP-3-AMC-LAUNCHER.md) — the `amc` subcommand
  and `bin/amc` wrapper the service drives.
- [`STEP-4-CALIB-OUTPUT-WIRING.md`](STEP-4-CALIB-OUTPUT-WIRING.md) — ingest
  of the calibration result.
- [`STEP-7-WEBAPP-INTEGRATION.md`](STEP-7-WEBAPP-INTEGRATION.md) — section G,
  the pattern section 9 follows.
- [`STEP-8-TRACKING-HISTORY.md`](STEP-8-TRACKING-HISTORY.md) — the ticket
  contract, `tracking.env`, and the tailnet exposure rules this step reuses.
- [`LIVE-BASELINE-AND-EDIT-PROTECTION.md`](LIVE-BASELINE-AND-EDIT-PROTECTION.md)
  — AMC's unauthenticated ports.
- [`AMC-WEB-PROJECTS.md`](AMC-WEB-PROJECTS.md) — the projects root, the
  project routes, any-project tickets and the `web_project` mode.
- [`DEVELOPMENT-STATUS.md`](DEVELOPMENT-STATUS.md) — what has run on the
  workstation.
- `installer/mv3dt_installer/steps/step9_amc_control.py`,
  `installer/mv3dt_installer/assets/amc_control/mv3dt_amc_control/` — the
  implementation.
