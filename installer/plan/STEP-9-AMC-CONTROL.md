# STEP 9 — AMC control service (start, stop and automate AutoMagicCalib)

Status: step spec, **implemented and unit-tested, never run on the
workstation**. It specifies the `step9_amc_control` module and the bundled
`mv3dt_amc_control` service package it installs. It builds on the contracts in
[`00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md) and follows
[`STEP-8`](STEP-8-TRACKING-HISTORY.md)'s structure wherever the two overlap;
those are not restated here.

The web app's Calibration page lets a permitted user start AutoMagicCalib
(AMC) on the workstation, see when it is ready, open it in a new browser tab
over the tailnet, and run AMC's workflow automatically up to the manual
alignment. **Step 9 installs the service that does this on the workstation.**
AMC is not streamed or proxied: the service only controls it, and the user's
browser loads AMC's own UI directly.

---

## 1. Module identity and scope

Module: `installer/mv3dt_installer/steps/step9_amc_control.py`, step id
`step9_amc_control`, order 9. Package:
`installer/mv3dt_installer/assets/amc_control/mv3dt_amc_control/`. Unit
template: `assets/systemd/mv3dt-amc-control.service.in`. Tests:
`installer/tests/test_step9_amc_control.py` and `installer/tests/amc_control/`.

In scope: installing and running the service, its sudo rule, its tailnet
exposure and its settings file. Out of scope: AMC itself (Step 3), ingesting
calibration results (Step 4), and the web app's routes and page.

## 2. What it installs

```
<install_dir>/amc_control/
  mv3dt_amc_control/        the package, replaced on upgrade
  requirements.txt
  venv/
  amc_automation.yaml       operator-managed settings, seeded once (section 6)
  state/                    launcher log and automation progress, mode 750
/etc/systemd/system/mv3dt-amc-control.service
/etc/sudoers.d/mv3dt-amc-control
```

Everything under `amc_control/` is owned by the invoking user, who is also
the service user. Nothing is written to `secrets/`: the service reads Step 8's
`secrets/tracking.env`.

## 3. Gate and prerequisites

Gated by `MV3DT_AMC_CONTROL` in `installer.conf` with the CLI flag
`--amc-control`, default `off`. Preflight requires:

| Requirement | Why |
|---|---|
| Python 3.10+ with `venv` | the service venv |
| `<install_dir>/bin/amc` (Step 3) | the launcher the service runs |
| `MV3DT_TRACKING_HISTORY=on` and a complete `secrets/tracking.env` (Step 8) | the web app signs AMC and tracking tickets with one `WORKSTATION_TICKET_SECRET`; `TRACKING_PROJECT_ID` and `ALLOWED_ORIGINS` apply to both services |
| `visudo` | the sudo rule is checked before it is installed |
| A Tailscale IPv4 (`tailscale ip -4`) | it is the address remote browsers use for AMC and is part of the sudo rule |

Network exposure follows Step 8's rules: the service binds `127.0.0.1:8091`
and is published only with `tailscale serve --bg --https=8444`, never Funnel.
Tracking keeps 8090 and 8443.

## 4. Package, venv and unit

The package and its pinned `requirements.txt` are copied and the venv built
exactly as in Step 8. The FastAPI stack uses the same versions as
`mv3dt_tracking`; `httpx` (AMC's REST API) and `Pillow` (floor plan
conversion) are added. `python-multipart` is test-only
(`requirements-dev.txt`).

The unit runs `venv/bin/python3 -m mv3dt_amc_control serve` as the invoking
user, loads `installer.conf` and `secrets/tracking.env` as environment files,
and sets `MV3DT_INSTALL_DIR`, `AMC_STATE_DIR`, `AMC_AUTOMATION_CONFIG`,
`AMC_PUBLIC_HOST` (the Tailscale IPv4) and `BIND`. `NoNewPrivileges` must stay
unset because the service uses `sudo`.

`serve` refuses to start without the ticket secret, the project id and the
allowed origins, with a wildcard origin, or on a non-loopback bind.

## 5. The sudo rule

`bin/amc` runs installer subcommands, which require root; the service does
not run as root. `/etc/sudoers.d/mv3dt-amc-control` (mode 440) allows the
service user exactly two command lines and nothing else:

```
<user> ALL=(root) NOPASSWD: <install_dir>/bin/amc --keep-up --no-open --non-interactive --host-ip <tailscale IPv4>, <install_dir>/bin/amc --down
```

The service runs these argument lists verbatim and never builds a command
from request data. The candidate rule is written under a dotted name (which
sudo ignores), checked with `visudo -cf`, then renamed into place. A user
name or install path that cannot be expressed safely in sudoers fails
preflight. When the Tailscale address changes, `pending_update()` re-runs the
step and rewrites the rule and the unit.

Note: `--host-ip` applies to that launch only; it does not change `HOST_IP`
in `installer.conf`
([`STEP-3` section 4.1](STEP-3-AMC-LAUNCHER.md#41-persistent-installer-configuration)).
An AMC started by hand with plain `bin/amc` therefore uses the LAN address,
and its UI fails for remote browsers. The service does not restart an AMC
that is already up, so stop it (`bin/amc --down`) and start it from the web
app, or set `HOST_IP` in `installer.conf` to the Tailscale IPv4 if manual
launches should be reachable remotely too.

## 6. Automation settings

Every setting, with documentation, is in the packaged
`mv3dt_amc_control/amc_automation.example.yaml`. The operator's overrides go
in `<install_dir>/amc_control/amc_automation.yaml`, which is merged on top and
never replaced by the installer. Unknown keys and invalid values are rejected.
Check edits with:

```
cd <install_dir>/amc_control && venv/bin/python3 -m mv3dt_amc_control check-config
```

The seeded file sets `project.mode: create_new`, so a first automated run
makes its own AMC project and cannot overwrite the live calibration that
[`STEP-4`](STEP-4-CALIB-OUTPUT-WIRING.md#7-later-recalibration)'s ingest timer
would install. Switch to `reuse_installer_project` once runs are trusted.
`layout.allowed_hosts` must list the web app's storage host before floor plans
can be downloaded.

The automated workflow, against AMC 3.3.0's seven steps:

| AMC step | Automation |
|---|---|
| 1 Project Setup | starts AMC if needed; reuses `AMC_PROJECT_ID` or creates a project |
| 2 Video Configuration | uploads the recorded footage as `cam_00.mp4`, …; uploads the project's floor plan as `layout.png` |
| 3 Parameters | posts a settings file with the floor plan scale; uploads focal lengths |
| 4 Rectification | configured API requests, or done in the UI (default) |
| 5 Manual Alignment | re-uploads a saved `alignment_data.json` if present; otherwise the user |
| 6 Execute | `verify_project` must be `READY`; `calibrate`; wait for `COMPLETED` |
| 7 Results | evaluation statistics; optional result ZIP; Step 4 ingests the installer's project |

Steps 1 to 5 all run first; if any needs the user the run pauses once before
step 6 with a to-do list and continues on request.

## 7. API and ticket contract

| Route | Purpose |
|---|---|
| `GET /health` | open; version and whether the secret is configured |
| `GET /amc/status`, `POST /amc/start`, `POST /amc/stop` | AMC's state, launch, shutdown |
| `GET /amc/automation/status`, `POST .../start`, `.../continue`, `.../cancel` | the automated run |
| `GET /amc/automation/config` | settings in effect |

Tickets are Step 8's ([`STEP-8` section 3.2](STEP-8-TRACKING-HISTORY.md)) with
`scope: "amc"`: HS256, `aud: "p2bp-workstation"`, `exp`, `sub`, `pid`, and
`iss` checked when `TICKET_ISSUER` is set. Wrong scope or project is 403, so a
tracking ticket cannot control AMC. `POST /amc/automation/start` also takes a
spec signed the same way with `scope: "amc-calibration-spec"`, carrying the
project title and a short-lived link to its floor plan.

CORS allows only `ALLOWED_ORIGINS`, and Chrome's Private Network Access
preflight is answered for those origins. `/amc/stop` is refused while a run is
executing or waiting for the user.

## 8. Run, verify, update

`run` phases: package and venv; settings and state; sudo rule; unit; tailnet
exposure (a failed `tailscale serve` is a warning with the manual command).
`verify` requires the unit enabled and active, the sudo rule present with
mode 440 and matching the current Tailscale address, and `GET /health`
answering with the secret configured. `pending_update()` re-runs a completed
step when the bundled package, requirements, unit or sudo rule differ.

## 9. Web app setup

`report` prints these. In the web app's environment:

| Variable | Value |
|---|---|
| `WORKSTATION_TICKET_SECRET` | already set for tracking (the same secret) |
| `WORKSTATION_BASE_URL` | `https://<tailnet name>:8444` |
| `WORKSTATION_AMC_TICKET_MINUTES` | optional, default 25 |

## 10. Not verified on the workstation

- Everything against the real AMC 3.3.0; the tests use a fake AMC built from
  NVIDIA's documented REST API (`skills/amc-run-video-calibration` in
  NVIDIA/DeepStream).
- Whether uploading videos to an existing AMC project replaces or appends.
- Rectification's API paths, which NVIDIA documents only through the UI; take
  them from `python3 -m mv3dt_amc_control openapi`.
- AMC's UI and API ports listen on every interface with no login
  ([`LIVE-BASELINE-AND-EDIT-PROTECTION.md`](LIVE-BASELINE-AND-EDIT-PROTECTION.md)),
  so tailnet ACLs decide who can open AMC once it is running.
