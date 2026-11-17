# AMC footage recording — record calibration footage from an automated run (owner: DevD)

Status: design spec, **not implemented**. It extends
[`AMC-WEB-PROJECTS.md`](AMC-WEB-PROJECTS.md) (one AMC project and folder per
web project) and reuses the recorder of
[`STEP-4-CALIBRATION-FOOTAGE.md`](STEP-4-CALIBRATION-FOOTAGE.md); it does
**not** restate either — it links back to them. It resolves open decision 1
of [`AMC-WEB-PROJECTS` section 12](AMC-WEB-PROJECTS.md#12-out-of-scope--open-decisions).

Today an automated run uploads whatever is in `<folder>/footage/` and waits
for the user when that is empty. This spec lets the user **record new
footage from every camera as part of starting a run**, straight into the
project's folder. The AMC control service runs as the invoking user and
cannot read the root-only camera credentials, so recording goes through
**one new fixed sudo command** that validates a request file and records
with root privileges dropped.

---

## 1. Scope and identity

| Repository | Area | Owner of |
|---|---|---|
| `P2BP-25_26-Hardware` | `installer/mv3dt_installer/record_footage.py`, `steps/step9_amc_control.py` | the `record-footage` subcommand, its sudo rule line |
| `P2BP-25_26-Hardware` | `installer/mv3dt_installer/assets/amc_control/mv3dt_amc_control/` | requesting a recording from step 2 of a run |
| `P2BP-25_26-Web` | `BetterPlacemaking.CLIENT.REACT/src/pages/projects/calibration/` | the record option when starting a run |

Out of scope is [section 9](#9-out-of-scope--open-decisions).

---

## 2. End-to-end flow

1. **Start**: on the Calibration page the user starts a run with **Record new
   footage** checked (the default when the folder holds no footage) and a
   length, after a reminder that people must walk through every camera's
   view for the whole recording.
2. **Request**: `POST /amc/automation/start` carries `Record: {Seconds}`
   ([section 6](#6-service-changes)). In step 2 the service writes the
   request file and runs `sudo -n <install_dir>/bin/mv3dt-installer
   record-footage --pending` ([section 4](#4-the-request-file-required)).
3. **Record**: the root command validates the request, reads the camera
   credentials and inventory, drops to the service user and records every
   camera at once into `<folder>/footage/`
   ([section 5](#5-the-record-footage-command)).
4. **Continue**: progress shows in step 2's message; on success the run
   uploads the new clips exactly as it uploads copied footage
   ([`AMC-WEB-PROJECTS` section 6](AMC-WEB-PROJECTS.md#6-locking-calibration-to-the-project)).

---

## 3. The sudo rule (LOCKED)

`/etc/sudoers.d/mv3dt-amc-control` gains exactly one command line, so the
rule becomes:

```
<user> ALL=(root) NOPASSWD: <install_dir>/bin/mv3dt-installer amc --keep-up --no-open --non-interactive --host-ip <tailscale IPv4>, <install_dir>/bin/mv3dt-installer amc --down, <install_dir>/bin/mv3dt-installer record-footage --pending
```

The same root-owned binary, the same ownership preflight and the same
`visudo -cf` check as
[`STEP-9` section 5](STEP-9-AMC-CONTROL.md#5-the-sudo-rule) apply. **The
command line never carries request data**: everything variable comes from
the request file, which the root command treats as untrusted input. A rule
on disk without the new line is an update, so `pending_update()` re-runs
Step 9.

---

## 4. The request file (REQUIRED)

Written by the service, atomically (temp file, `fsync`, `os.replace`), mode
600, at `<install_dir>/amc_control/state/record-request.json`:

```json
{"schema_version": 1, "folder": "automation-test", "seconds": 300, "requested_at": 1794886679}
```

| Field | Rule |
|---|---|
| `schema_version` | exactly the integer `1` |
| `folder` | matches [`AMC-WEB-PROJECTS` section 3](AMC-WEB-PROJECTS.md#3-folder-names-locked) |
| `seconds` | integer, 30 to 1800 inclusive |
| `requested_at` | integer Unix seconds, at most 300 seconds old |

The service deletes the file after the command exits; the root command never
writes or deletes anything in the service user's directories while it holds
root.

---

## 5. The `record-footage` command

`mv3dt-installer record-footage --pending`, registered like the other
subcommands, root required. In order:

1. **Who**: the service user is `SUDO_USER`; it must exist, not be root,
   and own `<install_dir>/amc_projects` (else exit `2`).
2. **Read the request** with `O_NOFOLLOW`: a regular file owned by that user,
   at most 4 KiB, valid against [section 4](#4-the-request-file-required)
   (else exit `2`).
3. **Check the folder**: `<install_dir>/amc_projects/<folder>` and its
   `footage/` are directories, not symlinks, owned by the user, and the folder
   holds a regular `project.json` (else exit `2`).
4. **Load inputs as root**: camera credentials with
   `camera_credentials.load_credentials()` and the inventory from
   `installer.conf`'s `CAMERAS_FILE` (none or empty is exit `2`); the 1 GiB
   per camera free-space check of
   [`STEP-4-CALIBRATION-FOOTAGE` section 2](STEP-4-CALIBRATION-FOOTAGE.md#2-pinned-values)
   on the folder's filesystem (exit `1` when short).
5. **Drop privileges, irrevocably**: `initgroups`, `setgid`, `setuid` to the
   user, then confirm `os.setuid(0)` fails. Everything after runs as the user.
6. **Replace old footage**: move every `*.mp4` and `.mv3dt-footage.json` in
   `footage/` into `footage/previous/`, emptying that directory first, so
   one earlier recording is kept.
7. **Record** with `footage.record(cameras, user=..., password=...,
   project_dir=<folder>/footage, project_name=<folder>, seconds=...)` and no
   `user_prefix`; `SIGTERM` and `SIGINT` both cancel it (every `.part` file
   removed, as in [`STEP-4-CALIBRATION-FOOTAGE` section 4](STEP-4-CALIBRATION-FOOTAGE.md#4-recording)).

**Output contract (REQUIRED)**, on stdout, one JSON object per line:

| Line | When |
|---|---|
| `{"progress": 0.42}` | about once per second while recording |
| `{"ok": true, "cancelled": false, "files": ["c1-left.mp4"], "failed": [], "seconds": 300}` | last line, always |

`failed` entries are `{"id", "ip", "error"}` with the password redacted.
Exit codes: `0` recorded, `1` some camera failed or no room, `2` invalid
request or setup, `3` cancelled. Diagnostics go to stderr.

---

## 6. Service changes

- **Start body**: `AutomationStart` gains optional `Record: {"Seconds": int}`
  (30 to 1800; otherwise `400`). Absent keeps today's behaviour.
- **Step 2 with `Record`** in `web_project` mode: write the request, run the
  command (through `sudo -n` unless `AMC_USE_SUDO=0`, as the controller
  does), set the step message from progress lines ("recording 2:10 of 5:00;
  keep people walking through every camera's view"), then delete the request
  file. Exit `0` continues to the existing upload; any other exit fails the
  step with the last line's reason and per-camera errors. `Record` outside
  `web_project` mode is `400`.
- **Cancel** during recording sends `SIGTERM` to the child and waits for it.
- **Resume**: the job records `footage_recorded` (persisted), so Continue
  never records again; a new recording needs a new run.
- **Busy**: recording happens inside a run, so the existing one-AMC-user rule
  ([`AMC-WEB-PROJECTS` section 5](AMC-WEB-PROJECTS.md#5-provisioning))
  already keeps provisioning out.

---

## 7. Web app changes

- `amcApi.ts`: `startAutomation(projectId, floorplanId?, options?)` with
  `options.recordSeconds` sending `Record: {Seconds}`.
- `Calibration.tsx`: the run panel's start controls gain **Record new
  footage** (checked by default when the folder's file list has no
  `footage/*.mp4`) and a length in minutes (1 to 30, default 5), with the
  walking reminder; starting with it checked asks for confirmation. Progress
  is the step message the page already shows.

---

## 8. Not verified on the workstation

- Recording while `mv3dt-pipeline@<slug>` holds the same RTSP streams. The
  cameras normally allow several sessions; if they refuse, the per-camera
  errors say so.
- CPU headroom for one `libx264` encode per camera next to the live pipeline.

---

## 9. Out of scope / open decisions

- Recording outside an automated run (a standalone "record" button).
- Choosing a subset of cameras; every inventory camera is recorded.
- Retention beyond one `previous/` recording.

---

## 10. Decomposition

| Unit | Branch | Files touched | Depends on | Wave |
|---|---|---|---|---|
| F0 This spec | `feat/installer-footage-recording-plan` | this doc | none | 1 |
| F1 Root command and sudo line | `feat/installer-record-footage-command` | `mv3dt_installer/record_footage.py`, `steps/step9_amc_control.py`, `tests/test_record_footage.py`, `tests/test_step9_amc_control.py`, `STEP-9-AMC-CONTROL.md` | F0 | 2 |
| F2 Recording from step 2 | `feat/installer-amc-run-recording` | `mv3dt_amc_control/automation.py`, `models.py`, `api.py`, `tests/amc_control/test_amc_control_automation.py` | F0 | 2 |
| F3 Record option on the Calibration page | `feat/calibration-record-footage` | `calibration/amcApi.ts`, `calibration/Calibration.tsx` | none | 2 |

Paths: `mv3dt_installer/` and `steps/` are under `installer/`;
`mv3dt_amc_control/` under `installer/mv3dt_installer/assets/amc_control/`;
`tests/` under `installer/`; the docs under `installer/plan/`; web paths
under `BetterPlacemaking.CLIENT.REACT/src/pages/projects/`.

Serialization points: none inside the waves. F1 and F2 meet only at the
request file and output contract (sections 4 and 5), and F3 meets F2 only
at the start body (section 6), so all three build in parallel. F1 must ship
in the same release as F2: without the sudo line, a run with `Record` fails
at step 2 with sudo's refusal.

---

## References

Recording behaviour is
[`STEP-4-CALIBRATION-FOOTAGE.md`](STEP-4-CALIBRATION-FOOTAGE.md)'s, as
implemented in `installer/mv3dt_installer/footage.py`.

- [sudoers(5)](https://www.sudo.ws/docs/man/sudoers.man/) — exact
  command-line matching behind [section 3](#3-the-sudo-rule-locked).
- [setuid(2)](https://man7.org/linux/man-pages/man2/setuid.2.html) — the
  irrevocable privilege drop in [section 5](#5-the-record-footage-command).
- [open(2) `O_NOFOLLOW`](https://man7.org/linux/man-pages/man2/open.2.html)
  — reading the request file without following a symlink.

Repo files referenced:

- [`AMC-WEB-PROJECTS.md`](AMC-WEB-PROJECTS.md) — folders, locking and the
  open decision this resolves.
- [`STEP-9-AMC-CONTROL.md`](STEP-9-AMC-CONTROL.md) — the sudo rule and its
  checks.
- [`STEP-4-CALIBRATION-FOOTAGE.md`](STEP-4-CALIBRATION-FOOTAGE.md) — the
  recorder, clip format and free-space rule.
- `installer/mv3dt_installer/footage.py`, `camera_credentials.py`,
  `cameras.py` — reused by the root command.
