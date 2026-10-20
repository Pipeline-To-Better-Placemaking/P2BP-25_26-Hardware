# STEP 5 — Start-or-close + per-project executables (owner: DevD)

Status: final step of the single-installer. This document specifies the
`step5_per_project_exes` module. It builds strictly on the contracts in
[`00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md) — the
step-module interface (§12), `Context` services, install-location config
(§11), the reporting/verify strings (§8.3–8.4), and the USER-ACTION display
(§9.3). Those contracts are **not** restated here; only Step 5's own scope is.

Step 5 is the "you're done — now use it" step. After Step 4 has rendered a
per-project DeepStream config and established `PROJECT_NAME` / `LOCATION_ID`,
Step 5 (a) offers the operator the final start-or-close choice, (b) generates
a **DeepStream pipeline executable named after the AMC project** in
`<install_dir>/bin/`, (c) records the project in a persistent **registry**,
and (d) exposes the reusable **AMC exe** flows for new/existing projects.

---

## 1. Inputs consumed and outputs produced

### 1.1 Inputs (from prior steps, via `Context`)

- `ctx.install_dir` / `ctx.conf` — the install root (default `/opt/mv3dt`),
  resolved through `config.load()` (framework §11). Never hardcoded.
- The **standalone AMC exe** dropped by Step 3 at `<install_dir>/bin/amc`
  (framework §12.4). Step 5 reuses this as the "run AMC" entry point — see
  [§5](#5-reusable-amc-exe-flows).
- The **rendered per-project DeepStream config** produced by Step 4 under
  `<install_dir>/deepstream/` (framework §12.4). This is the successor to the
  repo-tree `deepstream_app_config.rendered.txt` produced by
  [`laptop/scripts/40_export_watcher.sh`](../../laptop/scripts/40_export_watcher.sh).
- `PROJECT_NAME` and `LOCATION_ID` — established by Step 4. `PROJECT_NAME`
  is the human name the operator typed into the AMC calibration GUI
  (AMC "Project Setup" step); `LOCATION_ID` keys the calibration directory.
- The installer binary path itself (the frozen `mv3dt-installer`, framework
  §2), which the generated exes re-invoke with a subcommand.

### 1.2 Outputs (what Step 5 writes)

- `<install_dir>/bin/pipeline-<slug>` — the project-named DeepStream pipeline
  exe ([§3](#3-the-project-named-deepstream-exe)).
- `<install_dir>/bin/record-<slug>` — the project-named **tracking recorder**
  exe, and the sole owner of the bundled `60_record_tracking.sh`. It is
  generated in the same wrapper form as `pipeline-<slug>`
  ([§3.2](#32-generation-approach-locked-thin-generated-wrapper)) and runs the
  bundled script through the installer binary, passing the project's
  `LOCATION_ID` and writing under `<install_dir>/tracking_exports/`. The
  operator runs `record-<slug>`; they never type the script's name, and the
  script itself is never on any documented operator path. The artifacts it
  produces (`tracks.jsonl`, `tracks.csv`, `summary.json`) are what
  [`STEP-7` §E.1](STEP-7-WEBAPP-INTEGRATION.md#e1-what-gets-uploaded) uploads.
- `<install_dir>/projects/registry.json` — the project registry
  ([§4](#4-the-project-registry)), plus a per-project entry dir
  `<install_dir>/projects/<slug>/`.
- No new `state.json` writes — Step 5 returns a `StepResult` and the framework
  owns all state persistence (framework §12.2).

All files created for the operator are `chown`ed to the invoking user/group
via `ctx.run_as_user` bookkeeping (framework §9.2).

---

## 2. The final choice (start-or-close)

Product requirement: *"Once AMC is done and the file is chosen, the user can
either press to start the pipeline for the given AMC project, or just close
this exe."*

`run(ctx)` performs its generation work ([§3](#3-the-project-named-deepstream-exe),
[§4](#4-the-project-registry)) first, then presents a single terminal prompt
(this is the only "GUI-like" affordance, consistent with the TUI-first rule in
framework §2):

```
Installation complete for project: <PROJECT_NAME>
A pipeline executable was created: <install_dir>/bin/pipeline-<slug>

What next?
  [S] Start the DeepStream pipeline for <PROJECT_NAME> now
  [C] Close (you can start it later with the command above)
> _
```

### 2.1 Path S — start now

- The prompt shells out to the freshly-generated
  `<install_dir>/bin/pipeline-<slug>` (equivalent to running the project exe
  directly, [§3.3](#33-what-the-exe-does-at-runtime-pipeline-subcommand)). This `exec`s
  `deepstream-app`, so the installer process is replaced by the running
  pipeline — matching `50_start_pipeline.sh`'s terminal `exec` behavior.
- Because `run()` must return a `StepResult` to the framework, "start now"
  is implemented as: mark the step result `COMPLETE` and persist it *before*
  launching, then `os.execv` the pipeline exe as the final action. The
  installer's job is finished; the pipeline owns the terminal from here.
- `--non-interactive` / `--no-pause` (framework §3.3) default to **Close**
  (path C): generation is complete, no long-running pipeline is auto-started
  in an unattended run.

### 2.2 Path C — close

- Print the "how to start later" summary (the pipeline exe path, the
  `record-<slug>` recorder path, the `amc` exe path, and the stop command from
  [§3.4](#34-stopping-the-pipeline)) and return
  `StepResult(status=COMPLETE, ...)`. The dispatch loop then prints the
  final success banner (framework §6.3 `all_complete()`).

Either path leaves the same on-disk result: an AMC exe to run calibration and
a project-named DeepStream exe. That is the whole deliverable of Step 5.

---

## 3. The project-named DeepStream exe

The crux of Step 5. The completed artifact for a project is an executable in
`<install_dir>/bin/` **named after the AMC project**, that launches the full
DeepStream pipeline for that project's rendered config.

### 3.1 Naming and filename sanitization

- `PROJECT_NAME` is free-form operator text from the AMC GUI, so it MUST be
  sanitized into a safe filename `slug`:
  - lower-case; replace any run of non-`[a-z0-9]` chars with a single `-`;
    strip leading/trailing `-`; collapse repeats; truncate to 64 chars.
  - if the result is empty (e.g. name was all punctuation), fall back to the
    `LOCATION_ID`, then to `project`.
- Exe name: **`pipeline-<slug>`** (e.g. project `"North Lobby #2"` →
  `<install_dir>/bin/pipeline-north-lobby-2`). The `pipeline-` prefix keeps
  the `bin/` dir self-describing and avoids collisions with the reserved
  `amc` and `mv3dt-installer` names.
- The original, unsanitized `PROJECT_NAME` is preserved in the registry
  ([§4](#4-the-project-registry)) so the slug never has to be reversed.
- Collision policy: if `pipeline-<slug>` already exists for a *different*
  registry `PROJECT_NAME`, append `-<LOCATION_ID>`; if still colliding, this
  is a `FAILED` result asking the operator to pick a distinct project name.

### 3.2 Generation approach (LOCKED: thin generated wrapper)

The exe is a **thin generated wrapper script** that re-invokes the single
installer binary with a `pipeline` subcommand. It is **not** a second
PyInstaller build (that would re-package megabytes per project) and **not** a
copy of the bash logic (that would fork `50_start_pipeline.sh` maintenance).
This mirrors the framework's own hint that the Step 3 `amc` exe is "the same
PyInstaller binary invoked with an `amc` subcommand, or a thin generated
wrapper" (framework §12.4) — Step 5 uses the wrapper form for symmetry.

Generated file `<install_dir>/bin/pipeline-<slug>` (`chmod 0755`, chowned to
the invoking user):

```bash
#!/usr/bin/env bash
# Auto-generated by mv3dt-installer Step 5 (step5_per_project_exes).
# Project: <PROJECT_NAME>   LOCATION_ID: <LOCATION_ID>
# Do not edit; regenerate by re-running the installer or the AMC re-run flow.
exec "<install_dir>/bin/mv3dt-installer" pipeline --project "<PROJECT_NAME>" "$@"
```

- The installer binary itself is placed/symlinked at
  `<install_dir>/bin/mv3dt-installer` by the framework so the wrapper has a
  stable absolute target that survives the repo clone being removed
  (framework §4.2: "the operator does not need the repo checkout at runtime").
- All pipeline logic lives **once**, in Python, behind the `pipeline`
  subcommand (§3.3). Multiple project exes are all tiny wrappers differing
  only in `--project`.

### 3.3 What the exe does at runtime (`pipeline` subcommand)

The `pipeline --project <NAME>` subcommand is a faithful Python port of
[`laptop/scripts/50_start_pipeline.sh`](../../laptop/scripts/50_start_pipeline.sh),
resolving the config from the registry rather than the repo tree:

1. **Resolve the project** — look up `<NAME>` in `registry.json`
   ([§4](#4-the-project-registry)) to get its `rendered_config` path and
   `location_id`. Fail fast with a clear message if the project is unknown
   (points the operator at the `amc` exe to create/calibrate it first).
2. **Ensure Mosquitto is up** — idempotent `systemctl start mosquitto`
   (port `ensure_mosquitto()` from `50_start_pipeline.sh` lines 112–132),
   using `ctx.run_root` for the privileged start.
3. **Ping-sweep the cameras** — read the discovered inventory named by
   **`CAMERAS_FILE`** in `installer.conf`
   ([`00` §11.2](00-FRAMEWORK-AND-BOOTSTRAP.md#112-persistence--sharing-with-later-steps))
   and ping each enabled entry; warn on misses, do not block (port lines
   137–196). The inventory is produced by camera discovery
   ([`00` §15](00-FRAMEWORK-AND-BOOTSTRAP.md#15-camera-discovery)); the
   `pipeline` subcommand never reads
   [`laptop/config/cameras.yml`](../../laptop/config/cameras.yml) and never
   compiles in a path. If `CAMERAS_FILE` is unset or missing, the sweep is
   skipped with a warning pointing at the discovery scan — a stale inventory
   must not block a pipeline start.

   > **RESOLVED — the RTSP check is not orphaned; it moved into `cameras.py`.**
   > The ping sweep here is deliberately shallow, and the distinction this note
   > has always drawn still holds: *a ping proves the host is up, not that it
   > serves a decodable RTSP stream.* What changed is where that stronger check
   > lives. The `ffprobe`-over-RTSP probe from
   > [`20_verify_cameras.sh`](../../laptop/scripts/20_verify_cameras.sh) is
   > ported into the camera discovery module
   > ([`00` §15](00-FRAMEWORK-AND-BOOTSTRAP.md#15-camera-discovery)), which
   > probes every discovered camera and records the result as a `stream_ok`
   > field on its inventory entry. The pass/fail table becomes that field plus
   > the scan record, read back from `CAMERAS_FILE` — so the guarantee is
   > preserved by a module the installer calls, not by a script an operator
   > has to remember to run. `20_verify_cameras.sh` is therefore no longer
   > load-bearing; it is retained in git as a developer tool. Closes gap 2 in
   > [`DELETION-REVIEW` §6](DELETION-REVIEW.md#6-coverage-gaps-this-triage-exposed).
4. **Source the DeepStream env** — `. /etc/profile.d/deepstream.sh` (written
   by Step 2) so `deepstream-app` and `DEEPSTREAM_DIR` are in the environment
   (port lines 201–210).
5. **Print the §10.2 validation helpers** ([§6](#6-validation--monitoring)),
   then `cd <install_dir>/deepstream` and
   `exec deepstream-test5-app -c <rendered_config>` (port lines 374–401).

   > **RESOLVED (2026-10-20, live workstation): the pipeline runs
   > `deepstream-test5-app`, not `deepstream-app`.** Both read the same config
   > format, but plain `deepstream-app` never attaches the per-object event
   > metadata the `[sink1]` msgconv/msgbroker sink publishes: it connected to
   > Mosquitto and sent nothing. `deepstream-test5-app` builds a message per
   > tracked object, and NVIDIA's MV3DT reference app is built on it. Its
   > process name is truncated to `deepstream-test` (15 characters), which is
   > what the stop path's `pgrep -x` / `pkill -x` match. Delivery to the
   > broker is still to be confirmed on the workstation. Remaining
   > `deepstream-app` mentions in this doc describe the original port and
   > read as `deepstream-test5-app`.
- **Flags** the subcommand honors, mirroring `50_start_pipeline.sh`:
  `--preview` (display-enabled config while keeping MQTT publish),
  `--config <path>` (override), `--skip-ping`, `--dry-run`. These are passed
  through the wrapper via `"$@"`.

> **Step 6 handoff (24/7 supervision).** The foreground `exec deepstream-app`
> behavior above is the default (remote supervision **off**). When
> [Step 6](STEP-6-REMOTE-SUPERVISION.md) is enabled, `pipeline --project <NAME>`
> instead delegates to `systemctl start mv3dt-pipeline@<slug>` (a supervised,
> boot-enabled, auto-restarting unit), and the foreground logic in §3.3 steps
> 2–5 moves behind a non-interactive `--service-exec` mode that Step 6's unit
> `ExecStart` calls. A `--foreground` debug escape retains the original inline
> behavior. See Step 6 for the unit design and the remote MQTT control path.

### 3.4 Stopping the pipeline

Port [`laptop/scripts/99_stop_all.sh`](../../laptop/scripts/99_stop_all.sh)
behavior as a `pipeline --project <NAME> --stop` mode (and surface it in the
start-or-close summary and validation banner):

- `deepstream-app` — `SIGTERM`, wait up to 5s, then `SIGKILL` (port
  `99_stop_all.sh` lines 62–78).
- Optionally `docker compose down` the AMC stack and `systemctl stop
  mosquitto` behind `--stop-all` (equivalent to `99_stop_all.sh` with no skip
  flags); default `--stop` only stops `deepstream-app` so a shared broker/AMC
  serving other projects is left running. The skip-flag shape mirrors
  `--no-deepstream` / `--no-amc` / `--no-mosquitto`.
- When [Step 6](STEP-6-REMOTE-SUPERVISION.md) is enabled, `--stop` maps to
  `systemctl stop mv3dt-pipeline@<slug>` instead of signalling `deepstream-app`
  directly, so the supervisor does not immediately restart it.

---

## 4. The project registry

A persistent registry is what lets multiple project exes coexist, lets the
`amc` exe re-run an existing project, and gives each `pipeline-<slug>` exe a
stable place to resolve its config.

### 4.1 Path

- **`<install_dir>/projects/registry.json`** (the `projects/` dir is already
  reserved by framework §11.2). Root-owned, `chmod 0644`; the directory
  `chmod 0755`. Written with the **shared atomic-write helper** specified in
  [`00` §6.3](00-FRAMEWORK-AND-BOOTSTRAP.md#63-api-statepy) — `tmp` →
  `json.dump` → `flush` → **`os.fsync`** → `os.replace`. The `fsync` is not
  optional: a power loss between `write` and `replace` otherwise yields a
  zero-length registry, which orphans every generated exe. Readers are
  correspondingly forgiving — a missing or malformed file yields the empty
  registry, never an exception. It is **separate** from `state.json`: the
  registry is per-project data owned by Step 5, not installer step-state.
- Per-project scratch/entry dir: `<install_dir>/projects/<slug>/` (may hold a
  copy of that project's `cameras.yml` and a back-reference to its
  calibration dir).

### 4.2 Schema

```json
{
  "schema_version": 1,
  "updated_utc": "2026-07-01T22:41:12Z",
  "projects": {
    "North Lobby #2": {
      "slug": "north-lobby-2",
      "location_id": "north-lobby-2-loc",
      "exe": "/opt/mv3dt/bin/pipeline-north-lobby-2",
      "rendered_config": "/opt/mv3dt/deepstream/north-lobby-2/deepstream_app_config.rendered.txt",
      "calibration_dir": "/opt/mv3dt/deepstream/calibration/north-lobby-2-loc",
      "cameras_yml": "/opt/mv3dt/projects/north-lobby-2/cameras.yml",
      "created_utc": "2026-07-01T21:03:55Z",
      "updated_utc": "2026-07-01T22:41:12Z",
      "calib_runs": 2
    }
  }
}
```

- Key: the **original** `PROJECT_NAME` (unsanitized), so display and AMC
  re-run round-trip exactly. `slug` holds the filesystem-safe form (§3.1).
- `rendered_config` / `calibration_dir` point at the Step 4 outputs under
  `<install_dir>/deepstream/`; `location_id` is Step 4's `LOCATION_ID`.
- `exe` is the absolute path of the generated wrapper (§3.2).
- `created_utc` set on first registration; `updated_utc` and `calib_runs`
  bumped on every re-run (§5.2).

### 4.3 Registry API (`step5_per_project_exes.py` internal)

- `load_registry(install_dir) -> Registry` / `save_registry(...)` — atomic.
- `upsert(project_name, location_id, rendered_config, calibration_dir,
  cameras_yml) -> Entry` — creates or updates, computes/validates the slug,
  bumps timestamps + `calib_runs`.
- `get(project_name) -> Entry | None`, `list() -> list[Entry]`.

---

## 5. Reusable AMC exe flows

The `amc` exe from Step 3 (`<install_dir>/bin/amc`) is the single entry point
for calibration. Step 5 defines how it feeds new and existing projects back
through Step 4's ingest/wiring and this step's exe generation. AutoMagicCalib
is a web-based, human-driven workflow (see
[`30_start_amc.sh`](../../laptop/scripts/30_start_amc.sh) and the DS 9.1
AutoMagicCalib doc); the exe only brings the operator to the UI and then
ingests the result.

The `amc` exe (or `mv3dt-installer amc`) presents a small menu:

```
AMC — Auto Magic Calibration
  [N] New project        — calibrate a brand-new project
  [R] Re-run existing    — recalibrate a project from the registry
  [L] List projects      — show registered projects and their exes
  [X] Remove             — reconcile/remove projects deleted in the AMC GUI (§5.4)
> _
```

The menu runs `reconcile_registry(ctx)` (§5.4) before rendering, so any project
already deleted in the AMC GUI is cleaned up and never shown as runnable.

### 5.1 Flow (a) — NEW project

1. `[N]` brings up the AMC UI exactly as Step 3 does (docker compose up +
   open `http://localhost:<UI_PORT>`; port of
   [`30_start_amc.sh`](../../laptop/scripts/30_start_amc.sh)). The operator
   completes the 6-step AMC workflow in the browser (Project Setup → Video
   Upload → Parameters → Manual Align → Execute → Results/Export) and defines
   a brand-new `PROJECT_NAME` in "Project Setup".
2. On export, control returns through **Step 4's wiring** (place/ingest the
   AMC MV3DT export, render `deepstream_app_config`, patch the tracker
   `calibrationDirectory` / `LOCATION_ID`) — Step 5 does not duplicate that
   logic, it invokes the Step 4 ingest entry point for the new project.
3. Step 5 then runs `upsert(...)` ([§4.3](#43-registry-api-step5_per_project_exespy-internal))
   and generates `<install_dir>/bin/pipeline-<slug>`
   ([§3](#3-the-project-named-deepstream-exe)) for the new project, then
   offers the same start-or-close choice ([§2](#2-the-final-choice-start-or-close)).

### 5.2 Flow (b) — RE-RUN existing

1. `[R]` lists projects from `registry.json` and lets the operator pick one.
   The AMC UI comes up pre-seeded with that project's existing
   `PROJECT_NAME` / `LOCATION_ID` (AMC keys calibration by project; AMC can
   re-calibrate from archived videos per the DS 9.1 AutoMagicCalib doc, so a
   re-run does not require a fresh capture).
2. On completion, the **new** calibration is re-ingested and re-wired through
   Step 4 for the *same* project (overwrites/updates that project's
   `rendered_config` and `calibration_dir`).
3. Step 5 `upsert(...)`s the same registry key (bumping `updated_utc` and
   `calib_runs`) and **regenerates** `pipeline-<slug>`. Because the wrapper
   only references `--project <NAME>`, its content is stable across re-runs;
   regeneration is idempotent and simply refreshes the header/timestamps.

Both flows converge on: re-ingest via Step 4 → `upsert` registry → (re)generate
project exe → start-or-close.

### 5.3 Multiple coexisting projects

- Every project gets its own `pipeline-<slug>` in `<install_dir>/bin/`; they
  coexist freely because each resolves its own config via the registry key.
- `[L] List projects` (and CLI `mv3dt-installer projects --list`) prints a
  table from `registry.json`: `PROJECT_NAME`, `slug`, `exe`, `location_id`,
  last-calibrated timestamp, `calib_runs`. This is how the operator discovers
  and manages the exes.
- Project removal is **AMC-GUI-driven** and reconciled locally — see
  [§5.4](#54-project-removal-amc-gui-driven--reconciliation-check). Projects are
  not deleted by fiat here; a project goes away only because the operator
  deleted/removed it in the AMC calibration GUI, and Step 5 then cleans up the
  matching local artifacts (exe, rendered config, calibration dir, registry
  entry).

### 5.4 Project removal (AMC-GUI-driven) + reconciliation check

Removal is driven by the operator deleting/removing a project in the AMC
calibration GUI — not by a destructive command in this installer. AMC owns the
project under `$HOME/auto-magic-calib/projects/<PROJECT_NAME>/` (the
`PROJECT_DIR` templated by [`30_start_amc.sh`](../../laptop/scripts/30_start_amc.sh)),
so that directory is the **source of truth** for whether a project still exists.

Expected happy path: when a project is removed in the AMC GUI, its
`projects/<PROJECT_NAME>/` directory is deleted by AMC. Step 5's job is to make
the local install-side artifacts follow: the registry entry, the
`<install_dir>/bin/pipeline-<slug>` exe, the rendered config under
`<install_dir>/deepstream/<slug>/`, the calibration dir under
`<install_dir>/deepstream/calibration/<LOCATION_ID>/`, and
`<install_dir>/projects/<slug>/`.

Because AMC's own deletion cannot know about those install-side files, Step 5
adds an explicit **reconciliation check** so removal is not silently missed:

1. `reconcile_registry(ctx)` — for each project in `registry.json`, test whether
   its AMC project dir (`$AMC_ROOT/projects/<PROJECT_NAME>/`) still exists.
   - Still present → no action.
   - **Absent → the project was removed in the AMC GUI.** Treat it as a removal:
     enumerate the install-side artifacts above and remove them, then drop the
     registry key (atomic write, framework §6.3). If
     [Step 6](STEP-6-REMOTE-SUPERVISION.md) is enabled, first
     `systemctl disable --now mv3dt-pipeline@<slug>` and remove its unit file so
     a removed project is not left supervised/boot-enabled. In interactive mode,
     list what will be deleted and require a confirm; under `--non-interactive` /
     `--reconcile --yes`, proceed and report each deletion via the framework
     reporters. Missing artifacts are logged and skipped (idempotent).
2. When it runs (any of):
   - at the start of the `amc` exe menu and on `[L] List projects`, so stale
     entries never show as runnable;
   - as an explicit `mv3dt-installer projects --reconcile` (and the manual
     escape hatch `projects --remove <PROJECT_NAME>` for the case where the
     operator wants to remove the local artifacts even though AMC still has the
     project);
   - inside `verify()` (§7.3) as a read-only drift check that reports, but does
     not delete, any registry entry whose AMC project dir is gone.

This directly implements the requirement: if AMC removing a project also removed
the file-directory artifacts, reconciliation is a no-op; if AMC did **not** (or
only removed its own `projects/` dir), the reconciliation check catches the
drift and completes the cleanup. Reconciliation never touches AMC's own data —
it only removes install-side artifacts once AMC has already dropped the project.

The `amc` menu (§5) gains an `[X] Remove` affordance that runs the same
reconciliation, and the menu render always reconciles first so removed projects
disappear from the list.

---

## 6. Validation / monitoring

Per [`SCRIPTED-WORKFLOW.md`](../../laptop/docs/SCRIPTED-WORKFLOW.md) §10.2 and
the tail of `50_start_pipeline.sh`, the `pipeline` subcommand prints the
validation helpers to run in a second TTY **before** handing off to
`deepstream-app`:

```
[validation helpers — SCRIPTED-WORKFLOW §10.2]
  # MV3DT/SV3DT tracks (topic base <MQTT_TOPIC_BASE>, default mv3dt):
  mosquitto_sub -h 127.0.0.1 -t 'mv3dt/#' -v

  # GPU utilization / memory / temperature:
  watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv'

  # Broker health:
  systemctl status mosquitto --no-pager

  # Stop this pipeline:
  <install_dir>/bin/pipeline-<slug> --stop
```

Expected signal (SCRIPTED-WORKFLOW §10.2): one payload per tracked object per
frame on `mv3dt/<LOCATION_ID>/sv3d`, and once MV3DT fuses across cameras, on
`mv3dt/<LOCATION_ID>/fused`.

---

## 7. Step lifecycle (`preflight` / `run` / `verify` / `report`)

Implements the `Step` protocol (framework §12.1). `id =
"step5_per_project_exes"`, `title = "Per-project executables"`, `order = 5`.

### 7.1 `preflight(ctx)`

- Confirm Step 4 is `COMPLETE` and its outputs exist: the rendered config
  under `<install_dir>/deepstream/` and a non-empty `PROJECT_NAME` /
  `LOCATION_ID`. Missing → `USER_ACTION_REQUIRED` pointing back at Step 4.
- Confirm the Step 3 `amc` exe exists at `<install_dir>/bin/amc`; missing →
  `USER_ACTION_REQUIRED` pointing back at Step 3.
- Confirm `deepstream-app` is resolvable after sourcing
  `/etc/profile.d/deepstream.sh` (framework `require_tool` equivalent).
- All good → `COMPLETE` (ok to run).

### 7.2 `run(ctx)`

- Generate `pipeline-<slug>` (§3), `upsert` the registry (§4), then present
  the start-or-close choice (§2). Returns `COMPLETE` on Close, or persists
  `COMPLETE` and `execv`s the pipeline on Start.
- Report artifacts touched via the framework reporters (§8.3), e.g.
  `installed pipeline-north-lobby-2 version 1` for a newly generated exe and
  `already installed pipeline-north-lobby-2 version 1` when regeneration is a
  no-op on re-run.

### 7.3 `verify(ctx)`

Idempotent post-checks; returns `COMPLETE` only when all pass, using the
framework `verify_pinned` / reporter strings (§8.3–8.4):

- The project-named exe exists, is a regular file, and is executable
  (`os.access(exe, X_OK)`), and its wrapper references the expected
  `--project <PROJECT_NAME>`.
- The `amc` exe exists and is executable.
- A `registry.json` entry exists for `PROJECT_NAME` and its `rendered_config`
  / `calibration_dir` paths exist on disk.
- Read-only removal drift check (§5.4): report (do not delete) any registry
  entry whose AMC project dir (`$AMC_ROOT/projects/<PROJECT_NAME>/`) is gone, so
  a stale entry is surfaced rather than silently passing.
- Optional dry-run smoke: `pipeline-<slug> --dry-run` prints the final
  `deepstream-app -c <config>` command without launching (mirrors
  `50_start_pipeline.sh --dry-run`).

### 7.4 `report(ctx)`

Prints the human-facing summary (no side effects): the generated exe path,
the `amc` exe path, the registry path + this project's entry, and the "start
later / stop" commands from §2/§3.4.

---

## 8. References

DeepStream 9.1 official documentation only.

- DS 9.1 `deepstream-app` reference (the `deepstream-app -c <config>` entry
  point the project exe drives):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html>
- DS 9.1 Quickstart (`deepstream-app -c <path_to_config_file>` invocation):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html>
- DS 9.1 MV3DT (multi-view 3D tracking pipeline + `mv3dt/<LOCATION_ID>/*`
  topics the pipeline publishes):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_MV3DT.html>
- DS 9.1 AutoMagicCalib (project-based web workflow; MV3DT-compatible export;
  re-calibration from archived videos — basis for the re-run flow §5):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_AutoMagicCalib.html>

Repo files referenced (ported/reused by this step):

- [`laptop/scripts/50_start_pipeline.sh`](../../laptop/scripts/50_start_pipeline.sh)
  — pipeline launcher logic ported into the `pipeline` subcommand (§3.3).
- [`laptop/scripts/30_start_amc.sh`](../../laptop/scripts/30_start_amc.sh)
  — AMC bring-up reused by the `amc` exe flows (§5).
- [`laptop/scripts/99_stop_all.sh`](../../laptop/scripts/99_stop_all.sh)
  — teardown logic ported into the `--stop` / `--stop-all` modes (§3.4).
- [`laptop/config/cameras.yml`](../../laptop/config/cameras.yml)
  — camera inventory format; the ping sweep consumes the **generated**
  inventory named by `CAMERAS_FILE`, not this committed file (§3.3).
- [`laptop/scripts/60_record_tracking.sh`](../../laptop/scripts/60_record_tracking.sh)
  — the tracking recorder bundled into the installer binary and owned by the
  generated `record-<slug>` exe (§1.2).
- [`laptop/scripts/20_verify_cameras.sh`](../../laptop/scripts/20_verify_cameras.sh)
  — origin of the `ffprobe`-over-RTSP check, now carried by the camera
  discovery module's `stream_ok` probe (§3.3).
- [`laptop/docs/SCRIPTED-WORKFLOW.md`](../../laptop/docs/SCRIPTED-WORKFLOW.md)
  — §10 startup sequence and §10.2 validation helpers (§6).

---

## 9. Out of scope / flag for human

- **systemd supervision + remote control** of the per-project pipelines is NOW
  IN SCOPE and specified in [Step 6](STEP-6-REMOTE-SUPERVISION.md): the pipeline
  runs 24/7 as a supervised `mv3dt-pipeline@<slug>.service` unit, remotely
  run/stop/restarted via JSON commands over MQTT. Step 5 hands off to it (§3.3
  `--service-exec`, §3.4 stop mapping, §5.4 disable-on-removal). When Step 6 is
  disabled, Step 5's foreground-exe behavior is the default.
- Project **removal** is handled by the AMC-GUI-driven reconciliation check in
  [§5.4](#54-project-removal-amc-gui-driven--reconciliation-check).
- **Per-camera network/RTSP configuration** — the pipeline only consumes the
  inventory at `CAMERAS_FILE`; camera activation, OSD-disable, and stream
  profile stay manual (framework §13). Finding the cameras is **not** manual:
  that is [`00` §15](00-FRAMEWORK-AND-BOOTSTRAP.md#15-camera-discovery).
- **Alternate detectors** — PeopleNet-only, matching the DS 9.1 MV3DT
  reference; `yolo11n` remains future work (framework §13).
- **Shipping per-project artifacts to the web app** — the tracking exports,
  plots, and clips a running project produces are uploaded by
  [Step 7](STEP-7-WEBAPP-INTEGRATION.md), which keys its upload prefixes and
  its status payload off this step's `registry.json`
  ([`STEP-7` §E.1](STEP-7-WEBAPP-INTEGRATION.md#e1-what-gets-uploaded)). Step 5
  produces the artifacts; it does not transmit them.
