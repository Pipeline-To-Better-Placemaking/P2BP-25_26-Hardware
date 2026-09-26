# Step 4 — Calibration Output Placement and Config Wiring (owner: DevD)

Status: step spec. This document depends on the shared framework in
[`00` §12](00-FRAMEWORK-AND-BOOTSTRAP.md#12-step-module-interface-the-contract-for-steps-15)
and the AMC launcher contract in
[`STEP-3` §6](STEP-3-AMC-LAUNCHER.md#6-project-identity-and-step-4-contract).
It does **not** restate their state, privilege, or project-creation contracts.

Step 4 consumes the **persisted AMC 3.2.1 project identity**, waits for that
project to finish, downloads its MV3DT result through the local API, and wires
the result into the DeepStream configuration. It supersedes the operator-side
filesystem watcher in `laptop/scripts/40_export_watcher.sh`; a workstation
does not need a repository checkout or a manually exported directory.

---

## 1. Scope and identity

Module:
`installer/mv3dt_installer/steps/step4_calib_output_wiring.py`, registered as
`step4_calib_output_wiring` with `order = 4`.

**LOCKED:** Step 4 uses the AMC API contract produced by
[`STEP-3` §6](STEP-3-AMC-LAUNCHER.md#6-project-identity-and-step-4-contract).
It never guesses a project by name and never watches the obsolete
`$AMC_ROOT/projects/$PROJECT_NAME/exports/` path.

In scope:

1. **Resolve identity**: consume `LOCATION_ID`, `PROJECT_NAME`,
   `AMC_PROJECT_ID`, and `AUTO_MAGIC_CALIB_MS_PORT` from `installer.conf`.
2. **Wait for completion**: poll the persisted project through the local AMC
   API while preserving bounded, Ctrl-C, and non-interactive behavior.
3. **Ingest safely**: download the MV3DT ZIP, validate its structure, and
   atomically replace the calibration tree.
4. **Render configuration**: patch the tracker YAML and render camera RTSP
   sources into the DeepStream application config.
5. **Re-ingest later results**: install a timer and one-shot service that use
   the same API-based `ingest` subcommand.

Out of scope:

- Creating or selecting an AMC project — owned by
  [`STEP-3` §6](STEP-3-AMC-LAUNCHER.md#6-project-identity-and-step-4-contract).
- Driving the browser calibration workflow.
- Launching the DeepStream pipeline — owned by Step 5.
- Parsing calibration matrices or judging calibration quality.

---

## 2. Required inputs

The following values are **REQUIRED** before polling begins:

| Key | Source | Purpose |
|---|---|---|
| `LOCATION_ID` | Step 3 | calibration directory and MV3DT node identity |
| `PROJECT_NAME` | Step 3 | operator-facing label and unit slug |
| `AMC_PROJECT_ID` | Step 3 API response | exact AMC API resource |
| `AUTO_MAGIC_CALIB_MS_PORT` | Step 3 | localhost API port |
| `CAM_USER` | first-run camera credential capture | rendered RTSP credentials |
| `CAM_PASSWORD` | first-run no-echo camera credential capture | rendered RTSP credentials |
| `CAMERAS_FILE` | [`00` §15](00-FRAMEWORK-AND-BOOTSTRAP.md#15-camera-discovery) | enabled camera inventory |
| `AMC_EXPORT_WAIT_S` | optional, default `3600` | bounded interactive wait |

Missing required configuration is reported in one consolidated
`USER_ACTION_REQUIRED` result. Missing Step 3-owned values
direct the operator back through the automated Step 3 flow; they must never be
invented by editing `AMC_PROJECT_ID` or an API port manually. Camera
credentials are captured and stored securely during onboarding. When the
inventory is missing, Step 4 automatically runs camera discovery, RTSP
validation, and one-time guided position binding. If no camera is found, the
operator is asked to connect and activate the cameras and rerun the installer.

The standalone mode remains the explicit refresh path after initial setup:

```bash
sudo mv3dt-installer --scan-cameras
```

The operator is never asked to hand-write camera inventory YAML. A normal
run receives these values from prior installer collection and Step 3 project
creation.

---

## 3. Project-state polling

### 3.1 Status endpoint

Step 4 polls:

```text
GET http://localhost:<AUTO_MAGIC_CALIB_MS_PORT>/v1/get_project_info/<AMC_PROJECT_ID>
```

The response must be HTTP-successful JSON and contain exactly the documented
`project_info.project_state` shape. A top-level `project_state`, an alternate
`state` field, or a non-object `project_info` is invalid. State comparisons are
case-insensitive after normalization to uppercase.

| State | Behavior |
|---|---|
| `COMPLETED` | continue directly to result download |
| `ERROR` | fetch the calibration log and fail with bounded evidence |
| any other state | continue polling until the wait ends |

**REQUIRED — quiet polling.** The status request above and the calibration-log
request in [§3.2](#32-error-evidence) pass `stream=False` to `ctx.run_root`.
Their JSON replies are parsed, never echoed into the live window — a streamed
`{"code":0,"message":"Project info retrieved successfully",...}` reply every
few seconds would bury the wait description and the AMC UI hint for as long as
calibration takes. Before the completion wait starts, Step 4 sets
`ctx.progress.task("waiting for AMC project <PROJECT_NAME> to complete")` so
the live task line no longer reads "resolving AMC project and camera inputs"
while it waits. State changes are still logged once each.

### 3.2 Error evidence

When state is `ERROR`, Step 4 fetches:

```text
GET http://localhost:<AUTO_MAGIC_CALIB_MS_PORT>/v1/amc/calibrate/<AMC_PROJECT_ID>/log
```

The step returns `FAILED` and includes a bounded excerpt of the log. A failed
log request is itself included as evidence; an AMC error must never degrade
into another hour-long wait.

### 3.3 Wait outcomes

The shared wait helper from
[`00` §9](00-FRAMEWORK-AND-BOOTSTRAP.md#9-privilege-boundary-root-versus-invoking-user)
owns timing and cancellation.

| Outcome | Cause | Step result |
|---|---|---|
| `SATISFIED` | state reached `COMPLETED` | continue to `run()` |
| `TIMEOUT` | `AMC_EXPORT_WAIT_S` elapsed | `USER_ACTION_REQUIRED` |
| `CANCELLED` | operator pressed Ctrl-C | `USER_ACTION_REQUIRED` |
| `SKIPPED` | `--non-interactive` | immediate `USER_ACTION_REQUIRED` |

The hint names the persisted project and the AMC UI URL. It does not instruct
the operator to export a file or run a watcher script.

---

## 4. Result download and validation

### 4.1 Result endpoint

After `COMPLETED`, Step 4 performs a bounded localhost download:

```text
GET http://localhost:<AUTO_MAGIC_CALIB_MS_PORT>/v1/result/<AMC_PROJECT_ID>/mv3dt_result?result_type=amc
```

The response is written to a temporary file beside the final calibration
directory. HTTP failures, timeouts, missing output, and empty output are
`FAILED`; none may alter an existing working calibration.

### 4.2 ZIP safety

**REQUIRED:** validation happens before the destination is touched.

The archive is rejected when it contains any of the following:

- an absolute path, parent traversal, backslash, colon, Windows drive prefix,
  or duplicate path;
- a symbolic link, unsupported file type, or encrypted member;
- more than `4096` members or more than `1 GiB` expanded content;
- no root-level `transforms.yml`.

`transforms.yml` is the documented AMC 3.2.1 MV3DT result contract. The old
heuristic requiring one `camInfo` file per enabled camera is removed; it could
reject a valid current result.

Extraction copies regular files member by member. It does not call
`ZipFile.extractall()`.

### 4.3 Atomic replacement and idempotency

The archive is extracted into a sibling staging directory. Only after full
validation and extraction does Step 4 rename the existing destination aside
and atomically rename the staging payload into place. If the final rename
fails, the previous destination is restored.

The installed tree records:

| File | Purpose |
|---|---|
| `transforms.yml` | required AMC transform output |
| `.archive.sha256` | idempotent result identity |
| `.ingest.log` | UTC ingest project breadcrumb |

An archive with the same SHA-256 as the installed result is an idempotent
no-change. Stale files from a previous result cannot survive a successful
replacement, and invalid or partial results cannot replace a working tree.

---

## 5. Calibration destination

The default destination is:

```text
<install_dir>/deepstream/calibration/<LOCATION_ID>/
```

An interactive first run offers that default in the standard path prompt.
`--non-interactive` and the `ingest` subcommand use the default without
prompting. Once chosen, the absolute destination is persisted as
`CALIBRATION_DIR`; every later ingest reuses it.

Files are owned by the invoking user in accordance with
[`00` §9](00-FRAMEWORK-AND-BOOTSTRAP.md#9-privilege-boundary-root-versus-invoking-user).

---

## 6. DeepStream configuration wiring

### 6.1 Tracker YAML

Step 4 copies and patches the bundled `config_tracker_NvMOT.yml`:

| Field | Value |
|---|---|
| `SV3DT.calibrationDirectory` | `calibration/<LOCATION_ID>` for the default, otherwise the absolute destination |
| `MV3DT.nodeID` | `LOCATION_ID` |

The following equality checks remain unchanged:

| Field | Required value |
|---|---|
| `SV3DT.projectionType` | `homography` |
| `MV3DT.mqttBrokerIP` | `127.0.0.1` |
| `MV3DT.mqttBrokerPort` | `1883` |

### 6.2 Application config

`deepstream_app_config.rendered.txt` is regenerated from the bundled
`deepstream_app_config.txt`. It substitutes `CAM_USER`, `CAM_PASSWORD`, and
`LOCATION_ID`, then rewrites each enabled `[sourceN]` URI from `CAMERAS_FILE`
in inventory order.

The rendered config must keep relative references to
`config_tracker_NvMOT.yml`, `config_infer_primary.txt`, and
`msgconv_config.txt`. The latter two are copied verbatim beside it.

---

## 7. Later recalibration

**LOCKED:** the obsolete filesystem `.path` unit is replaced by API polling.
After the first successful ingest, Step 4 installs:

| Unit | Role |
|---|---|
| `mv3dt-ingest-<slug>.timer` | runs two minutes after boot and every 60 seconds thereafter |
| `mv3dt-ingest-<slug>.service` | invokes the one-shot non-interactive `ingest` subcommand |

The timer alone is enabled. The service performs one status request:

- a non-`COMPLETED` state exits successfully without downloading;
- `ERROR` logs calibration evidence and exits nonzero;
- `COMPLETED` downloads and ingests, with the SHA-256 guard making an
  unchanged result a no-op.

The service command is:

```text
<install_dir>/bin/mv3dt-installer ingest --project <PROJECT_NAME> --non-interactive --install-dir <install_dir>
```

This preserves automatic later re-ingest without watching a directory AMC
3.2.1 does not create.

On upgrade, Step 4 disables and removes only the matching legacy
`mv3dt-ingest-<slug>.path` unit before enabling the timer. A failure to disable
that unit is fatal, and unrelated projects' units are never removed.

---

## 8. Verification and failure surfaces

`verify()` returns `COMPLETE` only when:

- `CALIBRATION_DIR/transforms.yml` exists;
- the tracker config parses and every pinned field matches;
- the rendered app config contains no placeholder and references the tracker;
- every enabled camera source has a concrete `rtsp://` URI;
- the referenced PeopleNet and message-converter configs exist;
- both re-ingest units exist, the timer alone is enabled, and the obsolete
  matching `.path` unit is neither installed nor enabled.

| Failure | Result |
|---|---|
| Missing Step 3 identity | consolidated `USER_ACTION_REQUIRED` before waiting |
| Missing camera inventory | consolidated camera-discovery action |
| Status or result HTTP error | `FAILED` with curl evidence |
| AMC state `ERROR` | `FAILED` with calibration-log evidence |
| Timeout, Ctrl-C, or non-interactive wait | `USER_ACTION_REQUIRED` |
| Unsafe or incomplete ZIP | `FAILED`; prior calibration preserved |
| Render or verification mismatch | `FAILED` with the affected field/path |

No Step 4 failure requires reinstalling the NVIDIA driver, CUDA, cuDNN,
TensorRT, DeepStream, or the PeopleNet model.

---

## 9. Developer quick-reference

- [ ] `RUNNING` transitions to `COMPLETED` without restarting the installer.
- [ ] `ERROR` fetches and reports the calibration log immediately.
- [ ] Interactive wait is bounded and Ctrl-C is recoverable.
- [ ] Non-interactive preflight never waits for a human.
- [ ] HTTP and JSON failures preserve the prior calibration.
- [ ] Traversal, symlink, unsupported, encrypted, and oversized ZIPs fail.
- [ ] A root-level `transforms.yml` is required.
- [ ] Repeat ingest of the same archive is a no-change.
- [ ] Rendering and pinned tracker checks pass.
- [ ] The timer exists; no obsolete export-directory path unit is installed.
- [ ] An upgrade disables and removes only its matching legacy path unit.

---

## 10. Out of scope

- VGGT-specific result selection; the current request pins `result_type=amc`.
- Background services beyond the local systemd timer and one-shot ingest.
- Automatic remediation of a calibration algorithm failure.
- DeepStream runtime launch and pipeline validation, owned by Step 5.

---

## 11. Implementation decomposition

| Unit | Branch | Files touched | Depends on | Wave |
|---|---|---|---|---|
| Step 4 AMC API ingest | `feat/installer-step4-amc-api-ingest` | `installer/mv3dt_installer/steps/step4_calib_output_wiring.py`, `installer/tests/test_step4_calib_output_wiring.py`, `installer/plan/STEP-4-CALIB-OUTPUT-WIRING.md` | Step 3 AMC hardening | 2 |

This unit is serialized after `feat/installer-step3-amc-hardening` because it
imports and consumes Step 3's persisted project-id and microservice-port
contract. Its pull request therefore targets that dependency branch.

---

## References

NVIDIA's pinned AMC 3.2.1 repository is the authority for project state and
MV3DT result retrieval.

- [AutoMagicCalib 3.2.1 repository](https://github.com/NVIDIA-AI-IOT/auto-magic-calib/tree/0cfd2b790fd77598b0543340a65c2a0e1d192327) — pinned application source.
- [AMC video-calibration skill](https://github.com/NVIDIA-AI-IOT/auto-magic-calib/blob/0cfd2b790fd77598b0543340a65c2a0e1d192327/skills/amc-run-video-calibration/SKILL.md) — project status, calibration log, and MV3DT result endpoints.
- [Python `zipfile` documentation](https://docs.python.org/3/library/zipfile.html) — archive metadata and member-level extraction behavior.

Repo files referenced:

- [`installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md) — shared step, wait, privilege, and camera-discovery contracts.
- [`installer/plan/STEP-3-AMC-LAUNCHER.md`](STEP-3-AMC-LAUNCHER.md) — pinned AMC version and persisted project identity.
- [`laptop/scripts/40_export_watcher.sh`](../../laptop/scripts/40_export_watcher.sh) — superseded filesystem-export watcher.
- [`laptop/deepstream/config_tracker_NvMOT.yml`](../../laptop/deepstream/config_tracker_NvMOT.yml) — tracker template and pinned fields.
- [`laptop/deepstream/deepstream_app_config.txt`](../../laptop/deepstream/deepstream_app_config.txt) — application template.
