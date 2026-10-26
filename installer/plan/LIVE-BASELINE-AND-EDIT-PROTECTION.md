# Live baseline and edit protection — adopt the tuned workstation state (owner: DevD)

Status: cross-step design spec, **specified, not implemented, never run on
the workstation**. It amends the Step 4 rendering and re-ingest contracts in
[`STEP-4` section 6](STEP-4-CALIB-OUTPUT-WIRING.md#6-deepstream-configuration-wiring)
and [`STEP-4` section 7](STEP-4-CALIB-OUTPUT-WIRING.md#7-later-recalibration),
the AMC pin in [`STEP-3` section 2](STEP-3-AMC-LAUNCHER.md#2-upstream-pin),
and the pipeline unit in
[`STEP-6` section A.1](STEP-6-REMOTE-SUPERVISION.md#a1-the-templated-per-project-instance-unit).
It builds on the gate mechanism in
[`00` section 3.4](00-FRAMEWORK-AND-BOOTSTRAP.md#34-opt-in-step-gates) and
the redaction rule in
[`00` section 14.4](00-FRAMEWORK-AND-BOOTSTRAP.md#144-redaction-required).
Those contracts are **not** restated here; only the changes are.

The workstation pipeline runs today only because of hand edits made after
install: multi-view fusion sections in the tracker config,
`useBatchNumForFrameId: 0`, a longer shadow age, the BodyPose3DNet pose
estimator, a 0.4 detector threshold, per-camera dewarpers, the MV3DT pub/sub
files, a VLC viewer stream, two systemd drop-ins, a hand-written watchdog and
an AMC 3.3.0 calibration. A live snapshot of the installed tree, diffed
against the bundled templates, shows **every one of those edits is missing
from the installer**. This doc makes them the installer's baseline, so a new
release reproduces today's working pipeline on a fresh install.

It also closes the hazard that made the gap dangerous: the installer
**overwrites rendered files unconditionally**, and the per-project ingest
timer re-renders them every 60 seconds. After this work, **upgrading never
silently overwrites an operator edit**: an edited file is kept, and the
installer's version is written beside it for review.

---

## 1. Scope and the problem

### 1.1 Unconditional overwrite

`_write_text_report()` in
`installer/mv3dt_installer/steps/step4_calib_output_wiring.py` compares the
new text with the file only to choose between "installed" and "already
installed" in the report. It then writes the file in every case. Both
`_write_camera_models()` and `_render_configs()` go through it, so every
Step 4 render replaces the tracker config, the inference config, the
rendered application config and the camera models with the bundled
templates, discarding any edit.

### 1.2 The 60-second re-render

The ingest timer from
[`STEP-4` section 7](STEP-4-CALIB-OUTPUT-WIRING.md#7-later-recalibration)
runs `mv3dt-installer ingest` every 60 seconds. `handle_ingest_subcommand()`
calls `_wire_download()`, which re-renders every config and camera model
after the download, even when the archive SHA-256 guard of
[`STEP-4` section 4.3](STEP-4-CALIB-OUTPUT-WIRING.md#43-atomic-replacement-and-idempotency)
found the result unchanged. The guard protects the calibration tree, not the
configs.

### 1.3 The timer is still enabled on the workstation

On the live box the ingest timer is **enabled** and inactive only because it
was stopped by hand. A reboot starts it, and the first run wipes the tuning
of [section 3](#3-pinned-baseline-values). Until a release carrying this spec
is installed, the protection is the one-time command in
[section 8.1](#81-before-any-release-required).

In scope:

1. Edit protection for every file the installer renders
   ([section 4](#4-managed-files-edit-protection)).
2. The tuned values as the template baseline
   ([section 3](#3-pinned-baseline-values)).
3. Per-install generated files: fusion pub/sub, MQTT adaptor config,
   dewarpers, viewer sections, pose model
   ([section 5](#5-generated-per-install-files)).
4. AMC 3.3.0 as the pinned calibration stack
   ([section 2.3](#23-amc-330-pin-locked)).
5. Runtime and systemd changes: two new gates, pipeline logging, a watchdog
   ([section 6](#6-runtime-and-systemd)).
6. The documented migration of the live workstation
   ([section 8](#8-live-migration)).

---

## 2. Decisions

### 2.1 Dewarper from AMC rectification (LOCKED)

**LOCKED: dewarper files are generated from AMC 3.3.0's committed
rectification**, never hand-entered. Step 4 reads:

```text
GET /v1/rectification/{AMC_PROJECT_ID}/cameras
```

Verified live: each camera entry carries `committed: {model: radial, k1, k2}`
and the principal point `cx`, `cy` with `width` and `height`. AMC's radial
model is in pixel units; `nvdewarper` projection type 3 takes coefficients
normalized by the focal length. With the fixed focal length `f = 1600`:

| `nvdewarper` key | Value |
|---|---|
| `projection-type` | `3` |
| `focal-length` | `1600` |
| `distortion` | `k1*f^2;k2*f^4;0;0;0` |
| `src-x0` | `cx` |
| `src-y0` | `cy` |
| `width`, `height`, `output-width`, `output-height` | the camera's `width`, `height` |

Worked example from the live calibration: `k1 = -3.2126e-07` gives
`k1*1600^2 = -0.82243`, and `k2 = 1.0097e-13` gives `k2*1600^4 = 0.66173`,
the live `distortion=-0.82243;0.66173;0;0;0`. The conversion was verified to
match AMC's own rectification within 0.1 px on DeepStream 9.1. A camera
without a committed rectification gets no dewarper file and no
`[dewarper<N>]` section.

### 2.2 Two new gates (LOCKED)

**LOCKED: both default `off`.** They are behavior gates, not step gates: no
step is skipped. They are seeded and persisted exactly as in
[`00` section 3.4](00-FRAMEWORK-AND-BOOTSTRAP.md#34-opt-in-step-gates).

| Gate key | Values (default first) | CLI flag | Effect when `on` |
|---|---|---|---|
| `MV3DT_AUTO_INGEST` | `off` \| `on` | `--auto-ingest` | Step 4 enables `mv3dt-ingest-<slug>.timer`; when `off` it installs the units but disables the timer, and re-ingest is the manual `mv3dt-installer ingest` |
| `MV3DT_VIEWER_STREAM` | `off` \| `on` | `--viewer-stream` | Step 4 renders the viewer sections ([section 5.3](#53-viewer-stream)); `pipeline --service-exec` adds `-t` |

`MV3DT_AUTO_INGEST=off` amends
[`STEP-4` section 7](STEP-4-CALIB-OUTPUT-WIRING.md#7-later-recalibration),
which enables the timer unconditionally.

### 2.3 AMC 3.3.0 pin (LOCKED)

**LOCKED: AMC comes from the `NVIDIA/DeepStream` repository at tag
`v9.2.0`, sparse path `tools/auto-magic-calib`.** This reverses the
standalone-repository decision of
[`STEP-3` section 2](STEP-3-AMC-LAUNCHER.md#2-upstream-pin): the standalone
repository stops at 3.2.1, and only 3.3.0 has the rectification API of
[section 2.1](#21-dewarper-from-amc-rectification-locked).

| Value | Pin |
|---|---|
| Repository | `https://github.com/NVIDIA/DeepStream.git` |
| Tag | `v9.2.0`, recorded as its resolved commit |
| Sparse path | `tools/auto-magic-calib` |
| MS image | `nvcr.io/nvidia/vss-core/vss-auto-calibration:3.3.0` |
| UI image | `nvcr.io/nvidia/vss-core/vss-auto-calibration-ui:3.3.0` |
| Default `AMC_ROOT` | `<invoking user home>/amc-3.3.0/tools/auto-magic-calib` |

The DeepStream pipeline itself stays on 9.1; only the AMC stack moves. Step 4
ingest keeps `result_type=mv3dt_result`: the 3.3.0 ZIP layout was verified
identical (`camInfo/`, `transforms.yml`, `map.png`).

---

## 3. Pinned baseline values

**REQUIRED:** the bundled templates under `laptop/deepstream/` (packaged by
`installer/installer.spec`) carry these exact values, taken from the live
snapshot. "Template" is the value shipped today.

### 3.1 Tracker (`config_tracker_NvMOT.yml`)

| Key | Template | Baseline |
|---|---|---|
| `BaseConfig.useBatchNumForFrameId` | absent | `0` |
| `TargetManagement.minIouDiff4NewTarget` | `0.3686` | `0.2` |
| `TargetManagement.maxShadowTrackingAge` | `42` | `75` |
| `ObjectModelProjection.minPoseConfidence` | `0.925` | `0.5` |
| `ObjectModelProjection.cameraModelFilepath` | `- ${CAMERA_MODEL_FILEPATH}` | unchanged placeholder; Step 4 expands it to one path per camera |
| `PoseEstimator.poseEstimatorType` | `0` | `1` |
| `PoseEstimator.onnxFile` | SDK `samples/models/Tracker/` path | `<install_dir>/deepstream/models/bodypose3dnet/bodypose3dnet_accuracy.onnx` |
| `PoseEstimator.modelEngineFile` | SDK `samples/models/Tracker/` path | `<install_dir>/deepstream/models/bodypose3dnet/bodypose3dnet_accuracy.onnx_b1_gpu0_fp16.engine` |
| `PoseEstimator.poseInferenceInterval` | `-1` | `5` |

New `MultiViewAssociator:` section, every key exactly as live:

| Key | Value |
|---|---|
| `multiViewAssociatorType` | `1` |
| `enableLatePeerReAssoc` | `1` |
| `enableIDCorrection` | `1` |
| `enableSeeThrough` | `1` |
| `enableMsgSync` | `0` |
| `maxPeerTrackletSize` | `50` |
| `recentlyActiveAge` | `178` |
| `minCommonFrames4MatchScore` | `2` |
| `maxPeerToPredDistance4Fusion` | `2.0` |
| `minPeerVisibility4Fusion` | `0.15` |
| `minPeerTrackletMatchScore` | `0.35` |
| `maxTrackletMatchingTimeSearchRange` | `1` |
| `maxPeerFrameDiff4NoDet` | `2` |
| `communicatorInitSleepTime` | `0` |

New `Communicator:` section:

| Key | Value |
|---|---|
| `communicatorType` | `2` (MQTT) |
| `pubSubInfoConfigPath` | `<install_dir>/deepstream/mv3dt/pub_sub_info_config.yml` |
| `mqttProtoAdaptorConfigPath` | `<install_dir>/deepstream/mv3dt/config_mqtt.txt` |

The template header comment that says fusion is not configured and that
`poseEstimatorType` is `0` is rewritten to match. Every other key keeps
NVIDIA's reference value, as
[`STEP-4` section 6.1](STEP-4-CALIB-OUTPUT-WIRING.md#61-tracker-yaml) records.

### 3.2 Inference and application configs

| File | Key | Template | Baseline |
|---|---|---|---|
| `config_infer_primary.txt` | `pre-cluster-threshold` | `0.2` | `0.4` |
| `deepstream_app_config.txt` | `[tracker] display-tracking-id` | absent | `1` |

The viewer sections (`[tiled-display]` enabled, `[osd]`, `[sink2]`) are
**not** template defaults; the renderer injects them only when
`MV3DT_VIEWER_STREAM=on` ([section 5.3](#53-viewer-stream)).

**REQUIRED:** the header comment of `deepstream_app_config.txt` must not
contain the `${CAM_USER}` or `${CAM_PASSWORD}` placeholders. The renderer
substitutes every occurrence, so today the rendered config carries the
camera credentials in a comment as well as in the source URIs.

`tests/test_deepstream_templates.py` and the Step 4 `verify()` pins change
with these values: `poseEstimatorType` becomes `1`, and the assertion that no
MV3DT section exists is replaced by one that requires `MultiViewAssociator`
and `Communicator`.

---

## 4. Managed files (edit protection)

### 4.1 Manifest (LOCKED)

**LOCKED: dpkg-conffile behavior.** New module
`installer/mv3dt_installer/managed_files.py` owns a manifest at:

```text
/var/lib/mv3dt-installer/managed-files.json
```

It maps each absolute path the installer wrote to the SHA-256 of the exact
content it last wrote:

```json
{
  "version": 1,
  "files": {
    "<install_dir>/deepstream/config_tracker_NvMOT.yml": "<sha256>"
  }
}
```

The manifest is written atomically (temporary file plus rename), mode `644`,
owned by root.

### 4.2 `write_managed(path, text)` rules (REQUIRED)

| Current state of `path` | Action | Reported as |
|---|---|---|
| Missing | write, record hash | installed |
| Hash equals the recorded hash (untouched since the installer wrote it) | write, record hash | installed, or already installed when the text is equal |
| Content already equals `text` | record hash only | already installed |
| Anything else: edited by the operator, or present with no manifest entry (a pre-manifest install, such as the live workstation) | leave `path` untouched, write `<path>.installer-new`, log a warning | kept, with the `.installer-new` path |

A kept file is not a failure: the step completes and its report lists every
`.installer-new` written. A stale `.installer-new` is replaced on the next
render, and removed when the live file is accepted or already matches.

Callers routed through `write_managed`:

- Step 4 `_write_text_report()` and `_write_camera_models()`, and every
  generated file of [section 5](#5-generated-per-install-files).
- Unit files rendered by `systemd.install_unit` for
  `mv3dt-pipeline@.service` and `mv3dt-watchdog@.service`.

### 4.3 Ingest skips an unchanged archive (REQUIRED)

When `download_and_ingest()` reports the archive unchanged and every render
target exists, `_wire_download()` returns without rendering. A timer run
against an unchanged calibration therefore writes nothing at all.

### 4.4 `configs` subcommand

```bash
sudo mv3dt-installer configs status
sudo mv3dt-installer configs diff [<path>]
sudo mv3dt-installer configs accept <path>
sudo mv3dt-installer configs accept-all
```

| Action | Behavior |
|---|---|
| `status` | one line per managed path: `ok`, `edited`, `pending` (an `.installer-new` exists) or `missing` |
| `diff` | unified diff of the live file against its `.installer-new`, for one path or all pending |
| `accept <path>` | move `<path>.installer-new` over `<path>`, keeping owner and mode, and record its hash |
| `accept-all` | `accept` for every pending path |

**REQUIRED:** `diff` output passes through the redaction of
[`00` section 14.4](00-FRAMEWORK-AND-BOOTSTRAP.md#144-redaction-required):
rendered RTSP URIs carry camera credentials.

To keep an operator edit instead, the operator deletes the `.installer-new`;
the file stays `edited` and is never overwritten.

---

## 5. Generated per-install files

All are rendered by Step 4 under `<install_dir>/deepstream/` and written
through `write_managed`.

### 5.1 Fusion pub/sub and MQTT adaptor

`mv3dt/pub_sub_info_config.yml` covers the N enabled cameras in source
order. Camera k (1-based) publishes `/trck/cam<k>` and subscribes to every
other camera's topic, all on the local broker at port `1883`. Format and
topic template are those of NVIDIA's `deepstream-tracker-3d-multi-view`
generator. For two cameras:

```yaml
pubBrokerTopicStr:
- <loopback>:1883;/trck/cam1
- <loopback>:1883;/trck/cam2
subPeerBrokerTopicStrs:
- - <loopback>:1883;/trck/cam2
- - <loopback>:1883;/trck/cam1
```

`mv3dt/config_mqtt.txt` is the `[message-broker]` file for the local
anonymous broker installed by
[`STEP-1` section 3.2](STEP-1-PREREQUISITES.md#32-mosquitto-broker). Its
content is pinned exactly:

```ini
[message-broker]
username = user
password = password
#client-id = uniqueID
#enable-tls = 1
#tls-cafile =
#tls-capath =
#tls-certfile =
#tls-keyfile =
#share-connection = 1
#loop-timeout = 2000
#keep-alive = 60
set-threaded = 0
```

`username = user` and `password = password` are the fixed placeholders from
NVIDIA's MV3DT sample, not credentials: the broker allows anonymous access,
and the live pipeline log confirms it connects with exactly these values.
They are never replaced with a real credential. The golden render of
[section 7](#7-verification) compares `username`, `password` and
`set-threaded` exactly.

### 5.2 Dewarpers

For each camera with a committed rectification
([section 2.1](#21-dewarper-from-amc-rectification-locked)), Step 4 writes
`dewarper/dewarper_cam<N>.txt` (N is the 0-based source index):

```ini
[property]
output-width=1920
output-height=1080
num-batch-buffers=1

[surface0]
projection-type=3
width=1920
height=1080
focal-length=1600
distortion=-0.82243;0.66173;0;0;0
src-x0=960
src-y0=540
```

and a matching section in the rendered application config:

```ini
[dewarper0]
enable=1
gpu-id=0
source-id=0
num-surfaces-per-frame=1
num-output-buffers=4
nvbuf-memory-type=0
config-file=<install_dir>/deepstream/dewarper/dewarper_cam0.txt
```

An API error or a missing rectification endpoint logs a warning and renders
no dewarpers; it does not fail Step 4.

### 5.3 Viewer stream

When `MV3DT_VIEWER_STREAM=on`, the rendered application config gets:

| Section | Live values |
|---|---|
| `[tiled-display]` | `enable=1`, `rows=1`, `columns=N`, `width=1920`, `height=540` |
| `[osd]` | `enable=1`, `gpu-id=0`, `display-text=1`, `border-width=3`, `text-size=22`, `text-color=1;1;1;1`, `text-bg-color=0.3;0.3;0.3;1`, `font=Serif` |
| `[sink2]` | `enable=1`, `type=4`, `codec=1`, `enc-type=0`, `sync=0`, `bitrate=4000000`, `rtsp-port=8554`, `udp-port=5400` |

When `off`, `[tiled-display]` keeps the template's `enable=0` and neither
`[osd]` nor `[sink2]` is rendered.

### 5.4 BodyPose3DNet model

Step 4 downloads `bodypose3dnet_accuracy.onnx` from NGC into
`<install_dir>/deepstream/models/bodypose3dnet/`, owned by the invoking
user, with the same authenticated pattern as `_ensure_peoplenet_model`. A
present model is not downloaded again; `verify()` fails when it is missing.
DeepStream builds the engine beside it on first start.

---

## 6. Runtime and systemd

### 6.1 Pipeline unit logging

`assets/systemd/mv3dt-pipeline@.service.in` sets:

```ini
LogRateLimitIntervalSec=0
```

replacing `LogRateLimitIntervalSec=30s` and `LogRateLimitBurst=200`. The
MQTT callback logging otherwise exhausts the journal rate limit and hides the
`**PERF:` lines the watchdog reads. This replaces the hand drop-in
`no-log-ratelimit.conf`.

### 6.2 Viewer flag

`pipeline --service-exec`
([`STEP-5` section 3.3](STEP-5-PER-PROJECT-EXES.md#33-what-the-exe-does-at-runtime-pipeline-subcommand))
passes `-t` to `deepstream-test5-app` only when `MV3DT_VIEWER_STREAM=on`.
This replaces the hand drop-in `labels.conf`, which overrides `ExecStart`.

### 6.3 Watchdog

New subcommand `mv3dt-installer watchdog --project-slug <slug>`, a Python
port of the hand-deployed `mv3dt-watchdog.sh`:

| Parameter | Value |
|---|---|
| Check interval | 10 s |
| Grace after unit start | 30 s |
| Hung condition | the unit is active past the grace, and every `**PERF:` line in the last 30 s shows 0 fps on all sources |
| No `**PERF:` lines at all | not a hang; leave the unit alone |
| Restart | `systemctl stop`, 5 s pause, `systemctl start` |
| Rate limit | at most 5 restarts per 600 s, logged when reached |

`assets/systemd/mv3dt-watchdog@.service.in` runs it as root with
`After=mv3dt-pipeline@%i.service`, `Restart=always`, `RestartSec=10`. Step 6
installs and enables `mv3dt-watchdog@<slug>.service` beside the pipeline
instance, following
[`STEP-6` section A.4](STEP-6-REMOTE-SUPERVISION.md#a4-installer-integration-mirror-installsh).

---

## 7. Verification

1. **Every PR**: the full installer suite passes, apart from the known
   failure `tests/test_config.py::test_persist_value_preserves_existing_keys_and_overwrites_its_own`
   and the known flaky
   `tests/test_app.py::test_verbose_does_not_change_whether_run_root_streams`.
2. **Golden render (U5, U6)**: render the full config set on a development
   machine from a fixture of the live camera list and the live rectification
   response, then diff against the live snapshot. Tracker, inference,
   application config with the viewer on, pub/sub, MQTT and dewarper files
   must match apart from comments, redacted URIs and the expansion of the
   single camera-model placeholder.
3. **Edit protection (U2)**: tests for an untouched file updated, an
   operator-edited file kept with `.installer-new`, a pre-manifest install,
   `accept`, and an ingest with an unchanged archive writing nothing.
4. **On the workstation**, after the migration of
   [section 8](#8-live-migration):

- [ ] `configs status` lists only the expected `.installer-new` files
      before `accept-all`, and none after.
- [ ] The pipeline restarts with `Active sources : 2` and `**PERF:` near
      22 fps per source.
- [ ] The VLC stream on port 8554 works with the viewer gate on.
- [ ] `mv3dt-watchdog@<slug>.service` is active.
- [ ] `mv3dt-ingest-<slug>.timer` is disabled.
- [ ] A 60-second walk test still shows fused IDs on both cameras.

---

## 8. Live migration

Run by the operator on the workstation. Replace `<slug>` with the project
slug from [`DEVELOPMENT-STATUS.md`](DEVELOPMENT-STATUS.md).

### 8.1 Before any release (REQUIRED)

```bash
sudo systemctl disable --now mv3dt-ingest-<slug>.timer
systemctl is-enabled mv3dt-ingest-<slug>.timer
```

The second command must print `disabled`.

### 8.2 After the release

1. **Update the binary** by the procedure of
   [`00` section 5.1](00-FRAMEWORK-AND-BOOTSTRAP.md#51-operator-procedure).
   Do not use `--reset-state`.
2. **Set the AMC project and the gates** in `installer.conf`:

   ```bash
   sudoedit /opt/mv3dt/installer.conf
   ```

   ```ini
   AMC_ROOT=<invoking user home>/amc-3.3.0/tools/auto-magic-calib
   AMC_PROJECT_ID=<the AMC 3.3.0 project id>
   MV3DT_VIEWER_STREAM=on
   MV3DT_AUTO_INGEST=off
   ```

3. **Re-run Steps 4 and 6**:

   ```bash
   sudo ./mv3dt-installer --reset-step 4
   sudo ./mv3dt-installer --reset-step 6
   sudo ./mv3dt-installer --verbose
   ```

4. **Review the kept files.** Every live file is kept because none has a
   manifest entry. Each `.installer-new` should match its live file apart
   from comments:

   ```bash
   sudo /opt/mv3dt/bin/mv3dt-installer configs status
   sudo /opt/mv3dt/bin/mv3dt-installer configs diff
   sudo /opt/mv3dt/bin/mv3dt-installer configs accept-all
   ```

   Any difference beyond comments is a finding: stop and diagnose before
   accepting.
5. **Remove the hand-made pieces** and restart:

   ```bash
   sudo rm /etc/systemd/system/mv3dt-pipeline@<slug>.service.d/labels.conf
   sudo rm /etc/systemd/system/mv3dt-pipeline@<slug>.service.d/no-log-ratelimit.conf
   sudo rm /opt/mv3dt/bin/mv3dt-watchdog.sh
   sudo systemctl daemon-reload
   sudo systemctl restart mv3dt-pipeline@<slug>.service mv3dt-watchdog@<slug>.service
   ```

   The hand watchdog unit has the same name as the installer's
   `mv3dt-watchdog@.service`, so `accept-all` in step 4 already replaced it
   with the unit that runs the `watchdog` subcommand.
6. **Verify** with the checklist of [section 7](#7-verification), then update
   the workstation table in [`DEVELOPMENT-STATUS.md`](DEVELOPMENT-STATUS.md).

---

## 9. Out of scope / open decisions

- Recording calibration footage, the alignment points and the focal-length
  choice inside AMC remain operator work; only the committed result is read.
- No change to the Step 8 recorder or the web app; fusion only changes which
  IDs the existing topic carries.

Open decisions:

1. **Flagged: the `v9.2.0` commit hash.** U4 resolves the tag to a commit and
   records it in [`STEP-3` section 2](STEP-3-AMC-LAUNCHER.md#2-upstream-pin)
   as the equality pin, and confirms the compose file path under the sparse
   checkout.
2. **Flagged: fixed `f = 1600`.** It is the value that gave the best live
   calibration, not one read from AMC. Reading the focal length from the
   project is follow-up work.
3. **RESOLVED: existing installs.** A pre-manifest file is treated as
   operator-edited ([section 4.2](#42-write_managedpath-text-rules-required)),
   so the first upgrade never overwrites anything; the operator accepts once.

---

## 10. Documented drift

| Spec | Was | Becomes |
|---|---|---|
| [`STEP-3` section 2](STEP-3-AMC-LAUNCHER.md#2-upstream-pin) | standalone AMC 3.2.1 repository, sparse checkout rejected | `NVIDIA/DeepStream` `v9.2.0` sparse `tools/auto-magic-calib`, AMC 3.3.0 |
| [`STEP-4` section 6](STEP-4-CALIB-OUTPUT-WIRING.md#6-deepstream-configuration-wiring) | single-view 3D only, files overwritten on every render | fusion, dewarpers, optional viewer, managed writes |
| [`STEP-4` section 7](STEP-4-CALIB-OUTPUT-WIRING.md#7-later-recalibration) | timer always enabled, every run re-renders | timer behind `MV3DT_AUTO_INGEST`, unchanged archive renders nothing |
| [`STEP-6` section A.1](STEP-6-REMOTE-SUPERVISION.md#a1-the-templated-per-project-instance-unit) | journal rate limit 30 s / 200, no watchdog | rate limit off, `mv3dt-watchdog@.service` |

---

## 11. Decomposition

| Unit | Branch | Files touched | Depends on | Wave |
|---|---|---|---|---|
| U1 Spec | `feat/installer-live-baseline-spec` | `installer/plan/LIVE-BASELINE-AND-EDIT-PROTECTION.md`, `installer/plan/DEVELOPMENT-STATUS.md` | none | 1 |
| U2 Edit protection | `feat/installer-managed-files` | `mv3dt_installer/managed_files.py`, `steps/step4_calib_output_wiring.py` (write helpers + ingest skip), `app.py` (configs subcommand), `tests/test_managed_files.py`, `tests/test_step4_calib_output_wiring.py` | none | 1 |
| U3 Template baseline | `feat/installer-template-baseline` | `laptop/deepstream/config_tracker_NvMOT.yml`, `config_infer_primary.txt`, `deepstream_app_config.txt`, `tests/test_deepstream_templates.py` | none | 1 |
| U4 AMC 3.3.0 | `feat/installer-amc-330` | `steps/step3_amc_launcher.py`, `tests/test_step3_amc_launcher.py`, `installer/plan/STEP-3-AMC-LAUNCHER.md` | none | 1 |
| U5 Generated files + viewer + pose model | `feat/installer-fusion-dewarper-viewer` | `steps/step4_calib_output_wiring.py` (render), `config.py`, `app.py` (viewer gate), `tests/test_step4_calib_output_wiring.py`, `tests/test_config.py` | U2, U3 | 2 |
| U6 Runtime and systemd | `feat/installer-runtime-hardening` | `steps/step4_calib_output_wiring.py` (timer gate), `steps/step5_per_project_exes.py` (`-t`), `steps/step6_remote_supervision.py`, `mv3dt_installer/watchdog.py`, `assets/systemd/mv3dt-pipeline@.service.in`, `assets/systemd/mv3dt-watchdog@.service.in`, `config.py`, `app.py`, tests for steps 4/5/6 and watchdog | U5 | 3 |

Package paths are relative to `installer/mv3dt_installer/`; test paths are
relative to `installer/`.

Serialization points:

- `step4_calib_output_wiring.py`, `config.py` and `app.py` are shared by U2,
  U5 and U6, so those three are strictly sequential: U5 builds on U2's
  `write_managed`, and U6 adds the timer gate to the render U5 extends.
- U5 depends on U3 because its golden render diffs the new templates
  against the live snapshot.
- U3 and U4 touch disjoint files and run in wave 1 with U1 and U2.
- Recommended merge order: U1, U2, U3, U4, U5, U6, then a release.

---

## References

DeepStream facts here come from NVIDIA's `nvdewarper` and tracker
documentation and the `deepstream-tracker-3d-multi-view` sample; AMC facts
come from the AMC 3.3.0 API as observed on the workstation. Every baseline
value is copied from a snapshot of the live installed tree.

- [Gst-nvdewarper](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdewarper.html)
  — **backs [section 5.2](#52-dewarpers)**: `projection-type`,
  `focal-length`, `distortion`, `src-x0`, `src-y0`.
- [Gst-nvtracker](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html)
  — **backs [section 3.1](#31-tracker-config_tracker_nvmotyml)**:
  `PoseEstimator`, `MultiViewAssociator`, `Communicator`.
- [deepstream_reference_apps `deepstream-tracker-3d-multi-view`](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps)
  — the pub/sub file format and topic template of
  [section 5.1](#51-fusion-pubsub-and-mqtt-adaptor).
- [NVIDIA DeepStream repository](https://github.com/NVIDIA/DeepStream) — tag
  `v9.2.0`, `tools/auto-magic-calib` (AMC 3.3.0).
- [Debian conffile handling](https://www.debian.org/doc/debian-policy/ch-files.html#configuration-files)
  — the model for [section 4](#4-managed-files-edit-protection).
- [systemd.exec `LogRateLimitIntervalSec`](https://www.freedesktop.org/software/systemd/man/latest/systemd.exec.html)
  — **backs [section 6.1](#61-pipeline-unit-logging)**.

Repo files referenced:

- [`00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md) — sections
  3.4 gates, 5.1 operator procedure, 14.4 redaction.
- [`STEP-1-PREREQUISITES.md`](STEP-1-PREREQUISITES.md) — the anonymous local
  Mosquitto broker the fusion files point at.
- [`STEP-3-AMC-LAUNCHER.md`](STEP-3-AMC-LAUNCHER.md) — section 2, the AMC pin
  this doc replaces.
- [`STEP-4-CALIB-OUTPUT-WIRING.md`](STEP-4-CALIB-OUTPUT-WIRING.md) — sections
  4.3, 6 and 7, the render and re-ingest contracts amended here.
- [`STEP-5-PER-PROJECT-EXES.md`](STEP-5-PER-PROJECT-EXES.md) — section 3.3,
  the `pipeline` subcommand that gains `-t`.
- [`STEP-6-REMOTE-SUPERVISION.md`](STEP-6-REMOTE-SUPERVISION.md) — sections
  A.1 and A.4, the pipeline unit and unit installation.
- [`DEVELOPMENT-STATUS.md`](DEVELOPMENT-STATUS.md) — the dated workstation
  record, which carries this spec's status row.
- `installer/mv3dt_installer/steps/step4_calib_output_wiring.py` —
  `_write_text_report()`, `_write_camera_models()`, `_render_configs()`,
  `_wire_download()`, `handle_ingest_subcommand()`.
- `installer/mv3dt_installer/assets/systemd/mv3dt-pipeline@.service.in` — the
  unit whose rate limit changes.
- `laptop/deepstream/config_tracker_NvMOT.yml`,
  `laptop/deepstream/config_infer_primary.txt`,
  `laptop/deepstream/deepstream_app_config.txt` — the templates of
  [section 3](#3-pinned-baseline-values).
- `installer/installer.spec` — bundles the templates into the binary.
