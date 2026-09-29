# Step 4 — Calibration footage capture (owner: DevD)

Status: step spec, an extension of
[`STEP-4`](STEP-4-CALIB-OUTPUT-WIRING.md). It depends on the camera inventory
and credential contract in
[`00` §15](00-FRAMEWORK-AND-BOOTSTRAP.md#15-camera-discovery), the privilege
boundary in
[`00` §9.2](00-FRAMEWORK-AND-BOOTSTRAP.md#92-resolving-the-invoking-user--home),
and Step 4's project-state polling in
[`STEP-4` §3](STEP-4-CALIB-OUTPUT-WIRING.md#3-project-state-polling). It does
**not** restate those contracts — link back to them.

AMC's browser workflow needs one video per camera before it can calibrate
([`DEEPSTREAM-SETUP` §8.6](../../laptop/docs/DEEPSTREAM-SETUP.md), step 2
"Video Upload"). Until now the operator produced those clips by hand. This
spec makes Step 4 **record a synchronized five-minute clip from every enabled
camera** into the operator's `~/Downloads`, where the AMC upload dialog can
reach them, and **delete that footage automatically once Step 4 has ingested
a successful AMC export** — at which point the clips have served their only
purpose.

It ports the recording logic of `laptop/scripts/record_cameras_mp4.sh`
(`ffmpeg -rtsp_transport tcp ... -map 0:v:0 -c:v copy -an -movflags
+faststart`, one child per camera, all started together), with one change:
clips are re-encoded at 1920x1080 instead of stream-copied (§2).
[`DELETION-REVIEW` §8](DELETION-REVIEW.md#8-script-disposition-under-the-binary-distribution)
dropped that script from the binary for having "no plan coverage"; this
document is that coverage. The script itself stays a developer-only tool and
is **not** bundled — the logic is ported to Python, as `20_verify_cameras.sh`'s
probe was.

It also fixes a readability defect in the Step 4 wait screen (§8), because the
recording prompt and the upload hint are only useful if the operator can see
them.

---

## 1. Module identity

| Item | Value |
|---|---|
| New module | `installer/mv3dt_installer/footage.py` (framework helper, like `cameras.py`) |
| Consumer | `installer/mv3dt_installer/steps/step4_calib_output_wiring.py` |
| Tests | `installer/tests/test_footage.py`, `installer/tests/test_step4_calib_output_wiring.py` |

`footage.py` owns directory layout, recording, completeness, and deletion. It
has no knowledge of AMC or of step state; Step 4 decides **when** to call it.

---

## 2. Pinned values

| Setting | Value | Source |
|---|---|---|
| Clip length | `300` seconds | `CALIB_FOOTAGE_SECONDS` in `installer.conf`, optional |
| Footage root | `<invoking user home>/Downloads/mv3dt-calibration-footage` | `CALIB_FOOTAGE_DIR` in `installer.conf`, optional; when unset, Step 4 persists the resolved absolute default on first recording (§6) |
| Project directory | `<footage root>/<PROJECT_NAME>/` | `PROJECT_NAME` from [`STEP-4` §2](STEP-4-CALIB-OUTPUT-WIRING.md#2-required-inputs) |
| Clip file name | `<id>-<position>.mp4`, `unlabeled` for an empty position; characters outside `[A-Za-z0-9._-]` become `-` | camera inventory, [`00` §15.4](00-FRAMEWORK-AND-BOOTSTRAP.md#154-guided-position-binding-one-time) |
| In-progress file name | `<clip file name>.part` | — |
| Ownership marker | `.mv3dt-footage.json` in the project directory | — |
| RTSP socket timeout | `-timeout 5000000` (5 s, before `-i`) | matches [`00` §15.3](00-FRAMEWORK-AND-BOOTSTRAP.md#153-rtsp-probe) |
| Process timeout | clip length plus `60` seconds | — |
| Minimum free space | `1 GiB` per camera on the footage root's filesystem | — |
| Clip resolution | `1920x1080`, H.264, `yuv420p` | AMC input requirement ([AMC README, "Tracklet-Based Calibration: Input Video Requirements"](https://github.com/NVIDIA-AI-IOT/auto-magic-calib/blob/0cfd2b790fd77598b0543340a65c2a0e1d192327/README.md#tracklet-based-calibration-input-video-requirements)) |

The `ffmpeg` argv for each camera is exactly:

```bash
ffmpeg -hide_banner -loglevel error -nostats -y \
  -rtsp_transport tcp -timeout 5000000 \
  -i "rtsp://<user>:<pass>@<ip>:554<rtsp_path>" \
  -t <seconds> -map 0:v:0 -vf scale=1920:1080:out_range=tv \
  -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p \
  -an -movflags +faststart \
  -f mp4 "<project dir>/<clip file name>.part"
```

**RESOLVED — re-encode, don't stream-copy.** AMC 3.2.1 requires 1920x1080
input and its multi-view config pins `video_resolution: [1920, 1080]`. The
fleet's native main stream is 3072x1728, which `-c:v copy` passed through
unchanged. The clip is therefore always scaled and re-encoded, whatever the
camera sends: a camera already set to 1080p costs only the encode, and a
camera left at its native resolution still produces a valid clip.
`out_range=tv` with `-pix_fmt yuv420p` converts the cameras' full-range
`yuvj420p` to the standard limited range DeepStream and AMC decode;
`-pix_fmt` alone keeps the full-range tag. `veryfast` keeps each child well
under real time on CPU, so all children can run at once. `-f mp4` is
required because the `.part` suffix hides the container type from `ffmpeg`.

The marker is JSON:

```json
{
  "project_name": "Valencia-West",
  "seconds": 300,
  "recorded_utc": "2026-09-25T20:15:00Z",
  "cameras": [
    {"id": "c1", "mac": "d0:3b:f4:02:44:e2", "file": "c1-top-left.mp4"}
  ]
}
```

It is written with the
[`00` §6.3](00-FRAMEWORK-AND-BOOTSTRAP.md#63-api-statepy) `write_json_atomic`
helper, and **only** after every camera's clip succeeded.

---

## 3. When footage is recorded

**LOCKED — Step 4 preflight, after the camera inventory is resolved and before
the completion wait.** Recording is attempted only when all of the following
hold:

1. **The AMC project has not started**: one status request
   ([`STEP-4` §3.1](STEP-4-CALIB-OUTPUT-WIRING.md#31-status-endpoint)) returns
   `INIT`. Any other state means the operator has already moved past video
   upload.
2. **No complete set exists**: `footage.is_complete()` is false — the marker is
   absent, names a different `PROJECT_NAME` or clip length, lists a different
   set of enabled camera MACs, or a listed clip file is missing.
3. **The run is interactive.** Under `--non-interactive` recording is skipped
   with one `info` line, because a useful recording needs a person walking
   through the scene; an unattended run must never wait for a human
   ([`00` §15.4](00-FRAMEWORK-AND-BOOTSTRAP.md#154-guided-position-binding-one-time)
   applies the same rule to binding).
4. **There is room**: the footage root's filesystem has at least the §2
   minimum free space. Otherwise Step 4 returns `USER_ACTION_REQUIRED` naming
   the directory, the space found, and the space needed.

When all hold, Step 4 prompts once:

```
Record 5 minutes of calibration footage from 2 cameras into
/home/p2bp-admin/Downloads/mv3dt-calibration-footage/Valencia-West?
Have someone walk through the whole scene while it records.
Press Enter to start, or type s to skip:
```

`s` skips recording for this run and continues to the completion wait. Any
other answer starts recording. The minutes and camera count in the prompt are
computed, not fixed.

Recording uses every camera in the inventory with `enabled: true`. A camera
whose last probe recorded `stream_ok: false` is still attempted — the probe
may predate activation — and simply fails fast if it is still unreachable.

---

## 4. Recording

1. **Run as the invoking user.** The project directory is created and every
   `ffmpeg` child runs through the
   [`00` §9.2](00-FRAMEWORK-AND-BOOTSTRAP.md#92-resolving-the-invoking-user--home)
   invoking-user prefix (`sudo -u <user> -H`), so the files are owned by the
   operator and can be opened, moved, or deleted from the Files app.
2. **Start every camera together.** All children are spawned before any is
   waited on. Multi-view calibration matches the same moments across views,
   so clips must overlap in time; sequential recording would not.
3. **Report progress.** `ctx.progress.task()` names the recording and the
   camera count, and a determinate bar advances by elapsed seconds over the
   clip length about once per second, so a five-minute recording never looks
   hung.
4. **Finish atomically per camera.** A child that exits `0` with a non-empty
   `.part` file has its file renamed to the final clip name. A child that
   exits non-zero, produces an empty file, or exceeds the process timeout is
   killed if still running and its `.part` file is deleted.
5. **Write the marker only on full success.** If every camera succeeded, the
   marker is written and Step 4 logs the directory plus the hint
   "Upload these N files at the AMC Video Upload step." If any camera failed,
   the marker is not written, successful clips are kept, and Step 4 returns
   `USER_ACTION_REQUIRED` naming each failed camera by `id` and IP with the
   hint to check activation and camera credentials
   ([`00` §15.3](00-FRAMEWORK-AND-BOOTSTRAP.md#153-rtsp-probe)). The next run
   re-records the whole set, overwriting the kept clips, so every clip in a
   complete set comes from the same session.
6. **Ctrl-C is a clean cancel.** It terminates every child, deletes every
   `.part` file, writes no marker, and returns `USER_ACTION_REQUIRED` with the
   same resume hint as a cancelled wait
   ([`STEP-4` §3.3](STEP-4-CALIB-OUTPUT-WIRING.md#33-wait-outcomes)).

**REQUIRED — the camera password never reaches a log line or an error
message.** It appears only inside the argv handed to the child process, and
any `ffmpeg` stderr included in an error is passed through the
[`00` §14.4](00-FRAMEWORK-AND-BOOTSTRAP.md#144-redaction-required) redaction rule
first.

---

## 5. The completion-wait hint

When a complete set exists for the project, Step 4's wait hints
([`STEP-4` §3.3](STEP-4-CALIB-OUTPUT-WIRING.md#33-wait-outcomes)) gain one
action, placed first:

```
Upload the N clips in <project dir> at the AMC Video Upload step.
```

---

## 6. Deletion after a successful export

**REQUIRED.** The footage exists to be uploaded once. After Step 4 has
downloaded, validated, and installed an AMC export — the point where
`_wire_download()` returns `COMPLETE`, which covers both the interactive
`run()` and the timer-driven `ingest` subcommand
([`STEP-4` §7](STEP-4-CALIB-OUTPUT-WIRING.md#7-later-recalibration)) — Step 4
calls `footage.delete_if_owned()` on the project directory.

`delete_if_owned()` deletes the directory **only if it contains the §2
ownership marker**, so it can never remove a directory the installer did not
create, and it removes the footage root too if that leaves it empty. It
returns whether anything was deleted and logs the path it removed. A deletion
failure is a `warn` line, never a step failure — the calibration is already
installed.

> **RESOLVED — deletion reads the persisted root.** The timer-driven
> `ingest` runs as root with no `$SUDO_USER`, so the
> [`00` §9.2](00-FRAMEWORK-AND-BOOTSTRAP.md#92-resolving-the-invoking-user--home)
> invoking user resolves to root and the §2 default would point under
> `/root`. Step 4 therefore persists the resolved footage root to
> `CALIB_FOOTAGE_DIR` just before its first recording (an existing value is
> left alone), and deletion reads only that key. When the key is absent the
> installer never recorded footage, so deletion is skipped rather than
> falling back to the running user's home.

Footage is **not** deleted on a failed ingest, an AMC `ERROR` state, a wait
timeout, or a cancelled wait: in every one of those cases the operator may need
to upload it again.

---

## 7. `footage.py` API surface

```python
DEFAULT_FOOTAGE_SECONDS = 300
FOOTAGE_DIRNAME = "mv3dt-calibration-footage"
MARKER_NAME = ".mv3dt-footage.json"
MIN_FREE_BYTES_PER_CAMERA = 1 << 30

@dataclass(frozen=True)
class RecordResult:
    project_dir: Path
    recorded: list[Path]          # final clip paths
    failed: list[Camera]          # cameras whose clip did not complete
    cancelled: bool

def footage_root(home: Path, override: str | None = None) -> Path
def project_dir(root: Path, project_name: str) -> Path
def clip_name(camera: Camera) -> str
def is_complete(project_dir: Path, cameras: Sequence[Camera], *,
                project_name: str, seconds: int) -> bool
def has_room(root: Path, camera_count: int) -> tuple[bool, int, int]   # ok, free, needed
def record(cameras: Sequence[Camera], *, user: str, password: str,
           project_dir: Path, project_name: str, seconds: int,
           user_prefix: Sequence[str] = (),
           spawn=subprocess.Popen, clock=time.monotonic, sleep=time.sleep,
           on_progress: Callable[[float], None] = lambda fraction: None
           ) -> RecordResult
def delete_if_owned(project_dir: Path) -> bool
```

`Camera` is `cameras.Camera`
([`00` §15.6](00-FRAMEWORK-AND-BOOTSTRAP.md#156-api-surface-cameraspy)).
`user_prefix` is the invoking-user prefix (`("sudo", "-u", <user>, "-H")` in
production) prepended to each `ffmpeg` argv; `spawn`, `clock`, and `sleep` are
injected so tests never start a real process or wait real time. `record()`
creates `project_dir` itself; production callers pass a `project_dir` whose
parent was created through the same invoking-user prefix.

---

## 8. Quiet status polling

**REQUIRED.** Step 4's project-status and calibration-log requests
([`STEP-4` §3.1](STEP-4-CALIB-OUTPUT-WIRING.md#31-status-endpoint),
[§3.2](STEP-4-CALIB-OUTPUT-WIRING.md#32-error-evidence)) pass `stream=False`
to `ctx.run_root`. Their JSON replies are parsed, never echoed into the live
window. On the workstation, each poll's
`{"code":0,"message":"Project info retrieved successfully",...}` reply was
streamed into the window every few seconds, burying the wait description and
the AMC UI hint for as long as calibration took.

Before the completion wait starts, Step 4 also sets
`ctx.progress.task("waiting for AMC project <PROJECT_NAME> to complete")`, so
the live task line stops reading "resolving AMC project and camera inputs"
while it waits. State changes are still logged once each, as today.

---

## 9. Verification checklist

- [ ] A fresh `INIT` project with no footage prompts once, records every
      enabled camera concurrently, and writes the marker.
- [ ] Clips land in `~/Downloads/mv3dt-calibration-footage/<PROJECT_NAME>/`,
      owned by the invoking user.
- [ ] A rerun with a complete set does not prompt or record again.
- [ ] A project past `INIT`, `--non-interactive`, or an `s` answer skips
      recording.
- [ ] Insufficient free space returns `USER_ACTION_REQUIRED` before any
      recording starts.
- [ ] A failed or timed-out camera leaves no `.part` file, no marker, and a
      `USER_ACTION_REQUIRED` naming that camera.
- [ ] Ctrl-C during recording kills every child and leaves no `.part` file.
- [ ] The password never appears in a log line or error message.
- [ ] A successful ingest deletes the marked project directory and an empty
      footage root; an unmarked directory is never deleted.
- [ ] A failed ingest, `ERROR`, timeout, or cancel keeps the footage.
- [ ] The wait screen shows no raw status JSON, names the project, and lists
      the upload hint when footage exists.

---

## 10. Out of scope / open decisions

- Uploading the clips into AMC through its API. The operator still selects
  them in the Video Upload dialog.
- Transcoding or downscaling. Clips are stream copies; the VGGT VRAM note in
  [`DEEPSTREAM-SETUP` §8.6](../../laptop/docs/DEEPSTREAM-SETUP.md) is left to
  the operator.
- Audio, which AMC does not use.
- Deleting footage for any outcome other than a successful ingest (§6).
- Bundling `record_cameras_mp4.sh`; it remains a developer tool.

Open decisions:

1. **Flagged — automatic AMC upload.** If AMC 3.2.1's API accepts video
   uploads for a project, Step 4 could upload the set itself and remove the
   manual Video Upload step. Deferred until the endpoint is confirmed against
   the pinned AMC source.

---

## 11. Implementation decomposition

| Unit | Branch | Files touched | Depends on | Wave |
|---|---|---|---|---|
| Step 4 quiet status polling (§8) | `feat/installer-step4-quiet-poll` | `installer/mv3dt_installer/steps/step4_calib_output_wiring.py`, `installer/tests/test_step4_calib_output_wiring.py`, `installer/plan/STEP-4-CALIB-OUTPUT-WIRING.md` | None | 1 |
| Footage recorder module (§1, §2, §4, §6, §7) | `feat/installer-calibration-footage-recorder` | `installer/mv3dt_installer/footage.py`, `installer/tests/test_footage.py`, `installer/plan/STEP-4-CALIBRATION-FOOTAGE.md` | None | 1 |
| Step 4 footage wiring (§3, §4, §5, §6) | `feat/installer-step4-footage-wiring` | `installer/mv3dt_installer/steps/step4_calib_output_wiring.py`, `installer/tests/test_step4_calib_output_wiring.py`, `installer/plan/STEP-4-CALIB-OUTPUT-WIRING.md`, `installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md`, `installer/plan/DELETION-REVIEW.md` | Step 4 quiet status polling, Footage recorder module | 2 |

The two wave-1 units touch disjoint files. The wiring unit is serialized after
both: it imports `footage.py`, and it edits the same Step 4 module, test file,
and spec as the quiet-polling unit, so the two cannot be open against `main`
at the same time. Its pull request targets the quiet-polling branch and merges
the recorder branch in before building; merge order is quiet
polling, then recorder, then wiring. The wiring unit also adds
`CALIB_FOOTAGE_SECONDS` and `CALIB_FOOTAGE_DIR` to the
[`00` §11.2](00-FRAMEWORK-AND-BOOTSTRAP.md#112-persistence--sharing-with-later-steps)
key table, links this document from
[`STEP-4` §1](STEP-4-CALIB-OUTPUT-WIRING.md#1-scope-and-identity), and updates
the `record_cameras_mp4.sh` row of
[`DELETION-REVIEW` §8](DELETION-REVIEW.md#8-script-disposition-under-the-binary-distribution)
to "Ported to Python — not bundled".

---

## References

The AMC workflow facts come from the repository's DeepStream 9.1 setup guide
and NVIDIA's AutoMagicCalib documentation; the recording command is ported
from the developer harness.

- [DS 9.1 AutoMagicCalib](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_AutoMagicCalib.html) — **the six-step browser workflow whose Video Upload step this footage feeds.**
- [FFmpeg RTSP demuxer options](https://ffmpeg.org/ffmpeg-protocols.html#rtsp) — `-rtsp_transport tcp` and the `-timeout` socket option.
- [FFmpeg MOV/MP4 muxer](https://ffmpeg.org/ffmpeg-formats.html#mov_002c-mp4_002c-ismv) — `-movflags +faststart`.
- [AutoMagicCalib 3.2.1 README, "Tracklet-Based Calibration: Input Video Requirements"](https://github.com/NVIDIA-AI-IOT/auto-magic-calib/blob/0cfd2b790fd77598b0543340a65c2a0e1d192327/README.md#tracklet-based-calibration-input-video-requirements) — **the 1920x1080 input resolution the clips are scaled to (§2).**
- [FFmpeg scale filter](https://ffmpeg.org/ffmpeg-filters.html#scale-1) — `scale=1920:1080:out_range=tv`.

Repo files referenced:

- [`installer/plan/STEP-4-CALIB-OUTPUT-WIRING.md`](STEP-4-CALIB-OUTPUT-WIRING.md) — the step this extends: polling, wait outcomes, ingest, re-ingest.
- [`installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md) — camera inventory and probe (§15), privilege (§9.2), atomic JSON (§6.3), redaction (§14.4), config keys (§11.2).
- [`installer/plan/DELETION-REVIEW.md`](DELETION-REVIEW.md) — §8 records `record_cameras_mp4.sh` as dropped for lack of plan coverage.
- [`laptop/scripts/record_cameras_mp4.sh`](../../laptop/scripts/record_cameras_mp4.sh) — the developer script whose `ffmpeg` recording this ports.
- [`laptop/docs/DEEPSTREAM-SETUP.md`](../../laptop/docs/DEEPSTREAM-SETUP.md) — §8.6, the AMC workflow and the per-camera MP4 upload.
- [`installer/mv3dt_installer/cameras.py`](../mv3dt_installer/cameras.py) — the `Camera` type and RTSP URL shape reused here.
