# TRACKING FUSION AND CLEANING — fused, cleaned people from per-camera tracker output (owner: DevD)

Status: amendment spec, **specified, not implemented, not run on the
workstation**. It amends
[`STEP-8-TRACKING-HISTORY.md`](STEP-8-TRACKING-HISTORY.md): it replaces the
query-time stitching of
[`STEP-8` section 7](STEP-8-TRACKING-HISTORY.md#7-stitching) with a
fusion-and-cleaning engine, extends the `range` and `export` routes of
[`STEP-8` section 8](STEP-8-TRACKING-HISTORY.md#8-http-api), adds three
subcommands to [`STEP-8` section 10](STEP-8-TRACKING-HISTORY.md#10-cli), and
adds two files to the Step 8 install
([`STEP-8` section 11.2](STEP-8-TRACKING-HISTORY.md#112-runctx)). It also
carries one installer unit of upgrade fixes found while migrating the
workstation to 0.4.17. The recorder, storage, tickets, lease and network
exposure of Step 8 are **not** restated here — link back to them.

The recorded history is honest about what DeepStream publishes, and that is
not what an operator means by "people". DeepStream publishes per-camera
positions under shared MV3DT ids, ids churn, and some static objects are
detected as people for hours. **Fusion and cleaning run entirely on the
workstation, over DeepStream's recorded per-camera output, at query time.**
Raw rows in the day files are never modified.

The strict defaults below are taken from the prototype `clean.py` (a local
analysis script, not committed), which was run against a full recorded day
and an exported hour. The prototype keys tracklets by `(track, sensor)`,
associates them with duplicate merging and sequential stitching, and takes a
per-person median after association. This spec departs from it in five
places, listed in [section 5](#5-fusion-stages).

---

## 1. Scope

Module: a new `fuse.py`, `offsets.py` and `cleaning.py` in the bundled
service package `installer/mv3dt_installer/assets/tracking/mv3dt_tracking/`,
plus changes to that package's `api.py`, `cli.py` and `settings.py`, to
`steps/step8_tracking_history.py`, to `assets/systemd/mv3dt-tracking.service.in`,
and to the web app's Floor Plan Replay panel.

In scope:

1. The cleaning configuration file and its presets
   ([section 4](#4-cleaning-configuration)).
2. The resumable fusion engine and its per-stage drop counters
   ([section 5](#5-fusion-stages)).
3. Learned per-camera offset correction
   ([section 6](#6-offset-correction)).
4. Ignore-zone suggestions ([section 7](#7-zone-suggestions)).
5. API, CLI and installer passthrough changes
   ([section 8](#8-api-changes), [section 9](#9-cli-and-installer-passthrough)).
6. Step 8 install changes ([section 10](#10-step-8-changes)).
7. The installer upgrade fixes unit ([section 11](#11-installer-upgrade-fixes)).
8. The web Replay changes ([section 12](#12-web-replay-changes)).

Out of scope: any change to the DeepStream pipeline, tracker configuration,
calibration, or recorder ([section 3](#3-decisions)).

---

## 2. Problem and measured evidence

Replay of one real hour with one to two people present, one of them seated
on a couch, showed **562 "people"**. The shipped stitcher only joins tracker
ids with a 3 s / 1.2 m continuation rule
([`STEP-8` section 7](STEP-8-TRACKING-HISTORY.md#7-stitching)).

Analysis of a full recording (93,516 detections over 14.7 h) found:

| Measurement | Value |
|---|---|
| Points outside the 7.41 x 5.02 m room | 18% |
| Consecutive steps faster than 3 m/s | 14% |
| Tracker ids living under 2 s | 63% |
| Largest single "person" | a TV, seen only by camera 2: 19,398 samples at one fixed spot |
| Seated person | split into 363 tracker ids |
| Time both cameras see someone | 11.5% |
| Share of co-observed time MV3DT gives both cameras the same id | about one third |
| Same-id cross-camera distance, walking | about 0.9 m |
| Same-id cross-camera distance, seated | about 1.2 m |
| Steady cross-camera offset in same-id pairs | about 0.43 m |
| Calibration error at floor points | 3 to 16 cm |

DeepStream publishes **per-camera** positions on the `sv3d` topic: there is
no consolidated topic, and NVIDIA's documentation describes per-camera output
carrying shared MV3DT ids. Calibration itself is centimetre-level; the
cross-camera gap comes from DeepStream's model-based foot estimate, not from
the floor homography.

The prototype applied to the same hour, with its hard-coded TV zone,
gave:

| Metric | Before | After |
|---|---|---|
| People in the hour | 562 | 10 |
| Max people at once | not meaningful | 2, during real overlap |
| Points outside the room | 18% | 0 |
| Max step speed | unbounded | about 4 m/s |
| People in a ten-minute walking window | 40 | 3 |

---

## 3. Decisions

### 3.1 Workstation-only fusion (LOCKED)

**LOCKED: fuse and clean DeepStream's per-camera output on the workstation;
no DeepStream changes.** The pipeline, tracker config, message converter and
recorder stay exactly as Step 8 and the live baseline leave them. Fusion is a
pure function of recorded rows plus two workstation files.

**LOCKED: no 2D-box position bypass.** Floor positions remain DeepStream's
`footLocation3D` (or its `bbox3d` fallback,
[`STEP-8` section 5](STEP-8-TRACKING-HISTORY.md#5-decoding-nvframe)). The
engine never re-projects 2D boxes through the calibration to get its own
positions.

### 3.2 Strict now, relaxable later (LOCKED)

**LOCKED: the defaults are the strict prototype values.** Every threshold is
a key in `cleaning.json` ([section 4](#4-cleaning-configuration)), and a
single `preset` field switches between `strict` and `relaxed`. Relaxing is
expected once calibration footage is re-recorded and the cameras agree; it
must be a one-line edit with no restart.

### 3.3 Offset correction (LOCKED)

**LOCKED: the per-camera offset correction is learned, reported and
toggleable.** It is never applied silently: `learn-offsets` writes
`offsets.json` with before/after numbers, `cleaning.json` key
`offsets_apply` turns its use on or off, and every `range` summary says
whether it was applied. When the learned correction improves the median
cross-camera distance by **less than 0.1 m**, the file records
`"needed": false` and the UI says the correction is unnecessary.

### 3.4 Zones (LOCKED)

**LOCKED: ignore zones live in the workstation's `cleaning.json`, not in the
web app.** A `zones-suggest` command lists static hotspots as ready-to-paste
zone entries ([section 7](#7-zone-suggestions)); the operator decides what
to paste.

---

## 4. Cleaning configuration

### 4.1 File and reload (REQUIRED)

| Item | Value |
|---|---|
| Path | `<install_dir>/tracking/cleaning.json` (outside the package dir) |
| Env var | `CLEANING_FILE` |
| Owner | the invoking user, mode `644` |
| Seeded | once by Step 8; never overwritten ([section 10](#10-step-8-changes)) |
| Reload | re-read when its mtime changes; no restart |
| Invalid file | keep the last valid config, or strict defaults when none loaded yet; log the error and surface it as `summary.config_error` |
| Missing file | strict defaults |

Resolution order for every key: the value present in the file, else the
value of the selected `preset`, else the strict default. Keys absent from
the file follow the preset, so changing `preset` changes every key the
operator has not pinned. An unknown key is ignored, logged, and listed in
`summary.config_warnings` ([section 8.1](#81-range)).

### 4.2 Keys and strict defaults

| Key | Strict default | Unit | Stage | Meaning |
|---|---|---|---|---|
| `version` | `1` | | | schema version |
| `preset` | `"strict"` | | | `strict` \| `relaxed` |
| `room_margin_m` | `0.3` | m | 1 | margin around the room rectangle |
| `room` | `null` | m | 1 | fallback `{"x": [min, max], "y": [min, max]}` used only when calibration is unavailable |
| `zones` | `[]` | | 1 | ignore zones ([section 4.4](#44-zone-entries)) |
| `vmax_mps` | `3.0` | m/s | 2, 7 | max speed between kept samples |
| `speed_slack_m` | `0.3` | m | 2, 7 | distance allowed on top of `vmax_mps * dt` |
| `min_dt_s` | `0.04` | s | 2 | floor on `dt` in the stage 2 gate |
| `bucket_ms` | `200` | ms | 3, 6 | time bucket for tracklets and fusion |
| `min_tracklet_s` | `1.0` | s | 3 | drop shorter tracklets |
| `min_tracklet_samples` | `5` | buckets | 3 | drop tracklets with fewer buckets |
| `tracklet_gap_s` | `5.0` | s | 3 | a `(track, sensor)` silent this long starts a new tracklet |
| `offsets_apply` | `true` | | 4 | apply `offsets.json` when present |
| `shared_id_hint` | `true` | | 5a | join same-id tracklets across cameras |
| `shared_id_gate_m` | `1.5` | m | 5a | max median distance for a shared-id join |
| `dup_dist_m` | `1.2` | m | 5b | closeness for concurrent duplicates |
| `dup_frac` | `0.7` | ratio | 5b | share of overlap buckets that must be close |
| `dup_min_overlap_s` | `1.0` | s | 5b | minimum overlap to consider a duplicate |
| `stitch_gap_s` | `120.0` | s | 5c | max gap for a sequential stitch |
| `stitch_d0_m` | `0.8` | m | 5c | stitch distance at zero gap |
| `stitch_dv_mps` | `0.1` | m/s | 5c | stitch distance growth per second of gap |
| `stitch_dmax_m` | `3.5` | m | 5c | stitch distance cap |
| `stitch_exclusive` | `true` | | 5c | never stitch when the candidate's person and the target person overlap in time |
| `smooth_window` | `5` | buckets | 7 | rolling median window |
| `min_person_s` | `3.0` | s | 8 | drop shorter people |

`tracklet_gap_s` and `shared_id_gate_m` are the thresholds not taken from
the prototype: the prototype never split a tracklet on silence and has no
shared-id pass. For `shared_id_gate_m`, 1.5 m sits above the measured
seated same-id distance (about 1.2 m) so the hint keeps today's behavior,
and below the room's smallest plausible separation between two people who
were merged by an id mistake.

The room rectangle comes from the calibration: the floor plan size from
`calib.png_size` on `map.png` and the inverse of `calib.read_t_gt2px` from
`transforms.yml`
([`STEP-8` section 2.1](STEP-8-TRACKING-HISTORY.md#21-inputs-via-context)).
Without calibration output the configured `room` is used; when that is also
`null`, stage 1 skips the bounds check and counts nothing.

### 4.3 Relaxed preset (flagged decision)

> **Flagged decision:** the relaxed values are starting points, not measured
> ones. They are re-tuned after calibration footage is re-recorded and the
> cameras agree; the table is updated then.

| Key | Strict | Relaxed |
|---|---|---|
| `room_margin_m` | `0.3` | `0.5` |
| `vmax_mps` | `3.0` | `4.0` |
| `speed_slack_m` | `0.3` | `0.5` |
| `min_tracklet_s` | `1.0` | `0.5` |
| `min_tracklet_samples` | `5` | `3` |
| `shared_id_gate_m` | `1.5` | `1.0` |
| `dup_dist_m` | `1.2` | `0.8` |
| `dup_frac` | `0.7` | `0.6` |
| `stitch_gap_s` | `120.0` | `30.0` |
| `stitch_dmax_m` | `3.5` | `2.0` |
| `smooth_window` | `5` | `3` |
| `min_person_s` | `3.0` | `1.0` |

Every key not listed has the same value in both presets.

### 4.4 Zone entries

A zone is a rectangle or a polygon in AMC world metres, optionally limited
to some sensors. A sample is dropped when it falls inside a zone that
applies to its sensor.

```json
{
  "version": 1,
  "preset": "strict",
  "zones": [
    {"name": "static-1", "sensors": ["camera2"], "rect": {"x": [2.75, 3.75], "y": [-4.25, -3.25]}},
    {"name": "static-2", "polygon": [[0.0, 0.0], [0.6, 0.0], [0.6, -0.4]]}
  ]
}
```

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | label shown in reports and drop counters |
| `sensors` | no | sensor names the zone applies to; absent or empty means all |
| `rect` | one of | `{"x": [min, max], "y": [min, max]}`, inclusive |
| `polygon` | one of | at least three `[x, y]` vertices, even-odd rule |

A zone with both or neither of `rect` and `polygon` is invalid
([section 4.1](#41-file-and-reload-required)).

---

## 5. Fusion stages

`fuse.py` is stdlib only. It exposes a resumable `Fuser` that carries state
across time-ordered chunks, like `stitch.Stitcher` today, so `export` can
stream a 24 h window without loading it. The stages run in this order
(REQUIRED). The departures from the prototype are:

- **Shared-id pass** (stage 5a) with its `shared_id_gate_m` of 1.5 m; the
  prototype has none.
- **Offset correction** (stage 4); the prototype has none.
- **Inverse-variance weighted mean** across sensors (stage 6) instead of
  the prototype's per-person median.
- **Room bounds from the calibration** (stage 1) instead of hard-coded room
  dimensions.
- **An empty default `zones` list**; the prototype hard-coded one zone
  around the TV.

1. **Bounds and zones**: drop samples outside the room rectangle grown by
   `room_margin_m`, then samples inside any applicable zone
   ([section 4.4](#44-zone-entries)).
2. **Speed gate**: per `(track_id, sensor)`, keep a sample only when its
   distance from the last kept sample of that key is at most
   `vmax_mps * max(dt, min_dt_s) + speed_slack_m`.
3. **Tracklets**: group surviving samples by `(track_id, sensor)` into
   `bucket_ms` buckets (median position per bucket), starting a new
   tracklet after `tracklet_gap_s` of silence on that key; drop tracklets
   shorter than `min_tracklet_s` or with fewer than `min_tracklet_samples`
   buckets.
4. **Offset correction**: when `offsets_apply` is true and `offsets.json`
   is valid, apply each sensor's learned correction to its tracklets
   ([section 6](#6-offset-correction)), then re-check the room bounds so a
   corrected position cannot leave the room.
5. **Association into people**, a union-find over tracklets, in three
   passes:
   1. **Shared id**: when `shared_id_hint` is true, join tracklets with
      the same `track_id` on different sensors whose median distance over
      their common buckets is within `shared_id_gate_m`.
   2. **Concurrent duplicates**: join tracklets overlapping for at least
      `dup_min_overlap_s` when at least `dup_frac` of the common buckets are
      within `dup_dist_m`.
   3. **Sequential stitch**: a tracklet starting after another person's last
      bucket, within `stitch_gap_s`, joins the nearest such person whose
      end lies within `min(stitch_d0_m + stitch_dv_mps * gap, stitch_dmax_m)`.
      With `stitch_exclusive`, the conflict rule checks the whole person on
      both sides: the stitch is refused when any tracklet of the
      candidate's person overlaps in time any tracklet of the target
      person. A literal port of the prototype, which checked only the
      candidate tracklet, merged concurrent people already joined through
      the shared-id pass.
6. **Camera fusion**: per person per bucket, combine the sensors' positions
   as an inverse-variance weighted mean, using each sensor's `sigma_m` from
   `offsets.json` when applied, else equal weights. Record the number of
   sensors that contributed (`cameras`).
7. **Smoothing and post-merge speed gate**: rolling median over
   `smooth_window` buckets, then drop buckets that violate
   `vmax_mps * max(dt, bucket_ms / 1000) + speed_slack_m` against the last kept
   bucket, which removes jumps at join points.
8. **Minimum person duration**: drop people whose kept span is shorter
   than `min_person_s`.

People are numbered `1..N` by first appearance within one response or one
export.

### 5.1 Drop and merge counters (REQUIRED)

Every stage counts what it removed. The counts are returned in every fused
`range` summary and printed by `clean-report`.

| Counter | Stage | Counts |
|---|---|---|
| `out_of_bounds` | 1, 4 | samples; includes the post-correction bounds re-check of stage 4 |
| `zone` | 1 | samples, also broken down per zone `name` in `zones` |
| `speed` | 2 | samples |
| `short_tracklet` | 3 | tracklets (and their samples in `short_tracklet_samples`) |
| `shared_id_joins` | 5a | joins |
| `duplicate_joins` | 5b | joins |
| `stitch_joins` | 5c | joins |
| `post_merge_speed` | 7 | buckets |
| `short_person` | 8 | people (and their buckets in `short_person_buckets`) |

### 5.2 Streaming

A person is finalized and emitted once the gap since its last kept
tracklet exceeds `stitch_gap_s`, or at the end of the requested window.
Chunked processing produces exactly the one-shot result (REQUIRED).
Memory is bounded by the people open in the last `stitch_gap_s`, not by
the window length.
Because a person can only be finalized after that hold-back, `export` rows
are emitted in per-person blocks, each block time-ordered.

---

## 6. Offset correction

### 6.1 Learning method

`offsets.py` learns one 2D correction per sensor from **co-observed
same-id pairs**: samples of one `track_id` on two sensors in the same
`bucket_ms` bucket, after stages 1 to 3 of
[section 5](#5-fusion-stages).

1. **Collect pairs** over the requested window (default the last 24 h).
   Discard pairs farther apart than `shared_id_gate_m` (1.5 m strict).
2. **Fit**: solve per-sensor translations with a zero-sum constraint by
   least squares, so with two sensors each moves by half of the mean
   difference, toward the other. With `--model similarity`, also fit a small
   rotation and uniform scale per sensor; the fit falls back to translation
   when the rotation exceeds 5 degrees or the scale leaves `0.95..1.05`.
3. **Residual spread**: `sigma_m` per sensor is the median absolute
   residual after correction, used as the stage 6 weight.
4. **Report** the median cross-camera distance before and after, the gain,
   and the pair count. Refuse to write with fewer than `200` pairs.

`learn-offsets` is the only writer. Nothing applies the file except stage
4, and only while `offsets_apply` is true.

### 6.2 `offsets.json` schema

| Item | Value |
|---|---|
| Path | `<install_dir>/tracking/offsets.json` |
| Env var | `OFFSETS_FILE` |
| Reload | re-read when its mtime changes |

```json
{
  "version": 1,
  "learned_at_ms": 0,
  "window": {"from_ms": 0, "to_ms": 0},
  "model": "translation",
  "pairs": 0,
  "median_dist_before_m": 0.43,
  "median_dist_after_m": 0.12,
  "gain_m": 0.31,
  "needed": true,
  "sensors": {
    "camera1": {"dx": 0.0, "dy": 0.0, "theta_deg": 0.0, "scale": 1.0, "sigma_m": 0.0},
    "camera2": {"dx": 0.0, "dy": 0.0, "theta_deg": 0.0, "scale": 1.0, "sigma_m": 0.0}
  }
}
```

`needed` is `gain_m >= 0.1` (REQUIRED). A file with `needed: false` is
still applied when `offsets_apply` is true; the flag only drives the report
and the UI note.

---

## 7. Zone suggestions

`zones-suggest` grids the window's stage 1 and 2 survivors into 0.5 m cells
and reports connected groups of cells that meet **all** of:

| Criterion | Threshold |
|---|---|
| Single camera | at least 95% of the group's samples come from one sensor |
| Low motion | median per-tracklet speed at most 0.1 m/s |
| Long occupancy | occupied in at least 30% of the window's occupied seconds, or at least 10 minutes absolute |
| Id churn | at least 10 distinct tracker ids |

Each suggestion prints its evidence (sensor share, occupancy, id count) and
a ready-to-paste zone entry: a `rect` around the group padded by 0.25 m,
limited to the dominant sensor, named `suggested-<n>`. Nothing is written to
`cleaning.json`. A seated person can also be static; the single-camera rule
is what separates a fixed object from a person both cameras can see, and the
operator makes the final call.

---

## 8. API changes

Amends [`STEP-8` section 8](STEP-8-TRACKING-HISTORY.md#8-http-api). Auth,
lease and window limits are unchanged.

### 8.1 `range`

`GET /tracking/range?from&to&hz=5&mode=fused|raw`

| `mode` | Behavior |
|---|---|
| `fused` (default) | the engine of [section 5](#5-fusion-stages); each person carries `trackIds` and a per-sample `cameras` array |
| `stitched` | alias of `fused` for one release, then removed; the response reports `"mode": "fused"` |
| `raw` | unchanged: each tracker id is its own person |

Fusion always runs at `bucket_ms`; the response is downsampled to `hz` when
`hz` is coarser. A fused response adds `summary`:

| Field | Type | Meaning |
|---|---|---|
| `people` | int | people after stage 8 |
| `max_at_once` | int | max people present in any one second |
| `occupied_s` | number | seconds with at least one person |
| `persons` | array | per person: `id`, `duration_s`, `camera_share` (sensor name to fraction of buckets) |
| `dropped` | object | the counters of [section 5.1](#51-drop-and-merge-counters-required) |
| `offsets` | object | `applied`, `needed`, `gain_m`, `learned_at_ms`; `applied` is `false` when no valid file exists |
| `preset` | string | the active preset |
| `config_error` | string or null | the last `cleaning.json` load error |
| `config_warnings` | array of strings | non-fatal load warnings, e.g. `"ignored unknown key foo"` |

### 8.2 `export`

`GET /tracking/export?from&to&mode=fused|raw&format=csv|jsonl`

| `mode` | Rows | Columns |
|---|---|---|
| `fused` | one per person per bucket, streamed through `Fuser` in chunks | `ts_ms, person_id, x, y, cameras` |
| `stitched` | alias of `fused` for one release | as `fused` |
| `raw` | unchanged | unchanged |

The export's default `mode` stays what it is today; only the accepted
values change.

---

## 9. CLI and installer passthrough

The package `cli.py` gains three subcommands; Step 8's `tracking`
passthrough ([`STEP-8` section 10](STEP-8-TRACKING-HISTORY.md#10-cli))
forwards them unchanged:

```bash
mv3dt-installer tracking zones-suggest [--from T] [--to T]
mv3dt-installer tracking learn-offsets [--from T] [--to T] [--model translation|similarity]
mv3dt-installer tracking clean-report --from T --to T
```

| Command | Reads | Writes | Prints |
|---|---|---|---|
| `zones-suggest` | day files, `cleaning.json` | nothing | suggestions ([section 7](#7-zone-suggestions)) |
| `learn-offsets` | day files, `cleaning.json` | `offsets.json` | before/after, gain, pairs, `needed` |
| `clean-report` | day files, both files | nothing | the fused `summary` ([section 8.1](#81-range)) as JSON |

`T` is unix milliseconds or ISO 8601; an ISO time without an offset is
workstation local time. `--from` and `--to` default to the last 24 h where
optional. All three run as the invoking user against the read-only day
files, so they need neither the lease nor a ticket.

---

## 10. Step 8 changes

Amends [`STEP-8` section 11.2](STEP-8-TRACKING-HISTORY.md#112-runctx) and
[`STEP-8` section 4.4](STEP-8-TRACKING-HISTORY.md#44-the-unit).

1. **Seed `cleaning.json` once**: when absent, write
   `{"version": 1, "preset": "strict", "zones": []}` to
   `<install_dir>/tracking/cleaning.json`, owned by the invoking user. Only
   those keys are seeded, so a later `preset` change reaches every other
   key ([section 4.1](#41-file-and-reload-required)). An existing file is
   never touched, on re-run or on upgrade.
2. **Never create `offsets.json`**: only `learn-offsets` writes it.
3. **Unit environment**: `mv3dt-tracking.service.in` adds
   `Environment="CLEANING_FILE=@TRACKING_DIR@/cleaning.json"` and
   `Environment="OFFSETS_FILE=@TRACKING_DIR@/offsets.json"`;
   `settings.py` reads both, defaulting to those paths beside the package.
4. **Upgrade**: re-running Step 8 after an installer upgrade re-syncs the
   package and restarts the unit, leaving both files in place.
5. **Passthrough**: the `tracking` subcommand parser accepts the three
   commands of [section 9](#9-cli-and-installer-passthrough).

---

## 11. Installer upgrade fixes

Found while migrating the workstation to 0.4.17. Independent of fusion and
shipped as their own unit.

1. **Atomic binary replace (REQUIRED)**: replacing
   `<install_dir>/bin/mv3dt-installer` while that copy is running failed
   with `Text file busy`. Write the new binary to a temp file in the same
   directory, `chmod 755`, `fsync`, then `os.replace` it over the target,
   so the running process keeps its old inode.
2. **Interactive Step 5 prompt honors Ctrl-C and SIGTERM**: the interactive
   Step 5 prompt, with its progress spinner, did not react to Ctrl-C or
   `SIGTERM`, and blocked when stdout was piped. It must end the run with
   the step left not `COMPLETE`, and must not block on a non-tty stdout.
   F1 identifies the exact module responsible and lists it in its PR.
   `waitui.py` is not the bug: its existing `KeyboardInterrupt` to
   `CANCELLED` handling correctly halts without `COMPLETE` and is kept
   unchanged.
3. **Gated-off steps are not marked complete**: a step skipped because its
   gate is `off` is logged as skipped but not persisted as `COMPLETE`, so
   turning the gate on later runs it. This amends the auto-`COMPLETE` rule
   of [`00` section 3.4](00-FRAMEWORK-AND-BOOTSTRAP.md#34-opt-in-step-gates)
   and [`STEP-8` section 4.1](STEP-8-TRACKING-HISTORY.md#41-gate).
4. **Step 5 registry drift for AMC 3.3.0**: the reconciliation check of
   [`STEP-5` section 5.4](STEP-5-PER-PROJECT-EXES.md#54-project-removal-amc-gui-driven--reconciliation-check)
   looked only under the pre-3.3.0 project path, so live projects of the
   AMC 3.3.0 layout
   ([`LIVE-BASELINE-AND-EDIT-PROTECTION` section 2.3](LIVE-BASELINE-AND-EDIT-PROTECTION.md#23-amc-330-pin-locked))
   were reported as drifted. Resolve the project directory under the
   configured `AMC_ROOT` for the 3.3.0 layout before declaring drift; the
   exact 3.3.0 project path is confirmed on the workstation first.

---

## 12. Web Replay changes

Owned by the web app repository (`P2BP-25_26-Web`); stated here as the
contract the API serves. Files: `TrackingReplayPanel.tsx`,
`workstationTrackingApi.ts`, `trackingTypes.ts`.

- Mode toggle **Fused / Raw**, default Fused.
- Legend header `N people · max M at once` from `summary`.
- People numbered `1..N` by first appearance, with per-person duration and
  camera share.
- Trails break on a gap over 2 s or a step over 1 m.
- A small cleaning note: points dropped, offset applied yes/no, and
  "offset correction not needed" when `offsets.needed` is false.
- Export Fused or Raw.

---

## 13. Verification

1. **Unit tests per PR**: the installer suite passes except the known
   `tests/test_config.py::test_persist_value_preserves_existing_keys_and_overwrites_its_own`
   failure and the known flaky
   `tests/test_app.py::test_verbose_does_not_change_whether_run_root_streams`;
   web build, typecheck and lint pass.
2. **Synthetic metric tests** (`tests/tracking/test_fuse.py`,
   `test_offsets.py`, `test_cleaning.py`) on fixtures mirroring the real
   patterns: churning ids, a static single-camera object, a seated person,
   two concurrent walkers, and a 1 m injected camera offset. Assert the
   people count, max at once, zero out-of-bounds points, max step speed
   bounded by `vmax_mps` plus slack, the learned offset within 5 cm of the
   injected value, a `zones-suggest` hit on the static object only, and a
   preset change taking effect on the next call without a restart.
3. **Offline check** (local only, not committed) against the exported hour
   and a copy of the database: the hour gives about 10 people and at most
   2 at once; the walking window gives about 3 people; `clean-report`
   counters are plausible. These figures assume the suggested TV zone from
   `zones-suggest` is pasted into `cleaning.json`, as the prototype's
   hard-coded zone did; with the default empty `zones` list the count is
   higher.
4. **Live checks after release**:
   - [ ] Step 8 re-run seeds `cleaning.json` and leaves an existing one
         untouched.
   - [ ] `learn-offsets` reports a correction near 0.4 m with
         `needed: true`.
   - [ ] `zones-suggest` lists the static TV hotspot.
   - [ ] Replay of the walking window in Fused mode shows about 3 people
         with continuous paths; Raw mode is unchanged.
   - [ ] A fused CSV export opens with the five columns.
   - [ ] Setting `"preset": "relaxed"` changes the result without a
         restart.
   - [ ] Replacing the installed binary while it runs succeeds.

---

## 14. Decomposition

| Unit | Branch | Files touched | Depends on | Wave |
|---|---|---|---|---|
| C1 Spec | `feat/installer-tracking-fusion-spec` | `installer/plan/TRACKING-FUSION-AND-CLEANING.md`, `installer/plan/DEVELOPMENT-STATUS.md` | none | 1 |
| C2 Fusion engine | `feat/installer-tracking-fuse` | `mv3dt_tracking/fuse.py`, `offsets.py`, `cleaning.py` (config + zones + suggestions), tests `tests/tracking/test_fuse.py`, `test_offsets.py`, `test_cleaning.py` | none | 1 |
| F1 Installer fixes | `feat/installer-upgrade-fixes` | `steps/step3_amc_launcher.py` (binary replace), `app.py` (signals, gated-off steps), `steps/step5_per_project_exes.py` (registry drift), their tests | none | 1 |
| C3 API + CLI | `feat/installer-tracking-fused-api` | `mv3dt_tracking/api.py`, `cli.py`, `settings.py`, `tests/tracking/test_api.py`, `test_cli.py` | C2 | 2 |
| W1 Replay fused view | `feat/replay-fused-view` | client `TrackingReplayPanel.tsx`, `workstationTrackingApi.ts`, `trackingTypes.ts` | C3 (contract) | 2 |
| C4 Step 8 wiring | `feat/installer-step8-cleaning` | `steps/step8_tracking_history.py` (seeding, tracking passthrough), `assets/systemd/mv3dt-tracking.service.in`, `tests/test_step8_tracking_history.py` | C3 | 3 |

Package paths are relative to
`installer/mv3dt_installer/assets/tracking/` for `mv3dt_tracking/` and to
`installer/mv3dt_installer/` otherwise; test paths are relative to
`installer/`. W1 lives in the web app repository.

Serialization points:

- C3 imports `fuse`, `offsets` and `cleaning`, so it branches from C2.
- W1 depends only on C3's wire contract
  ([section 8](#8-api-changes)), not on its code.
- C4 seeds the file C2 reads and passes through the commands C3 adds, so it
  lands last.
- Recommended merge order: C1, C2, F1, C3, C4, then a release (0.4.18).
  W1 merges after that release is installed on the workstation.

---

## References

DeepStream facts here (per-camera output with shared MV3DT ids, the
`footLocation3D` field) come from NVIDIA's DeepStream documentation and the
`nv.Frame` schema already cited by
[`STEP-8`](STEP-8-TRACKING-HISTORY.md#references). The measured numbers
come from a local analysis of one recorded day and are not committed.

- [DeepStream MV3DT](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_MV3DT.html)
  — **backs
  [section 2](#2-problem-and-measured-evidence)**: MV3DT output is per
  camera, with ids shared across views.
- [Gst-nvmsgconv](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvmsgconv.html)
  — the converter producing the recorded per-camera messages.
- [Python `os.replace`](https://docs.python.org/3/library/os.html#os.replace)
  — **backs [section 11](#11-installer-upgrade-fixes) item 1**: atomic
  rename over a busy executable.

Repo files referenced:

- [`STEP-8-TRACKING-HISTORY.md`](STEP-8-TRACKING-HISTORY.md) — sections 2.1,
  4.1, 4.4, 5, 7, 8, 10 and 11.2, which this doc amends.
- [`00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md) — section
  3.4, the gate rule amended by
  [section 11](#11-installer-upgrade-fixes).
- [`STEP-5-PER-PROJECT-EXES.md`](STEP-5-PER-PROJECT-EXES.md) — section 5.4,
  the registry reconciliation check.
- [`LIVE-BASELINE-AND-EDIT-PROTECTION.md`](LIVE-BASELINE-AND-EDIT-PROTECTION.md)
  — section 2.3, the AMC 3.3.0 pin and `AMC_ROOT`.
- [`DEVELOPMENT-STATUS.md`](DEVELOPMENT-STATUS.md) — carries this doc's
  status row.
- `installer/mv3dt_installer/assets/tracking/mv3dt_tracking/stitch.py` —
  the `Stitcher` the resumable `Fuser` follows and the stitching it
  replaces.
- `installer/mv3dt_installer/assets/tracking/mv3dt_tracking/calib.py` —
  `read_t_gt2px` and `png_size`, the room bounds source.
- `installer/mv3dt_installer/assets/tracking/mv3dt_tracking/api.py`,
  `cli.py`, `settings.py` — the routes, commands and env settings changed
  here.
- `installer/mv3dt_installer/assets/systemd/mv3dt-tracking.service.in` —
  gains `CLEANING_FILE` and `OFFSETS_FILE`.
