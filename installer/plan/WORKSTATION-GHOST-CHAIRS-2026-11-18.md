# Living-room chair detections — workstation investigation, 2026-11-18

Status: incident record and applied workstation mitigation. Empty-room
verification passed; a current walk-and-sit acceptance test remains open.
This record does not restate the filter contract in
[`DETECTION-EXCLUSION` §3](DETECTION-EXCLUSION-ZONES.md#3-decisions).

The operator supplied an image of two person boxes on empty dining chairs.
SSH inspection of the workstation and its SQLite history reproduced both
the live false detection and the web application's seven-person replay.
This is a deployment-specific configuration change, not a new detector or
a global installer default.

---

## 1. Evidence and cause

| Observation | Verified value |
|---|---|
| Workstation | `p2bp-admin-workstation`, installer `0.4.30`, commit `5f183dc` |
| Pipeline | `mv3dt-pipeline@living-room.service`, `mv3dt-test5-app`, two cameras |
| Detector | PeopleNet, `pre-cluster-threshold=0.4` |
| Existing exclusion | Source 1 / `camera2`: TV rectangle only |
| Missing exclusion | Source 0 / `camera1`: the two empty dining chairs |
| Replay configuration | Strict fusion, `zones=[]`, `smooth_window=15` |
| Reproduced window | Unix milliseconds `1795009905000` through `1795013505000` |
| Fused result | 7 inferred people, peak 3 |
| Nearby raw query | 199 `(sensor, track_id)` combinations in the preceding hour |
| Detector-only probe | 200 logged boxes, chair confidence reaching 0.95 |

The fresh viewer image showed a person box on the rear empty chair. The
detector-only probe disabled tracking, MQTT and the RTSP output sink, so its
false boxes establish that the problem starts in PeopleNet. They were not
created only by replay fusion. The full-frame diagnostic exclusions logged
detector geometry before removing it and could not contaminate history.

**RESOLVED:** a confidence increase alone is not a reliable chair fix: the
model sometimes rates the chair at 0.95. The existing tracker has
`probationAge=2`; raising that may reject brief false detections, but does
not establish that a repeatedly detected chair is a person. Neither
threshold nor probation was changed in this incident.

Tracker IDs are target identifiers, not a count of distinct human beings.
The web application's `TrackingReplayPanel.tsx` displays the workstation's
`summary.people` from `fused_range()`; it does not derive seven people from
the largest tracker ID. False detections and fragmented tracks survive
fusion as extra inferred people. Even after fixing ghosts, reappearances
outside the fusion matching window can still become separate identities.

---

## 2. Applied configuration

At approximately **09:55:31 EST / 14:55:31 UTC**, two rectangles were added
to camera `c1` in `/opt/mv3dt/cameras.yml` and source 0 in
`/opt/mv3dt/deepstream/exclusion.conf`. The existing camera `c2` TV rule was
preserved. Only the living-room pipeline was restarted.

| Camera / source | Left | Top | Width | Height |
|---|---:|---:|---:|---:|
| `c1` / source 0, front chair | 450 | 470 | 390 | 455 |
| `c1` / source 0, rear chair | 660 | 375 | 300 | 530 |
| `c2` / source 1, existing TV | 530 | 175 | 245 | 225 |

Coordinates are lens-corrected 1920x1080 detector pixels, not pixels of the
side-by-side viewer. The exact camera inventory value is:

```yaml
exclude_boxes: "450,470,390,455;660,375,300,530"
```

**REQUIRED:** retain full-box containment as specified in
[`DETECTION-EXCLUSION` §3.2](DETECTION-EXCLUSION-ZONES.md#32-inside-only-rule-locked).
A detection extending beyond a rectangle is retained. This reduces the
impact on a standing person crossing a chair, but does not guarantee that
a seated or heavily occluded person's box cannot fit inside it. Such a
person still needs explicit acceptance testing. Camera 2 receives no new
exclusion and retains its view of the dining area.

The inventory retains these rules through rescans and supplies future
renders under
[`DETECTION-EXCLUSION` §7](DETECTION-EXCLUSION-ZONES.md#7-configuration-camerasyml-and-step-4).
The edited live output also benefits from the installer's managed-file edit
protection. The room-specific rectangles were deliberately not added to the
fleet seed or any global default. Recheck them after moving cameras or
furniture or changing rectification.

---

## 3. Validation and historical replay

The detector-only trial logged 542 removed chair boxes in approximately one
minute. After the production restart, startup logs confirmed both source 0
rectangles and the unchanged source 1 TV rectangle. The viewer's empty-room
image contained no person boxes. Both streams settled at approximately
22 FPS; the history service remained connected with no decode or sink
errors. The first complete minute logged 560 rejected chair detections.

For 225.7 seconds after the 15-second startup allowance, the empty room
produced **zero stored detections**. Subsequent activity produced fresh
records; a later viewer frame showed a real person near the doorway. The
zero-detection claim applies to the bounded empty interval, not all time
after the restart. `post-change.json` records the later, active interval.
The final post-change query, saved as `post-change-latest.json`, returned
one inferred person with a peak of one. Pipeline, tracking recorder and
watchdog services were all active.

A separate detector-only run on seconds 15–55 of the workstation's archived
`demo-project` calibration footage used the current detector and dewarpers.
The sample image shows three people. All 5,654 detector boxes across 2,004
camera frames lay outside full containment by the effective exclusions
(2,758 boxes from camera 1 and 2,896 from camera 2). Thus the rules would
retain all detections in that recorded sample. This is a geometric check
against detector metadata, not a current seated-person acceptance test.

Local regression checks for the existing exclusion parser, containment rule
and camera inventory passed: **88 tests** in
`tests/test_exclusion_rules_c.py` and `tests/test_cameras.py`.

**RESOLVED:** historical records were not deleted, relabeled or overwritten.
A temporary, unapplied `cleaning-trial.json` restricted two small world-space
chair zones to `camera1`. Reprocessing the fixed hour reduced the fused
result from 7 to 2. The remaining second track comprised short camera 1
fragments near the room boundary around 09:04:56–09:05:37 EST; there is not
enough evidence to classify those fragments as another ghost or as the same
real person. Expanding exclusions merely to force a result of one would be
unsupported.

The historical trial was not installed because a floor-position exclusion
would also suppress valid future camera 1 samples at those positions,
regardless of the person's full image bounding box. The live
`/opt/mv3dt/tracking/cleaning.json` remains unchanged. Old replay windows can
therefore still show their recorded ghosts; select a window beginning after
the restart to assess the new pipeline.

---

## 4. Evidence and rollback

The Mac evidence directory is
`/Users/kevintto/Desktop/living-room-diagnostics-2026-11-18/`. It contains
before/after viewer frames, sanitized detector logs, effective non-secret
configuration, the fixed-window replay results and the unapplied cleaning
trial. Diagnostic media and history are local; no external upload was made.

The workstation's original configuration is backed up under:

```text
/opt/mv3dt/backups/ghost-chairs-20261118T1500Z/
```

The directory suffix is an incident label; `applied-at-ms` records the
actual change time. `original-hashes.json` records the original SHA-256
digests. To revert this incident alone, first check that neither live file
has received subsequent operator edits, then restore both backups:

```bash
sudo cp -p /opt/mv3dt/backups/ghost-chairs-20261118T1500Z/cameras.yml /opt/mv3dt/cameras.yml
sudo cp -p /opt/mv3dt/backups/ghost-chairs-20261118T1500Z/exclusion.conf /opt/mv3dt/deepstream/exclusion.conf
sudo systemctl restart mv3dt-pipeline@living-room.service
```

---

## References

Live detector logs and read-only SQLite/fusion queries are the primary
evidence for this incident. NVIDIA's guidance supports the distinction
between confidence filtering and tracker late activation; it does not
validate these room-specific rectangles.

- [NVIDIA accuracy tuning](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_Accuracy.html)
  — confidence thresholds, probation and false-positive handling.

Repo files referenced:

- [`DETECTION-EXCLUSION-ZONES.md`](DETECTION-EXCLUSION-ZONES.md) — containment
  rule, coordinates, persistence and source mapping.
- [`exclusion/mv3dt_exclusion.c`](../mv3dt_installer/assets/deepstream/exclusion/mv3dt_exclusion.c)
  — pre-tracker removal and bounded debug output.
- [`tracking/api.py`](../mv3dt_installer/assets/tracking/mv3dt_tracking/api.py)
  — `fused_range()` and `summary.people` consumed by web replay.
- Web repo at `e900952`,
  `BetterPlacemaking.CLIENT.REACT/src/pages/projects/scanner/TrackingReplayPanel.tsx`
  — renders the workstation summary, with inferred people for fused mode
  and tracker IDs for raw mode.
