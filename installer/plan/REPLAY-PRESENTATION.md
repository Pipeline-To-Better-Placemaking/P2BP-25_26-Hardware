# Replay presentation — camera markers, comet trails and a dwell heatmap (owner: DevD)

Status: cross-repo design spec, **specified, not implemented**. It extends
the `cleaning.json` contract in
[`TRACKING-FUSION` section 4](TRACKING-FUSION-AND-CLEANING.md#4-cleaning-configuration),
the `/tracking/meta` response of
[`STEP-8` section 8](STEP-8-TRACKING-HISTORY.md#8-http-api), and the web Replay contract in
[`TRACKING-FUSION` section 12](TRACKING-FUSION-AND-CLEANING.md#12-web-replay-changes).
Those contracts are **not** restated here; this doc's spec unit amends the
rows of each that it changes. The web
units are owned by the web app repository (`P2BP-25_26-Web`) and are stated
here as the contract both sides build to.

The Floor Plan Replay draws two things badly. The camera markers sit where
the calibration's projection matrices place the cameras, which for the AMC
run 2 calibration is 0.4 to 0.6 m from where the cameras are mounted: one
marker lands outside the room. And every person's trail accumulates from
the start of the window to the playhead, so a five-minute replay ends as a
**scribble of lines** that hides where anyone is now.

This doc makes the marker positions a **workstation setting**, replaces the
accumulated trail with a **moving dot and a short fading tail** (the full
path stays available), and adds a **dwell heatmap** that summarizes where
people spent time over the loaded window.

---

## 1. Scope

In scope:

- a `cameras` key in `cleaning.json` that overrides the marker position of
  named cameras in `/tracking/meta`;
- a tail selector and comet-style trail rendering in the web Replay;
- a heatmap view of dwell time over the loaded range in the web Replay.

Out of scope: [section 7](#7-out-of-scope--open-decisions).

---

## 2. Measured evidence

From the workstation and the Replay on 2026-11-12, AMC run 2 calibration
live, floor-plan scale 99.98 px/m.

| Fact | Value | Source |
|---|---|---|
| `camInfo_00` marker (cam-4484, inventory label `kitchen-side`) | about (7.46, -0.82) m, outside the right wall, 0.62 m from the corner | camera centre from `camInfo_00.yml` |
| `camInfo_00` mounted | top-right corner, (7.05, -0.35) m | operator |
| `camInfo_01` marker (cam-4487, inventory label `back-wall`) | about (0.79, -4.21) m, 0.43 m inside the left wall | camera centre from `camInfo_01.yml` |
| `camInfo_01` mounted | (0.36, -4.21) m, on the left wall of the floor plan | operator |
| Room walls (`cleaning.json` `room`) | x 0.36 to 7.05, y -4.65 to -0.35 | `cleaning.json` |
| Five-minute fused replay, one person walking | one continuous path covering most of the room | Replay screenshot |
| Recorded positions inside a drawn object, same window | 48 of 3012, at most 24 cm deep | tracking database |

The last row shows the "walking through furniture" seen in the Replay is
mostly the accumulated line joining samples, not where the cameras placed
the person.

---

## 3. Decisions

### 3.1 Marker positions are a workstation setting (LOCKED)

**LOCKED: the override lives in `cleaning.json`.** It is already the
per-site file the tracking service re-reads without a restart, and its
`room` key already overrides a calibration-derived value. The override is
**display only**: fusion, cleaning and exports never read it, and a mistake
in it never changes how data is cleaned
([section 4.1](#41-key-required)).

### 3.2 Comet trail by default (RESOLVED)

**RESOLVED 2026-11-12, by the operator:** the default view is a moving dot
with a 5 s fading tail. The selector offers `2 s`, `5 s`, `15 s` and `Full`;
`Full` is today's accumulated trail.

### 3.3 Heatmap computed in the browser (LOCKED)

**LOCKED: the heatmap is built from the `range` response the Replay already
loads.** No new API route, no server-side aggregation. The data is the same
the trails use, so the heatmap and the trails always agree.

---

## 4. `cleaning.json` camera markers

### 4.1 Key (REQUIRED)

```json
{
  "cameras": {
    "camInfo_00": {"x": 7.05, "y": -0.35},
    "camInfo_01": {"x": 0.36, "y": -4.21}
  }
}
```

| Rule | Value |
|---|---|
| Key | `cameras`, an object; absent or `null` means no override |
| Entry name | a camera model stem as `/tracking/meta` reports it, for example `camInfo_00` |
| Entry value | an object with numeric `x` and `y` in world metres (booleans and `null` are not numeric); any other field is an error |
| Invalid value | **REQUIRED exception to [`TRACKING-FUSION` section 4.1](TRACKING-FUSION-AND-CLEANING.md#41-file-and-reload-required):** an invalid `cameras` value does not make the file invalid. The whole `cameras` value is ignored (no overrides), every other key still applies, and the problem is listed in `camera_override_warnings` and in `summary.config_warnings` |
| Preset | a strict-preset key with default `null`, like `room`; the relaxed preset inherits it |
| Reload | as the rest of the file: re-read when its mtime changes |

### 4.2 `/tracking/meta` (REQUIRED)

Each entry of `cameras` keeps its `name`, `x`, `y` and `z`, and gains
`source`: `"config"` when `cleaning.json` overrides its `x` and `y`, else
`"calibration"`. `z` always comes from the calibration. An override whose
name matches no camera model is ignored. Every ignored override, and an
invalid `cameras` value, is listed in the response as
`camera_override_warnings`, a list of strings; the key is omitted when the
list is empty. Meta reads the same cached `cleaning.json` as `range`. While
the file as a whole is invalid, meta serves the overrides of the last valid
config, as `range` does for every other key.

The web Replay does not display `source` or `camera_override_warnings`;
they are for the API and for checking the workstation by hand.

---

## 5. Comet trails (web)

### 5.1 Controls (REQUIRED)

A segmented control `Tail: 2 s · 5 s · 15 s · Full`, default `5 s`, beside
the speed buttons. It applies in Fused and Raw mode alike. It is not
persisted between visits.

### 5.2 Rendering (REQUIRED)

| Item | Rule |
|---|---|
| Dot | a person's dot and label are drawn while the playhead is inside one of their trail segments, or within 1 s after a segment's last sample; during a gap longer than that no dot is drawn. This applies to every tail setting, `Full` included |
| Tail window | samples with timestamps in `[playhead - tail, playhead]`, plus the interpolated head at the playhead |
| Breaks | the existing breaks apply inside the window: a gap over 2 s or a step over 1 m starts a new segment |
| Fade | opacity is a function of time only: 0 at the window start rising linearly to 1 at the head, in the person's colour; a segment that starts later in the window starts partly opaque, and a segment that ends before the head ends below full opacity |
| Head | the line from the last sample to the interpolated head is drawn at the head's opacity, 1 |
| Window edge | a line between the last sample before the window and the first inside it is cut at the window start, not dropped |
| `Full` | today's accumulated rendering, unchanged, including its 0.7 opacity |
| Scrubbing | moving the playhead backwards redraws correctly; nothing accumulates outside the tail window |

---

## 6. Dwell heatmap (web)

### 6.1 Controls (REQUIRED)

A segmented control `View: Movement · Heatmap`, default `Movement`. In
Heatmap view the playback controls and tail selector are disabled, and the
people legend stays as the summary of the window.

### 6.2 Computation (REQUIRED)

| Item | Rule |
|---|---|
| Input | every person in the loaded `range` response, Fused or Raw as selected |
| Grid | square cells of 0.25 m in world coordinates, aligned to the world origin: cell `(floor(x / 0.25), floor(y / 0.25))` |
| Extent | the world rectangle covered by the floor-plan image: its four pixel corners mapped through the inverse of `T_gt2px` |
| Weight | for consecutive samples of one person within one trail segment ([section 5.2](#52-rendering-required) breaks), add the time between them to the cell of the earlier sample; the breaks already bound it at 2 s |
| Outside | a sample outside the extent adds nothing |
| No floor plan | when `/tracking/meta` has no `floorplan`, the Heatmap option is disabled with the note `Heatmap needs a floor plan` |
| Unit | person-seconds per cell |
| Raw mode | each camera's track counts separately; the legend says so |

### 6.3 Rendering (REQUIRED)

Cells drawn over the floor plan with a single-hue sequential colour ramp,
transparent at zero and most opaque at the window maximum, so walls and
furniture stay visible. A legend shows the ramp with `0` and the maximum in
seconds. Cells with no time are not drawn.

---

## 7. Out of scope / open decisions

- Long-term re-identification across exits. The fused view already joins
  tracklets by place and time
  ([`TRACKING-FUSION` section 5](TRACKING-FUSION-AND-CLEANING.md#5-fusion-stages)).
- Persisting the tail or view choice per viewer.
- Exporting the heatmap.

Open decisions:

1. **Flagged:** heatmap cell size is fixed at 0.25 m. A selector can follow
   if larger spaces need coarser cells.

---

## 8. Verification

1. **Hardware**: tests for the `cameras` key (valid override, unknown name
   warning, invalid values making the file invalid, no effect on fusion)
   and for the `/tracking/meta` fields of
   [section 4.2](#42-trackingmeta-required).
2. **Web**: the client has no test runner, so verification is
   `npm run build` (type check and bundle), `npm run lint`, a throwaway
   script exercising the pure tail-window and heatmap functions against the
   tables above (run by the implementer and again by the reviewer, never
   committed), and a check by hand in the deployed Replay.
3. **Workstation, after release**: `cleaning.json` gains the two
   [section 4.1](#41-key-required) entries, and the Replay shows both
   markers on the walls at the mounted positions.

---

## 9. Decomposition

| Unit | Branch | Files touched | Depends on | Wave |
|---|---|---|---|---|
| P1 Spec | `feat/installer-replay-presentation-spec` | `installer/plan/REPLAY-PRESENTATION.md`, `installer/plan/TRACKING-FUSION-AND-CLEANING.md` (section 4.2 row, section 12 contract), `installer/plan/STEP-8-TRACKING-HISTORY.md` (section 8 meta row) | none | 1 |
| P2 Camera marker override | `feat/installer-camera-markers` | `assets/tracking/mv3dt_tracking/cleaning.py`, `assets/tracking/mv3dt_tracking/api.py`, `tests/tracking/test_cleaning.py`, `tests/tracking/test_api.py` | none | 1 |
| P3 Comet trails (web) | `feat/replay-comet-trails` | `BetterPlacemaking.CLIENT.REACT/src/pages/projects/scanner/TrackingReplayPanel.tsx` | none | 1 |
| P4 Dwell heatmap (web) | `feat/replay-heatmap` | `BetterPlacemaking.CLIENT.REACT/src/pages/projects/scanner/TrackingReplayPanel.tsx`, a new `BetterPlacemaking.CLIENT.REACT/src/pages/projects/scanner/replayHeatmap.ts` | P3 | 2 |

P1 and P2 are in this repository, with package paths relative to
`installer/mv3dt_installer/` and test paths relative to `installer/`. P3
and P4 are in `P2BP-25_26-Web`, paths relative to its root.

Serialization points:

- P3 and P4 both edit `TrackingReplayPanel.tsx`, so P4 is in wave 2 and
  builds on P3's branch: the heatmap view switches off the controls P3 adds.
- P2 and P3 are independent: the web client already ignores unknown
  `/tracking/meta` fields, so they can merge in either order.
- Recommended order: Hardware P1, P2, then a release; web P3, P4, then the
  Vercel deploy.

---

## References

Facts come from the tracking service code in this repository, the web
Replay code in `P2BP-25_26-Web`, and measurements on the workstation on
2026-11-12.

- [`deepstream_reference_apps`](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps)
  — the camera model format the marker positions are computed from.

Repo files referenced:

- [`TRACKING-FUSION-AND-CLEANING.md`](TRACKING-FUSION-AND-CLEANING.md) —
  sections 4, 5 and 12: `cleaning.json`, fusion stages and the Replay
  contract.
- [`STEP-8-TRACKING-HISTORY.md`](STEP-8-TRACKING-HISTORY.md) — the tracking
  service and its `/tracking/meta` route.
- [`DETECTION-EXCLUSION-ZONES.md`](DETECTION-EXCLUSION-ZONES.md) — the
  calibration and detector changes live when this was measured.
