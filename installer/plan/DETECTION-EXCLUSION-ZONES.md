# Detection exclusion zones — drop TV detections before the tracker (owner: DevD)

Status: cross-step design spec, **specified, prototype-validated on the
workstation, not implemented in the installer**. It amends the DeepStream
config render in
[`STEP-4` section 6](STEP-4-CALIB-OUTPUT-WIRING.md#6-deepstream-configuration-wiring),
the pipeline runtime in
[`STEP-5` section 3.3](STEP-5-PER-PROJECT-EXES.md#33-what-the-exe-does-at-runtime-pipeline-subcommand),
the camera inventory in
[`00` section 15.5](00-FRAMEWORK-AND-BOOTSTRAP.md#155-outputs), and the
tracker baseline in
[`LIVE-BASELINE` section 3.1](LIVE-BASELINE-AND-EDIT-PROTECTION.md#31-tracker-config_tracker_nvmotyml).
It reverses one clause of
[`TRACKING-FUSION` section 3.1](TRACKING-FUSION-AND-CLEANING.md#31-workstation-only-fusion-locked).
Edit protection is
[`LIVE-BASELINE` section 4](LIVE-BASELINE-AND-EDIT-PROTECTION.md#4-managed-files-edit-protection)
and the step update hook is `_pending_update()` in `app.py`; neither is
restated here.

PeopleNet detects people shown on the living-room TV. Each detection becomes
a tracker target, MV3DT publishes it to the peer camera, and the fused IDs of
real people nearby are disturbed: on 2026-11-12 a TV ghost appeared on
`camera2` in the same second that both real people in the room were given
new IDs. The workstation cleaning layer already drops these points from
history ([`TRACKING-FUSION` section 3.4](TRACKING-FUSION-AND-CLEANING.md#34-zones-locked)),
but by then the tracker has used them. This doc removes them **between the
primary detector and `nvtracker`**, so the tracker never sees them.

The rule is geometric and deliberately narrow: a box is dropped only when it
lies **entirely inside** a configured rectangle of one camera's frame. A
person standing in front of the TV overlaps the TV rectangle but extends below
it, so the person is kept. The filter is a pad probe compiled into a patched
`deepstream-test5-app`, built on the workstation by Step 5 from the
DeepStream sources the SDK already installs.

---

## 1. Scope

In scope:

- a C pad probe on the primary detector's `src` pad, and the two-line hook
  that attaches it in `deepstream_test5_app_main.c`;
- building that patched app on the workstation as
  `<install_dir>/bin/mv3dt-test5-app`, and running it in place of
  `deepstream-test5-app` when present;
- a per-camera `exclude_boxes` field in `cameras.yml`, rendered by Step 4
  into `<install_dir>/deepstream/exclusion.conf`;
- the tracker baseline change `maxPeerToPredDistance4Fusion: 2.0` to `1.0`.

Out of scope: [section 10](#10-out-of-scope--open-decisions).

---

## 2. Measured evidence

All values are from the workstation, 2026-11-12, AMC run 2 calibration live.

| Fact | Value | Source |
|---|---|---|
| TV location, `camera2` (`d0:3b:f4:02:44:87`, source 1) | x `541`–`762`, y `186`–`387` px in the 1920x1080 lens-corrected frame | rectified frame from AMC, matches `nvdewarper` within 0.1 px |
| TV ghost boxes | 65–110 px wide, 109–141 px tall, confidence 0.40–0.77 | prototype debug log |
| Prototype rectangle | `530,175,245,225` (left, top, width, height) | TV plus about 10 px margin |
| Prototype result, 3 min clip | 170 boxes dropped, all inside the TV; 0 boxes overlapping the rectangle kept | `MV3DT_EXCLUSION_DEBUG=1` log |
| Live ghosts in the same 3 min | 102 detections under 8 track IDs behind the TV wall | `/var/lib/mv3dt-tracking` |
| Streammux resolution | 1920x1080 | `deepstream_app_config.rendered.txt` `[streammux]` |
| DeepStream / CUDA | 9.1.0 / `/usr/local/cuda-13.2` | `/opt/nvidia/deepstream/deepstream/version` |
| Prototype throughput | about 400 fps per stream, file input, filter on | prototype run |

---

## 3. Decisions

### 3.1 Filter before the tracker (LOCKED)

**LOCKED: drop excluded boxes on the primary detector's `src` pad.** In
`deepstream-app`'s common pipeline the tracker is linked directly after the
primary detector. The other insertion points do not work: `[pre-process]`
ROIs blank image regions and would cut the upper body off a person standing
in front of the TV, `[ds-example]` sits after the tiler, and
`[nvds-analytics]` sits after the tracker.

### 3.2 Inside-only rule (LOCKED)

**LOCKED: a box is dropped only when all four edges lie inside one
rectangle**, inclusive:
`left >= r.left`, `top >= r.top`, `left + width <= r.right`,
`top + height <= r.bottom`. Overlap is never enough. Rectangles are in
streammux pixels, which are the lens-corrected frame because the dewarper
runs before the muxer.

### 3.3 Patched test5 app built on the workstation (LOCKED)

**LOCKED: Step 5 compiles a patched copy of DeepStream's own
`deepstream-test5` sources.** The patch is two additions to
`deepstream_test5_app_main.c` and one to its `Makefile`; every other file is
the SDK's. Building on the workstation keeps the binary matched to the
installed SDK, and nothing prebuilt from NVIDIA's sources ships in the
release.

### 3.4 Off unless configured (REQUIRED)

**REQUIRED: the filter never stops the pipeline.** An unset
`MV3DT_EXCLUSION_CONFIG`, an unreadable file, or a file without valid
rectangles leaves the probe unattached and logs one line. A failed build
leaves the stock `deepstream-test5-app` in use and Step 5 warns.

### 3.5 Amendment to TRACKING-FUSION section 3.1 (RESOLVED)

**RESOLVED 2026-11-12, by the operator:** the "no DeepStream changes" clause
of [`TRACKING-FUSION` section 3.1](TRACKING-FUSION-AND-CLEANING.md#31-workstation-only-fusion-locked)
is lifted for this filter and the fusion gate in
[section 8](#8-tracker-fusion-gate). The rest of that section, including no
2D-box position bypass, stands.

---

## 4. The filter

### 4.1 Files

Bundled under `installer/mv3dt_installer/assets/deepstream/exclusion/`:

| File | Contents |
|---|---|
| `mv3dt_exclusion_rules.h`, `mv3dt_exclusion_rules.c` | GLib-only: parse the config text into per-source rectangle lists; the inside test of [section 3.2](#32-inside-only-rule-locked). No DeepStream headers, so it compiles in tests. |
| `mv3dt_exclusion.h`, `mv3dt_exclusion.c` | `mv3dt_exclusion_attach()`, the pad probe, logging. |

### 4.2 Config format (REQUIRED)

A GLib key file. One group per source index, `[source<N>]`, each with a
`rects` key: entries `left,top,width,height` separated by `;`.

| Rule | Value |
|---|---|
| Numbers | non-negative decimal integers (digits only, no sign, no fraction) |
| Width, height | greater than zero |
| Whitespace | allowed around each number and each entry |
| Empty entries | ignored, so a trailing `;` is valid |
| Frame bounds | not checked by the filter; Step 4 checks them ([section 7.2](#72-render-required)) |

```ini
# Rendered by mv3dt-installer Step 4 from cameras.yml exclude_boxes.
[source1]
rects=530,175,245,225
```

A group whose `rects` value fails to parse is ignored as a whole and logged.
Group names other than `source<N>` with `N` below 64 are ignored and logged.

### 4.3 Runtime contract (REQUIRED)

| Item | Value |
|---|---|
| Config variable | `MV3DT_EXCLUSION_CONFIG`, absolute path |
| Debug variable | `MV3DT_EXCLUSION_DEBUG=1` logs each dropped box, and each kept box that overlaps a rectangle, up to 200 lines |
| Attach point | `appCtx[i]->pipeline.common_elements.primary_gie_bin.bin`, pad `src`, `GST_PAD_PROBE_TYPE_BUFFER` |
| Removal | `nvds_remove_obj_meta_from_frame()`, advancing the list iterator before removal |
| Frame match | `NvDsFrameMeta.source_id` equals the group's `N` |
| Startup log | one line per loaded rectangle: `mv3dt-exclusion: source <N> excludes boxes inside left=<l> top=<t> right=<r> bottom=<b>` |
| Periodic log | every 60 s: `mv3dt-exclusion: source <N> dropped <count> boxes so far` |

### 4.4 Hook into test5 (REQUIRED)

Applied to the copy in the build tree, never to `/opt/nvidia`. Anchors 1
to 3 must each occur exactly once; anchor 4 must occur at least once and
every occurrence is replaced. Otherwise the build stops with a message naming
the anchor and the DeepStream version.

1. **Include**: after the line `#include "deepstream_test5_app.h"`, add
   `#include "mv3dt_exclusion.h"`.
2. **Attach**: immediately after the `create_pipeline (...)` failure block
   that ends with `NVGSTDS_ERR_MSG_V ("Failed to create pipeline");`,
   `return_value = -1;`, `goto done;` and `}`, add
   `mv3dt_exclusion_attach (appCtx[i]->pipeline.common_elements.primary_gie_bin.bin);`.
3. **Makefile sources**: `SRCS:= deepstream_test5_app_main.c deepstream_utc.c`
   gains `mv3dt_exclusion.c mv3dt_exclusion_rules.c`.
4. **Makefile shared source**: every occurrence of the absolute path
   `/opt/nvidia/deepstream/deepstream/sources/libs` becomes `../../../libs`,
   so objects are written inside the build tree.

---

## 5. Build and install (Step 5)

### 5.1 Procedure (REQUIRED)

1. **Build dependencies**: install `libjson-glib-dev` and
   `libgstrtspserver-1.0-dev` with apt if missing. The list in
   [`STEP-1` section 3](STEP-1-PREREQUISITES.md#3-ds-91-41-apt-prerequisite-package-list)
   already has `gcc`, `make` and several `-dev` packages, but not these two,
   which only building `deepstream-test5` needs.
2. **Build tree**: recreate `<install_dir>/build/test5-exclusion/sources/`
   and copy `apps`, `includes` and `libs` from
   `/opt/nvidia/deepstream/deepstream/sources/` into it.
3. **Patch**: copy the four bundled files into
   `sources/apps/sample_apps/deepstream-test5/` and apply
   [section 4.4](#44-hook-into-test5-required).
4. **Compile**: in that directory, `make -j<nproc> CUDA_VER=<X.Y>`, where
   `X.Y` comes from the resolved `/usr/local/cuda` target `cuda-X.Y`.
5. **Install**: copy `deepstream-test5-app` to
   `<install_dir>/bin/mv3dt-test5-app`, mode `0755`, owner root; then write
   the stamp.

### 5.2 Stamp and update (REQUIRED)

The **build key** is one line: the sha256 of the four bundled files (sorted
by name, concatenated), a space, `EXCLUSION_PATCH_VERSION` (an integer
constant in `step5_per_project_exes.py`, bumped whenever the
[section 4.4](#44-hook-into-test5-required) hook logic changes), a space, and
the DeepStream version from `/opt/nvidia/deepstream/deepstream/version`.

| File | Written when | Holds |
|---|---|---|
| `<install_dir>/bin/mv3dt-test5-app.stamp` | a build and install succeeded | the build key it was built from |
| `<install_dir>/bin/mv3dt-test5-app.failed` | a build failed | the build key that failed |

Step 5 gains a `pending_update(ctx)` hook that returns
`"bundled detection filter changed"` when the stamp differs from the current
build key, **unless** `mv3dt-test5-app.failed` holds that same key. A failed
build is therefore attempted once per release and DeepStream version, not on
every run; `--reset-step step5_per_project_exes` forces a retry. A successful
build deletes the `.failed` file. The hook never builds; the update run
does.

### 5.3 Failure (REQUIRED)

A failed dependency install, anchor, or compile is a `warn` naming the stage
and the last compiler error line; Step 5 stays `COMPLETE`, and any previous
`mv3dt-test5-app` and stamp are left as they were. When `cameras.yml` has
`exclude_boxes` and no `mv3dt-test5-app` exists, the warning says the
exclusion zones are inactive.

---

## 6. Launch (Step 5 `pipeline`)

| Item | Rule |
|---|---|
| Binary | `<install_dir>/bin/mv3dt-test5-app` when it exists and is executable, else `deepstream-test5-app` |
| Arguments | unchanged |
| Environment | `MV3DT_EXCLUSION_CONFIG=<install_dir>/deepstream/exclusion.conf` when that file exists |
| Process name | `pgrep`/`pkill -x` match both `deepstream-test` (the 15-character truncation of `deepstream-test5-app`) and `mv3dt-test5-app` (exactly 15 characters) |

The startup log line names the binary used. A restart of
`mv3dt-pipeline@<slug>` is what applies a new config or a new build.

---

## 7. Configuration (`cameras.yml` and Step 4)

### 7.1 Inventory field (REQUIRED)

A camera entry may carry `exclude_boxes`, one quoted flat string in the
[section 4.2](#42-config-format-required) `rects` syntax, in lens-corrected
1920x1080 pixels of that camera:

```yaml
  - id: c2
    mac: d0:3b:f4:02:44:87
    ...
    exclude_boxes: "530,175,245,225"
```

`cameras.py` keeps it on `Camera` (`exclude_boxes: str = ""`), writes it back
only when non-empty, and **preserves it through `merge()`** so a camera rescan
never drops it. The seed `assets/cameras/cameras.yml` header documents the
field.

### 7.2 Render (REQUIRED)

Step 4 renders `<install_dir>/deepstream/exclusion.conf` with the managed
writer, alongside the other DeepStream configs. Enabled cameras are numbered
in inventory order, the same order as `[source<N>]` in the app config. A
camera with valid boxes gets a group; a camera with an invalid value gets a
`warn` and no group. Invalid means the
[section 4.2](#42-config-format-required) syntax fails, or an entry does not
fit inside the 1920x1080 frame (`left + width <= 1920`,
`top + height <= 1080`). With no boxes anywhere the file holds only its
header comment.

### 7.3 Re-render triggers (REQUIRED)

`exclusion.conf` is **out of date** when it is missing, or when a fresh
render differs from all three of: the file, its pending
`exclusion.conf.installer-new` (when one exists), and the content the
manifest records as last written by the installer (its sha256 in
`/var/lib/mv3dt-installer/managed-files.json`). Comparing with the pending
copy and the manifest stops an operator-edited file from triggering a
re-render on every check: once the installer has offered a render, either a
copy waits beside the file or, when the render equals what the manifest
already records, `write_managed` writes nothing new. The condition is false
until `cameras.yml` changes again.

- `_render_targets_exist()` returns false when `exclusion.conf` is out of
  date, so `ingest` re-renders after a `cameras.yml` edit even when the AMC
  archive is unchanged.
- Step 4 gains a `pending_update(ctx)` hook returning
  `"exclusion.conf out of date"` under the same condition.

---

## 8. Tracker fusion gate

The template `assets/deepstream/config_tracker_NvMOT.yml` changes
`maxPeerToPredDistance4Fusion: 2.0` to `1.0`. The row in
[`LIVE-BASELINE` section 3.1](LIVE-BASELINE-AND-EDIT-PROTECTION.md#31-tracker-config_tracker_nvmotyml)
follows. **RESOLVED 2026-11-12:** with the AMC run 2 calibration the cameras
agree to 0.2–0.4 m, and at 2.0 m a seated person's ID moved onto someone
walking past 1.5 m away. NVIDIA's MV3DT page gives 1.0 as the default and
0.2–1.0 m as typical for people. The live tracker has run with `1.0` since
2026-11-12 15:55 as a hand edit.

---

## 9. Verification

1. **Rules test**: `tests/test_exclusion_rules_c.py` compiles a small C
   driver against `mv3dt_exclusion_rules.c` with `gcc` and
   `pkg-config glib-2.0`, and checks parsing, rejection of bad entries, and
   the inside rule on edge-touching, overlapping and outside boxes. It skips
   when `gcc` or GLib headers are absent.
2. **Python tests**: Step 5 build orchestration, stamp and `pending_update`
   with a fake runner; launch binary and environment selection; `cameras.py`
   parse, render and merge of `exclude_boxes`; Step 4 render and
   re-render triggers; tracker template value.
3. **Workstation, before release**: build from the branch, run the live
   pipeline on it with `exclude_boxes` set for `camera2`, and confirm in the
   tracking database that no detections land behind the TV wall while
   people are still tracked in front of the TV.

---

## 10. Out of scope / open decisions

- Polygon zones. Rectangles cover the TV; a polygon can be added later
  without changing the inside-only rule.
- Hiding boxes from the viewer without dropping them. The viewer shows what
  the tracker tracks.
- Packaging a prebuilt binary in the release.

Open decisions:

1. **Flagged:** a DeepStream upgrade that moves the section 4.4 anchors stops
   the build with a clear warning, and the stock app runs without the
   filter. Re-anchoring is a follow-up per DeepStream version.

---

## 11. Decomposition

| Unit | Branch | Files touched | Depends on | Wave |
|---|---|---|---|---|
| E1 Spec | `feat/installer-exclusion-spec` | `installer/plan/DETECTION-EXCLUSION-ZONES.md`, `installer/plan/LIVE-BASELINE-AND-EDIT-PROTECTION.md` (section 3.1 row), `installer/plan/TRACKING-FUSION-AND-CLEANING.md` (section 3.1 amendment note) | none | 1 |
| E2 Filter sources | `feat/installer-exclusion-filter` | `assets/deepstream/exclusion/mv3dt_exclusion_rules.{c,h}`, `assets/deepstream/exclusion/mv3dt_exclusion.{c,h}`, `tests/test_exclusion_rules_c.py` | none | 1 |
| E3 Inventory and render | `feat/installer-exclusion-config` | `mv3dt_installer/cameras.py`, `assets/cameras/cameras.yml`, `steps/step4_calib_output_wiring.py`, `tests/test_cameras.py`, `tests/test_step4_calib_output_wiring.py` | none | 1 |
| E4 Fusion gate | `feat/installer-fusion-gate-1m` | `assets/deepstream/config_tracker_NvMOT.yml`, `tests/fixtures/live_baseline/config_tracker_NvMOT.yml`, `tests/test_deepstream_templates.py` | none | 1 |
| E5 Build and launch | `feat/installer-exclusion-build` | `steps/step5_per_project_exes.py`, `tests/test_step5_per_project_exes.py` | E2, E3 | 2 |

Package paths are relative to `installer/mv3dt_installer/`; test paths are
relative to `installer/`.

Serialization points:

- E5 depends on E2 because it copies and hashes E2's bundled files; its
  tests read the real asset paths.
- E5 depends on E3 because its inactive-zones warning reads
  `Camera.exclude_boxes`, which E3 adds.
- E3 and E5 both use the path `<install_dir>/deepstream/exclusion.conf`
  but touch different files: E3 renders it and E5 only checks it exists.
  Both pin the literal from [section 6](#6-launch-step-5-pipeline) and
  [section 7.2](#72-render-required).
- E1 to E4 touch disjoint files. Recommended merge order: E1, E2, E3, E4, E5,
  then a release.

---

## References

DeepStream behavior comes from NVIDIA's documentation and the DS 9.1
`deepstream-app` and `deepstream-test5` sources installed on the
workstation; measured values come from the 2026-11-12 prototype runs.

- [MV3DT](https://docs.nvidia.com/deepstream/9.2/text/DS_MV3DT.md) — **backs
  [section 8](#8-tracker-fusion-gate)**: peer matching by 3D foot location,
  `maxPeerToPredDistance4Fusion` default 1.0 and the typical 0.2–1.0 m range,
  incorrect associations among close targets.
- [Gst-nvtracker](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html)
  — the tracker consumes the object metadata the probe edits.
- [NvDsBatchMeta / `nvds_remove_obj_meta_from_frame`](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_metadata.html)
  — **backs [section 4.3](#43-runtime-contract-required)**.
- DS 9.1 `sources/apps/sample_apps/deepstream-app/deepstream_app.c` —
  **backs [section 3.1](#31-filter-before-the-tracker-locked)**: element
  order and the existing probe on the primary detector's `src` pad.

Repo files referenced:

- [`00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md) — section
  15.5, the camera inventory.
- [`STEP-1-PREREQUISITES.md`](STEP-1-PREREQUISITES.md) — section 3, the apt
  list this doc does not extend.
- [`STEP-4-CALIB-OUTPUT-WIRING.md`](STEP-4-CALIB-OUTPUT-WIRING.md) — section
  6, the render this doc extends.
- [`STEP-5-PER-PROJECT-EXES.md`](STEP-5-PER-PROJECT-EXES.md) — section 3.3,
  the `pipeline` subcommand.
- [`LIVE-BASELINE-AND-EDIT-PROTECTION.md`](LIVE-BASELINE-AND-EDIT-PROTECTION.md)
  — sections 3.1 and 4, the tracker baseline and managed files.
- [`TRACKING-FUSION-AND-CLEANING.md`](TRACKING-FUSION-AND-CLEANING.md) —
  sections 3.1 and 3.4, the decision amended here and the history-side TV
  zone.
