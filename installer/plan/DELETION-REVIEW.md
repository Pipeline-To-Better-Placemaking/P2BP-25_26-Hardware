# DELETION-REVIEW — Repo triage for the installer port (owner: shared)

Status: review artifact and standing record. It depends on the harvest
specified in [`STEP-7` §B–§F](STEP-7-WEBAPP-INTEGRATION.md) and
[`00` §14](00-FRAMEWORK-AND-BOOTSTRAP.md#14-web-app-connection-contract), and
does **not** restate those contracts — it links back.

**Execution state:**

| Section | State |
|---|---|
| [§2](#2-immediate-deletions-no-harvest-required) immediate deletions | **EXECUTED** — approved and removed |
| [§4](#4-explicit-calls-resolved--both-deleted) the two judgment calls | **EXECUTED** — both resolved to delete; camera facts harvested first ([§4.1](#41-camera-facts-harvested-before-deletion)) |
| [§3](#3-deletions-gated-on-the-harvest-the-jetson-tree) the Jetson tree | **EXECUTED** — gate verified against the sources immediately before removal ([§3.2](#32-what-must-be-captured-before-these-files-go)); `CLAUDE.md` rewritten per [§3.3](#33-documentation-that-must-change-in-the-same-commit) |
| [§8](#8-script-disposition-under-the-binary-distribution) script disposition | **SUPERSEDED** by [§9](#9-removal-of-the-laptop-harness) — still the record of which `laptop/scripts/` entries were bundled into the binary; its "retained in git" column and the two-copy rule in §8.1 are historical |
| [§9](#9-removal-of-the-laptop-harness) the `laptop/` harness | **EXECUTED** — whole tree removed after unit A (PR #56) moved the DeepStream templates into `assets/`; `DEEPSTREAM-SETUP.md` relocated to `installer/plan/` |

The triage is complete: **14,692 files removed**, working tree down from ~1 GB
to ~23 MB. What remained was `laptop/` (the DeepStream harness), `installer/`
(these specs), `CLAUDE.md`, and `README.md`;
[§9](#9-removal-of-the-laptop-harness) has since removed `laptop/` as well,
so the repo is now `installer/`, `CLAUDE.md`, and `README.md`.

Everything removed remains recoverable via `git revert` or from the parent
repository; no history rewrite has been performed, so `.git` is still ~892 MB
(see the reclaim note in [§2](#2-immediate-deletions-no-harvest-required)).

> **Counting correction.** `homographies/` was originally scored as "14
> `*.yml`". It held **15** files — a `4p-ground-truth.txt` was missed because
> the survey filter excluded `*.txt`. Same verdict (EPFL dataset sample, zero
> references), but worth recording: that filter is also what initially hid
> `laptop/deepstream/*.txt` from the survey. Any future triage should enumerate
> by `git ls-files` without extension filters.

This repo began as a **fork** (`KevinTTO5/P2BP-25_26-Hardware_Test`) whose
`main` later replaced `Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware`'s,
and the work moved to the DeepStream workstation. Every file removed here
survives in this repository's git history, so the governing question
is not "is deletion safe?" but **"does this file contribute code or a pattern
the workstation plan needs — above all, for talking to the web app?"**

The ordering constraint that remains is **doc coherence, not runtime safety**:
files cited by `installer/plan/*.md` as the ported source must not disappear
before the port is written, or the specs lose the implementation they were
written against. Hence two tiers — [§2](#2-immediate-deletions-no-harvest-required)
(unblocked) and [§3](#3-deletions-gated-on-the-harvest-the-jetson-tree) (gated).

---

## 1. Scope and the deletion test

**REQUIRED** — every file in the repo is placed in exactly one of four
categories by the single test below. A file is retained only if it earns
retention; "might be useful later" is not a category, because the git history
already serves that purpose.

| Category | Test | Outcome |
|---|---|---|
| **HARVEST** | Contributes code or a pattern the workstation plan needs | Ported into a named plan section, *then* the source is deleted ([§3](#3-deletions-gated-on-the-harvest-the-jetson-tree)) |
| **DELETE-NOW** | No references anywhere; nothing to harvest | Removed on sign-off ([§2](#2-immediate-deletions-no-harvest-required)) |
| **RETAIN** | Cited by a plan doc, bundled as an installer asset, or unported tooling | Kept ([§5](#5-retained-files-and-why)) |
| **CALL** | Retention is a genuine judgment call, not a mechanical one | Human decides ([§4](#4-explicit-calls-resolved--both-deleted)) |

Reference checks backing the DELETE verdicts were run with `grep -rIn` across
`*.py`, `*.sh`, `*.md`, `*.yml`, and `*.ps1`, excluding the `amc/` tree itself.

---

## 2. Immediate deletions (no harvest required)

These are unblocked: no plan doc cites them, no code path reads them, and there
is no pattern in them worth porting.

| Target | Size / count | Verdict basis |
|---|---|---|
| `amc/alignment_data (Copy)/`, `amc/single_view_results (Copy)/` | **964 MB, 14,606 tracked files** | AMC run output — `kitti_detector/` per-frame label dumps, overlay JPGs, per-camera YAMLs. The `(Copy)` suffixes mark a manual paste, not a curated fixture. [`STEP-4` §4](STEP-4-CALIB-OUTPUT-WIRING.md#4-result-download-and-validation) fetches the AMC result **at run time** through AMC's API, never from the repo tree. |
| `homographies/` (whole directory) | 15 files | EPFL multi-camera dataset samples — 14 `*-homography.yml` (`4p-c0`, `campus4-c*`, `passageway1-c*`, `terrace1-c*`) plus `4p-ground-truth.txt` (2,957 lines of frame/track/bbox annotations). The runtime resolved homographies **by MAC** — `camera_handler.py:241`, `aruco_scanner.py:101`, and `homography.py:760` all used `{safe_mac}_homography.yml`. No committed file matched that scheme, so nothing ever read these. |
| `.DS_Store` | 6 KB | macOS Finder artifact. Also absent from `.gitignore` — add it there in the same change. |
| `virtual-cameras/` (whole directory, incl. the 48 MB `mediamtx.exe`) | **48 MB** | Windows binary in a repo whose installer targets Ubuntu 24.04 `x86_64` ([`00` §5.2](00-FRAMEWORK-AND-BOOTSTRAP.md#52-what-the-binary-does-on-first-launch)); also exposed to line-ending mangling (see the hazard note below). Both POSIX launchers already used the `bluenviron/mediamtx:latest` **Docker image** (`start_rtsp_cams.sh:22-24`) and the Windows launcher already fell back to it (`start_rtsp_cams.ps1:35-36`), so the whole directory is replaceable by one `docker run` — see [§4](#4-explicit-calls-resolved--both-deleted) item 2. |
| `config/config.json`, `config/cameras_runtime.json` | small | Committed before `config/` was added to `.gitignore`, so they still override the ignore rule. Device-specific runtime state — `cameras_runtime.json` is written by `camera_scanner.py`. **Harvest first:** these two files were the sole source of the fleet MAC inventory and the native sensor resolution — extracted to `cameras.yml` per [§4.1](#41-camera-facts-harvested-before-deletion) before removal. |

> **Line-ending hazard (fix with the same change):** `.gitattributes` sets
> `* text=auto eol=lf` and does **not** list `*.exe` as binary, so
> `mediamtx.exe` is currently exposed to line-ending normalization. Deleting it
> removes the hazard; if any `.exe` is ever re-added, add `*.exe binary` first.

**Reclaim note:** deleting `amc/` at `HEAD` does not shrink existing clones —
the objects stay in history. Reclaiming the 964 MB requires a history rewrite,
which is a **separate decision** and is not proposed here.

---

## 3. Deletions gated on the harvest (the Jetson tree)

**Gate (REQUIRED):** proceed only once
[`STEP-7`](STEP-7-WEBAPP-INTEGRATION.md) and
[`00` §14](00-FRAMEWORK-AND-BOOTSTRAP.md#14-web-app-connection-contract) are
written **and** the five critical patterns in
[§3.2](#32-what-must-be-captured-before-these-files-go) verify. Until then
these files are the reference implementation the specs are written against.

### 3.1 What is removed

| Path | Contents |
|---|---|
| `scripts/` | 17 modules, ~5,700 lines — every pattern worth keeping is harvested into `STEP-7` / `00` §14 first |
| `scripts/json_models/` | 6 DTO modules; the `cloud_storage` and `heartbeat_payload` field sets are inlined into `STEP-7` §C/§D before removal |
| `scripts/camera_controllers/` | `annke_controller.py` — see [§4](#4-explicit-calls-resolved--both-deleted) |
| `services/` | 11 systemd units + the logrotate drop-in; `tracker.path` / `tracker-toggle.service` are inlined into [`STEP-6` §B.1](STEP-6-REMOTE-SUPERVISION.md#b1-the-agent-systemd-unit) first |
| `models/` | 22 MB OSNet ReID weights + encoder — not needed: the tracker uses NVIDIA's TAO ReidentificationNet instead. Note the original basis here, "DeepStream supplies its own ReID engine", was wrong for DS 9.1, which ships no tracker models; Step 4 downloads the model ([`STEP-4` §6.3](STEP-4-CALIB-OUTPUT-WIRING.md#63-reid-model)) |
| `install.sh` | Jetson installer; its credential-capture (`42-109`) and unit-enable (`147-207`) logic is ported to `00` §14 / [`STEP-6` §A.4](STEP-6-REMOTE-SUPERVISION.md#a4-installer-integration-mirror-installsh) |
| `requirements.txt` | `ultralytics` / `torch` / `playwright` — Jetson tracker dependencies with no workstation role |
| `config/`, `homographies/` | Runtime state and dead sample data ([§2](#2-immediate-deletions-no-harvest-required)) |

Rationale, per subsystem: these implement per-camera YOLO+OSNet tracking on
Jetson hardware. The DeepStream MV3DT pipeline replaces that function entirely,
and no step doc in `installer/plan/` targets a Jetson.

### 3.2 What must be captured before these files go

**REQUIRED** — these five behaviors are the non-obvious ones. If a plan doc
merely *names* the source file rather than describing the behavior, the harvest
has failed and deletion must not proceed.

- [ ] **Retry predicate** — retry on `429` and `5xx` only; fail fast on every
      other `4xx` (`cloud_storage_media.py:197-245`).
- [ ] **mtime edge-trigger** — fire a one-shot operation only when the config
      file is *freshly written* AND the flag is true, so a stale `true` never
      re-fires (`aruco_scanner.py:307-345`).
- [ ] **Signed-URL redaction** — strip everything from `?` onward before any
      URL reaches a log or transcript (`cloud_storage_media.py:46-51`).
- [ ] **Size + mtime dedupe with per-file cooldown** — re-upload only on a
      changed fingerprint; back off after repeated failures
      (`tracking_uploader.py:64-104`).
- [ ] **Exact DTO field names** — the PascalCase wire format the backend
      expects (`json_models/cloud_storage.py`, `json_models/heartbeat_payload.py`).

### 3.3 Documentation that must change in the same commit

[`CLAUDE.md`](../../CLAUDE.md) describes "**two independent subsystems**" and
instructs contributors to treat `scripts/` and `laptop/scripts/` as "separate
worlds". Both statements become false the moment this section executes. The
rewrite is specified in the plan and must land atomically with the deletion, or
the repo ships instructions referring to directories that no longer exist.

---

## 4. Explicit calls (RESOLVED — both deleted)

Two retention questions were genuine judgment rather than mechanical
application of [§1](#1-scope-and-the-deletion-test). **Both are RESOLVED:
delete.** The governing criterion, from the repo owner, was:

> Keep only what pertains to the workstation — for cameras, that means MAC
> addresses or information *about* the cameras that benefits the new plan.

That criterion is narrower than "keep the code" and is what decided both: the
*facts* the camera tooling encoded are worth keeping, the *tooling* is not.
Those facts were harvested into
[`laptop/config/cameras.yml`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/blob/57adf5d/laptop/config/cameras.yml) before deletion
— see [§4.1](#41-camera-facts-harvested-before-deletion).

1. **`scripts/camera_controllers/annke_controller.py` (1,533 lines).**
   Automated camera web-UI control — activation and OSD-text disable. The plans
   **deliberately** place per-camera network and stream configuration out of
   scope as manual work via each camera's web UI, in both
   [`00` §13](00-FRAMEWORK-AND-BOOTSTRAP.md#13-out-of-scope--defer-to-human)
   and [`STEP-5` §9](STEP-5-PER-PROJECT-EXES.md#9-out-of-scope--flag-for-human).
   This file is the only asset in the repo that could close that gap.
   **RESOLVED: delete.** The scope call was made twice and independently, and
   it is Playwright-driven browser automation — a heavyweight dependency
   (`playwright` + a browser download) for work the plans assign to a human.
   What mattered in it was never the automation but the four facts it encoded
   about the hardware; those are now in `cameras.yml`
   ([§4.1](#41-camera-facts-harvested-before-deletion)), including the manual
   procedure it automated, so the capability is documented rather than lost.

2. **`virtual-cameras/`.** Serves RTSP test streams via `mediamtx`, which made
   it a candidate for exercising Steps 5–7 without cameras attached.
   **RESOLVED: delete the whole directory**, not just `mediamtx.exe`. It sits
   outside `laptop/`, no plan doc references it, and its function is replaceable
   in one command — `mediamtx` runs from the upstream Docker image
   (`bluenviron/mediamtx:latest`), which is exactly what the shell launchers did
   anyway. Keeping a 48 MB vendored binary plus three launcher scripts to avoid
   one `docker run` is not a trade worth making in a workstation-only tree.

### 4.1 Camera facts harvested before deletion

Applying the "keep information *about* the cameras" criterion surfaced three
facts that existed **nowhere else in the repo** and were about to be deleted
with `config/` and the camera controller. All are now recorded in the header of
[`laptop/config/cameras.yml`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/blob/57adf5d/laptop/config/cameras.yml),
the retained camera inventory at the time (its successor is
[`assets/cameras/cameras.yml`](../mv3dt_installer/assets/cameras/cameras.yml),
[§9](#9-removal-of-the-laptop-harness)):

| Fact | Was only in | Why it benefits the new plan |
|---|---|---|
| **Fleet MAC inventory** — 8 cameras, OUI `d0:3b:f4` | `config/config.json` (`TrackingCameras`) | The MAC is the stable per-camera identity; `cameras.yml` previously listed IPs only |
| **Native sensor resolution 3072x1728** | `config/cameras_runtime.json` | `deepstream_app_config.txt` sets `[streammux] width=1920 height=1080`, so the pipeline **downscales** every source — a deliberate throughput choice that was nowhere written down |
| **Vendor/firmware + manual pre-flight** — ANNKE on Hikvision-OEM firmware; cameras ship un-activated; OSD text must be disabled | `annke_controller.py` | The OSD overlay is burned into the encoded stream, so leaving it on feeds timestamp/name pixels into PeopleNet detection and AMC feature matching |

A fourth finding is recorded there as a caveat rather than a fact: the
`169.254.0.0/16` link-local IPs pinned in `cameras.yml` are **volatile**. A
snapshot of four of these cameras in `cameras_runtime.json` held a completely
disjoint IP set from the eight pinned below it, and the addresses are not
derivable from the MAC. This is why the camera-node stack discovered by MAC
instead of trusting a static list — and it is a live risk for the new plan,
whose `cameras.yml` *is* a static list. Flagged, not solved.

> **No MAC-to-position mapping survived.** `config.json` keyed cameras by MAC,
> `cameras.yml` keys them by IP and position, and the two shared no common
> field. The MAC inventory is preserved; re-derive the mapping with
> `arp-scan -l` on the camera LAN if it is needed.

---

## 5. Retained files and why

> **Superseded by [§9](#9-removal-of-the-laptop-harness) (2026-11-11).** Every
> `laptop/` row below was retained when this section was written; the whole
> harness has since been removed. The table stays as the record of why each
> file was kept and what it was bundled as —
> [§9.2](#92-where-each-piece-lives-now) gives each file's current home.

| File(s) | Reason for retention |
|---|---|
| `laptop/deepstream/config_tracker_NvMOT.yml`, `config_infer_primary.txt`, `deepstream_app_config.txt`, `msgconv_config.txt` | Bundled as PyInstaller **data assets** to `assets/deepstream/` by [`00` §4.1](00-FRAMEWORK-AND-BOOTSTRAP.md#41-what-builds-the-binary). Named one by one in [`installer.spec`](../installer.spec), not collected by glob — nothing else under `laptop/deepstream/` ships. Permanent. |
| `laptop/mosquitto/mv3dt.conf` | Bundled to `assets/mosquitto/`. Consumed by the bundled `10_setup_mosquitto.sh` ([§8](#8-script-disposition-under-the-binary-distribution)). Permanent. |
| `laptop/config/cameras.yml` | Bundled to `assets/cameras/` by the [`installer.spec`](../installer.spec) entry that lands with `cameras.py`. Its header block (fleet MAC inventory, sensor resolution, volatility caveat, manual pre-flight — [§4.1](#41-camera-facts-harvested-before-deletion)) is copied into every generated `<install_dir>/cameras.yml`, and its pinned IPs seed the ARP-cache fallback. Stays exactly where it is in git. |
| `laptop/config/laptop.env.example` | **Not bundled.** Operator state moves to `installer.conf` ([`00` §11.2](00-FRAMEWORK-AND-BOOTSTRAP.md#112-persistence--sharing-with-later-steps)); the example file is retained for the developer harness only. |
| `laptop/docs/*`, `laptop/README.md` | Cited 22+ times across the plans; `DEEPSTREAM-SETUP.md` is the source of truth for the version pins in [`STEP-1` §2](STEP-1-PREREQUISITES.md#2-the-ds-91-dgpu-prerequisite-pins-equality). |
| `laptop/scripts/00_bootstrap.sh`, `lib/common.sh`, `30_start_amc.sh`, `50_start_pipeline.sh`, `99_stop_all.sh` | Ported by Steps 1, 2, 3, 5, and 6. Retained as the **developer harness** ([§8](#8-script-disposition-under-the-binary-distribution)); no operator procedure runs them. |
| `laptop/scripts/40_export_watcher.sh` | **Superseded operationally** — [`STEP-4` §7](STEP-4-CALIB-OUTPUT-WIRING.md#7-later-recalibration) installs a per-project `mv3dt-ingest-<slug>.timer` that polls AMC's API and ingests each completed recalibration, so no operator runs a watcher command. Retained in git as a developer tool. |
| `laptop/scripts/10_setup_mosquitto.sh` | **Bundled** into the binary as `assets/scripts/10_setup_mosquitto.sh` and owned by [`STEP-1` §3.2](STEP-1-PREREQUISITES.md#32-mosquitto-broker). The `laptop/` copy is the developer-harness original. |
| `laptop/scripts/20_verify_cameras.sh` | **Not bundled.** Its `ffprobe`-over-RTSP check becomes the RTSP probe in `cameras.py` ([§6](#6-coverage-gaps-this-triage-exposed) gap 2); the ping sweep is [`STEP-5` §3.3](STEP-5-PER-PROJECT-EXES.md#33-what-the-exe-does-at-runtime-pipeline-subcommand)'s. Retained in git as a developer tool. |
| `laptop/scripts/60_record_tracking.sh` | **Bundled** as `assets/scripts/60_record_tracking.sh` — it is the **producer of the artifacts** [`STEP-7` §E.1](STEP-7-WEBAPP-INTEGRATION.md#e1-what-gets-uploaded) uploads (`tracks.jsonl`, `tracks.csv`, `summary.json`). The `laptop/` copy is the developer-harness original. |
| `laptop/scripts/70_plot_floorplan.py`, `record_cameras_mp4.sh`, `view_cameras.sh` | **Not bundled** — `70_plot_floorplan.py` would drag matplotlib into a self-contained binary for work the web app already does; `record_cameras_mp4.sh`'s recording is ported to Python as `installer/mv3dt_installer/footage.py` ([`STEP-4-CALIBRATION-FOOTAGE`](STEP-4-CALIBRATION-FOOTAGE.md)); `view_cameras.sh` has no plan coverage. Retained in git as developer tools. |

> **RESOLVED — the bundling claim above was wrong.** This table previously
> asserted that `laptop/deepstream/*`, `laptop/config/*`, and
> `laptop/mosquitto/mv3dt.conf` were all bundled. Reading
> [`installer.spec`](../installer.spec) shows it bundles everything under
> `installer/mv3dt_installer/assets/`, **four named files** from
> `laptop/deepstream/`, and `laptop/mosquitto/mv3dt.conf` — nothing from
> `laptop/config/` at all. The rows above are corrected to what the spec
> actually does, and `cameras.yml` is additionally bundled to
> `assets/cameras/` by a new spec entry so the camera facts travel with the
> binary. `laptop.env.example` is deliberately still not bundled.

---

## 6. Coverage gaps this triage exposed

Surfaced by checking which `laptop/scripts/` entries no step doc claims. All
three gaps below are now **RESOLVED** with a named owner; the original gap
text is retained in each item so the reason the script was kept is not lost.

1. **Mosquitto install — RESOLVED, owned by
   [`STEP-1` §3.2](STEP-1-PREREQUISITES.md#32-mosquitto-broker).**
   The gap was real: [`STEP-4` §1](STEP-4-CALIB-OUTPUT-WIRING.md#1-scope-and-identity)
   deferred broker install to "Step 1/2 bootstrap", neither
   [`STEP-1`](STEP-1-PREREQUISITES.md) nor
   [`STEP-2`](STEP-2-DEEPSTREAM-SDK.md) mentioned it, and
   [`STEP-6` §E.1](STEP-6-REMOTE-SUPERVISION.md#e1-lifecycle) `preflight`
   **requires** a reachable broker while
   [`STEP-6` §D](STEP-6-REMOTE-SUPERVISION.md#d-security-remote-control-must-be-authenticated)
   **rewrites** its configuration. Step 1 now installs `mosquitto` and
   `mosquitto-clients` and drives the **bundled** `10_setup_mosquitto.sh`
   ([§8](#8-script-disposition-under-the-binary-distribution)), so the broker
   has one owner and the operator types nothing.

2. **Camera verification — RESOLVED, owned jointly by `cameras.py` and
   [`STEP-5` §3.3](STEP-5-PER-PROJECT-EXES.md#33-what-the-exe-does-at-runtime-pipeline-subcommand).**
   The gap was that Step 5 ported only the ping sweep from
   `20_verify_cameras.sh`. **The `ffprobe`-over-RTSP check and its pass/fail
   table are not lost — they move into `cameras.py`**, which discovers cameras
   by MAC OUI and then probes each one with
   `ffprobe -rtsp_transport tcp rtsp://<user>:<pass>@<ip>:554<rtsp_path>`,
   recording the result as `stream_ok` in `<install_dir>/cameras.yml` and
   rendering the same pass/fail table. A ping proves the host answers; the
   probe proves it serves a decodable stream — the distinction the original
   script existed to make. `20_verify_cameras.sh` itself is therefore dropped
   as an operator-run script: the capability survives with no command to type.
   `STEP-5` §3.3 now carries the resolved blockquote confirming this split —
   the ping sweep stays in `STEP-5`, the `ffprobe` check moved to
   `cameras.py` — so this row and `STEP-5` agree.

3. **The export watcher — RESOLVED, superseded operationally by
   [`STEP-4` §7](STEP-4-CALIB-OUTPUT-WIRING.md#7-later-recalibration)
   auto-ingest.**
   The earlier reading — "not superseded, do not delete" — was written before
   auto-ingest was designed. Step 4 now ingests in `run()`, and continuous
   re-ingest is a per-project `mv3dt-ingest-<slug>.timer` that polls AMC's
   API every 60 seconds. That replaced the interim design of a systemd `.path`
   unit watching an export directory AMC 3.2.1 does not create, and it
   supersedes the watcher's `inotifywait` loop operationally.
   [`40_export_watcher.sh`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/blob/57adf5d/laptop/scripts/40_export_watcher.sh)
   was retained in git as a developer tool at the time, on the grounds that
   its standalone watch mode had no installer equivalent; it was not
   load-bearing for the installer's own ingest path, and
   [§9](#9-removal-of-the-laptop-harness) removed it with the rest of the
   harness. It is recoverable from git history at `57adf5d`.

---

## 7. Execution and rollback

Ordered, one commit per group, so any step can be reverted independently:

1. **Harvest** — write [`STEP-7`](STEP-7-WEBAPP-INTEGRATION.md) and
   [`00` §14](00-FRAMEWORK-AND-BOOTSTRAP.md#14-web-app-connection-contract);
   apply the `STEP-4` / `STEP-5` / `STEP-6` edits.
2. **Verify the gate** — confirm every checkbox in
   [§3.2](#32-what-must-be-captured-before-these-files-go), then re-run the
   reference check immediately before removing anything:

   ```bash
   grep -rIn "homographies/4p-\|single_view_results\|alignment_data\|mediamtx.exe" \
     --include='*.py' --include='*.sh' --include='*.md' --include='*.yml' \
     --include='*.ps1' . | grep -v '^./amc/'
   ```

   Expect no hits outside `amc/` itself and `start_rtsp_cams.ps1`.
3. **Commit A** — [§2](#2-immediate-deletions-no-harvest-required) deletions
   plus the `.gitignore` addition of `.DS_Store`.
4. **Commit B** — [§3](#3-deletions-gated-on-the-harvest-the-jetson-tree)
   deletions plus the [`CLAUDE.md`](../../CLAUDE.md) rewrite, atomically.
5. **Confirm isolation holds** — `laptop/` and `installer/` must not reference
   any removed path:

   ```bash
   grep -rn "\.\./scripts/\|\.\./services/\|\.\./models/\|\.\./config/" laptop/ installer/
   bash -n laptop/scripts/*.sh && echo "laptop scripts parse OK"
   ```

**Rollback:** every deletion is recoverable via `git revert` of the relevant
commit, or from this repository's history. No history rewrite is performed, so no
deletion in this document is destructive to the object store.

---

## 8. Script disposition under the binary distribution

> **Superseded by [§9](#9-removal-of-the-laptop-harness) (historical).** The
> bundling decisions below still describe which scripts ship inside the
> binary and where each was ported. The "Retained in git?" column and the
> two-copy rule in [§8.1](#81-which-copy-do-i-edit) no longer hold: the
> `laptop/` originals were removed, so
> `installer/mv3dt_installer/assets/scripts/` is the only copy of each
> bundled script.

**LOCKED.** The operator artifact is a **prebuilt binary downloaded from the
repo's GitHub Releases page** ([`00` §5](00-FRAMEWORK-AND-BOOTSTRAP.md#5-distribution-the-github-release-binary)).
Nothing clones this repo onto a workstation, so no numbered script under
[`laptop/scripts/`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/tree/57adf5d/laptop/scripts/) is reachable by an operator unless
it is **bundled into the binary**. Exactly two are: `installer/mv3dt_installer/assets/scripts/`
holds `10_setup_mosquitto.sh`, `60_record_tracking.sh`, and a purpose-written
`lib/common.sh` sibling, landed by U12 (`feat/installer-bundled-scripts`,
commit `8157dfc`) and shell-linted (`bash -n`) on every push/PR by
`installer-tests.yml`.

Everything else stays in git — deleting a script that no longer ships is not
the same decision as deleting one nothing needs, and the developer harness is
still how this repo is exercised from a clone.

| Script | Disposition | Bundled as | Owner | Retained in git? |
|---|---|---|---|---|
| `00_bootstrap.sh` | Superseded — ported to Python | — | [`STEP-1`](STEP-1-PREREQUISITES.md), [`STEP-2`](STEP-2-DEEPSTREAM-SDK.md) | Yes, developer-only |
| `10_setup_mosquitto.sh` | **Bundled** | `assets/scripts/10_setup_mosquitto.sh` | [`STEP-1` §3.2](STEP-1-PREREQUISITES.md#32-mosquitto-broker) | Yes, developer-only original |
| `20_verify_cameras.sh` | **Dropped** as an operator script; the `ffprobe` check is ported | — | `cameras.py` RTSP probe + [`STEP-5` §3.3](STEP-5-PER-PROJECT-EXES.md#33-what-the-exe-does-at-runtime-pipeline-subcommand) ping sweep | Yes, developer-only |
| `30_start_amc.sh` | Superseded — ported to Python | — | [`STEP-3`](STEP-3-AMC-LAUNCHER.md) | Yes, developer-only |
| `40_export_watcher.sh` | Superseded operationally by auto-ingest | — | [`STEP-4` §7](STEP-4-CALIB-OUTPUT-WIRING.md#7-later-recalibration) | Yes, developer-only |
| `50_start_pipeline.sh` | Superseded — ported into the per-project exe | — | [`STEP-5` §3.3](STEP-5-PER-PROJECT-EXES.md#33-what-the-exe-does-at-runtime-pipeline-subcommand) | Yes, developer-only |
| `60_record_tracking.sh` | **Bundled** | `assets/scripts/60_record_tracking.sh` | [`STEP-5`](STEP-5-PER-PROJECT-EXES.md); its artifacts feed [`STEP-7` §E.1](STEP-7-WEBAPP-INTEGRATION.md#e1-what-gets-uploaded) | Yes, developer-only original |
| `70_plot_floorplan.py` | **Dropped** — matplotlib bloat; the web app visualizes | — | None (web app) | Yes, developer tool |
| `99_stop_all.sh` | Superseded — ported into the per-project exe | — | [`STEP-5` §3.4](STEP-5-PER-PROJECT-EXES.md#34-stopping-the-pipeline) | Yes, developer-only |
| `record_cameras_mp4.sh` | **Not bundled** — ported to Python as `installer/mv3dt_installer/footage.py` | — | [`STEP-4-CALIBRATION-FOOTAGE`](STEP-4-CALIBRATION-FOOTAGE.md) | Yes, developer tool |
| `view_cameras.sh` | **Dropped** — no plan coverage | — | None | Yes, developer tool |
| `lib/common.sh` | Not bundled; a purpose-written sibling is | `assets/scripts/lib/common.sh` (sibling, **not** a copy) | [`00` §4.2](00-FRAMEWORK-AND-BOOTSTRAP.md#42-locating-and-staging-bundled-assets-at-runtime) | Yes, developer-only |

### 8.1 Which copy do I edit?

> **RESOLVED by [§9](#9-removal-of-the-laptop-harness):** only the
> `assets/scripts/` copy exists now; edit it. The rule below is kept as the
> record of the two-copy period.

**REQUIRED.** Two of these scripts exist in two places, and the answer is
not "whichever one you opened".

| You are changing… | Edit | Do **not** edit |
|---|---|---|
| Behavior the shipped binary must have | `installer/mv3dt_installer/assets/scripts/<script>` | The `laptop/` original |
| Behavior only a developer clone needs | `laptop/scripts/<script>` | The `assets/scripts/` copy |

The `assets/scripts/` copies are **authoritative for the binary**; the
`laptop/` originals are **developer-only** and are not kept in sync
automatically. They are deliberately derived rather than copied, so the
difference is reviewable as a diff: `lib/common.sh`'s `repo_root()` and
`load_env()` cannot exist inside a binary with no repo root, and are replaced
by `asset_root()` and `load_installer_conf()` driven by the environment the
staging runner sets ([`00` §11.2](00-FRAMEWORK-AND-BOOTSTRAP.md#112-persistence--sharing-with-later-steps)).

> **Why not one copy?** Bundling the originals verbatim does not work: the
> runner stages assets into a temporary directory, so
> `source "$SCRIPT_DIR/lib/common.sh"` only resolves if the whole tree is
> staged, and `repo_root()`'s `cd ../../..` has nothing to land on. Making one
> file serve both worlds requires either a fake `laptop/`-shaped tree inside
> the bundle or patching the script at stage time — at which point it is not
> verbatim anyway. Two reviewable files with a written rule beats one file
> with a hidden lie.

---

## 9. Removal of the `laptop/` harness

**EXECUTED (2026-11-11).** The `laptop/` tree — the developer-only scripted
workflow the binary superseded
([§8](#8-script-disposition-under-the-binary-distribution)) — is removed in
full. Under the [§1](#1-scope-and-the-deletion-test) test no
file in it still earns retention: everything the plans needed was already
ported, bundled, or dropped, and the one doc still cited as a source of truth
was relocated rather than deleted
([§9.3](#93-the-deepstream-setupmd-relocation)).
The tree as it stood is browsable at
[`57adf5d`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/tree/57adf5d/laptop).

### 9.1 Why it could go

1. **Superseded by the binary**: every numbered script was ported into a step,
   bundled as an asset, or dropped from the operator path by
   [§8](#8-script-disposition-under-the-binary-distribution). The harness was
   how the repo was exercised by hand, never how an operator installs.
2. **No runtime dependency**: the workstation was verified on 2026-11-11 to
   run release `v0.4.20` (built from `main` at `57adf5d`) entirely through the
   installer binary and the systemd units it renders. Nothing on it reads
   `laptop/` — consistent with
   [`00` §4.1](00-FRAMEWORK-AND-BOOTSTRAP.md#41-what-builds-the-binary)'s rule
   that nothing clones this repo onto a workstation.
3. **No build or test dependency**: unit A (PR #56,
   `feat/installer-deepstream-templates-in-assets`) moved the four DeepStream
   templates from `laptop/deepstream/` into
   `installer/mv3dt_installer/assets/deepstream/` and repointed
   [`installer.spec`](../installer.spec) and the tests **first**. After it,
   `installer.spec` bundles only the package `assets/` tree and no build or
   test code reads `laptop/`.
4. **Everything else it held was a duplicate**: the binary already bundled
   authoritative copies of the two shipped scripts (`assets/scripts/`), the
   Mosquitto drop-in (`assets/mosquitto/mv3dt.conf`, identical content), and
   the camera inventory (`assets/cameras/cameras.yml`).

### 9.2 Where each piece lives now

**REQUIRED** — resurrect nothing from `laptop/` without checking this table
first. "Git history" means the file exists only at `57adf5d` and earlier.

| `laptop/` path | New home | Basis |
|---|---|---|
| `deepstream/config_tracker_NvMOT.yml`, `config_infer_primary.txt`, `deepstream_app_config.txt`, `msgconv_config.txt` | `installer/mv3dt_installer/assets/deepstream/` | Moved by PR #56 (`git mv`, history preserved) |
| `deepstream/calibration/`, `deepstream/models/` (empty `.gitkeep` dirs) | None in git | Runtime trees are `<install_dir>/deepstream/calibration/` and `models/` ([`STEP-4` §5](STEP-4-CALIB-OUTPUT-WIRING.md#5-calibration-destination), Step 2) |
| `mosquitto/mv3dt.conf` | `installer/mv3dt_installer/assets/mosquitto/mv3dt.conf` | Identical settings; owned by [`STEP-1` §3.2](STEP-1-PREREQUISITES.md#32-mosquitto-broker) |
| `config/cameras.yml` | `installer/mv3dt_installer/assets/cameras/cameras.yml` | The seed inventory and fleet-facts header ([§4.1](#41-camera-facts-harvested-before-deletion), [`00` §15](00-FRAMEWORK-AND-BOOTSTRAP.md#15-camera-discovery)) |
| `config/laptop.env.example` | Git history | Operator state is `installer.conf` ([`00` §11.2](00-FRAMEWORK-AND-BOOTSTRAP.md#112-persistence--sharing-with-later-steps)) |
| `scripts/10_setup_mosquitto.sh`, `scripts/60_record_tracking.sh` | `installer/mv3dt_installer/assets/scripts/` | Already the authoritative copies ([§8.1](#81-which-copy-do-i-edit)) |
| `scripts/lib/common.sh` | Git history | `assets/scripts/lib/common.sh` is a purpose-written sibling, not a copy |
| `scripts/00_bootstrap.sh` | Git history | Ported by [`STEP-1`](STEP-1-PREREQUISITES.md) and [`STEP-2`](STEP-2-DEEPSTREAM-SDK.md) |
| `scripts/20_verify_cameras.sh` | Git history | `cameras.py` RTSP probe + [`STEP-5` §3.3](STEP-5-PER-PROJECT-EXES.md#33-what-the-exe-does-at-runtime-pipeline-subcommand) ping sweep |
| `scripts/30_start_amc.sh` | Git history | Ported by [`STEP-3`](STEP-3-AMC-LAUNCHER.md) |
| `scripts/40_export_watcher.sh` | Git history | Superseded by [`STEP-4` §7](STEP-4-CALIB-OUTPUT-WIRING.md#7-later-recalibration) auto-ingest |
| `scripts/50_start_pipeline.sh`, `scripts/99_stop_all.sh` | Git history | Ported into the per-project exe ([`STEP-5` §3.3](STEP-5-PER-PROJECT-EXES.md#33-what-the-exe-does-at-runtime-pipeline-subcommand), [§3.4](STEP-5-PER-PROJECT-EXES.md#34-stopping-the-pipeline)) |
| `scripts/record_cameras_mp4.sh` | Git history | Ported to `installer/mv3dt_installer/footage.py` ([`STEP-4-CALIBRATION-FOOTAGE`](STEP-4-CALIBRATION-FOOTAGE.md)) |
| `scripts/70_plot_floorplan.py`, `scripts/view_cameras.sh` | Git history | Dropped: the web app visualizes; no plan coverage |
| `docs/DEEPSTREAM-SETUP.md` | `installer/plan/DEEPSTREAM-SETUP.md` | Relocated, [§9.3](#93-the-deepstream-setupmd-relocation) |
| `docs/SCRIPTED-WORKFLOW.md`, `docs/SCRIPTS-AND-CONFIG-REFERENCE.md`, `README.md` | Git history | Harness-only guides; the operator guide is [`installer/README.md`](../README.md) |
| `.gitignore` | None | Secrets are excluded by `installer/.gitignore`; the root `.gitignore`'s `!laptop/config/` allowlist is removed |

### 9.3 The `DEEPSTREAM-SETUP.md` relocation

**LOCKED** — moved with `git mv`, not deleted. [`CLAUDE.md`](../../CLAUDE.md)
names it the source of truth for the version pins, and the plans cite it
throughout (the pins in
[`STEP-1` §2](STEP-1-PREREQUISITES.md#2-the-ds-91-dgpu-prerequisite-pins-equality)
among them), so it lands beside them as
[`DEEPSTREAM-SETUP.md`](DEEPSTREAM-SETUP.md). Its content is unchanged apart
from link fixes: relative links to harness files are now permalinks pinned to
`57adf5d`. Passages that walk through the scripted harness are historical;
the pins are what it remains authoritative for.

### 9.4 What changed in the same PR

Matching [§3.3](#33-documentation-that-must-change-in-the-same-commit), the
references move in the same PR as the deletion so nothing on `main` points at
a missing path:

- [`CLAUDE.md`](../../CLAUDE.md), [`installer/README.md`](../README.md), and
  every `installer/plan/*.md` — links to `DEEPSTREAM-SETUP.md` repointed;
  provenance links to removed files ("ported from `00_bootstrap.sh` Phase
  10") converted to permalinks pinned to `57adf5d`; present-tense claims
  about `laptop/` rewritten.
- Header comments of the bundled `assets/cameras/cameras.yml`,
  `assets/mosquitto/mv3dt.conf`, and the four `assets/deepstream/` templates
  — they now name the installer as their consumer. Settings are
  byte-identical.
- Python comments and docstrings in `installer/mv3dt_installer/`
  (`cameras.py`, `config.py`, `logs.py`, `ngc.py`, `preflight.py`,
  `privilege.py`, `webapp.py`) and in `assets/scripts/lib/common.sh` — no
  longer describe `laptop/` as present; "ported from" provenance stays.
- Three user-facing strings repointed: the Step 1 unsupported-platform error
  now cites `installer/plan/DEEPSTREAM-SETUP.md`, the Step 5 validation
  banner cites `STEP-5` §6 instead of `SCRIPTED-WORKFLOW.md`, and the
  generated `installer.conf` header describes its own shape instead of
  naming `laptop.env.example`.
- `.gitignore` and `installer/.gitignore` — the `laptop/` allowlist and
  cross-reference removed.

**Rollback:** `git revert` the PR's commits, or restore the tree with
`git checkout 57adf5d -- laptop/`.

---

## References

Authority for this document is the repo itself: the verdicts rest on reference
checks against the working tree and on the scope decisions recorded in the
`installer/plan/` specs. No external documentation is cited — where a DeepStream
fact underlies a verdict, it is cited in the plan doc linked instead.

Repo files referenced:

- [`00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md) — §4.1 asset
  bundling, §5.1 target platform, §13 out-of-scope scope calls, and §14 the
  web-app credential contract that gates
  [§3](#3-deletions-gated-on-the-harvest-the-jetson-tree).
- [`STEP-4-CALIB-OUTPUT-WIRING.md`](STEP-4-CALIB-OUTPUT-WIRING.md) — §4 fetches
  the AMC result at run time (basis for deleting `amc/`); §7 is the
  auto-ingest that supersedes the export watcher operationally
  ([§6](#6-coverage-gaps-this-triage-exposed) gap 3).
- [`STEP-5-PER-PROJECT-EXES.md`](STEP-5-PER-PROJECT-EXES.md) — §3.3 the ping
  sweep half of camera verification, the other half being `cameras.py`'s RTSP
  probe ([§6](#6-coverage-gaps-this-triage-exposed) gap 2); §3.4 the stop path;
  §9 the per-camera-config scope call.
- [`STEP-6-REMOTE-SUPERVISION.md`](STEP-6-REMOTE-SUPERVISION.md) — §A.4 and
  §B.1 consume `install.sh` and the `services/` unit conventions before those
  files are removed; §D and §E.1 are why Mosquitto setup is load-bearing.
- [`STEP-7-WEBAPP-INTEGRATION.md`](STEP-7-WEBAPP-INTEGRATION.md) — the harvest
  target for every web-app pattern in the Jetson tree; the gate on
  [§3](#3-deletions-gated-on-the-harvest-the-jetson-tree); §E.1 names the
  artifacts `60_record_tracking.sh` produces, which is why that script is
  bundled ([§8](#8-script-disposition-under-the-binary-distribution)).
- [`STEP-1-PREREQUISITES.md`](STEP-1-PREREQUISITES.md) — §2 the version pins
  backing the `laptop/docs/` retention and the
  [§9.3](#93-the-deepstream-setupmd-relocation) relocation; §3.2 the named
  owner that closes [§6](#6-coverage-gaps-this-triage-exposed) gap 1.
- [`installer.spec`](../installer.spec) — the authority for every bundling
  claim in [§5](#5-retained-files-and-why) and
  [§8](#8-script-disposition-under-the-binary-distribution); read it, do not
  infer, before asserting that a file ships inside the binary.
- [`CLAUDE.md`](../../CLAUDE.md) — the "two independent subsystems" framing that
  must be rewritten atomically with the Jetson-tree deletion ([§3.3](#33-documentation-that-must-change-in-the-same-commit)).
  Rewritten again with the [§9](#9-removal-of-the-laptop-harness) harness
  removal.
- [`DEEPSTREAM-SETUP.md`](DEEPSTREAM-SETUP.md) — the DS 9.1 setup reference
  relocated from `laptop/docs/` by
  [§9.3](#93-the-deepstream-setupmd-relocation).
- [`installer/README.md`](../README.md) — the operator guide that replaces the
  harness guides removed by [§9](#9-removal-of-the-laptop-harness).
- [`mv3dt_installer/assets/`](../mv3dt_installer/assets/) — the new home of
  every `laptop/` file still needed ([§9.2](#92-where-each-piece-lives-now)).
- [`laptop/` at `57adf5d`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/tree/57adf5d/laptop)
  — the harness as it stood immediately before removal.
