# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

P2BP Senior Design (Fall '25 – Spring '26) hardware stack for a multi-camera
3D tracking system (MV3DT). It is now a **single subsystem**: the DeepStream
9.1 workstation.

1. **DeepStream 9.1 harness** (`laptop/`) — the working scripted workflow for
   multi-view calibration (AutoMagicCalib) and MV3DT inference. Start at
   [`laptop/README.md`](laptop/README.md) and
   [`laptop/docs/SCRIPTED-WORKFLOW.md`](laptop/docs/SCRIPTED-WORKFLOW.md).
2. **`installer/plan/`** — specs for the unified `mv3dt-installer` binary that
   supersedes the numbered `laptop/scripts/`. Steps 1–5 cover install and
   per-project pipelines; `STEP-6` adds systemd supervision + MQTT remote
   control; `STEP-7` adds the HTTP web-app data plane.

> **The Jetson camera-node stack has been removed** (`scripts/`, `services/`,
> `models/`, `install.sh`, `requirements.txt`, `config/`, `homographies/`,
> `virtual-cameras/`). Everything worth keeping was harvested into the plan
> docs first. **Read
> [`installer/plan/DELETION-REVIEW.md`](installer/plan/DELETION-REVIEW.md)
> before deleting or resurrecting anything** — it records what went, why, and
> where each pattern landed. The originals remain in this repository's git
> history, before their deletion commit.

## Working in this repo

- **The plan docs are the authoritative spec for the web-app connection.**
  [`STEP-7`](installer/plan/STEP-7-WEBAPP-INTEGRATION.md) (signed-URL upload,
  registration/status, web-app-initiated operations) and
  [`00` §14](installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md) (credentials,
  endpoint normalization, redaction) were ported from the deleted Jetson tree
  and are written to be implementable without it. Don't reinvent that contract.
- **The operator artifact is the GitHub Release binary.** A workstation
  downloads a prebuilt `mv3dt-installer` from the repo's Releases page,
  verifies the checksum, and runs it. **Nothing clones this repo onto a
  workstation** — don't write a doc, script, or step that assumes a checkout
  is present at run time.
- `laptop/scripts/` is the **developer harness**: numbered (`00_` … `99_`),
  run in order from a clone, each idempotent. It is how this repo is exercised
  by hand, not how an operator installs anything. See the script table in
  [`laptop/docs/SCRIPTED-WORKFLOW.md`](laptop/docs/SCRIPTED-WORKFLOW.md).
- **Exactly two `laptop/scripts/` entries are bundled into the binary:**
  `10_setup_mosquitto.sh` (owned by `STEP-1` §3.2) and `60_record_tracking.sh`
  (producer of the `tracks.jsonl` / `tracks.csv` / `summary.json` artifacts
  `STEP-7` §E.1 uploads). Their **authoritative copies live under
  `installer/mv3dt_installer/assets/scripts/`** (landed by U12,
  `feat/installer-bundled-scripts`); the `laptop/` originals stay
  developer-only and are not kept in sync automatically. Everything else is
  either superseded by a step or dropped from the binary —
  `70_plot_floorplan.py` (matplotlib bloat; the web app visualizes),
  `record_cameras_mp4.sh`, `view_cameras.sh`, and `20_verify_cameras.sh` as an
  operator-run script (its `ffprobe`-over-RTSP check moves into `cameras.py`
  as an RTSP probe). `40_export_watcher.sh` is superseded operationally by the
  STEP-4 auto-ingest. All dropped scripts stay in git as developer tools;
  the full table is
  [`DELETION-REVIEW.md` §8](installer/plan/DELETION-REVIEW.md#8-script-disposition-under-the-binary-distribution).
- PeopleNet is the only detector wired into the DeepStream pipeline; `yolo11n`
  is reserved for future work — don't add YOLO wiring without being asked.
- [`laptop/config/cameras.yml`](laptop/config/cameras.yml) is the camera
  inventory **and** the record of the fleet's hardware facts (MACs, native
  3072x1728 sensor resolution, vendor/firmware, required manual pre-flight).
  Its header is load-bearing documentation, not commentary. Note its caveat:
  the pinned `169.254.*` link-local IPs are volatile.
- `laptop/config/` is explicitly allowlisted in `.gitignore`; `laptop.env`
  (operator secrets) is not committed — only `laptop.env.example` is.
- Line endings are enforced LF via `.gitattributes`; images/weights/binaries
  are marked binary there — don't hand-edit or re-encode those.
- Whenever creating a git commit in this repo — a plain commit, a
  `parallel-worktree-dev` unit's commit, or a `merge-prs` squash-merge — use
  the `git-commits` skill. Commits carry no `Co-Authored-By` trailer, no
  `Claude-Session` link, no agent-authorship footer of any kind.
- **Release flow:** `parallel-worktree-dev` builds and reviews PRs,
  `/merge-prs <ordered list>` merges them, and `/release` then cuts the next
  patch release from the most recent `v*` tag (bump, commit, tag, wait for
  `release.yml`, verify, print the workstation update commands). `/release`
  refuses to run if that tag has no published release yet. Both commands
  live in `.claude/commands/` and only load when Claude Code is started
  from the repo root.

## Documentation

Whenever creating or substantially editing a Markdown doc in this repo, use
the `markdown-docs` skill to match the house style established by
[`installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md`](installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md)
and [`laptop/docs/SCRIPTED-WORKFLOW.md`](laptop/docs/SCRIPTED-WORKFLOW.md).

## Reference docs

- [`installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md`](installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md)
  — shared contracts for the unified installer (state machine, reporting,
  privilege, credentials).
- [`installer/plan/STEP-1…STEP-7`](installer/plan/) — per-step specs; STEP-6
  is the MQTT control plane, STEP-7 the HTTP data plane.
- [`installer/plan/DELETION-REVIEW.md`](installer/plan/DELETION-REVIEW.md) —
  what was removed from this fork and why.
- [`laptop/docs/SCRIPTED-WORKFLOW.md`](laptop/docs/SCRIPTED-WORKFLOW.md) —
  developer harness guide + script run order + flow diagram.
- [`laptop/docs/DEEPSTREAM-SETUP.md`](laptop/docs/DEEPSTREAM-SETUP.md) — DS 9.1
  package/OS setup and AMC workflow; source of truth for version pins.
- [`laptop/docs/SCRIPTS-AND-CONFIG-REFERENCE.md`](laptop/docs/SCRIPTS-AND-CONFIG-REFERENCE.md)
  — per-script/config field reference.
