# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

P2BP Senior Design (Fall '25 – Spring '26) hardware stack for a multi-camera
3D tracking system (MV3DT). It is now a **single subsystem**: the DeepStream
9.1 workstation, installed and run entirely by **`installer/`** — the
`mv3dt-installer` binary, the operator path. All seven steps are implemented
and released; the specs live in `installer/plan/`. Steps 1–5 cover install
and per-project pipelines; `STEP-6` adds systemd supervision + MQTT remote
control; `STEP-7` adds the HTTP web-app data plane. Start at
[`installer/README.md`](installer/README.md).

> **Shipped is not validated.** What has actually run on the workstation —
> and what has not, such as a completed calibration — is the dated table in
> [`installer/plan/DEVELOPMENT-STATUS.md`](installer/plan/DEVELOPMENT-STATUS.md).
> Treat any workstation fact (camera IPs, ports, project names) older than
> its "last verified" date as unverified, and check it on the machine first.

> **The Jetson camera-node stack has been removed** (`scripts/`, `services/`,
> `models/`, `install.sh`, `requirements.txt`, `config/`, `homographies/`,
> `virtual-cameras/`). Everything worth keeping was harvested into the plan
> docs first. **Read
> [`installer/plan/DELETION-REVIEW.md`](installer/plan/DELETION-REVIEW.md)
> before deleting or resurrecting anything** — it records what went, why, and
> where each pattern landed. The originals remain in this repository's git
> history, before their deletion commit.

> **The `laptop/` developer harness has been removed too** (numbered
> scripts, `laptop/config/`, `laptop/mosquitto/`, its docs). The binary
> superseded it; every piece still needed lives under `installer/` now. See
> [`DELETION-REVIEW.md` §9](installer/plan/DELETION-REVIEW.md#9-removal-of-the-laptop-harness)
> for where each piece went; the originals are in git history at `57adf5d`.

## Working in this repo

- **The plan docs are the authoritative spec for the desktop side of the
  web-app connection.**
  [`STEP-7`](installer/plan/STEP-7-WEBAPP-INTEGRATION.md) (signed-URL upload,
  registration/status, web-app-initiated operations) and
  [`00` §14](installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md) (credentials,
  endpoint normalization, redaction) were ported from the deleted Jetson tree
  and are written to be implementable without it. Don't reinvent that
  contract. **Route names, auth, and wire shapes are owned by the web app
  repo** (`P2BP-25_26-Web`); STEP-7 §G mirrors it and records the gaps.
- **The operator artifact is the GitHub Release binary.** A workstation
  downloads a prebuilt `mv3dt-installer` from the repo's Releases page,
  verifies the checksum, and runs it. **Nothing clones this repo onto a
  workstation** — don't write a doc, script, or step that assumes a checkout
  is present at run time.
- **Exactly two shell scripts ship inside the binary:**
  `10_setup_mosquitto.sh` (owned by `STEP-1` §3.2) and `60_record_tracking.sh`
  (producer of the `tracks.jsonl` / `tracks.csv` / `summary.json` artifacts
  `STEP-7` §E.1 uploads). They live under
  `installer/mv3dt_installer/assets/scripts/` with a purpose-written
  `lib/common.sh`, and are the only copies. Every other former
  `laptop/scripts/` entry was ported into a step or dropped; the per-script
  table is
  [`DELETION-REVIEW.md` §8](installer/plan/DELETION-REVIEW.md#8-script-disposition-under-the-binary-distribution).
- PeopleNet is the only detector wired into the DeepStream pipeline; `yolo11n`
  is reserved for future work — don't add YOLO wiring without being asked.
- [`installer/mv3dt_installer/assets/cameras/cameras.yml`](installer/mv3dt_installer/assets/cameras/cameras.yml)
  is the authoritative camera inventory seed **and** the record of the
  fleet's hardware facts (MACs, native 3072x1728 sensor resolution, the
  configured 1920x1080 main stream, vendor/firmware, required manual
  pre-flight). Its header is load-bearing documentation, not commentary: the
  installer copies it into every generated `<install_dir>/cameras.yml`. Note
  its caveat: the pinned `169.254.*` link-local IPs are volatile.
- Operator secrets are never committed: the installer writes them under
  `<install_dir>/secrets/`, which `installer/.gitignore` excludes.
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
and the other `installer/plan/` specs.

## Reference docs

- [`installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md`](installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md)
  — shared contracts for the unified installer (state machine, reporting,
  privilege, credentials).
- [`installer/plan/STEP-1…STEP-7`](installer/plan/) — per-step specs; STEP-6
  is the MQTT control plane, STEP-7 the HTTP data plane.
- [`installer/plan/DELETION-REVIEW.md`](installer/plan/DELETION-REVIEW.md) —
  what was removed from this fork and why, including the `laptop/` harness.
- [`installer/plan/DEEPSTREAM-SETUP.md`](installer/plan/DEEPSTREAM-SETUP.md) —
  DS 9.1 package/OS setup and AMC workflow; source of truth for version pins.
