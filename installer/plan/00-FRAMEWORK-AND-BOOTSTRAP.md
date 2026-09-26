# 00 — Installer Framework and Bootstrap (owner: DevA)

Status: shared foundation. The five step docs
(`STEP-1-PREREQUISITES.md` … `STEP-5-PER-PROJECT-EXES.md`) depend on the
contracts defined here. Do not restate these contracts in step docs — link
back to this document. See
[`AI-AGENT-SKILLS-ASSESSMENT.md`](AI-AGENT-SKILLS-ASSESSMENT.md) for a
sibling, non-step tooling reference on DS 9.1's Agentic Skills.

This document specifies the **single self-contained installer binary** that
unifies the loose numbered scripts under
[`laptop/scripts/`](../../laptop/scripts/) into one resumable, CD-style
installer for a brand-new Ubuntu 24.04 workstation
(target GPU: NVIDIA RTX PRO 4500 Blackwell) targeting DeepStream 9.1 +
AutoMagicCalib.

It supersedes and wraps-and-ports the phase logic in
[`laptop/scripts/00_bootstrap.sh`](../../laptop/scripts/00_bootstrap.sh) and
the numbered runtime scripts. The DeepStream/OS reference facts come from the
DS 9.1 docs (see [References](#references)); all version pins were
cross-checked directly against NVIDIA's published DS 9.1 Installation page.

---

## 1. Goals and non-goals

Goals:

- One artifact the operator runs: a single binary that behaves like a GUI
  installer for path selection but is TUI/CLI-first, so it runs reliably
  before the NVIDIA driver / desktop session is fully up.
- Fully resumable: survive the multiple reboots that the driver/CUDA install
  requires, and resume at the exact step on the next launch.
- Verbose, auditable: every dependency touch is reported with name + exact
  version, in a fixed human-readable format, and written to a transcript log.
- Five independently-buildable steps behind a stable module contract.

Non-goals (see [§13](#13-out-of-scope--defer-to-human)): hardware/BIOS setup,
per-camera network configuration, production hardening, systemd
supervision, and anything the DS 9.1 docs mark as manual.

---

## 2. Executable form (LOCKED)

- A **Python application** packaged into a **single self-contained binary**
  via **PyInstaller** (`--onefile`). Binary name: `mv3dt-installer`.
- **TUI/CLI-first.** Prompts are plain terminal prompts with sane defaults;
  the only "GUI-like" affordance is guided path selection with a default.
  No X/Wayland dependency — it must run from a TTY before the desktop is up.
- It **supersedes / wraps-and-ports** the logic in
  [`laptop/scripts/`](../../laptop/scripts/). Bash that still has to run
  verbatim (e.g. NVIDIA `update_rtpmanager.sh` invocation, apt transactions)
  is bundled as data assets and shelled out to; new orchestration lives in
  Python.

---

## 3. Single-binary architecture

### 3.1 Module layout

All installer code lives under `installer/` (a sibling of `laptop/`, so the
existing harness is untouched during the port):

```
installer/
├── plan/                           # these spec docs (this file is 00-*)
├── installer.spec                  # PyInstaller build recipe
├── pyproject.toml                  # deps + console-script entrypoint
└── mv3dt_installer/                # the Python package
    ├── __main__.py                 # `python -m mv3dt_installer` entrypoint
    ├── app.py                      # top-level TUI orchestrator / dispatch loop
    ├── state.py                    # StateMachine + state.json (§6)
    ├── logs.py                     # shared logger + transcript (§8)
    ├── report.py                   # verify_pinned / report_* helpers (§8.3)
    ├── reboot.py                   # boot-id reboot detection (§7)
    ├── privilege.py                # root + $SUDO_USER resolution (§9)
    ├── preflight.py                # OS/arch + invoking-user gates (§3.2)
    ├── onboarding.py               # first-run credential capture (§10, §14)
    ├── ngc.py                      # NGC key capture + secure store (§10)
    ├── webapp.py                   # web-app credential store (§14)
    ├── cameras.py                  # MAC-OUI camera discovery (§15)
    ├── waitui.py                   # blocking wait/poll screen (§3.2)
    ├── systemd.py                  # unit render + idempotent install
    ├── config.py                   # install-location config (§11)
    ├── shellout.py                 # bundled-asset locator + subprocess runner
    ├── steps/
    │   ├── __init__.py             # Step protocol, StepResult, STEP_REGISTRY
    │   ├── step1_prerequisites.py       # DevA
    │   ├── step2_deepstream_sdk.py       # DevB
    │   ├── step3_amc_launcher.py         # DevC
    │   ├── step4_calib_output_wiring.py  # DevD
    │   └── step5_per_project_exes.py     # DevD
    └── assets/                     # bundled bash/config, staged at runtime
        ├── scripts/                # installer-side bash (idempotent)
        │   ├── lib/
        │   │   └── common.sh       # installer-side sibling of laptop's common.sh
        │   ├── 10_setup_mosquitto.sh
        │   └── 60_record_tracking.sh
        ├── systemd/                # @PLACEHOLDER@ unit templates
        │   ├── mv3dt-ingest.path.in
        │   └── mv3dt-ingest.service.in
        ├── cameras/
        │   └── cameras.yml         # seed inventory + fleet header (§15)
        ├── deepstream/             # config templates (tracker/infer/app)
        └── mosquitto/              # mv3dt.conf drop-in
```

There is no `bootstrap/` directory: nothing clones this repo onto a
workstation and nothing builds the binary there (§4.1, §5).

### 3.2 Entrypoint and dispatch

- `__main__.py` → `app.main()`. `app.main()` runs, in this exact order:
  1. **Parse CLI flags** (§3.3).
  2. **Load the state file** into the `StateMachine` (`state.py`, §6).
  3. **`--status`** prints the state table and exits — deliberately still
     **pre-root**, so an operator can read install progress without `sudo`.
  4. **`privilege.require_root()`** (§9.1).
  5. **`onboarding.run_platform_preflight()`** — the Ubuntu 24.04 / `x86_64`
     gate plus the real-`$SUDO_USER` gate, delegating to `preflight.py`. It
     replaces the bare `privilege.resolve()` call and returns the same
     `InvokingUser` (§9.2).
  6. **Apply the reset flags** (`--reset-state`, `--reset-step N`).
  7. **Load the install-location config**
     (`config.load(gate_overrides=...)`, §11) — also where the opt-in gates
     are seeded from flag, environment, or prompt (§3.4).
  8. **Open the transcript logger** (`logs.open_transcript()`, §8).
  9. **`onboarding.onboard()`** — first-run credential capture: the NGC key
     (§10), camera RTSP credentials (§15), and, when its gate is `on`, the
     web-app credential (§14).
  10. **`reboot.reconcile()`** to clear any satisfied reboot-pending marker
      (§7).
  11. Enter the **dispatch loop** over `STEP_REGISTRY` (steps 1→7 in order).
      Steps 6 and 7 are opt-in and auto-skip when their gate is `off` (§3.4).

- **Onboarding runs after the transcript opens, deliberately.** Every prompt
  and its redacted outcome is then part of the auditable record §8.2
  requires; capturing credentials before `open_transcript()` would leave the
  one interactive moment of the whole install unlogged. Onboarding is a
  no-op on every launch after the first (§10.2, §14.3), so the ordering
  costs a resumed run nothing.

- `--scan-cameras` (§15) is a standalone mode shaped like `--status`, but it
  runs **after** `require_root()` because ARP scanning needs raw sockets.

- **Dispatch loop** (the core of the state machine, §6):
  For each step in order, if `state.status(step) == COMPLETE`, skip and emit
  the "already complete" log line. Otherwise call the step lifecycle
  (`preflight → run → verify → report`, §12) and record the returned
  `StepResult`. The loop **halts** the moment a step returns anything other
  than `COMPLETE`:
  - `REBOOT_REQUIRED` → write reboot marker (§7), print the USER-ACTION
    block telling the operator to reboot then **run the installer again to
    continue**, and exit 0.
  - `USER_ACTION_REQUIRED` → print the required manual actions (§9 display
    contract) and exit 0 with the "run the installer again to continue"
    line.
  - `FAILED` → print the failure + remediation and exit non-zero.
  On the next launch the loop resumes at the first non-`COMPLETE` step.

### 3.3 CLI flags (framework-level)

Steps may add their own, but the framework owns these:

- `--install-dir PATH` — override the default install location (§11).
- `--resume` — default behavior; explicit for clarity.
- `--status` — print the state table and exit.
- `--reset-state` — wipe `state.json` (mirrors `00_bootstrap.sh --reset-state`).
- `--reset-step N` — clear one step's completion so it re-runs.
- `--non-interactive` — never prompt; use defaults/config; fail if a required
  value is missing (mirrors `00_bootstrap.sh --non-interactive`).
- `--no-pause` — skip "press Enter" confirmations.
- `--verbose` — show every line of command output instead of the rolling
  8-line window (section 8.5). There is deliberately no `--quiet`.
- `--log-dir PATH` — override the transcript directory (§8).
- `--remote-supervision {off,local,remote}` — set the Step 6 gate (§3.4).
- `--webapp-integration {off,on}` — set the Step 7 gate (§3.4). Both gate
  flags default to `None` rather than `off`, so "not passed" stays
  distinguishable from "passed `off`" — the distinction the precedence in
  §3.4 depends on.
- `--scan-cameras` — run camera discovery, write the inventory, and exit
  (§15).
- `--camera-scan-cidr CIDR` — override the discovery sweep range (default
  `169.254.0.0/16`); persisted as `CAMERA_SCAN_CIDR` (§11.2).
- `--camera-scan-iface IFACE` — restrict discovery to one interface;
  persisted as `CAMERA_SCAN_IFACE` (§11.2).
- `-h/--help`, `--version`. `--version` carries **build provenance** in a
  release build — the tag, the short commit, and the UTC build time stamped
  in by CI (§4.1):

  ```
  mv3dt-installer 0.2.0 (v0.2.0, commit a1b2c3d, built 2026-08-19T14:02:11Z)
  ```

  In a source checkout the stamp is absent and it degrades to
  `mv3dt-installer 0.2.0`, which is how an operator tells a downloaded
  release binary from a developer build.

### 3.4 Opt-in step gates

Steps 1–5 always run. Steps 6 and 7 are **opt-in**: a workstation used only for
local calibration and ad-hoc pipeline runs needs neither 24/7 supervision nor a
web-app connection. Each is gated by an `installer.conf` key (§11.2) with a
matching CLI flag (§3.3):

| Gate key | Values (default first) | CLI flag | Owning step |
|---|---|---|---|
| `MV3DT_REMOTE_SUPERVISION` | `off` \| `local` \| `remote` | `--remote-supervision` | [`STEP-6` §E.2](STEP-6-REMOTE-SUPERVISION.md#e2-gating-opt-in) |
| `MV3DT_WEBAPP_INTEGRATION` | `off` \| `on` | `--webapp-integration` | [`STEP-7` §H.2](STEP-7-WEBAPP-INTEGRATION.md#h2-gating-opt-in) |

When a gate is `off` the dispatch loop treats that step as auto-`COMPLETE`
(skipped) with a one-line log, using the same skip discipline as a genuinely
completed step (§3.2).

**RESOLVED — how a gate value is chosen.** `config.load(gate_overrides=...)`
seeds each gate key on first write only, preserving the "capture once, then
read back" discipline: a key already present in `installer.conf` is never
re-seeded. Four tiers, highest first:

| Tier | Source | Notes |
|---|---|---|
| 1 | CLI flag (`--remote-supervision`, `--webapp-integration`) | The supported path. `default=None` is what distinguishes "not passed" from "passed `off`" |
| 2 | Environment (`MV3DT_REMOTE_SUPERVISION`, `MV3DT_WEBAPP_INTEGRATION`) | Compatibility affordance so `sudo -E ./mv3dt-installer` keeps working for scripted runs |
| 3 | Interactive prompt with `off` prefilled | Skipped entirely under `--non-interactive` |
| 4 | `off` | The default, and what `--non-interactive` leaves behind |

This **replaces the `exec sudo -E` hand-off in
`installer/bootstrap/bootstrap.sh`**, which was the only mechanism carrying
`MV3DT_WEBAPP_INTEGRATION` from the operator's shell into the process where
`config.load()` seeded it. With the bootstrap deleted (§5.3), tier 2 alone
would never reach the installer at all — hence tier 1. A flag passed on a run
where the key is **already persisted** overwrites the stored value and logs
the change, so a gate can be turned on later without hand-editing
`installer.conf`.

Under `--non-interactive` an unset gate stays `off`, so an unattended run
never enables long-running services or outbound network connections the
operator did not ask for.

---

## 4. PyInstaller packaging

### 4.1 What builds the binary

**GitHub Actions builds the binary; the target machine never builds anything**
(LOCKED). The build lives in
[`.github/workflows/release.yml`](../../.github/workflows/release.yml); the
operator's only artifact is the file attached to a GitHub Release (§5).

| Aspect | Contract |
|---|---|
| Publish trigger | `push` on tags matching `v*` |
| Dry-run triggers | `pull_request` touching `installer/**` (builds, uploads a 7-day artifact, never publishes); `workflow_dispatch` with a `draft` input |
| Runner | **`ubuntu-24.04`, pinned — never `ubuntu-latest`** |
| Version gate | Fails the build unless `"v" + mv3dt_installer.__version__` equals `$GITHUB_REF_NAME` |
| Build command | `pyinstaller installer/installer.spec --distpath dist --workpath /tmp/mv3dt-build --clean --noconfirm` |
| Smoke test | As the unprivileged runner user: `--version` contains the tag's version and `--status` exits 0 |
| Release assets | `mv3dt-installer` and `mv3dt-installer.sha256` |
| Publish command | `gh release create "$GITHUB_REF_NAME" --generate-notes --verify-tag dist/mv3dt-installer dist/mv3dt-installer.sha256` |

**Why the runner is pinned (LOCKED).** PyInstaller `--onefile` links its
bootloader against the **build host's glibc**. A binary built on a newer
runner image will not start on the Ubuntu 24.04 target, and `ubuntu-latest`
floats — it would one day silently produce an unusable release. The pin is
the only thing preventing that, so the workflow line carries a comment saying
so.

`--status` is the right smoke probe precisely because `app.main()` runs it
**before** `privilege.require_root()` (§3.2): it exercises argparse,
`state.py`'s forgiving reader, and `sys._MEIPASS` unpacking without root,
apt, or a GPU.

**Versioning and stamping.** `mv3dt_installer/__init__.py::__version__` is the
single source of truth; `pyproject.toml` reads it dynamically so the two
cannot drift. Bumping the version is an ordinary source commit — tagging is
what publishes. CI writes an **untracked** `mv3dt_installer/_buildinfo.py`
(`TAG`, `COMMIT`, `BUILT_UTC`) which `__init__.py` imports inside a
`try/except ImportError` with a source-mode fallback; that is what `--version`
renders (§3.3). No workflow ever writes a tracked file.

`installer.spec` (data-only summary; DevA owns the exact spec):

- `Analysis(['installer/mv3dt_installer/__main__.py'], ...)`.
- `datas` bundles the runtime assets so they ship inside the binary:
  - `installer/mv3dt_installer/assets/**` → `assets/` (the whole tree, §3.1)
  - the DeepStream config templates from
    [`laptop/deepstream/`](../../laptop/deepstream/) (tracker/infer/app/msgconv)
  - [`laptop/mosquitto/mv3dt.conf`](../../laptop/mosquitto/mv3dt.conf)
  - [`laptop/config/cameras.yml`](../../laptop/config/cameras.yml) →
    `assets/cameras/cameras.yml`, the seed inventory §15 reads
- `--onefile` → one executable at `dist/mv3dt-installer`. The build step
  passes `--distpath dist` with no `working-directory` override, so the
  output directory is `dist` at the **repo root**, not under `installer/`.
  (`installer.spec`'s own build-header comment still says `installer/dist`;
  that comment is stale against the actual workflow and is not fixed here —
  it lives in a different plan unit's file.) The repo-root `dist/` is not
  covered by any `.gitignore` rule today: `installer/.gitignore`'s `dist/`
  line is scoped to the `installer/` subtree and only ignores
  `installer/dist/`. A developer running the build command locally from the
  repo root gets the same `dist/mv3dt-installer` the workflow does.
- No hidden GUI toolkits and no third-party parsers on the hot path: the
  camera inventory reader in `cameras.py` is hand-rolled rather than pulling
  in PyYAML, so the frozen binary gains no hidden-import surface. Keep the dep
  list minimal so the binary is small and boots fast on a bare TTY.

### 4.2 Locating and staging bundled assets at runtime

PyInstaller `--onefile` unpacks datas into a temp dir exposed as
`sys._MEIPASS`. `shellout.py` MUST resolve asset paths through a single
helper so both "frozen" and "run-from-source" modes work:

```python
def asset_path(*parts: str) -> pathlib.Path:
    base = getattr(sys, "_MEIPASS", None)
    if base:                       # frozen binary
        return pathlib.Path(base, "assets", *parts)
    return pathlib.Path(__file__).parent / "assets" / *parts   # dev mode
```

- **`stage_assets(*parts, prefix=...) -> Path` (REQUIRED for bash).** Copies a
  bundled asset **directory** out of `sys._MEIPASS` into a fresh temp dir,
  preserving structure, `chmod 0755` on `*.sh` and `0644` otherwise. Staging
  the whole tree is what makes `source "$SCRIPT_DIR/lib/common.sh"` resolve —
  copying a single file leaves its `lib/` sibling behind in the bundle, and
  the staged script dies on its first line.
- **`run_bundled_script(..., tree=None, inherit_env=True, cleanup=True)`.**
  `tree=("scripts",)` stages the tree and runs from inside it. The staged bash
  is pointed at its own root with `MV3DT_ASSET_ROOT` and at the shared config
  with `MV3DT_INSTALLER_CONF=<install_dir>/installer.conf` (§11.2) — the two
  variables that replace the repo-root and `laptop.env` lookups the original
  `laptop/scripts/lib/common.sh` did.
- **`inherit_env=True` merges `env` over `os.environ`** instead of replacing
  it. Replacement semantics strip `PATH`, which breaks every `command -v` in
  the staged bash; `inherit_env=False` is the explicit opt-out for a
  deliberately sterile environment.
- **Transcript capture (REQUIRED).** The command line and both output streams
  are logged through `logs.log`, satisfying §8.2's requirement that
  shelled-out stdout/stderr reach the transcript. Redaction is applied to any
  `KEY=value` occurrence of `NGC_API_KEY`, `API_KEY`, `CAM_PASSWORD`, or
  `MQTT_PASSWORD`, which print as `<redacted>` (§10.1, §14.4).
- **`cleanup=True` removes the staging dir in a `finally`.** `cleanup=False`
  preserves it for debugging and is what tests use. Never execute directly
  from `sys._MEIPASS` — stage first, always.
- The binary is the unit of delivery; the operator does **not** need the repo
  checkout at runtime, and there is no checkout to fall back on. Nothing
  clones this repo onto a workstation (§5.3), so a staged tree plus
  `installer.conf` is the entire world a bundled script sees.

---

## 5. Distribution: the GitHub Release binary

One artifact, downloaded from this repo's **Releases** page onto a bare
Ubuntu 24.04 workstation that has `git` and nothing else. This section
supersedes the deleted `installer/bootstrap/bootstrap.sh`; §5.3 records what
that script did and where each responsibility went.

### 5.1 Operator procedure

1. **Open the Releases page** in a browser on the target workstation:
   <https://github.com/KevinTTO5/P2BP-25_26-Hardware_Test/releases/latest>.
2. **Download both assets** attached to the release: the executable
   `mv3dt-installer` and its checksum `mv3dt-installer.sha256`.
3. **Verify, mark executable, run** — from the download directory:

   ```bash
   sha256sum -c mv3dt-installer.sha256
   chmod +x mv3dt-installer
   sudo ./mv3dt-installer
   ```

   `sha256sum -c` must print `mv3dt-installer: OK`. If it does not, the
   download is corrupt or tampered with — re-download; do not run it.

That is the whole delivery mechanism. There is **no `curl | bash` one-liner**
— a piped shell script cannot be checksum-verified before it executes, which
is exactly the property the release asset buys — and **no separate USB
variant**: an offline lab install moves the same two files by whatever means,
then verifies and runs them identically. No step of this procedure clones a
repository, installs a compiler, or builds anything (§5.3).

### 5.2 What the binary does on first launch

Everything the bootstrap used to do before handing off, now inside the exe
and inside the transcript (§3.2 gives the exact ordering):

1. **Platform preflight** — Ubuntu 24.04 + `x86_64`, running as root via
   `sudo`, and `$SUDO_USER` resolving to a real non-root login (§9.2).
   `MV3DT_SKIP_PLATFORM_CHECK=1` is the **planned** developer escape hatch
   that downgrades the platform half to a warning; it arrives with
   `preflight.py` and exists nowhere in the tree today. The `$SUDO_USER`
   gate gets no escape hatch at all, because without it every secret and
   every per-user artifact lands in `/root`.
2. **Install location** — prompts with `/opt/mv3dt` prefilled (§11.1) and
   writes `installer.conf` (§11.2).
3. **Opt-in gates** — asks once for remote supervision and web-app
   integration, `off` prefilled, unless a flag or environment variable
   already answered (**§3.4**).
4. **NGC API key** — secret prompt, no echo, stored `chmod 600`; the key is
   required, so a blank answer re-prompts (**§10**).
5. **Web-app credential** — only when its gate is `on`: API key plus
   endpoint, normalized and stored beside the NGC key (**§14**).
6. **Dispatch** — the step loop takes over (§3.2).

On every launch after the first, preflight re-checks the platform and
onboarding is a **no-op**: "have we already asked?" is answered by the
existence of the secret file, not by parsing its contents (§10.2, §14.3), so
nothing re-prompts and the run resumes at the first non-`COMPLETE` step.

### 5.3 What the bootstrap did, and where it went (LOCKED)

| Bootstrap responsibility | Disposition |
|---|---|
| `git clone` of this repo onto the target | **Removed.** The binary is self-contained and the assets ship inside it (§4.2). Nothing clones this repo onto a workstation. |
| apt-installing build deps (`python3-venv`, `python3-pip`, `build-essential`, `curl`, `ca-certificates`) | **Removed.** A frozen PyInstaller binary needs no Python, venv, or toolchain at runtime, and the one real apt transaction belongs to [`STEP-1` §3](STEP-1-PREREQUISITES.md#3-ds-91-41-apt-prerequisite-package-list). |
| Building the binary with PyInstaller on the target | **Removed.** CI builds it once on the pinned runner (§4.1). |
| Ubuntu / arch preflight | **Kept**, in `preflight.py`, reading `/etc/os-release` rather than shelling to `lsb_release` (not installed on minimal Ubuntu 24.04). |
| `$SUDO_USER` validation | **Kept**, in `preflight.py`. Not cosmetic: `privilege.resolve()` falls back to `getpass.getuser()`, which under a bare root shell returns `root` and would put `secrets/`, `~/.ngc/config`, and the AMC clone in `/root`. |
| NGC key capture | **Kept**, in `onboarding.py` (§10.2). |
| Web-app credential capture | **Kept**, in `onboarding.py` (§14.3). |
| `exec sudo -E` to carry gate environment variables into the installer | **Replaced** by real CLI flags with a four-tier precedence (§3.4). |
| `--rebuild` / re-clone idempotency | **Not applicable.** Re-running means re-running one downloaded file; upgrading means downloading a newer release. |

---

## 6. Resumable state machine

### 6.1 State file

- Canonical path: **`/var/lib/mv3dt-installer/state.json`** (root-owned,
  `chmod 0644`; the directory is created `chmod 0755`). This is the direct
  successor to the existing line-based marker
  `/var/lib/mv3dt-laptop-bootstrap.state` used by
  [`00_bootstrap.sh`](../../laptop/scripts/00_bootstrap.sh) via
  [`lib/common.sh`](../../laptop/scripts/lib/common.sh)
  (`phase_done` / `mark_phase_done`); it upgrades that flat pattern to JSON so
  it can also hold reboot markers, timestamps, and the chosen install path.
- If `--install-dir` moves the install root, the state file stays at the
  canonical `/var/lib/mv3dt-installer/state.json` path (a stable, absolute,
  root-owned location survives home-dir/desktop churn during driver install).

### 6.2 Schema

```json
{
  "schema_version": 1,
  "install_dir": "/opt/mv3dt",
  "created_utc": "2026-07-01T22:00:00Z",
  "updated_utc": "2026-07-01T22:41:12Z",
  "steps": {
    "step1_prerequisites":    { "status": "COMPLETE", "finished_utc": "..." },
    "step2_deepstream_sdk":   { "status": "PENDING" },
    "step3_amc_launcher":     { "status": "PENDING" },
    "step4_calib_output_wiring": { "status": "PENDING" },
    "step5_per_project_exes": { "status": "PENDING" },
    "step6_remote_supervision":  { "status": "PENDING" },
    "step7_webapp_integration":  { "status": "PENDING" }
  },
  "reboot_pending": {
    "requested_by": "step1_prerequisites",
    "boot_id_at_request": "3f6b...c1",
    "requested_utc": "..."
  }
}
```

- `steps[*].status` ∈ the `StepStatus` enum (§12.2).
- `reboot_pending` is `null` unless a step requested a reboot (§7). While it
  is non-null, the dispatch loop refuses to advance past the requesting step.

### 6.3 API (`state.py`)

- `load() -> State` / `save(State)` — atomic write, per the shared helper
  below.
- `status(step_id) -> StepStatus` / `set_status(step_id, status)`.
- `mark_complete(step_id)` — sets `COMPLETE` + `finished_utc`.
- `set_reboot_pending(step_id, boot_id)` / `clear_reboot_pending()`.
- `all_complete() -> bool` — used to print the final success banner.

**REQUIRED — the shared atomic-write helper.** `state.json`, the Step 5
registry ([`STEP-5` §4.1](STEP-5-PER-PROJECT-EXES.md#41-path)), and every
run-state file written by [`STEP-7` §F.2](STEP-7-WEBAPP-INTEGRATION.md#f2-state-file-fan-in)
use one implementation. The `flush` + `fsync` pair before `os.replace` is the
part that matters: without it a power loss mid-install can leave a
zero-length `state.json` and lose every completed step.

```python
def write_json_atomic(path: pathlib.Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)          # atomic within a filesystem
```

Readers are correspondingly forgiving: a missing file yields the empty default
and a `json.JSONDecodeError` is treated as "absent", never as a fatal error —
so a partially-written file can never wedge the installer.

Re-running the binary after a reboot resumes at the first non-`COMPLETE`
step; completed steps log the standard "already complete" line and are
skipped, exactly like the phase-skip behavior in `00_bootstrap.sh`.

---

## 7. Reboot detection / continuation contract

This is the contract **Step 1 consumes** (driver/CUDA install requires
reboots) and any later step may use.

### 7.1 Signaling "reboot required"

A step signals a reboot by returning `StepResult(status=REBOOT_REQUIRED,
message=..., user_actions=[...])`. The framework then:

1. Reads the current boot id and stores it:
   `state.set_reboot_pending(step_id, boot_id=current_boot_id())`.
2. Prints the USER-ACTION reboot block (§9.4) including the exact line
   **"run the installer again to continue"**.
3. Exits 0 (a pending reboot is a normal, expected pause — not a failure).

### 7.2 Detecting the reboot actually happened

`reboot.current_boot_id()` reads, in priority order:

1. **`/proc/sys/kernel/random/boot_id`** — a fresh random UUID generated by
   the kernel on every boot. This is the primary signal.
2. Fallback: **`btime`** from `/proc/stat` (boot wall-clock epoch), used only
   if `boot_id` is unreadable.

`reboot.reconcile(state)` runs early in `app.main()` (step 5 of §3.2):

- If `state.reboot_pending` is `null` → nothing to do.
- Else compare `state.reboot_pending.boot_id_at_request` to
  `current_boot_id()`:
  - **Different** → the machine really rebooted. Mark the requesting step
    `COMPLETE` (its work finished; the reboot was its tail), call
    `state.clear_reboot_pending()`, log "reboot confirmed", and let dispatch
    continue to the next step.
  - **Same** → the operator re-ran the binary **without** rebooting. Do NOT
    advance. Re-print the reboot USER-ACTION block and exit 0. This blocks the
    next step until the reboot is confirmed.

### 7.3 Guarantee for step authors

A step that returns `REBOOT_REQUIRED` is guaranteed that, on the next launch
that actually follows a real reboot, it is treated as `COMPLETE` and the next
step runs. If a step needs to do post-reboot verification itself (rather than
being auto-completed), it instead returns `USER_ACTION_REQUIRED` for the
reboot and re-runs its own `verify()` on next launch — Step 1 will document
which it uses.

---

## 8. Verbose logging + reporting contract

### 8.1 Shared logger (`logs.py`)

- Colour-aware, level-prefixed lines to **stderr** so stdout stays reserved
  for machine-parsable output — same discipline as
  [`lib/common.sh`](../../laptop/scripts/lib/common.sh) (`log_info`,
  `log_warn`, `log_error`, `die`). Python equivalents: `log.info/warn/error`
  and `die(msg)` (logs error + non-zero exit).
- Every emitted line is **also** appended to the transcript (§8.2).

### 8.2 Transcript log file

- Directory: **`/var/lib/mv3dt-installer/logs/`** (override with `--log-dir`).
- Per-run file: `install-YYYYMMDD-HHMMSS.log` plus a `latest.log` symlink.
- The transcript captures every logger line, every prompt + chosen value
  (secrets redacted — see §10), and the stdout/stderr of every shelled-out
  bash fragment. It is the auditable record of the whole CD-style install.

### 8.3 Reporting format for dependencies (REQUIRED, exact strings)

Every step reports dependency state through two shared helpers so the wording
is identical everywhere. These are the required human-facing strings:

- **Newly installed** dependency — reported with name + version:

  ```
  installed <dependency> version <version>
  ```

- **Pre-existing** dependency — reported as:

  ```
  already installed <dependency> version <version>
  ```

Concrete examples:

```
installed cuda-toolkit-13-2 version 13.2
already installed gstreamer1.0-tools version 1.24.2
already installed deepstream-9.1 version 9.1.0-1
```

Helper signatures (`report.py`):

- `report_installed(dependency: str, version: str) -> None`
  → prints/logs `installed <dependency> version <version>`.
- `report_already_installed(dependency: str, version: str) -> None`
  → prints/logs `already installed <dependency> version <version>`.

### 8.4 "Verify at exact pinned version" helper (REQUIRED)

A single reusable helper does equality-pinned verification, porting
`require_version_eq` from
[`lib/common.sh`](../../laptop/scripts/lib/common.sh):

- `verify_pinned(label: str, actual: str, expected: str) -> bool`
  - On match: logs `Version OK: <label> == <actual>` and returns `True`.
  - On mismatch: logs
    `Version check failed: <label> — expected '<expected>', got '<actual>'`
    and returns `False` (the caller decides whether to `die` or surface a
    `USER_ACTION_REQUIRED`).

Every step MUST use `verify_pinned` in its `verify()` and one of the two §8.3
reporters after every dependency it touches, so the transcript is uniform and
greppable. Steps use **equality pins** (DS 9.1 refuses older/newer minors of
the driver and `libnvinfer*`; see [References](#references)).

### 8.5 Verbosity and live command output (REQUIRED)

The transcript contract above is about the **record**. This subsection is
about the **screen**, and it owns exactly two framework-level facts: the
`--verbose` flag ([section 3.3](#33-cli-flags-framework-level)), and what
`Context.run_root`'s `stream=None` resolves to. The rendering itself —
bars, spinners, the rolling window, the step and phase banner — is
specified in
[`08` section 4](08-PROGRESS-AND-OBSERVABILITY.md#4-streaming-command-output)
and
[`08` section 8](08-PROGRESS-AND-OBSERVABILITY.md#8-verbosity),
and is not restated here.

`--verbose` replaces the rolling window with the full stream. The window's
size and its fixed-height rule are
[`08` section 8](08-PROGRESS-AND-OBSERVABILITY.md#8-verbosity)'s and are
deliberately not repeated here, so there is one place to change them.
`--verbose` is what an operator is told to re-run with when reporting a
problem, in place of being asked to find and tail the transcript by hand.

**There is no `--quiet` (RESOLVED).** It has no consumer. The systemd units
in [`STEP-6`](STEP-6-REMOTE-SUPERVISION.md) pass no `--non-interactive`;
what covers them is `StandardError=journal`, which is not a tty, so they
already take the last row of the table below. And the transcript holds every
line whatever the screen shows
([`08` section 7.1](08-PROGRESS-AND-OBSERVABILITY.md#71-live-rendering-must-never-cost-the-transcript-required)).
Adding a flag that can only make a silent installer more silent would
reintroduce the failure mode doc `08` exists to remove.

#### AUTO mode, resolved

[`08` section 4.1](08-PROGRESS-AND-OBSERVABILITY.md#41-the-one-seam-locked)
described `stream=None` as "stream when stderr is a tty and the caller asked
for captured output", and
[`08` section 7](08-PROGRESS-AND-OBSERVABILITY.md#7-behaviour-off-a-tty-required)
requires that a non-tty run still get per-line output, plain. Read together,
a tty-gated AUTO sends nothing at all to an operator whose **stderr** is not
a terminal: `mv3dt-installer 2>&1 | tee install.log`, any run redirected to
a file, CI, and the journald-backed STEP-6 units. That is exactly the
silence that caused two interrupted installs. Section 7 is REQUIRED and
section 4.1's tty clause was prose inside it, so the narrower statement
gives way.

> A plain `mv3dt-installer | tee install.log` is **not** one of these cases,
> and it is worth stating because it is the example that first motivated
> this resolution and it is wrong: a pipe redirects stdout, leaving stderr
> on the terminal, so the old gate would have streamed there normally. The
> resolution is unchanged — the cases above are real — but the reasoning
> should not rest on a case that does not exist.

**RESOLVED: AUTO asks one question, and a terminal is not part of it.**
`stream=None` streams whenever the call is one the tee runner can serve —
`capture_output=True` **and** text mode — because only then can it return a
`CompletedProcess` shaped exactly as `subprocess.run` would have. Nothing in
the resolution reads `isatty`. Whether a streamed line is drawn into a live
region or written out plain is the renderer's decision, made once per run
from the same two inputs section 7 already names.

| Context | `--verbose` | AUTO streams | On screen | Bars / spinners | Transcript |
|---------|-------------|--------------|-----------|-----------------|------------|
| Interactive tty | off | Yes | Last 8 lines, rolling | Yes | Every line |
| Interactive tty | on | Yes | Every line, scrolling above the live region | Yes | Every line |
| Interactive tty, `--non-interactive` | off or on | Yes | Every line, plain | No | Every line |
| Not a tty (pipe, CI, `tee`) | off or on | Yes | Every line, plain | No | Every line |

Three consequences, stated so they need not be re-derived:

1. **`--verbose` is observable only on an interactive tty.** Everywhere else
   the full stream is already what reaches the screen, because there is no
   cursor to redraw a window with. The flag is accepted and changes nothing
   there, which is the correct behavior rather than an omission.
2. **`--non-interactive` suppresses drawing, never streaming.** It is a
   separate question from verbosity and answers only "may anything be drawn
   in place", matching section 7's third row.
3. **`stream=False` is absolute.** A step author's explicit "this probe's
   output is noise" is not overridden by `--verbose`; the line still reaches
   the transcript, so the record loses nothing. `stream=True` is the mirror
   escape hatch, and it raises rather than streaming if the call is not one
   the tee can return unchanged.

---

## 9. USER-ACTION display + privilege/sudo contract

### 9.1 Privilege handling

- The installer must run as **root** (installs packages, writes
  `/etc/profile.d/*`, `/var/lib/*`). `privilege.require_root()` mirrors
  `require_root` in [`lib/common.sh`](../../laptop/scripts/lib/common.sh):
  exit with a clear message if `os.geteuid() != 0`.
- The operator launches it via `sudo ./mv3dt-installer`. `-E` is **no longer
  required for the opt-in gates** — `--remote-supervision` and
  `--webapp-integration` are real flags (§3.4), and there is no longer a
  bootstrap `exec sudo -E` to carry the environment across. Use `sudo -E`
  only to pass a pre-set `NGC_API_KEY` through the environment instead of
  answering the prompt; `onboarding.py` is what honors it, before prompting
  (§10.2).

### 9.2 Resolving the invoking user / home

Ported directly from the `$SUDO_USER` logic in
[`00_bootstrap.sh`](../../laptop/scripts/00_bootstrap.sh) (LOCAL_DEB_DIR
resolution) and the per-user NGC handling:

`privilege.resolve() -> InvokingUser` returns:

- `name`: `$SUDO_USER` if set and not `"root"`, else the current user.
- `home`: resolved from the passwd DB, equivalent to
  `getent passwd "$SUDO_USER" | cut -d: -f6` (NOT `$HOME`, which is
  root's under sudo).
- `uid`/`gid`: the invoking user's, for `chown`-ing artifacts back.

Rules for step authors:

- Anything that must run "as the user" (the `ngc` CLI, `docker` without sudo,
  files under the user's home, the AMC clone under `$HOME/auto-magic-calib`)
  MUST be run via `privilege.run_as_user(...)` (equivalent to
  `sudo -u "$SUDO_USER" -H ...`), exactly as Phases 6/10 of
  `00_bootstrap.sh` do for `ngc`.
- Files created for the operator (env files, models, per-project exes) MUST be
  `chown`ed to the invoking user/group.

### 9.3 USER-ACTION display contract

When a step needs the operator to do something by hand (PATH edits, a command
to run, placing an NGC download, a reboot), it returns
`USER_ACTION_REQUIRED` with a `user_actions: list[UserAction]`. The framework
renders a single, unmissable block to stderr + transcript:

```
+---------------------------------------------------------------+
|  ACTION REQUIRED                                              |
+---------------------------------------------------------------+
  Step: <step title>
  Why:  <one-line reason>

  Do the following, in order:
    1. <action text>            (command, if any, shown verbatim)
    2. <action text>

  Then run the installer again to continue.
+---------------------------------------------------------------+
```

- The final line is ALWAYS exactly: **"Then run the installer again to
  continue."** (the phrase "run the installer again to continue" is the
  contract).
- `UserAction` fields: `text: str`, optional `command: str` (rendered
  verbatim, copy-pasteable), optional `path: str` (for "edit this file"
  actions like appending CUDA/DS exports to a profile).

### 9.4 Reboot block

The `REBOOT_REQUIRED` path (§7.1) renders the same frame with the reason
"a reboot is required before the next step can run" and the same closing
"run the installer again to continue" line, matching the reboot banners
already in [`00_bootstrap.sh`](../../laptop/scripts/00_bootstrap.sh)
(Phases 2 and 3).

---

## 10. NGC API key capture + local secure storage

The key is captured on first launch by `onboarding.py` (§5.2) and consumed
by **Step 2** for the PeopleNet model fetch and for the Docker install method
(both are NGC-gated), and by later steps that pull gated NGC content. The DS
SDK deb/tar artifacts themselves are **public GitHub Release assets** and do
not need this key — see [`STEP-2`](STEP-2-DEEPSTREAM-SDK.md#1-locked-facts-and-pins-from-ds-91-docs)
for the acquisition split.

### 10.1 Storage contract

- Canonical secret file: **`<install_dir>/secrets/ngc.env`** owned by the
  invoking user, permissions **`chmod 600`**, directory `chmod 700`. Single
  line: `NGC_API_KEY=<key>`.
- If `<install_dir>` is inside a git working tree, the secret dir MUST be
  gitignored. Add `installer/**/secrets/` and `secrets/ngc.env` to the
  relevant `.gitignore` (the existing
  [`laptop/.gitignore`](../../laptop/.gitignore) already demonstrates the
  pattern by ignoring `config/laptop.env`). The key is **NEVER** committed.
- The key is **NEVER** written to the transcript log. Any prompt echo is
  suppressed (secret input), and any log line that would contain it prints
  `NGC_API_KEY=<redacted>`.

### 10.2 Capture + handoff API (`ngc.py`)

The key is **REQUIRED** — there is no manual fallback. `capture_key()` never
returns without one:

- `capture_key(non_interactive: bool) -> str` — secret prompt; a blank
  answer re-prompts rather than being accepted. Under `--non-interactive`
  there is no human to re-prompt, so this dies immediately (§4: "fail if a
  required value is missing") unless the caller has already resolved a key
  some other way (see below).
- `store_key(key: str, install_dir) -> Path` — writes `secrets/ngc.env`
  (`chmod 600`), chowned to the invoking user.
- `load_key() -> str | None` — read back for a step; `None` only means
  onboarding has not run yet. Every step that runs after onboarding can
  treat a present key as guaranteed.
- `configure_ngc_cli()` — for steps that call `ngc`, writes
  `~/.ngc/config` (`chmod 600`) as the invoking user, mirroring Phase 10 of
  [`00_bootstrap.sh`](../../laptop/scripts/00_bootstrap.sh).

**The caller is `onboarding.ensure_ngc_key()`** (§3.2, §5.2), which runs on
every launch and must be silent after the first. Two rules make that true:

- **"Have we already asked?" is the existence of
  `<install_dir>/secrets/ngc.env`, not the result of `load_key()`** —
  though since a key is mandatory, the file now only ever exists holding a
  real key, so the two checks agree in practice; the distinction only
  matters mid-run, before `store_key()` has been called.
- **A pre-set `NGC_API_KEY` in the environment is honored before prompting**
  (`sudo -E`, §9.1), then stored through `store_key` like any other value —
  this is also what makes a non-interactive first run succeed without
  `capture_key()` ever being reached.

---

## 11. Install-location config

### 11.1 Default and selection

- Default install directory: **`/opt/mv3dt`**.
- On first run (or when `--install-dir` is not given and no config exists),
  the TUI prompts for the path with `/opt/mv3dt` prefilled (GUI-installer-like
  path selection). `--non-interactive` uses the default silently.

### 11.2 Persistence + sharing with later steps

- The chosen path is written into `state.json` as `install_dir` (§6.2) — the
  single source of truth — and mirrored into a small
  `<install_dir>/installer.conf` (KEY=VALUE) for human inspection and for
  bundled bash fragments (`set -a; . installer.conf`), same shape as
  [`laptop/config/laptop.env`](../../laptop/config/laptop.env.example). It is
  the file `MV3DT_INSTALLER_CONF` points staged bash at (§4.2).
- `config.load()` resolves precedence: `--install-dir` > `state.json` >
  `installer.conf` default > `/opt/mv3dt`. Every consumer of a path under the
  install root MUST take it from `config.load()`; a module-level default read
  unconditionally silently ignores `--install-dir` (§14.3 records one such
  bug).
- Layout under the install dir (created by the framework, consumed by steps):

  ```
  <install_dir>/
  ├── installer.conf        # chosen path + shared vars (table below)
  ├── cameras.yml           # runtime camera inventory, MAC-keyed (§15)
  ├── cameras.scan.json     # raw scan record: MACs, ifaces, unmatched (§15)
  ├── cameras/              # still frames captured for position binding (§15)
  ├── secrets/
  │   ├── ngc.env           # NGC key, chmod 600 (§10)
  │   ├── camera.env        # CAM_USER + CAM_PASSWORD, chmod 600 (§15)
  │   └── webapp.env        # API_KEY + ENDPOINT, chmod 600 (§14)
  ├── bin/
  │   ├── mv3dt-installer   # the installed copy of the release binary
  │   └── ...               # per-project exes dropped here (Steps 3 & 5)
  ├── deepstream/           # rendered DS configs + calibration (Step 4)
  ├── projects/             # per-project registry entries (Step 5)
  ├── agent/                # MQTT control agent env/config (Step 6)
  ├── webapp/               # upload queue + run-state files (Step 7)
  └── run/                  # worker state files, fan-in source (Step 7 §F.2)
  ```

- `<install_dir>/bin/mv3dt-installer` is the stable, absolute path a systemd
  unit's `ExecStart` can name — the operator's downloaded copy may live
  anywhere, or be deleted.
- Shared variables written into `installer.conf`:

  | Key | Meaning |
  |---|---|
  | `MV3DT_INSTALL_DIR` | The resolved install root; every other path is relative to it |
  | `MV3DT_REMOTE_SUPERVISION` | Step 6 gate (§3.4) |
  | `MV3DT_WEBAPP_INTEGRATION` | Step 7 gate (§3.4) |
  | `CAMERAS_FILE` | Absolute path to the runtime camera inventory, normally `<install_dir>/cameras.yml` (§15) |
  | `CAMERA_SCAN_CIDR` | Sweep range for discovery; default `169.254.0.0/16` (§15.2) |
  | `CAMERA_SCAN_IFACE` | Restrict discovery to one interface; empty means every candidate interface (§15.2) |
  | `CALIB_FOOTAGE_SECONDS` | Optional calibration clip length in seconds; default `300` ([`STEP-4-CALIBRATION-FOOTAGE` §2](STEP-4-CALIBRATION-FOOTAGE.md#2-pinned-values)) |
  | `CALIB_FOOTAGE_DIR` | Optional calibration footage root; default `<invoking user home>/Downloads/mv3dt-calibration-footage`, persisted by Step 4 on first recording when unset ([`STEP-4-CALIBRATION-FOOTAGE` §2](STEP-4-CALIBRATION-FOOTAGE.md#2-pinned-values), [§6](STEP-4-CALIBRATION-FOOTAGE.md#6-deletion-after-a-successful-export)) |

  Camera RTSP credentials are intentionally absent from `installer.conf`;
  they live in `secrets/camera.env` and are loaded only in memory (§15.3).

- **Steps 3 and 5** drop per-project executables into `<install_dir>/bin/`;
  **Step 4** writes rendered DeepStream config + calibration under
  `<install_dir>/deepstream/`. They MUST read the path via `config.load()`,
  never hardcode it.

---

## 12. Step-module interface (the contract for Steps 1–5)

Steps are independently buildable against this interface. Each
`stepN_*.py` module exposes one class implementing the `Step` protocol and
registers it in `STEP_REGISTRY` (ordered 1→5).

### 12.1 Protocol

```python
class Step(Protocol):
    id: str          # e.g. "step2_deepstream_sdk"  (matches state.json key)
    title: str       # human title for logs/USER-ACTION blocks
    order: int       # 1..5

    def preflight(self, ctx: Context) -> StepResult: ...
    def run(self, ctx: Context) -> StepResult: ...
    def verify(self, ctx: Context) -> StepResult: ...
    def report(self, ctx: Context) -> None: ...
```

- **`preflight(ctx)`** — cheap checks that prerequisites from prior steps /
  the environment are satisfied (tools present, previous step COMPLETE,
  required inputs available). May return `USER_ACTION_REQUIRED` (e.g. missing
  NGC key with no fallback) or `FAILED`; returns `COMPLETE` to mean "ok to
  run".
- **`run(ctx)`** — does the work (apt transactions, NGC downloads, docker
  compose, rendering configs, dropping exes). Uses `report_installed` /
  `report_already_installed` (§8.3) for every dependency touched. May return
  `REBOOT_REQUIRED` / `USER_ACTION_REQUIRED` / `FAILED` / `COMPLETE`.
- **`verify(ctx)`** — idempotent post-checks using `verify_pinned` (§8.4)
  against the DS 9.1 pins. Returns `COMPLETE` only when every pin matches.
- **`report(ctx)`** — prints the step's human-facing summary block (what was
  installed/skipped, versions, where files landed). No side effects.

### 12.2 `StepResult` and status recorded by the state machine

```python
class StepStatus(Enum):
    PENDING = "PENDING"
    COMPLETE = "COMPLETE"
    REBOOT_REQUIRED = "REBOOT_REQUIRED"
    USER_ACTION_REQUIRED = "USER_ACTION_REQUIRED"
    FAILED = "FAILED"

@dataclass
class StepResult:
    status: StepStatus
    message: str = ""
    user_actions: list[UserAction] = field(default_factory=list)
```

- The dispatch loop (§3.2) runs `preflight → run → verify`; the **effective**
  result is the first non-`COMPLETE` result, else `COMPLETE`. On `COMPLETE`
  it calls `report()` and `state.mark_complete(step.id)`. Any other status is
  recorded verbatim and halts the loop with the appropriate USER-ACTION /
  reboot / failure handling.
- Steps never write `state.json` directly — they only return a `StepResult`;
  the framework owns all persistence.

### 12.3 `Context` object passed to every step

`Context` bundles the shared services so steps stay decoupled from globals:

- `install_dir: Path`, `conf: dict` (from `config.load()`, §11).
- `user: InvokingUser` (§9.2) and `run_as_user(...)`, `run_root(...)`.
- `log` (§8.1), `report_installed`, `report_already_installed`,
  `verify_pinned` (§8.3–8.4).
- `ngc` handle (`load_key`, `configure_ngc_cli`, §10 — the key is required,
  so a present key is guaranteed by the time any step runs).
- `webapp` handle (`load_credentials`, `enabled`, §14) — the web-app API key
  and normalized endpoint, for steps that talk to the backend.
- `asset_path(...)` (§4.2) to locate bundled bash/config.
- `reboot.request()` helper (which just returns `REBOOT_REQUIRED`; the
  framework does the boot-id bookkeeping, §7).

### 12.4 What each step owns (cross-doc map)

Each step doc defines its own internals but consumes only the contracts above:

- **Step 1 — Prerequisites (DevA):** driver `595.58.03` / CUDA `13.2` /
  cuDNN `9.20.0.48` / TensorRT `10.16.0.72-1+cuda13.2` / GStreamer `1.24.2` +
  apt prereqs; Blackwell driver-support check for the RTX PRO 4500;
  verification via `verify_pinned`; uses the **reboot gate** (§7).
- **Step 2 — DeepStream SDK (DevB):** DS 9.1 install via deb / tar / docker
  (auto-detect or prompt) — deb/tar are public GitHub Release downloads, the
  docker image is NGC-gated using the key from §10 (always present after
  onboarding) — install-path, post-install (`update_rtpmanager.sh`,
  `ldconfig`, `/etc/profile.d/deepstream.sh`), version and model verification.
- **Step 3 — AMC launcher (DevC):** docker compose AMC bring-up, open the
  localhost UI, keep the service up until the browser closes; standalone AMC
  exe dropped in `<install_dir>/bin/` (§11).
- **Step 4 — Calib output wiring (DevD):** place the AMC output file (prompt +
  default), render `deepstream_app_config`, patch
  `config_tracker_NvMOT.yml` `calibrationDirectory` / `LOCATION_ID`; writes
  under `<install_dir>/deepstream/`.
- **Step 5 — Per-project exes (DevD):** start-or-close, project-named
  DeepStream exe + reusable AMC exe, project registry / re-run; writes to
  `<install_dir>/bin/` and `<install_dir>/projects/`.
- **Step 6 — Remote supervision (DevD, opt-in §3.4):** per-project pipelines
  become boot-enabled `mv3dt-pipeline@<slug>.service` instances plus an MQTT
  control agent — the **control plane** (run / stop / restart). See
  [`STEP-6`](STEP-6-REMOTE-SUPERVISION.md).
- **Step 7 — Web-app integration (DevD, opt-in §3.4):** the HTTP **data
  plane** — registration/status, signed-URL artifact upload, and
  web-app-initiated one-shot operations, against the credential contract in
  §14. See [`STEP-7`](STEP-7-WEBAPP-INTEGRATION.md).

---

## 13. Out of scope / defer to human

- Everything the DS 9.1 Installation page treats as manual and everything in
  Notion §1–3 / [`DEEPSTREAM-SETUP.md`](../../laptop/docs/DEEPSTREAM-SETUP.md)
  §1–3: hardware selection, BIOS (Secure Boot, virtualization, discrete-GPU
  primary), and the Ubuntu 24.04 dual-boot install itself. The installer
  assumes it is already booted into Ubuntu 24.04.
- **Per-camera setup through each camera's web UI stays manual** (Notion
  §7.1–7.4): activation (the fleet ships un-activated and refuses RTSP until
  an admin password is set), disabling OSD text, and stream-profile
  selection. **Camera IP discovery is NOT out of scope** — it is owned by
  §15. There is no IP to configure by hand: the fleet self-assigns
  `169.254.*` link-local addresses that are neither DHCP-reserved nor
  derivable from the MAC, so the installer discovers cameras by MAC OUI
  instead of trusting a pinned list. The pipeline consumes the resulting
  RTSP URLs.
- Secure Boot MOK enrollment and any BIOS interaction — surfaced as
  `USER_ACTION_REQUIRED` by Step 1, performed by the operator.
- `ufw`, Tailscale/NoMachine, and dashboards — see the "Out of scope" /
  "Future work" sections of
  [`SCRIPTED-WORKFLOW.md`](../../laptop/docs/SCRIPTED-WORKFLOW.md).
  (systemd supervision and Mosquitto ACL/password hardening are **no longer**
  out of scope — [`STEP-6`](STEP-6-REMOTE-SUPERVISION.md) owns both.)
- **The web app itself** — its UI, its API implementation, cloud auth
  infrastructure, and API-key issuance/rotation. The framework captures and
  stores the credential (§14) but never provisions it; the desktop-side half
  of the integration is [`STEP-7`](STEP-7-WEBAPP-INTEGRATION.md), and its own
  exclusions are listed in
  [`STEP-7` §I](STEP-7-WEBAPP-INTEGRATION.md#i-out-of-scope--flag-for-human).
- Alternate detectors (`yolo11n`): explicitly deferred; PeopleNet is the only
  detector installed, matching NVIDIA's DS 9.1 MV3DT reference.
- The NGC account itself and any credential the operator must obtain from
  NVIDIA — the installer captures and stores the key locally (§10) but never
  provisions it.

---

## 14. Web-app connection contract

The desktop talks to a cloud web app over **HTTP with a bearer API key**. This
section owns the credential and endpoint contract — the *transport* and the
*routes* belong to [`STEP-7`](STEP-7-WEBAPP-INTEGRATION.md), and
[`STEP-6`](STEP-6-REMOTE-SUPERVISION.md) owns the separate MQTT control plane.
It is deliberately a sibling of §10 (NGC key): same storage discipline, same
secrecy rules, different consumer.

> **Provenance:** this contract is ported from the P2BP Jetson camera-node
> agent, which has been running against the same backend. The capture flow
> mirrors that tree's `install.sh` (credential prompt → normalize → atomic
> write → `chmod 600`) and the endpoint rule mirrors its `_normalize_endpoint`.
> Those source files are being removed from this fork — see
> [`DELETION-REVIEW` §3](DELETION-REVIEW.md#3-deletions-gated-on-the-harvest-the-jetson-tree)
> — so the behavior is specified here in full rather than by reference.

### 14.1 Storage contract

- Canonical secret file: **`<install_dir>/secrets/webapp.env`**, owned by the
  invoking user (§9.2), permissions **`chmod 600`**, directory `chmod 700` —
  identical to §10.1. Exactly two lines:

  ```
  API_KEY=<key>
  ENDPOINT=<normalized-base-url>
  ```

- The file is `KEY=VALUE` with no quoting or expansion, so it is directly
  consumable both by `EnvironmentFile=` in a systemd unit and by
  `set -a; . webapp.env` in a bundled bash fragment (§11.2).
- If `<install_dir>` is inside a git working tree, the secrets directory MUST
  be gitignored, exactly as §10.1 requires for `ngc.env`. The key is **NEVER**
  committed.
- Only the API key is secret. `ENDPOINT` is not, but it lives in the same file
  so there is one place to look and one file to `chmod`.

### 14.2 Endpoint normalization (REQUIRED)

Operators paste base URLs inconsistently — with a trailing slash, or with the
`/api` suffix already attached, because that is what the browser address bar
shows. Every consumer joins routes as `<endpoint> + /api/...`, so a
non-normalized value silently produces `https://host/api/api/...`.

Normalization is applied **once at capture** and **again on every load** (cheap,
and it protects against a hand-edited file):

1. Strip surrounding whitespace.
2. Strip **all** trailing `/` characters.
3. If the result ends in `/api` (case-insensitive), remove that suffix, then
   strip trailing `/` again.
4. An empty result is an error, not an empty endpoint.

| Operator input | Normalized `ENDPOINT` |
|---|---|
| `https://host` | `https://host` |
| `https://host/` | `https://host` |
| `https://host/api` | `https://host` |
| `https://host/api/` | `https://host` |
| `https://host/API//` | `https://host` |

Route joining is correspondingly strict: `join(endpoint, path)` right-strips
`/` from the base, left-pads a single `/` onto the path, and concatenates —
never `urljoin`, whose relative-reference semantics would discard a base path
component.

### 14.3 Capture + handoff API (`webapp.py`)

Mirrors the §10.2 shape so the two credential handles behave identically. The
caller is `onboarding.ensure_webapp_credentials()` (§3.2, §5.2), and only when
the §3.4 gate is `on`:

- `capture_credentials(non_interactive: bool, install_dir=...) -> Credentials`
  — the prompt flow. **When a value already exists, ask before replacing it**
  rather than forcing re-entry: print that a credential was found and offer to
  keep it, defaulting to keep. The key prompt suppresses echo; the endpoint
  prompt does not. Under `--non-interactive`, existing values are kept
  silently and missing ones are left unset. A blank endpoint writes nothing
  and warns — `store_credentials` requires both fields — and
  [`STEP-7`](STEP-7-WEBAPP-INTEGRATION.md) surfaces its USER-ACTION block on
  first run.
- `store_credentials(creds, install_dir) -> Path` — writes `secrets/webapp.env`
  via the §6.3 atomic-write discipline, then `chmod 600` and `chown` to the
  invoking user. Write-then-permission ordering matters: create the file with a
  restrictive mode from the start (`os.open` with `0o600`) so the key is never
  briefly world-readable.
- `load_credentials(install_dir=...) -> Credentials | None` — reads back,
  re-normalizes the endpoint (§14.2), and returns `None` when either value is
  missing. Steps treat `None` as "not configured" and surface a USER-ACTION
  block (§9.3), not a crash. This one **already** takes `install_dir` and
  honors it.
- `enabled(gate_value: str, install_dir=...) -> bool` — the §3.4 gate combined
  with a successful `load_credentials()`. This is what a step's `preflight`
  checks. `gate_value` is **mandatory and stays first**; see the callout.

> **`install_dir` parameter (REQUIRED — fixes a real bug).** The defect is
> confined to `capture_credentials` and `enabled`: each calls a **bare
> `load_credentials()`** internally, so both read the module-level
> `DEFAULT_INSTALL_DIR` and **ignore `--install-dir`** (§3.3).
> `load_credentials` itself is already correct. With a non-default install
> root, `capture_credentials` compares against — and `enabled` reads — a
> `secrets/webapp.env` under `/opt/mv3dt` while every other component uses the
> chosen root, so a captured credential appears to vanish between capture and
> use. Both gain the same **optional** `install_dir`, defaulting to
> `DEFAULT_INSTALL_DIR` so the zero-argument call sites documented above keep
> working, and both pass it through to `load_credentials`. Framework callers
> MUST pass the path resolved by `config.load()` (§11.2).

> **`gate_value` is not replaced (REQUIRED).** `enabled` keeps
> `gate_value: str` as its first parameter and gains `install_dir` alongside
> it. `webapp.py` must not import `config.py`, so the gate cannot be resolved
> inside this module: the caller (`app.py`, which has both) resolves
> `MV3DT_WEBAPP_INTEGRATION` and passes the string in. A signature that
> dropped `gate_value` in favor of `install_dir` would force exactly the
> import the module is designed to avoid. `app.WebappHandle.enabled()`'s
> existing install-dir-bound workaround stays as it is — it is correct and
> documented (§12.3) — the underlying function simply becomes bindable too.

### 14.4 Redaction (REQUIRED)

The transcript log (§8.2) captures every prompt and every shelled-out command's
output, so redaction is a contract, not a nicety. Two distinct rules:

- **The API key** — any log line that would contain it prints
  `API_KEY=<redacted>`. The capture prompt echoes nothing.
- **Signed URLs** — the upload flow
  ([`STEP-7` §B](STEP-7-WEBAPP-INTEGRATION.md#b-signed-url-upload)) receives
  pre-signed URLs whose **query string is itself the credential**. Before any
  such URL reaches a log, transcript, or error message, everything from the
  first `?` onward is replaced:

  ```python
  def redact_url(url: str) -> str:
      if not url:
          return ""
      q = url.find("?")
      return url if q < 0 else url[:q] + "?<redacted>"
  ```

  This applies to failure paths too — an exception message containing a raw
  signed URL is the most likely way one leaks, so the redaction happens in the
  logging helper, not at each call site.

---

## 15. Camera discovery

Owner: framework (`cameras.py`). Consumed by
[`STEP-4`](STEP-4-CALIB-OUTPUT-WIRING.md) when rendering source URIs and by
[`STEP-5` §3.3](STEP-5-PER-PROJECT-EXES.md#33-what-the-exe-does-at-runtime-pipeline-subcommand)
for its pre-flight reachability check. `--scan-cameras` (§3.3) runs it
standalone: discover, probe, bind, write both artifacts, print the table, exit.

**LOCKED — the MAC is the camera's identity.** Every merge, every lookup, and
every persisted binding keys on the MAC address, never on the IP.
[`laptop/config/cameras.yml`](../../laptop/config/cameras.yml) says so in its
own header ("MACs are the STABLE identifier"), and everything in this section
follows from it.

### 15.1 Why discovery, not a pinned list

The fleet is eight ANNKE cameras on Hikvision-OEM firmware, OUI **`d0:3b:f4`**,
on a LAN with no DHCP server. They self-assign `169.254.0.0/16` link-local
addresses that are neither reserved nor derivable from the MAC, and a camera
can land on a different address after a power cycle. `cameras.yml`'s own
caveat records a prior snapshot of four of these cameras whose IP set was
**completely disjoint** from the eight pinned addresses beside it. A static
list is therefore a starting hint, not a source of truth — which is why §13
keeps per-camera web-UI setup manual but puts IP discovery in scope.

> **No MAC-to-position mapping survived the Jetson deletion.** The removed
> `config.json` keyed cameras by MAC; `cameras.yml` keys them by IP and
> position; the two files shared no common field. See
> [`DELETION-REVIEW` §4.1](DELETION-REVIEW.md#41-camera-facts-harvested-before-deletion).
> This is the entire reason the binding in §15.4 must be established once by a
> human rather than derived — there is nothing to derive it from.

### 15.2 Discovery mechanism

1. **Candidate interfaces.** `candidate_interfaces()` enumerates
   `/sys/class/net` and drops `lo`, `docker*`, `veth*`, `br-*`, `virbr*`,
   anything whose `carrier` is not `1` (no link, or administratively down),
   and anything whose `flags` carry `IFF_NOARP` or `IFF_POINTOPOINT` (tun
   devices such as `tailscale0` cannot answer ARP). An interface **without an
   IPv4 address is kept** so step 2 can report it. `CAMERA_SCAN_IFACE`
   (§11.2) replaces this list with exactly one interface.
2. **Classify each candidate by address.** `discover()` reads every IPv4
   address on the interface (`ip -4 -o addr show dev <if>`):

   | Interface state | Action | Recorded in |
   |---|---|---|
   | Has an address inside `<cidr>` | Swept, with that address as the ARP sender | `ScanResult.interfaces` |
   | Has addresses, none inside `<cidr>` | Skipped; swept only when named by `CAMERA_SCAN_IFACE` | `ScanResult.skipped` |
   | Link up, no IPv4 address | Never swept; warns with the `nmcli ... ipv4.method link-local` fix | `ScanResult.unaddressed` |

   When nothing qualifies, no scan runs, `ScanResult.tool` is `none`, and a
   warning names `--camera-scan-iface` / `--camera-scan-cidr`.
   An unparseable `<cidr>` is handled the same way. `--camera-scan-cidr`
   rejects one at argument parsing, before it can be persisted to
   `installer.conf`.
3. **Primary: `arp-scan`.** `arp-scan --interface <if> --arpspa <addr>
   <cidr>` per swept interface (`--arpspa` is omitted for an explicitly named
   interface with no in-range address). This needs raw sockets — the
   installer is already root (§9.1). `arp-scan` joins the apt list in
   [`STEP-1` §3](STEP-1-PREREQUISITES.md#3-ds-91-41-apt-prerequisite-package-list).
   A full `/16` sweep takes roughly two minutes, so the range is
   configurable via `CAMERA_SCAN_CIDR` and step-driven calls wrap it in the
   blocking wait screen (`waitui.py`, §3.1).

   **RESOLVED — never `--localnet`.** An earlier revision swept
   `--localnet` on every candidate. On a workstation whose PoE port had no
   address, that port was dropped and `--localnet` ran on the campus Wi-Fi's
   `/8` instead: 21 minutes, about 508,000 junk replies, zero cameras, and an
   ARP sweep of a network the installer has no business probing. Sweeping
   only `<cidr>`, only on interfaces that sit in it, bounds every scan by
   configuration rather than by whatever network an adapter joined.
4. **Fallback: the kernel ARP cache.** When `arp-scan` is absent, prime the
   cache by pinging `prime_ips` — the last-known IPs from the previous
   inventory, or the bundled seed (§4.1) on first run — then read `ip -4 neigh
   show`. This **cannot** find a camera that moved to an address nobody has
   seen; `ScanResult.tool` records which mechanism ran so the limitation is
   visible rather than implied.
5. **Filter by OUI**, case- and separator-insensitively (`d0:3b:f4`,
   `D0-3B-F4`, and `d03b.f401.5279` are the same prefix). Non-matching hosts
   are retained in `ScanResult.unmatched`, de-duplicated by MAC (a proxy-ARP
   device answers for every address with one MAC), so an operator can tell
   "the scan found nothing" apart from "the scan did not run".

### 15.3 RTSP probe

Discovery proves a host answers ARP; it does not prove the camera serves
video. Each discovered camera is probed with
`ffprobe -rtsp_transport tcp` against
`rtsp://<user>:<pass>@<ip>:554<rtsp_path>`, and the result is recorded as
`stream_ok` in the inventory. This is the check ported out of
`laptop/scripts/20_verify_cameras.sh`, and it is the distinction
[`STEP-5` §3.3](STEP-5-PER-PROJECT-EXES.md#33-what-the-exe-does-at-runtime-pipeline-subcommand)
flags between a ping sweep and a usable stream. Credentials are captured
during onboarding and stored in `<install_dir>/secrets/camera.env` with mode
`0600`; the password is loaded into memory for Steps 4 and 5 and redacted in
every log line (§4.2). `ffmpeg` joins the same apt list
([`STEP-1` §3](STEP-1-PREREQUISITES.md#3-ds-91-41-apt-prerequisite-package-list)).

**REQUIRED — every RTSP child is bounded.** `probe_rtsp()` and
`grab_still()` both pass `-timeout <timeout_us>` (default 5 s, an input
option placed before `-i`) and a process timeout of `timeout_us` plus 10 s; a
camera that accepts the connection and then stalls yields `stream_ok: false`
or no still rather than a hung scan. Each probe and still grab is logged as
it starts so a long scan reads as progress.

A failing probe on an activated camera is usually the manual pre-flight in
§13 not having been done — the camera ships un-activated and refuses RTSP
until an admin password is set.

### 15.4 Guided position binding (one time)

For each newly discovered MAC with no persisted position:

1. `grab_still()` captures a single frame (`ffmpeg -frames:v 1`) to
   `<install_dir>/cameras/still-<mac>.jpg` — only for a camera found in this
   scan whose §15.3 probe did not fail, since any other grab can only time
   out.
2. `bind_positions()` prints that path and asks the operator which position
   the camera occupies — one of the known positions from the seed inventory
   (`top-left`, `top-right`, `middle-top-left`, and so on) or a newly typed
   one. One camera at a time; eight prompts, once, ever.
3. The resulting `mac → id → position` binding persists to
   `<install_dir>/cameras.yml`.

**Later scans match on MAC and refresh only the IP.** An existing entry keeps
its `id`, `position`, `enabled`, and `rtsp_path`; a camera that has gone
missing is **retained and flagged, never deleted**, because a powered-off
camera is not a decommissioned one.

Under `--non-interactive`, binding is skipped rather than blocked on: ids are
assigned `c<N>` in MAC-sorted order and `position` is left empty with an
inline comment telling the operator to re-run `--scan-cameras` interactively
to label them. An unattended run must never wait for a human.

### 15.5 Outputs

| Artifact | Path |
|---|---|
| Runtime inventory (MAC-keyed, hand-editable positions) | `<install_dir>/cameras.yml` |
| Raw scan record — MACs, swept/`unaddressed`/`skipped` interfaces, unmatched hosts, timestamps | `<install_dir>/cameras.scan.json`, written with the §6.3 `write_json_atomic` helper |
| Still frames captured for binding | `<install_dir>/cameras/still-<mac>.jpg` |
| Pointer consumed by steps and bundled bash | `CAMERAS_FILE` in `installer.conf` (§11.2) |
| Scan tuning | `CAMERA_SCAN_CIDR`, `CAMERA_SCAN_IFACE` in `installer.conf` (§11.2) |

When the inventory is absent, Step 4 invokes this discovery flow
automatically before polling the AMC result. The standalone
`--scan-cameras` mode remains available for later fleet refreshes.

The generated `<install_dir>/cameras.yml` carries a
`# Generated by mv3dt-installer --scan-cameras at <utc>; do not hand-edit IPs`
banner above the header block inherited from the seed inventory — the fleet
MAC list, the 3072x1728 sensor note, the volatility caveat, and the manual
activation / OSD pre-flight, all of which
[`CLAUDE.md`](../../CLAUDE.md) treats as load-bearing documentation rather
than commentary.

[`laptop/config/cameras.yml`](../../laptop/config/cameras.yml) itself **stays
exactly where it is**, unmoved and undeleted; it is additionally bundled into
the binary as `assets/cameras/cameras.yml` (§4.1) and used for exactly two
things: that header block, and its eight pinned IPs as `prime_ips` for the
§15.2 fallback.

### 15.6 API surface (`cameras.py`)

```python
CAMERA_OUI = "d0:3b:f4"
DEFAULT_SCAN_CIDR = "169.254.0.0/16"
KNOWN_POSITIONS = ("top-left", "top-right", "middle-top-left", ...)   # seed inventory

@dataclass(frozen=True)
class Camera:
    id: str; mac: str; ip: str; position: str
    rtsp_path: str = "/Streaming/Channels/101"
    enabled: bool = True
    stream_ok: bool | None = None          # §15.3 probe result

@dataclass
class ScanResult:
    cameras: list[Camera]; unmatched: list[str]; tool: str   # "arp-scan" | "ip-neigh" | "none"
    interfaces: list[str]                  # swept (§15.2 step 2)
    unaddressed: list[str] = []            # link up, no IPv4
    skipped: list[str] = []                # address outside the scan CIDR

def normalize_mac(raw) -> str
def matches_oui(mac, oui=CAMERA_OUI) -> bool
def parse_inventory(text) -> list[Camera]
def render_inventory(cameras, *, header) -> str
def candidate_interfaces(*, runner=subprocess.run) -> list[str]
def discover(*, oui, cidr, interfaces=None, prime_ips=(), runner=...) -> ScanResult
def probe_rtsp(camera, *, user, password, timeout_us=5_000_000, runner=...) -> bool
def grab_still(camera, *, user, password, dest, timeout_us=5_000_000, runner=...) -> Path | None
def bind_positions(cameras, *, prompt, non_interactive, runner=...) -> list[Camera]
def merge(previous, discovered) -> list[Camera]
def refresh(install_dir, *, seed_header=None, prompt=input, runner=...) -> ScanResult
```

Every subprocess goes through an injected `runner` (production callers pass
`ctx.run_root`, §12.3) and the inventory parser is hand-rolled rather than
PyYAML-backed, for the packaging reason in §4.1.

---

## References

DeepStream 9.1 official documentation. Reference DS 9.1 only — NVIDIA's
current release.

- DS 9.1 Installation (dGPU Ubuntu prerequisites + three install methods:
  Debian package, tar package, Docker; `update_rtpmanager.sh`;
  `sudo apt-get install ./deepstream-9.1_9.1.0-1_amd64.deb`; deb/tar
  published as GitHub Release assets):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html>
- DS 9.1 Quickstart (Triton Docker image
  `nvcr.io/nvidia/deepstream:9.1-triton-multiarch`):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html>
- DS 9.1 `deepstream-app` reference:
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html>
- DS 9.1 Release Notes:
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Release_notes.html>
- DS 9.1 MV3DT:
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_MV3DT.html>
- DS 9.1 AutoMagicCalib:
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_AutoMagicCalib.html>
- DS 9.1 Docker containers:
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html>
- NVIDIA/DeepStream GitHub Releases (deb/tar distribution, v9.1.0):
  <https://github.com/NVIDIA/DeepStream/releases/tag/v9.1.0>

Verified DS 9.1 prerequisite pins (DS 9.1 Installation page, dGPU Ubuntu →
Prerequisites): Ubuntu 24.04, GStreamer 1.24.2, NVIDIA driver 595.58.03, CUDA
13.2, TensorRT 10.16.0.72-1+cuda13.2, cuDNN 9.20.0.48.

Repo files referenced:

- [`laptop/scripts/00_bootstrap.sh`](../../laptop/scripts/00_bootstrap.sh) —
  phased resumable installer this binary ports.
- [`laptop/scripts/lib/common.sh`](../../laptop/scripts/lib/common.sh) —
  logging, `require_root`/`require_tool`, `require_version_eq`, and the
  `phase_done`/`mark_phase_done` state pattern.
- [`laptop/docs/DEEPSTREAM-SETUP.md`](../../laptop/docs/DEEPSTREAM-SETUP.md) /
  [`laptop/docs/SCRIPTED-WORKFLOW.md`](../../laptop/docs/SCRIPTED-WORKFLOW.md)
  — source workflow the installer unifies.
- [`laptop/.gitignore`](../../laptop/.gitignore) — precedent for gitignoring
  local secrets/env (`config/laptop.env`).
- [`.github/workflows/release.yml`](../../.github/workflows/release.yml) —
  the CI build that produces and publishes the release binary (§4.1); the
  only place the binary is ever built.
- [`laptop/config/cameras.yml`](../../laptop/config/cameras.yml) — retained
  fleet inventory: the `d0:3b:f4` OUI, the MAC-as-identity rule, and the
  link-local volatility caveat that §15 exists to survive.
- [`STEP-6-REMOTE-SUPERVISION.md`](STEP-6-REMOTE-SUPERVISION.md) — the opt-in
  control plane (§3.4 gate, §12.4 map).
- [`STEP-7-WEBAPP-INTEGRATION.md`](STEP-7-WEBAPP-INTEGRATION.md) — the opt-in
  data plane; sole consumer of the §14 credential contract.
- [`08-PROGRESS-AND-OBSERVABILITY.md`](08-PROGRESS-AND-OBSERVABILITY.md) —
  the progress and streaming foundation section 8.5 configures; its sections
  4.1 and 7 are the two the AUTO-mode resolution reconciles.
- [`DELETION-REVIEW.md`](DELETION-REVIEW.md) — records which Jetson-tree files
  the §14 contract was harvested from, and why they are being removed.

> **Attribution — sources no longer in this fork.** The §14 credential flow,
> the §6.3 atomic-write helper, and the §14.4 redaction rule were ported from
> the P2BP Jetson camera-node agent (`scripts/cloud_storage_media.py`,
> `scripts/heartbeat.py`, `scripts/config_io.py`, and `install.sh` at the repo
> root). Those files are removed from this fork per
> [`DELETION-REVIEW` §3](DELETION-REVIEW.md#3-deletions-gated-on-the-harvest-the-jetson-tree)
> and remain available in the parent repository. They are named here as
> provenance only — deliberately not linked, so the links cannot rot.
