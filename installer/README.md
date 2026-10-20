# mv3dt-installer — operator download-and-run guide

Status: operator guide for the `mv3dt-installer` binary. This is the entry
point for anyone setting up a workstation; it does **not** restate the
framework contracts — those live in
[`00`](plan/00-FRAMEWORK-AND-BOOTSTRAP.md) and the per-step specs under
[`plan/`](plan/), and are linked from here section by section.

`mv3dt-installer` is a **single self-contained binary** that takes a bare
Ubuntu 24.04 workstation to a working DeepStream 9.1 / AutoMagicCalib MV3DT
setup. It is built once in CI and published as a GitHub Release asset, so the
target machine downloads one file and runs it — never building anything and
never needing a checkout of this repository.

This guide covers the operator path: where to get the binary, how to verify
and run it, every flag and subcommand it accepts, what the first run asks
for, and where it keeps state and logs. Developers building or testing the
installer from a clone want [§8](#8-developer-workflow).

> **REQUIRED reading — what ships today.** Every behavior in this guide is
> in the current release: the framework, all seven steps
> ([`STEP-1`](plan/STEP-1-PREREQUISITES.md) through
> [`STEP-7`](plan/STEP-7-WEBAPP-INTEGRATION.md)), and the release job. What
> is still open — and what has and has not been proven on real hardware —
> is tracked in
> [`DEVELOPMENT-STATUS` §5](plan/DEVELOPMENT-STATUS.md#5-what-remains), not
> here. A feature that has shipped is not the same as one that has been
> validated on the workstation.

---

## 1. What this binary is

`mv3dt-installer` is the unified installer described by
[`00` §2](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#2-executable-form-locked): one
PyInstaller `--onefile` executable containing the Python framework, the
bundled bash fragments, and the DeepStream/Mosquitto config templates it
needs. It supersedes the numbered `laptop/scripts/` harness for operator use.

It is a **resumable state machine**
([`00` §6](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#6-resumable-state-machine)):
every run picks up where the last one stopped, re-runs nothing that already
completed, and survives the reboots the driver/CUDA install requires. Running
it a second time is always safe.

> **LOCKED — you do not clone this repo on the workstation.** The Release
> binary is the unit of delivery, built once in CI
> ([`00` §4.1](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#41-what-builds-the-binary)),
> with no `git clone`, no `pip install`, and no local build on the target
> machine; the binary carries its own assets and unpacks them at runtime
> ([`00` §4.2](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#42-locating-and-staging-bundled-assets-at-runtime)).

---

## 2. Requirements

| Requirement | Value | Why |
|---|---|---|
| OS | Ubuntu 24.04 LTS | The release binary is built on a pinned `ubuntu-24.04` runner and links against that release's glibc ([`00` §4.1](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#41-what-builds-the-binary)) |
| Architecture | `x86_64` | The only architecture the release build targets |
| Privilege | `sudo` from a real login user, not a bare root shell | Package installs and writes to `/etc/profile.d` and `/var/lib` ([`00` §9.1](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#91-privilege-handling)) |
| GPU | NVIDIA, Ampere or newer | DeepStream 9.1 pins ([`laptop/docs/DEEPSTREAM-SETUP.md`](../laptop/docs/DEEPSTREAM-SETUP.md)) |
| Network | Outbound HTTPS | Package repositories, the DS SDK release assets, and the NGC model fetch |

Run the installer under `sudo` from your normal user account. It resolves the
invoking user from `SUDO_USER` and writes every secret, and every file under
that user's home, as that user
([`00` §9.2](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#92-resolving-the-invoking-user--home)).
From a bare root shell where `SUDO_USER` is unset, the resolver falls back to
the current user, and those files land in `/root` instead.

A startup preflight enforces this: it refuses an Ubuntu release other than
24.04, a non-`x86_64` machine, and an invoking user that resolves to `root`
(the unset-`SUDO_USER` case above), rather than proceeding
([`00` §5.2](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#52-what-the-binary-does-on-first-launch)).

---

## 3. Download, verify, run

Releases are published at:

```
https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/releases
```

Pushing a `v*` tag runs
[`release.yml`](../.github/workflows/release.yml), which builds the binary on
a pinned Ubuntu 24.04 runner and publishes it
([`00` §4.1](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#41-what-builds-the-binary)).
Each release carries exactly two assets — the binary `mv3dt-installer` and
its checksum `mv3dt-installer.sha256`. Download both into the same directory,
verify, then run:

```bash
cd ~/Downloads
curl -fLO https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/releases/latest/download/mv3dt-installer
curl -fLO https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/releases/latest/download/mv3dt-installer.sha256
sha256sum -c mv3dt-installer.sha256
chmod +x mv3dt-installer
sudo ./mv3dt-installer
```

`sha256sum -c` must print exactly:

```
mv3dt-installer: OK
```

Anything else — `FAILED`, or `No such file or directory` — means the download
is incomplete or the two files came from different releases. Delete both and
download them again; do not run a binary that failed verification.

To pin a specific release instead of `latest`, replace `latest/download` with
`download/<tag>`, for example `download/v0.4.10`. The tag and the binary's own
`--version` output always agree, because the release job asserts it and fails
the build on drift.

---

## 4. Command-line flags

Framework-level flags
([`00` §3.3](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#33-cli-flags-framework-level)).
`./mv3dt-installer --help` on the build you actually have is authoritative.

| Flag | Argument | What it does |
|---|---|---|
| `--install-dir` | `PATH` | Override the default install location (`/opt/mv3dt`); highest-precedence tier of [`00` §11.2](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#112-persistence--sharing-with-later-steps) |
| `--status` | — | Print the state table and exit; read-only, and the one flag that does not need `sudo` |
| `--resume` | — | Default behavior, stated explicitly for clarity; effectively a no-op |
| `--reset-state` | — | Wipe `state.json` and start fresh; the recorded install directory is preserved across the reset |
| `--reset-step` | `N` | Clear one step's completion by its order number so it re-runs, then exit |
| `--non-interactive` | — | Never prompt; use defaults and already-persisted config, and fail if a required value is missing |
| `--no-pause` | — | Skip the "press Enter" confirmations |
| `--verbose` | — | Show every line of command output instead of the rolling 8-line window ([`00` §8.5](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#85-verbosity-and-live-command-output-required)); re-run with this when reporting a problem |
| `--log-dir` | `PATH` | Override the transcript directory ([§6.2](#62-logs)) |
| `--version` | — | Print the version and exit; a release binary also prints the tag, commit, and build time CI stamped into it |
| `--remote-supervision` | `off`, `local`, `remote` | Set the Step 6 gate ([`00` §3.4](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#34-opt-in-step-gates)) without being prompted; overrides an already-persisted value and logs the change |
| `--webapp-integration` | `off`, `on` | Set the Step 7 gate the same way |
| `--scan-cameras` | — | Refresh the camera fleet after initial automatic Step 4 discovery, probe RTSP, bind new positions, write the inventory, print the table, and exit; needs `sudo` for raw sockets |
| `--camera-scan-cidr` | `CIDR` | Override the discovery sweep range (default `169.254.0.0/16`); persisted as `CAMERA_SCAN_CIDR` |
| `--camera-scan-iface` | `IFACE` | Restrict discovery to one interface; persisted as `CAMERA_SCAN_IFACE` |

### 4.1 Subcommands

`mv3dt-installer <subcommand> [...]` bypasses the install loop and runs one
step's tool directly, long after install. Several are what the installed
systemd units invoke; an operator rarely needs those by hand. Each accepts
`--help`. All need `sudo` unless the table says otherwise.

| Subcommand | What it does | Usually run by | Spec |
|---|---|---|---|
| `amc` | Start (or with `--down`, stop) the AutoMagicCalib stack and open its UI | Operator, via `<install_dir>/bin/amc` | [`STEP-3` §8](plan/STEP-3-AMC-LAUNCHER.md#8-launcher-and-verification) |
| `ingest` | One status check against AMC; downloads and installs a `COMPLETED` result | `mv3dt-ingest-<slug>.timer` | [`STEP-4` §7](plan/STEP-4-CALIB-OUTPUT-WIRING.md#7-later-recalibration) |
| `pipeline` | Start, stop, or dry-run one project's DeepStream pipeline; `--service-exec` is the non-root mode the pipeline unit runs | Operator, via `<install_dir>/bin/pipeline-<slug>`; `mv3dt-pipeline@<slug>.service` | [`STEP-5` §3.3](plan/STEP-5-PER-PROJECT-EXES.md#33-what-the-exe-does-at-runtime-pipeline-subcommand) |
| `record` | Run the bundled `60_record_tracking.sh` for a project, writing under `<install_dir>/tracking_exports/` | Operator | [`STEP-5` §1.2](plan/STEP-5-PER-PROJECT-EXES.md#12-outputs-what-step-5-writes) |
| `projects` | `--list`, `--reconcile`, or `--remove <NAME>` registered projects | Operator | [`STEP-5` §5.4](plan/STEP-5-PER-PROJECT-EXES.md#54-project-removal-amc-gui-driven--reconciliation-check) |
| `agent` | MQTT control agent: run/stop/restart pipelines on command. **Runs as the invoking user, not root** | `mv3dt-agent.service` | [`STEP-6` §B](plan/STEP-6-REMOTE-SUPERVISION.md#b-the-control-agent-remote) |
| `reporter` | Web-app status/heartbeat loop (`--once` for a single cycle) | `mv3dt-reporter.service` | [`STEP-7` §D](plan/STEP-7-WEBAPP-INTEGRATION.md#d-registration-and-status) |
| `uploader` | Web-app artifact upload loop (`--once` for a single pass) | `mv3dt-uploader.service` | [`STEP-7` §E](plan/STEP-7-WEBAPP-INTEGRATION.md#e-artifact-upload-daemon) |

Not passing one of the two gate flags differs from passing `off` explicitly:
an unpassed flag leaves the value to the next tier of precedence —
environment variable, then an interactive prompt, then `off` — while
`--webapp-integration off` sets it explicitly. A gate key already recorded
in `installer.conf` is unaffected by the environment-variable tier and is
only ever changed by a flag; a gate key absent from `installer.conf` is
seeded from the like-named environment variable if one is set, else `off`.

`--status` is deliberately allowed without `sudo`. It runs before the root
check, performs no writes, and reads a world-readable `state.json`, so an
operator or an unattended health check can inspect progress at any time:

```bash
./mv3dt-installer --status
```

---

## 5. First run: what it asks for

The first run captures a handful of answers, writes them down, and never asks
again. Later runs are silent on all of them — a prompt reappearing on a
second launch is a bug, not the design.

1. **Install directory.** Prompted once with `/opt/mv3dt` prefilled
   ([`00` §11.1](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#111-default-and-selection)).
   Blank accepts the default. Everything the installer owns lives under this
   directory, and the choice is recorded so later runs need no flag.
2. **Remote supervision gate** (`off`, `local`, `remote`). Prompted once
   with `off` prefilled if no flag or environment variable already answered
   it. `off` skips Step 6 entirely — systemd supervision and the MQTT
   control plane ([`STEP-6`](plan/STEP-6-REMOTE-SUPERVISION.md)). Take the
   default unless you have been told otherwise.
3. **Web-app integration gate** (`off`, `on`). Same terms as prompt 2. `off`
   skips Step 7, the HTTP data plane
   ([`STEP-7`](plan/STEP-7-WEBAPP-INTEGRATION.md)).
4. **NGC API key.** A secret prompt — nothing is echoed as you type, and the
   key is never written to the transcript
   ([`00` §10.1](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#101-storage-contract)).
   **The key is required**: a blank answer re-prompts rather than being
   accepted, and under `--non-interactive` with no key available (no
   `NGC_API_KEY` in the environment) the run fails outright rather than
   proceeding without one
   ([`00` §10.2](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#102-capture--handoff-api-ngcpy)).
   `onboarding.py` runs this on every launch; once `secrets/ngc.env` exists,
   it is a silent no-op.
5. **Camera credentials.** The shared RTSP username and a no-echo password.
   Both are required and stored once in `secrets/camera.env`; the operator is
   never asked to put the password in `installer.conf`. On reaching Step 4,
   a missing camera inventory triggers discovery, RTSP validation, and the
   one-time position-binding prompts automatically. `--scan-cameras` remains
   the explicit refresh command for later fleet changes.
6. **Web-app credential.** Endpoint plus API key, and **only** if you set
   the web-app gate to `on` in prompt 3. A blank endpoint writes nothing and
   warns; Step 7 then surfaces its own USER-ACTION block on first run
   ([`00` §14.3](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#143-capture--handoff-api-webapppy)).
   Same capture-once rule: once `secrets/webapp.env` exists,
   `onboarding.py` never re-prompts.

Under `--non-interactive` nothing is prompted. A gate already recorded in
`installer.conf` keeps the value it has; a gate key absent from
`installer.conf` is seeded from the like-named environment variable
(`MV3DT_REMOTE_SUPERVISION` / `MV3DT_WEBAPP_INTEGRATION`) if one is set in
the installer's own environment, else `off` — the same precedence
[§4](#4-command-line-flags) describes
([`00` §3.4](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#34-opt-in-step-gates)). A
required value that is missing fails the run rather than blocking on a human.

> **Still manual, by design:** camera activation, disabling the on-screen
> display, and setting each camera's stream profile happen once in the vendor
> tool before the installer runs. See
> [`00` §13](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#13-out-of-scope--defer-to-human)
> and the header of
> [`laptop/config/cameras.yml`](../laptop/config/cameras.yml), which is the
> record of the fleet's hardware facts.

---

## 6. State, logs, and secrets

### 6.1 State

| Path | Contents |
|---|---|
| `/var/lib/mv3dt-installer/state.json` | Per-step completion, the recorded install directory, and any pending-reboot marker |

The path is fixed and independent of `--install-dir`
([`00` §6.1](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#61-state-file)). The file is
world-readable, which is what lets `--status` work without `sudo`. A missing
or corrupt file is read as "nothing has run yet" rather than as an error.

> **Do not delete `state.json` by hand — it is not equivalent to
> `--reset-state`.** `--reset-state` clears step completion but writes the
> recorded install directory back into the fresh state file. Deleting the
> file loses that record, so the next run falls through to the lower
> precedence tiers
> ([`00` §11.2](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#112-persistence--sharing-with-later-steps))
> and can silently resolve to `/opt/mv3dt` instead of the directory a live
> install actually occupies. Use `--reset-state`; if the file is already
> gone, pass `--install-dir` explicitly on the next run.

### 6.2 Logs

| Path | Contents |
|---|---|
| `/var/lib/mv3dt-installer/logs/` | One timestamped transcript per run |
| `/var/lib/mv3dt-installer/logs/latest.log` | Symlink to the most recent transcript |

`--log-dir PATH` moves both
([`00` §8.2](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#82-transcript-log-file)). The
transcript captures the installer's own output and the stdout and stderr of
every command it runs, including the bundled bash fragments.

Secrets are redacted before anything is written, in both the installer's own
lines and the captured command output: a line that would carry the NGC key or
the web-app API key prints `NGC_API_KEY=<redacted>` or `API_KEY=<redacted>`
instead. When reporting a problem, attach `latest.log`:

```bash
sudo tail -n 200 /var/lib/mv3dt-installer/logs/latest.log
```

### 6.3 Secrets

Credentials live under the install directory, owned by the invoking user,
directory `0700` and files `0600`
([`00` §10.1](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#101-storage-contract),
[`00` §14.1](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#141-storage-contract)):

| Path | Contents |
|---|---|
| `<install_dir>/secrets/ngc.env` | `NGC_API_KEY=...` — the key is required, so a blank answer re-prompts rather than writing anything |
| `<install_dir>/secrets/camera.env` | `CAM_USER=...` and `CAM_PASSWORD=...`, the shared RTSP credential ([§5](#5-first-run-what-it-asks-for), prompt 5) |
| `<install_dir>/secrets/webapp.env` | `API_KEY=...` and `ENDPOINT=...`, only when the web-app gate is `on` |
| `<install_dir>/agent/agent.env` | The Step 6 agent's broker credentials, only when the remote-supervision gate is `remote` ([`STEP-6` §D](plan/STEP-6-REMOTE-SUPERVISION.md#d-security-remote-control-must-be-authenticated)) |
| `<install_dir>/installer.conf` | Non-secret `KEY=VALUE` config: the two gates, the resolved install directory, and the shared values later steps read |

---

## 7. Resuming, re-running, and resetting

- **After a reboot.** When a step needs a reboot the installer says so and
  exits cleanly. Reboot, then run `sudo ./mv3dt-installer` again — it detects
  that the reboot happened and continues
  ([`00` §7](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#7-reboot-detection--continuation-contract)).
  Step 1's NVIDIA driver install is the step that asks for one.
- **After a manual action.** A USER-ACTION block lists exactly what to do by
  hand; do it, then run the installer again. Nothing already complete is
  repeated.
- **Force one step to re-run.** `sudo ./mv3dt-installer --reset-step N` with
  that step's order number, then run normally.
- **Start over.** `sudo ./mv3dt-installer --reset-state` clears all step
  completion while keeping the recorded install directory ([§6.1](#61-state)).
  Captured credentials and gates are not cleared — remove
  `<install_dir>/secrets/` and `<install_dir>/installer.conf` by hand if that
  is genuinely what you want.
- **Upgrade the installer.** Download the newer release, verify it as in
  [§3](#3-download-verify-run), and run it; state carries forward.

---

## 8. Developer workflow

Everything in this section assumes a clone of this repository on a
**development** machine. None of it is part of the operator path
([§1](#1-what-this-binary-is)).

### 8.1 Run from source

```bash
cd installer
python3 -m mv3dt_installer --status
```

`--status` is the safe probe: it exits before the root check and writes
nothing. There is deliberately no `mv3dt-installer` console script, so a pip
install can never shadow the real binary on an operator's `PATH`.

### 8.2 Tests

```bash
cd installer
python3 -m pytest tests/ -v
```

The suite injects every external dependency — no test touches `/var/lib`,
`/opt/mv3dt`, apt, systemd, or a socket, and root is always faked.

### 8.3 Local build

From the repository root:

```bash
python3 -m pip install --user pyinstaller
pyinstaller installer/installer.spec --distpath installer/dist --workpath /tmp/mv3dt-build
installer/dist/mv3dt-installer --version
installer/dist/mv3dt-installer --status
```

This produces the same `--onefile` executable the release job publishes
([§3](#3-download-verify-run)), but it is not a substitute for a released
one: a locally built binary links against the build host's glibc, so only a
build on Ubuntu 24.04 is guaranteed to start on the target
([`00` §4.1](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#41-what-builds-the-binary)).
Use a local build to confirm a newly added asset is actually bundled, and the
release job for anything an operator will run.

---

## 9. Out of scope for this guide

- **What each step installs.** The per-step specs under [`plan/`](plan/) own
  that: [`STEP-1`](plan/STEP-1-PREREQUISITES.md) through
  [`STEP-7`](plan/STEP-7-WEBAPP-INTEGRATION.md).
- **The DeepStream 9.1 version pins** and the manual NVIDIA prerequisites —
  [`laptop/docs/DEEPSTREAM-SETUP.md`](../laptop/docs/DEEPSTREAM-SETUP.md) is
  the source of truth.
- **The scripted `laptop/` harness**, which remains a developer tool and is
  not the operator path; see
  [`laptop/docs/SCRIPTED-WORKFLOW.md`](../laptop/docs/SCRIPTED-WORKFLOW.md).
- **Camera activation, OSD, and stream-profile setup**, which stay manual
  ([`00` §13](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#13-out-of-scope--defer-to-human)).

---

## References

Release artifacts and their verification come from this repository's own
release job. The framework behavior summarized above is specified in
[`00`](plan/00-FRAMEWORK-AND-BOOTSTRAP.md), which is authoritative wherever
this guide condenses it.

- <https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/releases> — the
  Releases page; **the only supported source** of the `mv3dt-installer`
  binary and its `.sha256` checksum.
- <https://pyinstaller.org/en/stable/usage.html#options> — backs the
  `--onefile` packaging model and the runtime asset unpacking the bundled
  scripts and config templates depend on.

Repo files referenced:

- [`installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md`](plan/00-FRAMEWORK-AND-BOOTSTRAP.md)
  — the framework contracts this guide summarizes: distribution, CLI flags,
  gates, state, logging, privilege, and credentials.
- [`installer/plan/STEP-1-PREREQUISITES.md`](plan/STEP-1-PREREQUISITES.md)
  through
  [`installer/plan/STEP-7-WEBAPP-INTEGRATION.md`](plan/STEP-7-WEBAPP-INTEGRATION.md)
  — what the gated and ungated steps actually do.
- [`installer/installer.spec`](installer.spec) — the PyInstaller recipe used
  by both the release job and the local build in §8.3.
- [`.github/workflows/release.yml`](../.github/workflows/release.yml) — the
  release job: tag-triggered build, version gate, and asset publish.
- [`installer/plan/DEVELOPMENT-STATUS.md`](plan/DEVELOPMENT-STATUS.md) —
  what remains, and what has actually been proven on the workstation.
- [`laptop/config/cameras.yml`](../laptop/config/cameras.yml) — the camera
  inventory and the record of the fleet's hardware facts and required manual
  pre-flight.
- [`laptop/docs/DEEPSTREAM-SETUP.md`](../laptop/docs/DEEPSTREAM-SETUP.md) —
  DS 9.1 package pins and the manual NVIDIA prerequisites.
- [`laptop/docs/SCRIPTED-WORKFLOW.md`](../laptop/docs/SCRIPTED-WORKFLOW.md) —
  the developer-side scripted harness the binary supersedes for operators.
