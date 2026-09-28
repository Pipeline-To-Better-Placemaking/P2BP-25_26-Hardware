# mv3dt-installer — operator download-and-run guide

Status: operator guide for the `mv3dt-installer` binary. This is the entry
point for anyone setting up a workstation; it does **not** restate the
framework contracts — those live in
[`00`](plan/00-FRAMEWORK-AND-BOOTSTRAP.md) and the per-step specs under
[`plan/`](plan/), and are linked from here section by section.

`mv3dt-installer` is a **single self-contained binary** that takes a bare
Ubuntu 24.04 workstation to a working DeepStream 9.1 / AutoMagicCalib MV3DT
setup. **(planned)** It is meant to be built once in CI and published as a
GitHub Release asset, so the target machine downloads one file and runs it —
never building anything and never needing a checkout of this repository. See
[§1](#1-what-this-binary-is) for how that compares to today's actual build
model.

This guide covers the operator path: where to get the binary, how to verify
and run it, every flag it accepts, what the first run asks for, and where it
keeps state and logs. Developers building or testing the installer from a
clone want [§8](#8-developer-workflow).

> **REQUIRED reading — planned versus shipped.** The installer is being built
> in stages, so this guide describes a system that is partly in flight.
> Anything marked **(planned)** is specified in [`plan/`](plan/) and being
> built, but is **not** in a binary you can run today; everything unmarked is
> current behavior. The largest planned piece is the step modules themselves:
> today's binary parses flags, resolves its install location, manages state,
> and dispatches an empty step registry, so it installs no DeepStream yet.
> [§4](#4-command-line-flags)'s table carries the same distinction in a
> `Status` column. Where a citation into
> [`00`](plan/00-FRAMEWORK-AND-BOOTSTRAP.md) backs a **(planned)**
> statement, that section of the spec is itself being rewritten by the same
> work and may still read the other way today.

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

> **(planned) LOCKED — you do not clone this repo on the workstation.** The
> Release binary is meant to be the unit of delivery, built once in CI, with
> no `git clone`, no `pip install`, and no local build on the target
> machine; the binary carries its own assets and unpacks them at runtime.
> This is the target design, not what
> [`00` §4](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#4-pyinstaller-packaging)
> describes today: on `main`, §4.1 still has the binary built **during**
> bootstrap, on the target machine, from a cloned copy of this repo, and
> §4.2 says that clone is deliberately left in place afterward for
> logs/debugging.
> [`PR #31`](https://github.com/KevinTTO5/P2BP-25_26-Hardware_Test/pull/31)
> is rewriting doc 00 to match the CI-build model this callout describes;
> until it merges, treat this callout as forward-looking, not settled fact.

---

## 2. Requirements

| Requirement | Value | Why |
|---|---|---|
| OS | Ubuntu 24.04 LTS | **(planned)** The published binary is meant to be built once in CI against this release's glibc; on `main` today, [`00` §4.1](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#41-what-builds-the-binary) still describes the binary being built on the target machine itself during bootstrap, so pin to Ubuntu 24.04 either way |
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

**(planned)** A startup preflight that refuses an unsupported Ubuntu release,
a non-`x86_64` machine, or exactly that unset-`SUDO_USER` case, rather than
proceeding. Until it lands, the table above is yours to honor.

---

## 3. Download, verify, run

Releases are published at:

```
https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/releases
```

> **(planned) — the release job.** The CI workflow that builds the binary on
> a pinned Ubuntu 24.04 runner and publishes it is in flight. Until it lands
> there is nothing on the Releases page to download, and the only way to
> obtain a binary is the local build in [§8.3](#83-local-build). The
> procedure below is the operator path once that job exists, and is what
> [`00` §4.1](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#41-what-builds-the-binary)
> is being rewritten to specify.

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
`download/<tag>`, for example `download/v0.2.0`. **(planned)** The tag and
the binary's own `--version` output will always agree, because the release
job asserts it and fails the build on drift.

---

## 4. Command-line flags

Framework-level flags
([`00` §3.3](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#33-cli-flags-framework-level)).
`Status` separates what the binary accepts today from what is specified and
still landing — check `./mv3dt-installer --help` on the build you actually
have.

| Flag | Argument | What it does | Status |
|---|---|---|---|
| `--install-dir` | `PATH` | Override the default install location (`/opt/mv3dt`); highest-precedence tier of [`00` §11.2](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#112-persistence--sharing-with-later-steps) | shipped |
| `--status` | — | Print the state table and exit; read-only, and the one flag that does not need `sudo` | shipped |
| `--resume` | — | Default behavior, stated explicitly for clarity; effectively a no-op | shipped |
| `--reset-state` | — | Wipe `state.json` and start fresh; the recorded install directory is preserved across the reset | shipped |
| `--reset-step` | `N` | Clear one step's completion by its order number so it re-runs, then exit | shipped |
| `--non-interactive` | — | Never prompt; use defaults and already-persisted config, and fail if a required value is missing | shipped |
| `--no-pause` | — | Skip the "press Enter" confirmations | shipped |
| `--log-dir` | `PATH` | Override the transcript directory ([§6.2](#62-logs)) | shipped |
| `--version` | — | Print the version and exit | shipped; the stamped form carrying tag, commit, and build timestamp is **planned**, arriving with the release job |
| `--remote-supervision` | `off`, `local`, `remote` | Set the Step 6 gate ([`00` §3.4](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#34-opt-in-step-gates)) without being prompted; overrides an already-persisted value and logs the change | shipped |
| `--webapp-integration` | `off`, `on` | Set the Step 7 gate the same way | shipped |
| `--scan-cameras` | — | Refresh the camera fleet after initial automatic Step 4 discovery, probe RTSP, bind new positions, write the inventory, print the table, and exit; needs `sudo` for raw sockets | shipped |
| `--camera-scan-cidr` | `CIDR` | Override the discovery sweep range (default `169.254.0.0/16`); persisted as `CAMERA_SCAN_CIDR` | shipped |
| `--camera-scan-iface` | `IFACE` | Restrict discovery to one interface; persisted as `CAMERA_SCAN_IFACE` | shipped |

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
transcript captures the installer's own output. **(planned)** Capturing the
stdout and stderr of the bash fragments it shells out to as well — today that
output is collected and then discarded rather than logged.

Secrets are redacted before anything is written: a log line that would carry
the NGC key or the web-app API key prints `NGC_API_KEY=<redacted>` or
`API_KEY=<redacted>` instead, and the modules that own those secrets do that
redaction themselves. **(planned)** Extending the same redaction over the
shelled-out script output described above, once it is captured. When
reporting a problem, attach `latest.log`:

```bash
sudo tail -n 200 /var/lib/mv3dt-installer/logs/latest.log
```

### 6.3 Secrets

Credentials live under the install directory, owned by the invoking user,
directory `0700` and files `0600`
([`00` §10.1](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#101-storage-contract),
[`00` §14.1](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#141-storage-contract)):

| Path | Contents | Status |
|---|---|---|
| `<install_dir>/secrets/ngc.env` | `NGC_API_KEY=...` — the key is required, so a blank answer re-prompts rather than writing anything | shipped |
| `<install_dir>/secrets/webapp.env` | `API_KEY=...` and `ENDPOINT=...`, only when the web-app gate is `on` | shipped |
| `<install_dir>/installer.conf` | Non-secret `KEY=VALUE` config: the two gates, the resolved install directory, and the shared values later steps read | shipped |

---

## 7. Resuming, re-running, and resetting

- **After a reboot.** When a step needs a reboot the installer says so and
  exits cleanly. Reboot, then run `sudo ./mv3dt-installer` again — it detects
  that the reboot happened and continues
  ([`00` §7](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#7-reboot-detection--continuation-contract)).
  The detection is shipped; the steps that request a reboot are **(planned)**
  with the step modules.
- **After a manual action.** A USER-ACTION block lists exactly what to do by
  hand; do it, then run the installer again. Nothing already complete is
  repeated. Same split as above: the block is shipped, its callers are
  **(planned)**.
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

This produces the same `--onefile` executable the release job will publish,
and until that job lands it is the only way to get a binary at all
([§3](#3-download-verify-run)). It is still not a substitute for a released
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
  While the step modules are still **(planned)**, it is also the only way to
  actually stand the pipeline up.
- **Camera activation, OSD, and stream-profile setup**, which stay manual
  ([`00` §13](plan/00-FRAMEWORK-AND-BOOTSTRAP.md#13-out-of-scope--defer-to-human)).

---

## References

Release artifacts and their verification come from this repository's own
release job. The framework behavior summarized above is specified in
[`00`](plan/00-FRAMEWORK-AND-BOOTSTRAP.md), which is authoritative wherever
this guide condenses it; on a **(planned)** item the spec states the intent
while this guide records what the binary does meanwhile.

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
- [`laptop/config/cameras.yml`](../laptop/config/cameras.yml) — the camera
  inventory and the record of the fleet's hardware facts and required manual
  pre-flight.
- [`laptop/docs/DEEPSTREAM-SETUP.md`](../laptop/docs/DEEPSTREAM-SETUP.md) —
  DS 9.1 package pins and the manual NVIDIA prerequisites.
- [`laptop/docs/SCRIPTED-WORKFLOW.md`](../laptop/docs/SCRIPTED-WORKFLOW.md) —
  the developer-side scripted harness the binary supersedes for operators.
