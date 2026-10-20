# STEP 1 — Prerequisites (owner: DevA)

Status: step spec. Depends on the shared contracts in
[`00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md) — this doc
does **not** restate the framework (state machine, reboot detection, logging,
`Context`, `StepResult`). It links back to it and specifies only what Step 1
owns.

Step 1 installs and verifies **every DeepStream 9.1 dGPU prerequisite** on a
brand-new Ubuntu 24.04 workstation **before** the DeepStream SDK is pulled
(that is Step 2's job). Every prerequisite fact below is drawn directly from
the DS 9.1 Installation page (see [References](#references)).

Target machine (LOCKED): Ubuntu 24.04, GPU = NVIDIA RTX PRO 4500 Blackwell.

---

## 1. Module identity

Per the step-module interface in
[`00` §12](00-FRAMEWORK-AND-BOOTSTRAP.md#12-step-module-interface-the-contract-for-steps-15):

- Module: `installer/mv3dt_installer/steps/step1_prerequisites.py`
- `id = "step1_prerequisites"` (matches the `state.json` key in
  [`00` §6.2](00-FRAMEWORK-AND-BOOTSTRAP.md#62-schema)).
- `title = "Prerequisites (driver / CUDA / cuDNN / TensorRT / GStreamer)"`
- `order = 1` — the first step the dispatch loop runs.
- Consumes: `Context` (`install_dir`, `run_root`, `run_as_user`, `log`,
  `report_installed`, `report_already_installed`, `verify_pinned`,
  `asset_path`), Step 1's private continuation evidence ([§6](#6-reboot-gating-step-1-owned-continuation)),
  and the exact reporting strings
  ([`00` §8.3–8.4](00-FRAMEWORK-AND-BOOTSTRAP.md#83-reporting-format-for-dependencies-required-exact-strings)).
- Produces no NGC/download artifacts (Step 1 needs no NGC key). It writes only
  system state (packages, `/etc/modprobe.d/*`, `/etc/profile.d/*`) — never
  `state.json` directly.

This step is the port of Phases 0–4 and Phase 8 of
[`laptop/scripts/00_bootstrap.sh`](../../laptop/scripts/00_bootstrap.sh)
(preflight, base deps, nouveau cleanup, NVIDIA stack, runtime/GStreamer
prereqs, version audit), rebuilt against the framework contract and the DS 9.1
GA equality pins.

---

## 2. The DS 9.1 dGPU prerequisite pins (EQUALITY)

These are the exact versions the DS 9.1 Installation page (`dGPU Setup for
Ubuntu → Prerequisites`) pins. They are **equality pins**, not minimums — DS
9.1's runtime loader refuses older or newer minors of the driver and
`libnvinfer*`. Per the DS 9.1 Installation page: *"Before installing the
DeepStream SDK, ensure you have Ubuntu 24.04, GStreamer 1.24.2, NVIDIA driver
595.58.03, CUDA 13.2, and TensorRT 10.16.0.72 installed."* Also transcribed in
[`DEEPSTREAM-SETUP.md` §4](../../laptop/docs/DEEPSTREAM-SETUP.md).

| Component | Pinned version | Install source | Verify command | `verify_pinned` label / expected |
|-----------|----------------|----------------|----------------|----------------------------------|
| NVIDIA driver | `595.58.03` | `.run` installer (`NVIDIA-Linux-x86_64-595.58.03.run`) | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` | `verify_pinned("NVIDIA driver", <actual>, "595.58.03")` |
| CUDA Toolkit | `13.2` (`cuda-toolkit-13-2`) | NVIDIA `ubuntu2404/x86_64` apt repo | `/usr/local/cuda-13.2/bin/nvcc --version` → `release X.Y` | `verify_pinned("CUDA (nvcc release)", <actual>, "13.2")` |
| cuDNN | `9.20.0.48` (apt `9.20.0.48-1`) | NVIDIA CUDA 13 apt packages | `dpkg-query -W -f='${Version}' libcudnn9-cuda-13` | `verify_pinned("cuDNN (libcudnn9-cuda-13)", <normalized>, "9.20.0.48")` |
| TensorRT | `10.16.0.72-1+cuda13.2` | apt (all `libnvinfer*` pinned) | `dpkg -l \| grep libnvinfer10` | `verify_pinned("TensorRT (libnvinfer10)", <actual>, "10.16.0.72-1+cuda13.2")` |
| GStreamer | `1.24.2` | apt (`gstreamer1.0-*`) | `gst-inspect-1.0 --version` | `verify_pinned("GStreamer", <actual>, "1.24.2")` |
| OS | Ubuntu 24.04 / `x86_64` | (precondition) | `lsb_release -rs`, `uname -m` | preflight check, not `verify_pinned` |

### 2.1 TensorRT package set (all pinned to one version)

Every `libnvinfer*` package MUST be pinned to the same string
`10.16.0.72-1+cuda13.2` (from the DS 9.1 Installation page):

```
version="10.16.0.72-1+cuda13.2"
libnvinfer-dev                       libnvinfer-dispatch-dev
libnvinfer-dispatch10                libnvinfer-headers-dev
libnvinfer-headers-plugin-dev        libnvinfer-safe-headers-dev
libnvinfer-lean-dev                  libnvinfer-lean10
libnvinfer-plugin-dev                libnvinfer-plugin10
libnvinfer-vc-plugin-dev             libnvinfer-vc-plugin10
libnvinfer10                         libnvonnxparsers-dev
libnvonnxparsers10                   tensorrt-dev
libnvinfer-headers-python-plugin-dev libnvinfer-win-builder-resource10
```

The cuDNN transaction pins the complete CUDA 13.2 package dependency chain to
apt version `9.20.0.48-1`:

| Package | Dependency role |
|---------|-----------------|
| `cudnn9-cuda-13` | CUDA 13 meta-package |
| `cudnn9-cuda-13-2` | CUDA 13.2 meta-package |
| `libcudnn9-static-cuda-13` | Static library package required by the CUDA 13.2 meta-package |
| `libcudnn9-dev-cuda-13` | Development package required by the static library package |
| `libcudnn9-headers-cuda-13` | Header package required by the development package |
| `libcudnn9-cuda-13` | Concrete runtime package required by the development package |

Pinning every package prevents the meta-packages' greater-than-or-equal
dependency from resolving a newer cuDNN release. This transaction alone uses
`apt-get install --allow-downgrades` so a partially installed newer CUDA 13
cuDNN chain is reconciled to the DS 9.1 equality pin. TensorRT and general apt
transactions do not permit downgrades. The concrete runtime package is the
presence and version probe; the virtual `libcudnn9` name and shell globs are
not valid install or verification targets on Ubuntu 24.04. The probe
normalizes only the exact apt value `9.20.0.48-1` to the DeepStream component
pin `9.20.0.48`; a different Debian revision remains unnormalized and fails
verification. Dependency reporting uses the component pin.

### 2.2 Reporting each pin

For every component above, `run()` emits exactly one of the two required
strings from
[`00` §8.3](00-FRAMEWORK-AND-BOOTSTRAP.md#83-reporting-format-for-dependencies-required-exact-strings):

- Newly installed → `report_installed(dep, version)` →
  `installed <dependency> version <version>`
- Already at the pinned version → `report_already_installed(dep, version)` →
  `already installed <dependency> version <version>`

The "already installed" path is taken when a probe (see §7.2) finds the
component present **at the exact pinned version**. A transaction that upgrades
or otherwise changes an existing package reports `installed`, not `already
installed`. Examples:

```
installed cuda-toolkit-13-2 version 13.2
already installed gstreamer1.0-tools version 1.24.2
installed libnvinfer10 version 10.16.0.72-1+cuda13.2
already installed nvidia-driver version 595.58.03
```

`verify()` then re-checks every pin with `verify_pinned` (§7.3) and only
returns `COMPLETE` when all match.

---

## 3. DS 9.1 §4.1 apt prerequisite package list

Transcribed from
[`DEEPSTREAM-SETUP.md` §4.1](../../laptop/docs/DEEPSTREAM-SETUP.md) and
cross-checked against the DS 9.1 Installation page "Install prerequisite
packages" block. `run()` installs these in a single apt
transaction, reporting each with the §8.3 strings.

```bash
apt install \
  libssl3 libssl-dev libcurl4-openssl-dev libgles2-mesa-dev \
  libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
  libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-0 \
  libjansson4 libyaml-cpp-dev libjsoncpp-dev protobuf-compiler \
  libmosquitto1 gcc make git python3 \
  mosquitto mosquitto-clients arp-scan ffmpeg
```

### 3.1 Authoritative vs. repo-added packages

The DS 9.1 Installation page's own "Install prerequisite packages" list is:

```
libssl3 libssl-dev libcurl4-openssl-dev libgstreamer1.0-0 gstreamer1.0-tools
gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
gstreamer1.0-libav libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-0
libjansson4 libyaml-cpp-dev libmosquitto1
```

The following are **repo additions** carried by
[`DEEPSTREAM-SETUP.md` §4.1](../../laptop/docs/DEEPSTREAM-SETUP.md) and
[`00_bootstrap.sh` Phase 4](../../laptop/scripts/00_bootstrap.sh), not on the
DS 9.1 §4.1 authoritative list, but harmless/required downstream:
`libgles2-mesa-dev`, `libjsoncpp-dev`, `protobuf-compiler`, `gcc`, `make`,
`git`, `python3`. Keep them (they satisfy build tooling and the DS msgconv /
protobuf paths), and report each with the §8.3 strings like any other
dependency. See the open decision in [§9](#9-open-decisions-for-the-human).

Four further repo additions are carried for the installer's own subsystems
rather than for DeepStream itself. They are on the §3 list, are not on the DS
9.1 §4.1 authoritative list, and each has exactly one consumer:

| Package | Why Step 1 installs it | Consumer |
|---------|------------------------|----------|
| `mosquitto` | the MQTT **broker daemon** the MV3DT sink publishes into; without it there is no listener on `127.0.0.1:1883` | [§3.2](#32-mosquitto-broker), [`STEP-6` §E.1](STEP-6-REMOTE-SUPERVISION.md#e1-lifecycle) |
| `mosquitto-clients` | `mosquitto_sub` / `mosquitto_pub`, used by the broker reachability probe and by every documented validation helper | [`STEP-5` §6](STEP-5-PER-PROJECT-EXES.md#6-validation--monitoring) |
| `arp-scan` | the MAC-OUI sweep that finds the cameras on the link-local segment; needs raw sockets, which the installer already has | [`00` §15](00-FRAMEWORK-AND-BOOTSTRAP.md#15-camera-discovery) |
| `ffmpeg` | supplies `ffprobe` for the RTSP stream probe and `ffmpeg` for the single-frame still capture used in guided position binding | [`00` §15](00-FRAMEWORK-AND-BOOTSTRAP.md#15-camera-discovery), [`STEP-5` §3.3](STEP-5-PER-PROJECT-EXES.md#33-what-the-exe-does-at-runtime-pipeline-subcommand) |

> `libmosquitto1` MUST NOT be omitted — it is the DS MQTT protocol client lib
> required by `Gst-nvmsgbroker`
> ([`DEEPSTREAM-SETUP.md` §4.1](../../laptop/docs/DEEPSTREAM-SETUP.md)). It is
> **not** the broker: it is a shared library that `libnvds_mqtt_proto.so`
> links against, and installing it leaves nothing listening on port `1883`.
> The broker daemon is the separate `mosquitto` package — see
> [§3.2](#32-mosquitto-broker).

`gstreamer1.0-plugins-rtp` / `gstreamer1.0-rtsp` from
[`00_bootstrap.sh` Phase 4](../../laptop/scripts/00_bootstrap.sh) are optional
extras and may be included in the same transaction; they are not on the DS 9.1
§4.1 list.

### 3.2 Mosquitto broker

**LOCKED — Step 1 owns the Mosquitto broker.** Installing the broker daemon
and its `/etc/mosquitto/conf.d/mv3dt.conf` drop-in is Step 1 work, executed as
part of this step's `run()`. This closes gap 1 in
[`DELETION-REVIEW` §6](DELETION-REVIEW.md#6-coverage-gaps-this-triage-exposed),
which recorded that no step claimed the broker while
[`STEP-6` §E.1](STEP-6-REMOTE-SUPERVISION.md#e1-lifecycle) `preflight`
**requires** a reachable broker and
[`STEP-6` §D](STEP-6-REMOTE-SUPERVISION.md#d-security-remote-control-must-be-authenticated)
**rewrites** its configuration.

The `libmosquitto1` line in §3 is a different thing and does not satisfy this
requirement. Keep the two straight:

| Package | Role | Provides |
|---------|------|----------|
| `libmosquitto1` | DeepStream's MQTT **client library** (DS 9.1 §4.1 prerequisite) | the shared object `libnvds_mqtt_proto.so` links against |
| `mosquitto` | the MQTT **broker daemon** | `mosquitto.service`, the `127.0.0.1:1883` listener, `/etc/mosquitto/conf.d/` |

#### Execution: the bundled script, never a typed command

Step 1 does not re-implement the broker setup in Python. It runs the bundled
script, staged as a **tree** so `source "$SCRIPT_DIR/lib/common.sh"` resolves
([`00` §4.2](00-FRAMEWORK-AND-BOOTSTRAP.md#42-locating-and-staging-bundled-assets-at-runtime)):

```python
shellout.run_bundled_script(
    "scripts", "10_setup_mosquitto.sh",
    args=["--non-interactive"], tree=(),
)
```

`tree=()` stages the whole `assets/` tree, not just `scripts/` — required because
the script resolves its config from `$(asset_root)/mosquitto/mv3dt.conf`, a
sibling directory of `scripts/`. `tree=("scripts",)` would stage only the
`scripts/` subtree and the script's own `die "Missing source config..."` path
would fire on every real invocation.

The script ships inside the release binary
([`00` §5](00-FRAMEWORK-AND-BOOTSTRAP.md#5-distribution-the-github-release-binary)),
so the operator never types a script name and no repo checkout is involved.
Its stdout and stderr land in the transcript through the shared logger
([`00` §8.2](00-FRAMEWORK-AND-BOOTSTRAP.md#82-transcript-log-file)).

**REQUIRED — installer-owned execution is always non-interactive.**
The script calls `pause_for_config_review` after installing the drop-in,
which prints the installed config and then, unless the script's own
`NONINTERACTIVE` variable is `1`, blocks on `read -r -p "Press Enter to
continue..." </dev/tty`. `NONINTERACTIVE` starts at `0` and is set to `1`
only by the script's `--non-interactive` CLI flag. The parent TUI owns
installer interaction and streams the child output, so that child prompt is
not a usable input surface. Step 1 therefore always passes the flag. Running
the bundled script directly retains its config-review prompt and standalone
interactive behavior.

#### What the script actually does, in order

Read against
[`10_setup_mosquitto.sh`](../../laptop/scripts/10_setup_mosquitto.sh), the
developer-harness original. Step 1 runs the bundled copy at
`installer/mv3dt_installer/assets/scripts/10_setup_mosquitto.sh`, a ported
variant functionally equivalent to the original but not byte-identical — it
resolves its source config from the staged `asset_root` rather than the repo
tree, and adds the `MV3DT_NO_PAUSE` gate noted above.

1. **Packages.** `dpkg -s mosquitto` gates the install. Missing → `apt-get
   install -y --no-install-recommends libmosquitto1 mosquitto
   mosquitto-clients`. Present → the apt call is skipped, and `libmosquitto1`
   alone is installed if `dpkg -s` shows it absent.
2. **Drop-in.** `install -d -m 0755 /etc/mosquitto/conf.d`, then `mktemp` in
   the destination directory, `cp` the bundled `mv3dt.conf` in, `chmod 0644`,
   `chown root:root`, and `mv -f` onto `/etc/mosquitto/conf.d/mv3dt.conf`. The
   replace is atomic (a same-filesystem rename), so a half-written config is
   never visible to a restarting broker. It is also **unconditional**: the
   script never compares against the file already there, and rewrites
   byte-identical content on every run.
3. **Config review.** `pause_for_config_review` prints the destination path,
   the config's purpose, its full contents, and the customisation hints, then
   blocks on the terminal read unless suppressed as above.
4. **Service.** `systemctl enable mosquitto`, `systemctl restart mosquitto`,
   `sleep 1`, then `systemctl is-active --quiet mosquitto`. On failure it
   prints `systemctl status --no-pager mosquitto` and `die`s, exiting non-zero.
5. **Firewall (optional, off by default).** `--with-firewall` opens `1883/tcp`
   and `9001/tcp` via `ufw`. Step 1 does not pass it; the default posture
   assumes a workstation on a private segment.

#### What Step 1 adds on top of the script (REQUIRED)

The script emits **none** of the §8.3 reporting strings, and step 2 above
makes no already-installed distinction a caller could consume — its own log
lines are free-form and its drop-in rewrite is unconditional. Change detection
and reporting are therefore **Step 1's** work, done in Python around the
shell-out. Nothing here re-implements the install:

| Fact Step 1 needs | How Step 1 obtains it | Reported as |
|---|---|---|
| broker newly installed vs already present | `dpkg -s mosquitto` **before** the shell-out; read the `Version:` field **after** it | `installed mosquitto version <ver>` / `already installed mosquitto version <ver>` |
| drop-in changed vs unchanged | sha256 of `/etc/mosquitto/conf.d/mv3dt.conf` **before** the shell-out (absent counts as changed), compared against the sha256 of the bundled asset | `installed mv3dt.conf version <sha256[:12]>` / `already installed mv3dt.conf version <sha256[:12]>` |

Both probes are taken before `run_bundled_script` is called, because after it
runs the on-disk state is identical either way. A non-zero exit from the
script is mapped to `FAILED` with the script's own output already in the
transcript; Step 1 never returns `USER_ACTION_REQUIRED` for the broker, since
there is no manual fallback to point the operator at.

Re-running the step is safe: apt is skipped by the script's own `dpkg -s`
gate, the drop-in rewrite is byte-identical, and `restart` is unconditional
but cheap. Only the reported string differs between the first run and later
ones.

`verify()` (§7.3) treats the broker as a pass/fail check, not a pinned
version: `systemctl is-active --quiet mosquitto` succeeds and
`/etc/mosquitto/conf.d/mv3dt.conf` is byte-identical to the bundled asset.
Ubuntu 24.04's `mosquitto` version is not pinned — nothing in the DS 9.1 docs or in
[`mv3dt.conf`](../../laptop/mosquitto/mv3dt.conf) depends on a specific
broker minor, so an equality pin here would create false failures with no
corresponding benefit.

---

## 4. First-install caveats (install-time preconditions)

The DS 9.1 Installation page assumes a workstation with the standard dev
toolchain, no previous NVIDIA stack, Secure Boot disabled, and `nouveau`
already out of the way. On a brand-new Ubuntu 24.04 box those assumptions do
not hold. Step 1 MUST satisfy each of the following **before** running the
`.run` driver installer (§5 step 4). These mirror the caveat table in
[`DEEPSTREAM-SETUP.md` §4 first-install caveats](../../laptop/docs/DEEPSTREAM-SETUP.md)
and Phases 1–2 of
[`00_bootstrap.sh`](../../laptop/scripts/00_bootstrap.sh).

| # | Precondition | Action Step 1 takes | Outcome |
|---|--------------|---------------------|---------|
| 1 | Kernel headers for `.run` to build `nvidia.ko` | `apt install build-essential dkms linux-headers-$(uname -r)` | reported via §8.3 |
| 2 | `add-apt-repository` / `apt-key` present on minimal 24.04 | `apt install software-properties-common ca-certificates gnupg curl` | reported via §8.3 |
| 3 | Distro `nvidia-*` / `libnvidia-*` conflict with `.run` | `apt purge 'nvidia-*' 'libnvidia-*'` + `apt autoremove` | may return `USER_ACTION_REQUIRED` for reboot (§5) |
| 4 | `nouveau` will abort the `.run` installer | write `/etc/modprobe.d/blacklist-nouveau.conf` (`blacklist nouveau` + `options nouveau modeset=0`), `update-initramfs -u` | `USER_ACTION_REQUIRED` for reboot if nouveau was loaded (§5) |
| 5 | Secure Boot → unsigned `nvidia.ko` | probe `mokutil --sb-state`; if enabled, surface `USER_ACTION_REQUIRED` (disable Secure Boot in BIOS **or** complete MOK enrollment on next boot) | operator action; MOK/BIOS is out of scope ([`00` §13](00-FRAMEWORK-AND-BOOTSTRAP.md#13-out-of-scope--defer-to-human)) |
| 6 | GDM/Xorg holding the GPU during `.run` | From a desktop session, stage and asynchronously start `mv3dt-driver-handoff.service`; its root-owned worker stops the active display manager and Xorg after the foreground installer exits. TTY/SSH sessions retain the synchronous path. | desktop-safe precondition for §5 step 7 |
| 7 | CUDA 13.2 not on `PATH` / `LD_LIBRARY_PATH` | write `/etc/profile.d/cuda.sh` exporting `/usr/local/cuda-13.2/bin` and `/usr/local/cuda-13.2/lib64` | enables `nvcc` in `verify()` |

Notes:

- **Nouveau blacklist + `update-initramfs -u`** and the **distro-`nvidia-*`
  purge** are the two reboot triggers (§5). They match Phase 2 of
  [`00_bootstrap.sh`](../../laptop/scripts/00_bootstrap.sh).
- **Secure Boot / MOK enrollment and any BIOS interaction are out of scope**
  ([`00` §13](00-FRAMEWORK-AND-BOOTSTRAP.md#13-out-of-scope--defer-to-human));
  Step 1 detects and surfaces them as `USER_ACTION_REQUIRED`, it does not
  perform them.
- Config files written here (`blacklist-nouveau.conf`, `cuda.sh`) reuse the
  same canonical paths and "Managed by …" header discipline the existing
  bootstrap uses.

---

## 5. Order of operations

Step 1 mirrors the DS 9.1 Installation page ordering (apt prereqs → CUDA repo
→ driver → reboot → TensorRT/cuDNN → verify). Because the driver `.run`
requires a reboot mid-step, the work is split across **two launches** using
Step 1's idempotent driver probe and private handoff marker. It deliberately
does not use the framework auto-complete reboot gate, as [§6](#6-reboot-gating-step-1-owned-continuation)
defines.

**Launch A — up to and including the driver install:**

1. **Preflight** (§7.1): Ubuntu 24.04 + `x86_64`; root; NVIDIA GPU present via
   `lspci` (driver not required yet).
2. **Base deps + caveats 1–2** (§4): `build-essential`, `dkms`,
   `linux-headers-$(uname -r)`, `software-properties-common`,
   `ca-certificates`, `gnupg`, `curl`.
3. **DS 9.1 §4.1 apt prerequisites** (§3), single transaction.
4. **CUDA repo + keyring** and `apt-get install cuda-toolkit-13-2`
   (DS 9.1 §4.2). Write `/etc/profile.d/cuda.sh` (caveat 7). Any failed
   keyring, apt update, or apt install transaction returns `FAILED` before
   later work or dependency reporting.
5. **Nouveau + old-NVIDIA cleanup** (caveats 3–4). If nouveau was loaded or a
   distro `nvidia-*` package was purged → **return `USER_ACTION_REQUIRED`**
   with reboot instructions now
   (before touching the `.run`), because the `.run` cannot build against a
   live nouveau.
6. **Secure Boot check** (caveat 5). If enabled → `USER_ACTION_REQUIRED`.
7. **Install the driver** (caveat 6): a desktop launch copies the verified
   runfile and bundled worker into the root-owned handoff directory, starts
   `mv3dt-driver-handoff.service` with `--no-block`, and returns
   `USER_ACTION_REQUIRED` before the service stops GDM/Xorg. A TTY or SSH
   launch stops GDM/Xorg and runs the same `.run` synchronously.
8. **Reboot after success**: the handoff worker records `succeeded` and
   automatically reboots. The synchronous path returns reboot instructions.
   On handoff failure it records the runfile exit code, restores the display
   manager, and does not reboot.

**Launch B — after the confirmed reboot (§6):**

9. **CUDA recovery**: probe `/usr/local/cuda-13.2/bin/nvcc`; when the toolkit
   is missing after an interrupted Launch A, repeat the idempotent CUDA repo
   and toolkit install, then write `/etc/profile.d/cuda.sh`.
10. **TensorRT** (all `libnvinfer*` pinned to `10.16.0.72-1+cuda13.2`) and
    **cuDNN** (the concrete CUDA 13 package set at apt `9.20.0.48-1`) via apt
    (DS 9.1 §4.4). A failed transaction returns `FAILED` immediately and does
    not report a dependency as installed.
11. **GStreamer** pin is satisfied by the §3 prereq set; confirm `1.24.2`.
12. **Mosquitto broker** ([§3.2](#32-mosquitto-broker)): run the bundled
    `10_setup_mosquitto.sh` to install the daemon, drop in `mv3dt.conf`, and
    enable + restart the service. Placed here because nothing in the NVIDIA
    stack depends on it and the broker survives the driver reboot untouched;
    it must nevertheless complete inside Step 1, since
    [`STEP-6` §E.1](STEP-6-REMOTE-SUPERVISION.md#e1-lifecycle) `preflight`
    fails without a reachable broker.
13. **`verify()`** (§7.3): every pin via `verify_pinned`; confirm the driver
    now loads (`nvidia-smi` succeeds and reports `595.58.03`) and the broker
    is active.
14. On all-match → `COMPLETE`; the dispatch loop advances to Step 2.

> The single reboot in this spec is the **driver `.run`** reboot (step 8). The
> nouveau/purge reboot (step 5) only fires on machines that shipped with
> nouveau loaded or a distro driver preinstalled; on a clean image it is a
> no-op and Launch A proceeds straight to the driver `.run`. Both use the same
> continuation flow, so a machine may legitimately require **two** reboots before
> Step 1 completes; the framework resumes at the first incomplete step each
> time.

### 5.1 Driver install method: `.run` runfile (LOCKED, verified against NVIDIA docs)

This spec installs the driver with the **`.run` runfile installer**, matching
the DS 9.1 Installation page verbatim. Verified against the authoritative
source (DS 9.1 Installation page, "Install the DeepStream SDK → Install
Dependencies → Install NVIDIA driver 595.58.03"), which prescribes exactly:

```bash
$ chmod 755 NVIDIA-Linux-x86_64-595.58.03.run
$ sudo ./NVIDIA-Linux-x86_64-595.58.03.run --no-cc-version-check
```

The DS 9.1 Installation page documents **no** `.deb`/apt path for the *driver*
— only CUDA Toolkit `13.2`, TensorRT `10.16.0.72`, and cuDNN `9.20.0.48` are
installed from apt/deb packages (§3, §5). The existing
[`00_bootstrap.sh` Phase 3](../../laptop/scripts/00_bootstrap.sh) diverges by
using the NVIDIA **local-repo `.deb`** (an older `cuda-drivers-590` pin); Step
1 supersedes that and uses the `.run` because "use what the NVIDIA docs say"
is the ruling constraint. Only the `.run` path is documented.

### 5.1b Display-manager unit names (RESOLVED)

Ubuntu 24.04 ships the GNOME display manager as **`gdm3`**, not `gdm`.
Stopping it by trying `gdm` and falling back to `lightdm` therefore found
nothing on every real workstation, and -- because "neither unit stopped"
was treated as failure -- Step 1 reported *"could not stop the desktop
session"* and told the operator to switch to a virtual console they were
already sitting on.

Two corrections, both REQUIRED:

- Units are tried in the order `gdm3`, `gdm`, `lightdm`, `sddm`, and only
  the one that is actually active is stopped (`systemctl is-active` first).
- **No display manager running is success, not failure.** There is nothing
  to stop, which is precisely the state after the operator has done what
  [§5.1a](#51a-desktop-safe-systemd-handoff-resolved)
  asked of them. Failure is reserved for an active display manager that
  refuses to stop.

---

### 5.1a Desktop-safe systemd handoff (RESOLVED)

Two operational departures from the verbatim DS 9.1 command are **LOCKED**,
both found on real hardware rather than in review:

1. **`--silent` is added** to the `.run` invocation. The runfile is
   interactive by default and can stop on ncurses questions (DKMS
   registration, 32-bit compatibility libraries, an existing driver).
   `run_root` captures output, so such a question is drawn on no screen at
   all: the install blocks forever with nothing to show why, which from the
   outside is indistinguishable from a slow kernel-module build. `--silent`
   implies `--no-questions` and accepts the licence, making that class of
   hang impossible.
2. **A desktop launch hands the disruptive work to systemd before GDM is
   touched.** `service gdm stop` tears down the X session and every terminal
   emulator in it, including the foreground installer. Step 1 therefore
   stages the bundled worker, a freshly verified copy of the runfile, a
   status marker, and the persistent log under
   `/var/lib/mv3dt-installer/driver-handoff/`. It installs
   `mv3dt-driver-handoff.service`, starts it asynchronously, and exits. The
   unit waits ten seconds before the worker stops the display manager, runs
   the exact [§5.1](#51-driver-install-method-run-runfile-locked-verified-against-nvidia-docs)
   command, records `succeeded:<boot-id>`, and automatically reboots only
   after exit zero. A failure writes `failed:runfile:<code>`, attempts to
   restart the display manager, records a `:desktop-restore` suffix if that
   recovery also fails, and leaves the log
   at `/var/lib/mv3dt-installer/driver-handoff/driver-install.log`.

The session check still treats a virtual console (`/dev/ttyN`) and an SSH
session as safe synchronous paths. SSH is detected by walking `/proc` for an
`sshd` ancestor rather than reading `SSH_CONNECTION`/`SSH_TTY`, because
`sudo`'s default `env_reset` strips those variables. A desktop terminal with
an active display manager selects the handoff path instead of being refused.

| Handoff status | Next launch behavior |
| --- | --- |
| absent | Stage and schedule the worker when the desktop hazard is detected |
| `scheduled` / `running` | Refuse a duplicate handoff and show the journal/log locations |
| `succeeded:<current-boot-id>` | Show the reboot action as a fallback |
| `succeeded:<prior-boot-id>` but driver absent | Report that the module did not load and show Secure Boot/MOK remediation |
| `failed:<reason>` | Keep Step 1 pending; show the persistent log and marker-reset command |

---

### 5.2 Runfile acquisition: installer-fetched (RESOLVED)

The `.run` file is **neither bundled nor operator-staged by default**: Step 1
downloads it itself, from the version-stamped public runfile mirror

```
https://us.download.nvidia.com/XFree86/Linux-x86_64/595.58.03/NVIDIA-Linux-x86_64-595.58.03.run
```

Bundling was rejected — the runfile is ~400MB, and [`00`
§4.1](00-FRAMEWORK-AND-BOOTSTRAP.md#41-what-builds-the-binary)
makes the Release binary the single artifact every operator downloads, so
bundling would tax every install for a file apt already parallels for CUDA /
TRT / cuDNN. Requiring the operator to stage it by hand was equally rejected:
every other dependency in this step is acquired automatically, and a manual
step here is the one thing that stops an unattended run.

`DRIVER_DOWNLOAD_URL` is **derived from the `DRIVER_VERSION` constant**, never
hardcoded, so the [§2](#2-the-ds-91-dgpu-prerequisite-pins-equality) equality pin stays
the single source of truth — a pin bump cannot leave the URL pointing at the
superseded build. The mirror path is used rather than the
`driver/details/<id>` search result because the latter is a per-build
redirect page, not a stable fetchable asset.

**Compatibility is enforced on the file itself, not on the URL** (REQUIRED).
A `.run` runfile is a self-extracting shell archive whose plain-text header
names the build it was cut from; `_verify_driver_run` reads the first 8KB and
refuses anything whose stamped version is not exactly `595.58.03`, before the
file is made executable or run. This catches a mirror redirect, a withdrawn
build, a resumed partial fetch, and an HTML error page saved under the right
filename. An optional `DRIVER_RUN_SHA256` constant adds a full-content pin
when a human has verified a checksum out of band; it is `None` by default
because NVIDIA publishes no stable checksum manifest for this mirror.

| Condition | Outcome |
| --- | --- |
| No file staged | Download, verify, proceed |
| File staged, version matches | Verify only — no re-download |
| File staged, version wrong | Discard, re-download, verify |
| Download fails, or verify fails | `USER_ACTION_REQUIRED` — manual staging |

The download lands on a sibling `.part` path and is renamed into place only
after verification passes, so a failed or interrupted fetch never leaves
something a later launch would mistake for a good staged file.

Every fetch in this step shells out to **`curl`**, never `wget` (REQUIRED).
Step 1 runs before anything beyond a base Ubuntu image can be assumed, so it
may only use fetch tools it installs itself: `curl` is in
`BASE_TOOLING_PACKAGES` ([§4](#4-first-install-caveats-install-time-preconditions)
caveat 2) and `wget` is not. A minimal or server image need not ship `wget`,
and the resulting exit 127 surfaces as a download failure rather than as a
missing binary. `curl -fL` is the required form: `-f` turns an HTTP error
into a non-zero exit instead of a saved error page, `-L` follows the mirror
redirect.

> **Fallback is retained, not removed.** An air-gapped host, an outbound proxy,
> or a 404 on a withdrawn build still ends in the original
> `USER_ACTION_REQUIRED` pointing at NVIDIA's driver download search
> (<https://www.nvidia.com/en-us/drivers/>) and the exact staging path. A
> failed download is never `FAILED` — only the `.run` exiting non-zero is
> ([§5](#5-order-of-operations)).

---

## 6. Reboot gating (Step 1-owned continuation)

Step 1 **does not use** the framework's auto-complete reboot variant
([`00` §7](00-FRAMEWORK-AND-BOOTSTRAP.md#7-reboot-detection--continuation-contract)).
That contract marks the requesting step `COMPLETE` when a new boot is
detected. Step 1 still owns TensorRT, cuDNN, Mosquitto, and final verification
after the driver reboot, so auto-completion would skip required work.

### 6.1 Reboot outcomes

Both Step 1 reboot points keep the framework step state pending:

| Reboot point | Result | Continuation evidence |
| --- | --- | --- |
| Nouveau loaded or distro driver purged | `USER_ACTION_REQUIRED` with `sudo reboot` | Idempotent cleanup probes are clear on the next launch |
| Synchronous driver install succeeds | `USER_ACTION_REQUIRED` with `sudo reboot` | `nvidia-smi` reports exactly `595.58.03` on the next launch |
| Desktop handoff succeeds | Worker writes `succeeded:<boot-id>` and requests reboot | New boot ID plus `nvidia-smi` reporting exactly `595.58.03` |

The framework alone owns `state.json`
([`00` §12.2](00-FRAMEWORK-AND-BOOTSTRAP.md#122-stepresult-and-status-recorded-by-the-state-machine)).
The desktop worker's marker is operational handoff state under
`/var/lib/mv3dt-installer/driver-handoff/status`, not step-completion state.

### 6.2 User actions

The synchronous driver-success action list contains:

1. **Driver load**: reboot so `nvidia.ko` loads before TensorRT/cuDNN.
2. **CUDA shell path**: `/etc/profile.d/cuda.sh` applies to new login shells.
3. **Reboot command**: `sudo reboot`.

The desktop path instead tells the operator to save other work and wait. The
worker closes the desktop after its ten-second grace period and reboots
automatically after success. The operator logs in and runs the installer
again after either reboot path.

### 6.3 Next-launch routing

`run()` always starts from idempotent evidence:

- **Driver `595.58.03` loaded**: route directly to Launch B, regardless of a
  retained handoff marker.
- **Handoff status `scheduled` or `running`**: refuse a duplicate worker.
- **Handoff success from the current boot**: retain the Step 1 reboot gate and show
  `sudo reboot` as a fallback if the automatic request did not occur.
- **Handoff success from a prior boot but driver absent**: report that the
  module did not load and show Secure Boot/MOK remediation.
- **Handoff failure**: keep Step 1 pending and show the persistent log and
  marker-reset command. A `:desktop-restore` suffix states that graphical
  recovery also failed; it is never described as successful.
- **No handoff state**: continue Launch A normally.

---

## 7. Lifecycle behavior (`preflight`/`run`/`verify`/`report`)

Against the protocol in
[`00` §12.1](00-FRAMEWORK-AND-BOOTSTRAP.md#121-protocol).

### 7.1 `preflight(ctx)`

- Assert Ubuntu 24.04 + `x86_64` (`lsb_release`, `uname -m`); NVIDIA GPU via
  `lspci`. On mismatch → `FAILED` with a pointer to
  [`DEEPSTREAM-SETUP.md` §2–3](../../laptop/docs/DEEPSTREAM-SETUP.md).
- Determine internal stage: **Stage A** (driver not yet installed/loaded) vs
  **Stage B** (driver `595.58.03` present and `nvidia-smi` loads). The probe
  is idempotent so re-runs are safe.
- Returns `COMPLETE` ("ok to run") unless the OS/arch gate fails.

### 7.2 `run(ctx)`

- Executes the §5 order for the current stage. Bundled bash fragments (apt
  transactions, nouveau blacklist, the `.run` invocation) are shelled out via
  `ctx.run_root(...)` after being located with `ctx.asset_path(...)`
  ([`00` §4.2](00-FRAMEWORK-AND-BOOTSTRAP.md#42-locating-and-staging-bundled-assets-at-runtime)).
- The Mosquitto broker is installed through the bundled script
  ([§3.2](#32-mosquitto-broker)) rather than open-coded apt/systemctl calls.
- Every dependency touched is reported with `report_installed` /
  `report_already_installed` (§2.2, §3, §3.2). The "already installed" path is
  taken when a pre-check (`dpkg -s` / `nvidia-smi` / `nvcc` /
  `gst-inspect-1.0`) shows the component already at the pinned version.
- May return `USER_ACTION_REQUIRED` (reboot, active handoff, Secure Boot,
  GDM-stop failure, or driver-load failure), `FAILED` (apt, synchronous
  `.run`, or handoff scheduling error), or `COMPLETE`. It never returns
  `REBOOT_REQUIRED`; [§6](#6-reboot-gating-step-1-owned-continuation) defines
  why.

### 7.3 `verify(ctx)` — the pinned checklist

Idempotent; returns `COMPLETE` only when **every** check passes. Each maps a
verification command to `verify_pinned`
([`00` §8.4](00-FRAMEWORK-AND-BOOTSTRAP.md#84-verify-at-exact-pinned-version-helper-required)):

```
nvidia-smi --query-gpu=driver_version --format=csv,noheader
        -> verify_pinned("NVIDIA driver", <out>, "595.58.03")
/usr/local/cuda-13.2/bin/nvcc --version   (parse "release X.Y")
        -> verify_pinned("CUDA (nvcc release)", <out>, "13.2")
dpkg-query -W -f='${Version}' libcudnn9-cuda-13
        -> normalize exact "9.20.0.48-1" to "9.20.0.48"
        -> verify_pinned("cuDNN (libcudnn9-cuda-13)", <normalized>, "9.20.0.48")
dpkg -s libnvinfer10   (Version:)
        -> verify_pinned("TensorRT (libnvinfer10)", <out>, "10.16.0.72-1+cuda13.2")
gst-inspect-1.0 --version   (parse "version X.Y.Z")
        -> verify_pinned("GStreamer", <out>, "1.24.2")
```

Any `verify_pinned` returning `False` → `verify()` returns
`USER_ACTION_REQUIRED` (or `FAILED`) with the mismatch, never `COMPLETE`. This
is the hard gate that Phase 8 of
[`00_bootstrap.sh`](../../laptop/scripts/00_bootstrap.sh) performs, ported to
the framework helper.

### 7.4 `report(ctx)`

Prints the human summary block (no side effects): each pin and whether it was
newly installed vs already present, plus where the two config files landed
(`/etc/profile.d/cuda.sh`, `/etc/modprobe.d/blacklist-nouveau.conf`).

---

## 8. Blackwell (RTX PRO 4500) compatibility check

**Result: the pinned driver `595.58.03` and CUDA `13.2` support the RTX PRO
4500 Blackwell. No pin change is required.**

- DS 9.1 **Platform and OS Compatibility → dGPU**: the dGPU compatibility
  table lists supported architectures as *"Turing, Ampere, Hopper, ADA,
  Blackwell"* alongside Ubuntu 24.04, GCC 11.4.0, CUDA 13.2, cuDNN 9.20.0.48,
  TRT 10.16.0.72, and Display Driver R595.58.03.
- DS 9.1 **Installation**: the dGPU setup introduction lists *"NVIDIA
  GeForce® RTX pro 4500 and GeForce®/NVIDIA RTX/QUADRO"* among the supported
  dGPU products.

So Step 1 keeps the exact pins in §2. The RTX PRO 4500 is named as a supported
device at these exact versions; there is no evidence a newer driver/CUDA minor
is required for Blackwell.

`verify()` MAY additionally record the GPU's compute capability for the
transcript (informational, not a gate):

```
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

> Escalation rule (per LOCKED scope): if a future device or a driver
> regression means `595.58.03` / CUDA `13.2` do **not** enumerate the RTX PRO
> 4500 (e.g. `nvidia-smi` shows the GPU as unsupported, or CUDA cannot see
> it), Step 1 must **not** silently bump the pins — it returns
> `USER_ACTION_REQUIRED` and the version bump is escalated as an open decision
> ([§9](#9-open-decisions-for-the-human)).

---

## 9. Open decisions for the human

DevA does not invent new scope; the following need a human decision (per the
LOCKED constraints):

1. **Blackwell driver support — RESOLVED as a confirmation, not a change.** DS
   9.1 docs explicitly list the RTX PRO 4500 / Blackwell as supported at driver
   `595.58.03` + CUDA `13.2` + cuDNN `9.20.0.48` + TRT `10.16.0.72` (§8). No pin
   change. Flagged here only so the human can confirm the workstation's actual
   `nvidia-smi` output matches on first real hardware run; if it does not, the
   pin bump is a human decision, not an automatic one.
2. **Driver install method — RESOLVED: `.run` runfile.** Verified against the
   DS 9.1 Installation page: NVIDIA documents only the `.run`
   runfile for the driver (`NVIDIA-Linux-x86_64-595.58.03.run --no-cc-version-check`);
   there is no documented `.deb`/apt driver path. Step 1 uses the `.run`
   accordingly and supersedes the local-repo `.deb` driver install in the
   existing [`00_bootstrap.sh`](../../laptop/scripts/00_bootstrap.sh). This
   accepts the GDM-stop / TTY / MOK friction (caveats 5–6) as the documented
   cost. The residual bundled-vs-download choice is now also **RESOLVED:
   installer-fetched** from the version-stamped NVIDIA runfile mirror, with
   header-based version verification and operator staging retained as the
   fallback — see [§5.2](#52-runfile-acquisition-installer-fetched-resolved).
3. **apt prereq superset.** Step 1 installs the DS 9.1 §4.1 list **plus** the
   repo additions in §3.1 (`libgles2-mesa-dev`, `libjsoncpp-dev`,
   `protobuf-compiler`, `gcc`, `make`, `git`, `python3`). Confirm the superset
   is acceptable (recommended — they are needed downstream) or trim to the DS
   9.1 §4.1 authoritative list only. The four installer-subsystem additions
   (`mosquitto`, `mosquitto-clients`, `arp-scan`, `ffmpeg`) are **not** part of
   this open decision — they are RESOLVED and required by
   [§3.2](#32-mosquitto-broker) and
   [`00` §15](00-FRAMEWORK-AND-BOOTSTRAP.md#15-camera-discovery).
4. **Documented drift from the existing script** — see [§10](#10-documented-drift-from-00_bootstrapsh);
   confirm the equality-pin enforcement replaces the old `>= 550` / `>= 12.4`
   preflight.

---

## 10. Documented drift from `00_bootstrap.sh`

The existing
[`00_bootstrap.sh`](../../laptop/scripts/00_bootstrap.sh) preflight and
[`laptop/deepstream/*`](../../laptop/deepstream/) still target the older DS
9.0 pins and NGC-based acquisition — that script/config tree has not yet been
updated to match this spec (noted in
[`DEEPSTREAM-SETUP.md` §5.2 "Known drift"](../../laptop/docs/DEEPSTREAM-SETUP.md)).

**Step 1 targets the DS 9.1 GA equality pins in §2**: driver
**`== 595.58.03`**, CUDA **`== 13.2`**, cuDNN `== 9.20.0.48`, TRT
`== 10.16.0.72-1+cuda13.2`, GStreamer `== 1.24.2`, all enforced through
`verify_pinned` (§7.3). Until `00_bootstrap.sh` is updated to match, running it
against this spec's expectations will under-provision the driver/CUDA/TRT/cuDNN
stack and `verify()` will fail.

Note: Phase 8 of the current script hard-gates equality values via
`require_version_eq`, but those values are still the DS 9.0 pins — the drift
now spans the full stack (driver, CUDA, cuDNN, TensorRT, DS SDK acquisition),
not just the preflight thresholds. Step 1 makes the DS 9.1 equality pin the
single, consistent gate from preflight through verify; bringing
`00_bootstrap.sh` up to match is a follow-up implementation task, not
something this spec change performs by itself.

---

## 11. What the operator must do after this step

- Nothing further **if** `verify()` returned `COMPLETE` — the dispatch loop
  advances straight to Step 2.
- If Step 1 returned `USER_ACTION_REQUIRED` after cleanup or a synchronous
  driver install: `sudo reboot`, then run the installer again to continue.
- If Step 1 returned `USER_ACTION_REQUIRED` for **Secure Boot / MOK**: disable
  Secure Boot in BIOS, **or** complete MOK Manager enrollment on the next
  boot, then re-run. (BIOS/MOK are out of scope for the installer,
  [`00` §13](00-FRAMEWORK-AND-BOOTSTRAP.md#13-out-of-scope--defer-to-human).)
- If Step 1 returned `USER_ACTION_REQUIRED` for the **desktop handoff**: save
  other open work and wait. The desktop closes after the ten-second grace
  period and the workstation reboots automatically after a successful driver
  install. Log in and run the installer again to continue. On failure, the
  desktop is restored and the displayed persistent log explains the cause.
- New login shells pick up CUDA 13.2 automatically from
  `/etc/profile.d/cuda.sh`; a shell open from before the reboot must
  `source /etc/profile.d/cuda.sh` or re-login for `nvcc` to be on `PATH`.

---

## 12. `verify()` checklist (developer quick-reference)

Step 1 is `COMPLETE` iff **all** of these pass:

- [ ] `lsb_release -rs` = `24.04` and `uname -m` = `x86_64`.
- [ ] `nvidia-smi` runs (driver loaded) and driver_version == `595.58.03`.
- [ ] `/usr/local/cuda-13.2/bin/nvcc --version` release == `13.2`; new shells
      also receive CUDA on `PATH` via `/etc/profile.d/cuda.sh`.
- [ ] `dpkg-query -W -f='${Version}' libcudnn9-cuda-13` is exactly apt
      `9.20.0.48-1`, normalized to the DeepStream pin `9.20.0.48` for the
      verification and dependency report.
- [ ] `dpkg -s libnvinfer10` Version == `10.16.0.72-1+cuda13.2` (and the full
      `libnvinfer*` set from §2.1 all at that version).
- [ ] `gst-inspect-1.0 --version` == `1.24.2`.
- [ ] DS 9.1 §4.1 apt prereqs (§3) all installed, including `mosquitto`,
      `mosquitto-clients`, `arp-scan`, and `ffmpeg` (§3.1).
- [ ] `systemctl is-active --quiet mosquitto` succeeds and
      `/etc/mosquitto/conf.d/mv3dt.conf` matches the bundled config (§3.2).
- [ ] No `reboot_pending` marker for `step1_prerequisites`
      ([`00` §7](00-FRAMEWORK-AND-BOOTSTRAP.md#7-reboot-detection--continuation-contract)).
- [ ] (informational) `nvidia-smi --query-gpu=name,compute_cap` recorded for
      the RTX PRO 4500 (§8).

Each checked item is reported with the §8.3 strings and validated with
`verify_pinned` (§8.4).

---

## References

DeepStream 9.1 official documentation only. Reference DS 9.1 only.

- DS 9.1 Installation (authority for the dGPU prerequisites §4.1–4.4, the
  driver/CUDA/cuDNN/TRT/GStreamer pins, and install ordering):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html>
- DS 9.1 Overview (platform support incl. RTX PRO 4500 / Blackwell):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Overview.html>
- DS 9.1 Release Notes:
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Release_notes.html>
- DS 9.1 Quickstart (DeepStream startup and SDK usage reference):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html>

Facts confirmed directly against the pages above:

- "Before installing the DeepStream SDK, ensure you have Ubuntu 24.04,
  GStreamer 1.24.2, NVIDIA driver 595.58.03, CUDA 13.2, and TensorRT
  10.16.0.72 installed."
- CUDA Toolkit 13.2 via `cuda-toolkit-13-2` from the `ubuntu2404/x86_64` repo.
- TensorRT `version="10.16.0.72-1+cuda13.2"` across all `libnvinfer*` packages.
- dGPU compatibility table: "Turing, Ampere, Hopper, ADA, Blackwell" with
  Ubuntu 24.04, GCC 11.4.0, CUDA 13.2, cuDNN 9.20.0.48, TRT 10.16.0.72,
  Display Driver R595.58.03.

Repo files referenced:

- [`installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md)
  — shared framework contracts consumed by this step.
- [`laptop/docs/DEEPSTREAM-SETUP.md`](../../laptop/docs/DEEPSTREAM-SETUP.md) —
  §4 pins, §4.1 apt list, first-install caveats, §5.2 known drift.
- [`laptop/scripts/00_bootstrap.sh`](../../laptop/scripts/00_bootstrap.sh) —
  Phases 0–4 + Phase 8 this step ports; source of the `>= 550` / `>= 12.4`
  preflight drift.
- [`laptop/scripts/lib/common.sh`](../../laptop/scripts/lib/common.sh) —
  `require_version_eq` (ported to `verify_pinned`), logging, `require_root`.
- [`laptop/scripts/10_setup_mosquitto.sh`](../../laptop/scripts/10_setup_mosquitto.sh)
  — the original of the bundled broker-setup script this step now owns and
  runs (§3.2).
- [`laptop/mosquitto/mv3dt.conf`](../../laptop/mosquitto/mv3dt.conf) — the
  drop-in installed to `/etc/mosquitto/conf.d/mv3dt.conf` (§3.2).
