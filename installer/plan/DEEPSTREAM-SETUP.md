# Laptop DeepStream 9.1 Setup

This document is the **DS 9.1 laptop setup reference** for the scripted
testing harness. It is a standalone mirror of Notion page
`337b5d58-7212-81e1-b07a-d510d9605bbb`: §1–4 are manual prerequisites that
operators complete **outside this repo** before running any script; §5–10
are the parts that [`laptop/scripts/`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/tree/57adf5d/laptop/scripts/) automates.

Every external DS 9.1 URL in this doc is drawn from
[`.cursor/skills/deepstream-9-docs/reference.md`](../../.cursor/skills/deepstream-9-docs/reference.md)
so the link set stays in sync with NVIDIA's authoritative index. Every plugin
field and config fact was cross-checked directly against NVIDIA's published
DS 9.1 documentation before being written.

This doc and [`my-docs/02-LAPTOP-DEEPSTREAM-SETUP.md`](../../my-docs/02-LAPTOP-DEEPSTREAM-SETUP.md)
are both DS 9.1 references kept in sync against the
[DS 9.1 Installation page](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html).
This doc is scoped to the scripted harness under [`laptop/scripts/`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/tree/57adf5d/laptop/scripts/);
`my-docs/02` is the long-form operator manual. The two share the same driver /
CUDA / TensorRT / DS 9.1 pins (§4, §5) — if one is edited, re-sync the other.

## Overview

Target platform (versions are **equality pins** from the
[DeepStream 9.1 Installation page](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html),
`dGPU Setup for Ubuntu → Prerequisites`; DS 9.1 will not start against older
driver / CUDA / TRT minors):

- Ubuntu 24.04 dual-boot (§3)
- Ampere-or-newer NVIDIA dGPU (§1–2); Turing is supported per the DS 9.1
  Platform Compatibility table but is not the reference configuration for this
  harness
- NVIDIA driver **595.58.03** (§4) — from the `.run` installer at
  [nvidia.com/en-us/drivers](https://www.nvidia.com/en-us/drivers/) (resolve
  the exact `driver/details/<id>` URL for this build)
- CUDA Toolkit **13.2** (§4) — `cuda-toolkit-13-2` from NVIDIA's
  `ubuntu2404/x86_64` apt repo
- TensorRT **10.16.0.72-1+cuda13.2** (§4) — all `libnvinfer*` packages pinned
  to this version
- cuDNN **9.20.0.48** (§4) — per the DS 9.1 dGPU compatibility table
- DeepStream 9.1 + GStreamer 1.24.2 (§5) — `deepstream-9.1_9.1.0-1_amd64.deb`
  from [GitHub Releases](https://github.com/NVIDIA/DeepStream/releases/tag/v9.1.0)
  (public, no NGC account needed)
- Mosquitto 2.x broker + `libmosquitto1` client lib (§6)
- 8× IP cameras on `192.168.10.101..108` (§7.2)
- Docker Engine + NVIDIA Container Toolkit (§8.2) — for AMC
- `NVIDIA/DeepStream` monorepo, `tools/auto-magic-calib` (§8.3) — cloned
  (sparse checkout) into `$HOME/auto-magic-calib/`
- PeopleNet v2.6.3 deployable (§9.3, only detector this harness installs)

## §1–4  Manual prerequisites

The laptop must be booted into Ubuntu 24.04 **before** cloning this repo.
Sections §1–§3 (hardware, BIOS, dual-boot) are manual. The full NVIDIA driver,
CUDA, cuDNN, and TensorRT stack in §4 is **automated** by
[`laptop/scripts/00_bootstrap.sh`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/blob/57adf5d/laptop/scripts/00_bootstrap.sh) (Phases 2b–3, 4,
8) once you place the required `.deb` files in `LOCAL_DEB_DIR`; see the callout
in §4 below.

### §1 Overview / §2 Hardware & BIOS

- Ampere-or-newer NVIDIA discrete GPU, CUDA compute capability ≥ 8.0
  (`nvidia-smi --query-gpu=compute_cap --format=csv,noheader`).
- BIOS: enable virtualization extensions (VT-x / AMD-V) and disable Secure
  Boot if your distro-signed NVIDIA driver path requires it.
- ≥ 32 GB RAM recommended when running 8 cameras through DeepStream + AMC.

### §3 Dual-boot Ubuntu 24.04

Standard Ubuntu 24.04 LTS install from an installer USB (Rufus or
balenaEtcher). Shrink the existing Windows partition, let the Ubuntu
installer use the freed space. After first boot, `sudo apt update && sudo
apt full-upgrade`.

### §4 NVIDIA driver + CUDA + cuDNN + TensorRT

> **Semi-automated by `laptop/scripts/00_bootstrap.sh`.** The script currently
> installs and verifies the driver, CUDA, cuDNN, TensorRT, and GStreamer stack
> from pre-downloaded local-repo `.deb` files, and downloads the DeepStream
> `.deb` itself in Phase 6. **It has not yet been updated to the DS 9.1 pins
> and acquisition method below** — see the "Known drift" callout in §5.2
> before running it end-to-end. See the script header and
> `laptop/config/laptop.env.example` for `LOCAL_DEB_DIR`.

Follow the
[DeepStream 9.1 Installation page](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html)
`dGPU Setup for Ubuntu` section exactly. The version numbers below are the
values that section pins for DS 9.1; these are **not** minimums — DS 9.1's
runtime loader refuses older or newer minors of `libnvinfer*` and the driver.

| Component | Pinned version | Verify with |
|-----------|----------------|-------------|
| NVIDIA driver | `595.58.03` | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` |
| CUDA Toolkit | `13.2` (`cuda-toolkit-13-2`) | `nvcc --version` |
| TensorRT | `10.16.0.72-1+cuda13.2` | `dpkg -l \| grep libnvinfer10` |
| cuDNN | `9.20.0.48` | `dpkg -l \| grep libcudnn9` |
| GStreamer | `1.24.2` | `gst-inspect-1.0 --version` |

The exact commands (transcribed from the DS 9.1 Installation page, in the
required order):

```bash
# 4.1 — Prerequisite packages (note: libmosquitto1 is required for the DS
#       MQTT protocol lib; do not omit it)
sudo apt install \
    libssl3 libssl-dev libcurl4-openssl-dev libgles2-mesa-dev \
    libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-0 \
    libjansson4 libyaml-cpp-dev libjsoncpp-dev protobuf-compiler \
    libmosquitto1 gcc make git python3

# 4.2 — CUDA 13.2
sudo apt-key adv --fetch-keys \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub
sudo add-apt-repository \
    "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /"
sudo apt-get update
sudo apt-get install cuda-toolkit-13-2

# 4.3 — NVIDIA driver 595.58.03 (stop GDM/LightDM/Xorg first!)
#       Download NVIDIA-Linux-x86_64-595.58.03.run from
#       https://www.nvidia.com/en-us/drivers/ (resolve the exact
#       driver/details/<id> URL for this build)
sudo service gdm stop || sudo service lightdm stop
sudo pkill -9 Xorg || true
chmod 755 NVIDIA-Linux-x86_64-595.58.03.run
sudo ./NVIDIA-Linux-x86_64-595.58.03.run --no-cc-version-check
sudo reboot

# 4.4 — TensorRT (all libnvinfer* pinned to the same version)
version="10.16.0.72-1+cuda13.2"
sudo apt-get install \
    libnvinfer-dev=${version} libnvinfer-dispatch-dev=${version} \
    libnvinfer-dispatch10=${version} libnvinfer-headers-dev=${version} \
    libnvinfer-headers-plugin-dev=${version} libnvinfer-safe-headers-dev=${version} \
    libnvinfer-lean-dev=${version} libnvinfer-lean10=${version} \
    libnvinfer-plugin-dev=${version} libnvinfer-plugin10=${version} \
    libnvinfer-vc-plugin-dev=${version} libnvinfer-vc-plugin10=${version} \
    libnvinfer10=${version} libnvonnxparsers-dev=${version} \
    libnvonnxparsers10=${version} tensorrt-dev=${version} \
    libnvinfer-headers-python-plugin-dev=${version} \
    libnvinfer-win-builder-resource10=${version}
```

If `nvidia-smi` reports driver `595.58.03`, `nvcc --version` reports release
`13.2`, and `dpkg -l | grep libnvinfer10` shows `10.16.0.72-1+cuda13.2`,
§1–4 are complete.

> **Pitfall — do not use `cuda-keyring` alone.** The DS 9.1 Installation page
> uses the older `apt-key adv --fetch-keys` + `add-apt-repository` flow above.
> Installing `cuda-keyring_1.1-1_all.deb` alone does not pull in the CUDA
> toolkit from that repo the way this flow does. This does not affect the
> DeepStream `.deb`/tar itself — those are downloaded from
> [GitHub Releases](https://github.com/NVIDIA/DeepStream/releases/tag/v9.1.0)
> (§5), not from `developer.download.nvidia.com` or NGC.

### §4 first-install caveats (not in NVIDIA's DS 9.1 Installation page)

The DS 9.1 Installation page assumes a freshly-installed Ubuntu 24.04
workstation with the standard dev toolchain, no previous NVIDIA stack,
Secure Boot disabled, and `nouveau` already out of the way. A first-time
laptop install almost always hits at least one of the following — fix them
**before** running the `.run` installer in `4.3`. The long-form operator
manual [`my-docs/02-LAPTOP-DEEPSTREAM-SETUP.md` §4.7](../../my-docs/02-LAPTOP-DEEPSTREAM-SETUP.md#47-first-install-caveats-not-in-nvidias-ds-90-installation-page)
has the expanded version; this is the condensed checklist for the scripted
harness.

| Pitfall | Fix |
|---------|-----|
| `.run` installer needs kernel headers to build `nvidia.ko` | `sudo apt install -y build-essential dkms linux-headers-$(uname -r)` |
| `add-apt-repository` / `apt-key` missing on minimal 24.04 | `sudo apt install -y software-properties-common ca-certificates gnupg curl` |
| Distro `nvidia-driver-5xx` preinstalled by Ubuntu installer conflicts with `.run` | `sudo apt purge -y 'nvidia-*' 'libnvidia-*' && sudo apt autoremove -y && sudo reboot` |
| `nouveau` loaded — `.run` installer will abort | Write `/etc/modprobe.d/blacklist-nouveau.conf` (`blacklist nouveau` + `options nouveau modeset=0`), `sudo update-initramfs -u`, reboot |
| Secure Boot enabled → `nvidia.ko` unsigned after install | Either disable Secure Boot in BIOS, or let the `.run` installer generate a MOK keypair and complete MOK Manager enrollment on the next boot (`mokutil --sb-state` to check) |
| `.run` installer fails under GNOME/KDE even after `service gdm stop` | `Ctrl+Alt+F3` → log in at TTY → run `sudo ./NVIDIA-Linux-x86_64-595.58.03.run --no-cc-version-check` from there |
| `deepstream-9.1_9.1.0-1_amd64.deb` (§5.1) download fails from this machine | It is a public GitHub Release asset — no sign-in needed. Download it from [github.com/NVIDIA/DeepStream/releases/tag/v9.1.0](https://github.com/NVIDIA/DeepStream/releases/tag/v9.1.0) on any machine with internet access, `scp` onto the laptop if needed |
| `cuda-toolkit-13-2` does not touch `PATH` / `LD_LIBRARY_PATH` | Append `export PATH=/usr/local/cuda-13.2/bin:$PATH` and `export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:$LD_LIBRARY_PATH` to `~/.bashrc` |
| `apt-key` deprecation warning on 24.04 | Cosmetic — the key is installed. 24.04-native equivalent: write the de-armoured key to `/etc/apt/keyrings/nvidia-cuda.gpg` and reference it via `[signed-by=...]` in a `sources.list.d/` file |
| Disk space | Budget ≥ 12 GB free after Ubuntu install (CUDA ~5 GB, DS SDK ~2 GB, TRT ~1.5 GB, AMC Docker images ~4 GB, PeopleNet ~1 GB) |
| Optimus / hybrid graphics | Enable discrete GPU as primary renderer in BIOS; `xrandr --setprovideroutputsource` if HDMI-out is needed |

**At this point you may clone this repo and run
`sudo bash laptop/scripts/00_bootstrap.sh`.** Everything from §5 onward is
automated (with the DS 9.1 .deb install noted in §5 below).

## §5  DeepStream 9.1 + GStreamer 1.24  _(semi-automated by `00_bootstrap.sh`)_

The
[DS 9.1 Installation page](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html)
documents **three** install methods for x86_64: the Debian package, the tar
archive, and the DeepStream Docker image. The Debian package and tar archive
are published as **public GitHub Release assets** — there is no `apt install
deepstream-9.1` entry, and no NGC account is needed for these two methods.
Only the Docker image remains NGC-hosted (§8.2 covers the NGC login that
method needs).

### §5.1  Manual step (before `00_bootstrap.sh` can finish DS install)

1. Download `deepstream-9.1_9.1.0-1_amd64.deb` from the
   [NVIDIA/DeepStream v9.1.0 release](https://github.com/NVIDIA/DeepStream/releases/tag/v9.1.0)
   to the laptop (any directory is fine — place it next to the cloned repo so
   the bootstrap script can find it). No sign-in required.
2. Install it with apt so prereqs from §4.1 are resolved as transitive
   dependencies:

   ```bash
   sudo apt-get install ./deepstream-9.1_9.1.0-1_amd64.deb
   ```

3. After the .deb install, apply NVIDIA's RTSP jitter-buffer workaround
   (from the "Install prerequisite packages" note on the Installation page):

   ```bash
   sudo /opt/nvidia/deepstream/deepstream/update_rtpmanager.sh
   ```

> **Tar-archive alternative** (Method 2 on the Installation page):
>
> ```bash
> sudo tar -xvf deepstream_sdk_v9.1.0_x86_64.tbz2 -C /
> cd /opt/nvidia/deepstream/deepstream-9.1/
> sudo ./install.sh
> sudo ldconfig
> ```

### §5.2  What `00_bootstrap.sh` does after §5.1

Run `00_bootstrap.sh` **after** you have already installed DS 9.1 from the
`.deb` in §5.1. The script:

- Preflights §1–4 (Ubuntu 24.04, NVIDIA driver, CUDA, cuDNN, TensorRT
  presence) and refuses to continue if the driver / CUDA / TRT are not
  detected.
- Installs the GStreamer 1.24 plugin set (`gstreamer1.0-{tools,plugins-*,libav,rtsp}`,
  `libgstrtspserver-1.0-0`) — typically a no-op because the DS 9.1 prereq
  list in §4.1 already pulls these in.
- Writes `/etc/profile.d/deepstream.sh` exporting `DEEPSTREAM_DIR` and
  prepending `/opt/nvidia/deepstream/deepstream-9.1/{bin,lib}` onto `PATH` /
  `LD_LIBRARY_PATH`.
- Installs `mosquitto`, `mosquitto-clients`, Docker Engine, and the NVIDIA
  Container Toolkit (§6 + §8.2).

> **Known drift between the current script and this doc.** This doc now
> targets **DS 9.1** (driver `595.58.03` / CUDA `13.2` / cuDNN `9.20.0.48` /
> TensorRT `10.16.0.72-1+cuda13.2`, deb/tar from GitHub Releases), but
> [`laptop/scripts/00_bootstrap.sh`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/blob/57adf5d/laptop/scripts/00_bootstrap.sh) and
> [`laptop/deepstream/*`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/tree/57adf5d/laptop/deepstream/) have **not yet been updated** to
> match — the script still targets the older DS 9.0 pins and downloads the DS
> 9.0 `.deb` via the NGC CLI (`ngc registry resource download-version`), and
> its preflight thresholds predate even that GA release. Running the script
> as-is will install DS 9.0, not the DS 9.1 this doc now describes. Re-sync
> `laptop/scripts/00_bootstrap.sh` (and the hardcoded
> `/opt/nvidia/deepstream/deepstream-9.0` paths in
> [`laptop/deepstream/deepstream_app_config.txt`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/blob/57adf5d/laptop/deepstream/deepstream_app_config.txt)
> and
> [`config_tracker_NvMOT.yml`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/blob/57adf5d/laptop/deepstream/config_tracker_NvMOT.yml))
> against the values in §4–§5 before running it end-to-end.

Reference docs (via the skill catalog):

- DS 9.1 Installation (the authoritative source for §4 and §5.1):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html>
- DS 9.1 Quickstart (sample-app smoke test on x86):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html>
- DS 9.1 deepstream-app:
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html>
- DS 9.1 Release Notes:
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Release_notes.html>
- DS 9.1 Application Migration 9.0 → 9.1:
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Application_migration.html>
- NVIDIA/DeepStream GitHub Releases (deb/tar distribution, v9.1.0):
  <https://github.com/NVIDIA/DeepStream/releases/tag/v9.1.0>

### DS 9.1 breaking changes that affect this harness

Per the [skill's quick-reference table](../../.cursor/skills/deepstream-9-docs/reference.md)
("DS 9.1 Breaking Changes"):

| Change | How this harness deals with it |
|--------|--------------------------------|
| `pyds` (Python bindings) deprecated in favor of `pyservicemaker` | No Python pipeline code. All pipeline assembly is via `deepstream-app` configs. |
| Graph Composer removed | N/A — we author configs by hand. |
| TF/UFF/Caffe model formats removed | PeopleNet ETLT still works (TAO-encrypted format is not UFF). |
| INT8 calibration removed (TAO) | [`config_infer_primary.txt`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/blob/57adf5d/laptop/deepstream/config_infer_primary.txt) honours the PeopleNet INT8 cache if present; flip `network-mode=2` (FP16) if you hit issues. |
| YOLOv3/v4, SSD, FasterRCNN removed | Not used — PeopleNet only. |
| DLA not supported on Jetson Thor | N/A (this is the laptop harness). |
| `Gst-nveglglessink` deprecated in favor of `Gst-nv3dsink` | Jetson-only; not applicable to this x86_64 dGPU harness. |

## §6  Mosquitto  _(automated by `00_bootstrap.sh` + `10_setup_mosquitto.sh`)_

`00_bootstrap.sh` installs `mosquitto` and `mosquitto-clients`.
`10_setup_mosquitto.sh` installs
[`laptop/mosquitto/mv3dt.conf`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/blob/57adf5d/laptop/mosquitto/mv3dt.conf) into
`/etc/mosquitto/conf.d/mv3dt.conf` and restarts the service.

### `mv3dt.conf` (simple-testing posture)

```conf
listener 1883 0.0.0.0
listener 9001
protocol websockets
allow_anonymous true
max_inflight_messages 100
max_queued_messages 1000
max_packet_size 268435456
persistence false
log_dest file /var/log/mosquitto/mosquitto.log
log_type error
log_type warning
log_type notice
```

### Production hardening (reference — not applied by the scripts)

For a hardened production deployment, replace the simple-testing posture
with a bound listener, a password file, and an ACL file:

```conf
listener 1883 127.0.0.1
listener 9001 127.0.0.1
protocol websockets

password_file /etc/mosquitto/passwd
acl_file /etc/mosquitto/aclfile
allow_anonymous false
```

Then:

```bash
sudo mosquitto_passwd -c /etc/mosquitto/passwd mv3dt
sudo systemctl restart mosquitto
```

A minimal ACL:

```conf
user mv3dt
topic readwrite mv3dt/#
```

This deliberately lives in this doc (not [`my-docs/`](../../my-docs/)) so the
`laptop/` tree is self-contained.

### Firewall (Notion §6.3)

If the operator is running `ufw`, run `10_setup_mosquitto.sh --with-firewall`
to `ufw allow` 1883/tcp and 9001/tcp. Skipped by default because the
simple-testing posture does not assume `ufw` is active.

### DS 9.1 IoT / broker docs

- Gst-nvmsgbroker (MQTT proto lib):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvmsgbroker.html>
- Gst-nvmsgconv:
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvmsgconv.html>
- IoT / Edge-to-Cloud Messaging:
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_IoT.html>

## §7  Cameras  _(§7.5 automated by `20_verify_cameras.sh`)_

### §7.1–7.4  Per-camera manual setup (out of scope)

Static IP assignment and stream-profile selection are performed through each
camera's own web UI. Assign:

| Camera | IP | Stream path (default) |
|--------|-----|-----------------------|
| C1 | 192.168.10.101 | `/stream1` |
| C2 | 192.168.10.102 | `/stream1` |
| C3 | 192.168.10.103 | `/stream1` |
| C4 | 192.168.10.104 | `/stream1` |
| C5 | 192.168.10.105 | `/stream1` |
| C6 | 192.168.10.106 | `/stream1` |
| C7 | 192.168.10.107 | `/stream1` |
| C8 | 192.168.10.108 | `/stream1` |

All eight rows (including the `position` label and the `enabled` flag) are
encoded in [`laptop/config/cameras.yml`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/blob/57adf5d/laptop/config/cameras.yml). Override
`rtsp_path` per camera for mixed fleets.

### §7.5  Verification (automated)

```bash
bash laptop/scripts/20_verify_cameras.sh
# or, for partial rigs:
bash laptop/scripts/20_verify_cameras.sh --allow-partial
```

## §8  AutoMagicCalib  _(§8.2 automated by `00_bootstrap.sh`; §8.3–8.7 by `30_`/`40_`)_

### §8.2  Docker + NVIDIA Container Toolkit

`00_bootstrap.sh` installs:

- Docker Engine + `docker-compose-plugin` from Docker's Ubuntu apt repo.
- `nvidia-container-toolkit`, and runs `nvidia-ctk runtime configure --runtime=docker`.
- Adds the invoking (`$SUDO_USER`) user to the `docker` group — **log out /
  back in** once before running `30_start_amc.sh`, or use `sudo docker` for
  that first run.

Reference:
<https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html>

### §8.3–8.5  AMC bring-up (`30_start_amc.sh`)

`30_start_amc.sh` is a runtime orchestrator; **AMC itself is not vendored
into this repo**. AMC now lives inside the `NVIDIA/DeepStream` monorepo at
`tools/auto-magic-calib` rather than as its own standalone repo — see the
[DS 9.1 AutoMagicCalib page](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_AutoMagicCalib.html).
The script:

1. Clones `https://github.com/NVIDIA-AI-IOT/auto-magic-calib.git` into
   `$HOME/auto-magic-calib/` if missing (refuses to place the clone under
   the repo working tree). **Known drift:** this still clones the old
   standalone repo rather than a sparse checkout of the `NVIDIA/DeepStream`
   monorepo's `tools/auto-magic-calib` — re-sync before relying on it, per
   [`installer/plan/STEP-3-AMC-LAUNCHER.md` §4](STEP-3-AMC-LAUNCHER.md).
2. Creates `projects/` and `models/` subdirs and `chown 1000:1000`s them
   (Notion §8.3 — the in-container UID is 1000).
3. `docker login nvcr.io` using `NGC_API_KEY` if set.
4. Writes `$HOME/auto-magic-calib/compose/.env` from
   [`laptop/config/laptop.env`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/blob/57adf5d/laptop/config/laptop.env.example) with:
   - `HOST_IP`
   - `AUTO_MAGIC_CALIB_MS_PORT=8000`
   - `AUTO_MAGIC_CALIB_UI_PORT=5000`
   - `PROJECT_DIR=$AMC_ROOT/projects`
   - `MODEL_DIR=$AMC_ROOT/models`
   - `NVIDIA_VISIBLE_DEVICES=all`
   - `PROJECT_NAME=$PROJECT_NAME`

   Upstream AMC's own `.env.example` is diffed at run time and a warning is
   printed for any key that no longer exists upstream.
5. `docker compose pull && docker compose up -d`.
6. Opens `http://localhost:5000` via `xdg-open` (or prints the URL on
   headless sessions).

Reference:
<https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_AutoMagicCalib.html>

### §8.6  AMC 6-step workflow (**human-driven in the browser**)

1. **Project Setup** — name: `$PROJECT_NAME`; pick #cameras = 8.
2. **Video Upload** — one MP4 per camera, ≥ 15 s, stationary scene.
3. **Parameters** — intrinsic guess: let AMC auto-detect; extrinsic guess:
   pick the nearest preset layout.
4. **Manual Align** — click-to-match ≥ 6 correspondences per pair so AMC
   has enough to bootstrap.
5. **Execute** — run calibration. This is where VGGT runs.

   > **VGGT note (via skill):** if Execute times out, the common cause is
   > GPU VRAM pressure from the VGGT stage. Reduce the in-frame resolution
   > of the uploaded clips or restart AMC before retry.

6. **Results / Export** — review RMSE, export MV3DT artefacts into
   `$HOME/auto-magic-calib/projects/$PROJECT_NAME/exports/`.

### §8.7  Export ingest (`40_export_watcher.sh`)

`40_export_watcher.sh`:

1. Watches `$HOME/auto-magic-calib/projects/$PROJECT_NAME/exports/` with
   `inotifywait` (polls every 5 s if not installed).
2. On a change, first tries `$AMC_ROOT/scripts/export_mv3dt.py
   --project $PROJECT_NAME --output <REPO_ROOT>/laptop/deepstream/calibration/$LOCATION_ID/`.
   The AMC upstream is the ground truth — if that script was renamed in a
   newer release, the watcher falls back to a raw `cp -a` from `exports/`.
3. Renders `laptop/deepstream/deepstream_app_config.rendered.txt` from the
   committed template, substituting `${CAM_USER}` / `${CAM_PASSWORD}` /
   `${LOCATION_ID}` and rewriting each `[sourceN]` `uri=...` with the per-
   camera URL from [`cameras.yml`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/blob/57adf5d/laptop/config/cameras.yml).

## §9  MV3DT directory layout + models  _(PeopleNet: Phase 10 of `00_bootstrap.sh`)_

### §9.2  Directory layout

```
laptop/deepstream/
  deepstream_app_config.txt              # committed template (8 sources, NvMOT, MQTT sink)
  deepstream_app_config.rendered.txt     # generated by 40_export_watcher.sh (gitignored)
  config_infer_primary.txt               # PeopleNet only
  config_tracker_NvMOT.yml               # NvDCF + ReID + SV3DT + MV3DT
  msgconv_config.txt
  calibration/
    .gitkeep
    <LOCATION_ID>/                       # written by 40_export_watcher.sh
  models/
    .gitkeep
    peoplenet/                           # written by 00_bootstrap.sh Phase 10
```

### §9.3  Detector model

**PeopleNet only** — matches NVIDIA's DS 9.1 MV3DT reference documentation
and is intentionally enforced by this plan. `00_bootstrap.sh` (Phase 10) runs:

```bash
ngc registry model download-version "$PEOPLENET_NGC_TAG"
# default: nvidia/tao/peoplenet:deployable_quantized_v2.6.3
```

into `laptop/deepstream/models/peoplenet/` and writes `labels.txt`
(`person`, `bag`, `face`). Override the NGC tag via `PEOPLENET_NGC_TAG` in
[`laptop.env`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/blob/57adf5d/laptop/config/laptop.env.example) without editing the script.

#### yolo11n — future work, NOT installed

`yolo11n` (ultralytics `yolo11n.pt`) is the **single approved alternative**
detector reserved for future work. It is explicitly **not** installed,
exported, or configured by any script in this plan. See the
`marcoslucianops/DeepStream-Yolo` entry in
[`.cursor/skills/deepstream-9-docs/reference.md`](../../.cursor/skills/deepstream-9-docs/reference.md)
(Third-Party / Community) when wiring it in a future iteration.

### DS 9.1 plugin reference (verified via skill)

- Gst-nvvideo4linux2 (decoder):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvvideo4linux2.html>
- Gst-nvstreammux:
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux.html>
- Gst-nvinfer (TensorRT inference):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html>
- Gst-nvtracker:
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html>
- MV3DT 9.1:
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_MV3DT.html>

### NvMultiObjectTracker YAML (reference view)

Full template:
[`laptop/deepstream/config_tracker_NvMOT.yml`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/blob/57adf5d/laptop/deepstream/config_tracker_NvMOT.yml).
Key blocks:

```yaml
SV3DT:
  enable: 1
  calibrationDirectory: calibration/${LOCATION_ID}
  projectionType: homography

MV3DT:
  enable: 1
  mqttBrokerIP: 127.0.0.1
  mqttBrokerPort: 1883
  nodeID: ${LOCATION_ID}
  fusionUpdateRate: 30
  globalIDNegotiationTimeout: 100
```

`calibrationDirectory` is populated by `40_export_watcher.sh` from AMC's
export. `mqttBrokerIP`/`Port` match the local Mosquitto listener installed
by `10_setup_mosquitto.sh`.

## §10  Startup + monitoring  _(automated by `50_start_pipeline.sh`)_

### §10.1  Startup sequence

```bash
bash laptop/scripts/50_start_pipeline.sh
```

Internally:

1. `sudo systemctl start mosquitto` (idempotent).
2. Ping-sweep C1..C8 from [`cameras.yml`](https://github.com/Pipeline-To-Better-Placemaking/P2BP-25_26-Hardware/blob/57adf5d/laptop/config/cameras.yml).
3. Source `/etc/profile.d/deepstream.sh`.
4. `cd laptop/deepstream/ && exec deepstream-app -c deepstream_app_config.rendered.txt`.

Use `--force-template` for a dry-run against the unrendered committed
template (placeholder credentials), `--skip-ping` to skip the sweep, and
`--dry-run` to print the final command without launching.

### §10.2  Validation / monitoring

In a second tty:

```bash
mosquitto_sub -h 127.0.0.1 -t 'mv3dt/#' -v
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv'
systemctl status mosquitto --no-pager
```

`50_start_pipeline.sh` prints these same commands as a block before
executing `deepstream-app` so the operator has them at hand.

## References

This doc's external links come from the catalog in
[`.cursor/skills/deepstream-9-docs/reference.md`](../../.cursor/skills/deepstream-9-docs/reference.md).
For edits to this doc, the canonical DS 9.1 lookup path is (in priority
order):

1. **Context7** — library `/websites/nvidia_metropolis_deepstream_dev-guide`
   (20,418 snippets, high reputation) for any conceptual / API question.
2. **WebFetch** the specific URL from the skill's catalog.
3. **GitHub** samples in the skill's repo catalog
   (e.g. `NVIDIA-AI-IOT/deepstream_reference_apps/deepstream-tracker-3d-multi-view`
   for the MV3DT reference pipeline).

See [`../../.cursor/skills/deepstream-9-docs/SKILL.md`](../../.cursor/skills/deepstream-9-docs/SKILL.md)
for the full routing rules.
