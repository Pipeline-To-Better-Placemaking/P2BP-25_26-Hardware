# Step 3 — AutoMagicCalib launcher (owner: DevC)

Status: Step specification; depends on [`00` §8](00-FRAMEWORK-AND-BOOTSTRAP.md#8-logging--reporting-contract) and [`00` §9](00-FRAMEWORK-AND-BOOTSTRAP.md#9-privilege-and-user-action-contract), and does **not** restate those shared logging or privilege contracts.

This step installs and launches NVIDIA AutoMagicCalib (AMC) from the
**standalone, equality-pinned AMC 3.2.1 checkout**. It replaces
`laptop/scripts/30_start_amc.sh` with one resumable path shared by the installer
and `<install_dir>/bin/amc`.

The operator still performs calibration in the AMC browser UI. Step 3 owns the
container prerequisites, deterministic project identity, service readiness,
browser lifetime, and the durable handoff consumed by Step 4.

---

## 1. Module identity and boundary

| Field | Required value |
|---|---|
| Module | `mv3dt_installer.steps.step3_amc_launcher` |
| Step id | `step3_amc_launcher` |
| Order | `3` |
| Title | `AutoMagicCalib launcher` |
| Prerequisite | Step 2 install-method key is persisted |
| Deliverable | `<install_dir>/bin/amc` |

**LOCKED:** Step 3 brings the AMC stack to a verified ready state and opens the
UI. It does not upload camera media, run calibration, or ingest results. Current
result retrieval belongs to Step 4.

`launch_amc(...)` is shared by the optional installer launch and the
`mv3dt-installer amc` subcommand. The installer launch keeps AMC running after
the browser closes so Step 4 can consume its API. A standalone `amc` launch
retains close-to-stop behavior unless `--keep-up` is explicit. Declining the
immediate launch still completes the durable launcher installation.

---

## 2. Upstream pin

**LOCKED:** use the active standalone repository, not a sparse checkout of the
DeepStream monorepo. The DeepStream repository records the same AMC commit as a
submodule.

| Value | Pin |
|---|---|
| Repository | `https://github.com/NVIDIA-AI-IOT/auto-magic-calib.git` |
| AMC release | `3.2.1` |
| Commit | `0cfd2b790fd77598b0543340a65c2a0e1d192327` |
| Compose file | `<AMC_ROOT>/compose/compose.yml` |
| MS image | `nvcr.io/nvidia/auto-magic-calib:3.2.1` |
| UI image | `nvcr.io/nvidia/auto-magic-calib-ui:3.2.1` |

Every Git command checks its exit status and includes bounded stderr/stdout in
the failure. An existing checkout is accepted only when its `origin` matches
the pinned repository. A clean checkout at another revision is fetched and
detached at the pin. Tracked source changes remain fatal, while untracked data
created by AMC under `projects/` and `models/` and the installer-managed
`compose/.env` are accepted on rerun. An unrelated directory, invalid origin,
or partial clone fails without deleting or overwriting operator data; failures
name the unexpected paths.

`verify()` records the resolved commit. A launch must use the equality pin;
tracking `main` is forbidden.

---

## 3. Container prerequisites

### 3.1 Ubuntu 24.04 package contract

**REQUIRED:** missing prerequisites are installed automatically and
idempotently during Step 3. There is no legacy `docker-compose` fallback.

| Capability | Ubuntu 24.04 package or source | Verification |
|---|---|---|
| Docker Engine | `docker.io` | invoking-user `docker info` succeeds |
| Compose v2 | `docker-compose-v2` | version is at least `2.20.3` |
| Git and download tools | `git`, `curl`, `ca-certificates`, `gnupg` | package installed |
| NVIDIA runtime | official `nvidia.github.io/libnvidia-container/stable/deb` repository, package `nvidia-container-toolkit` | Docker lists runtime `nvidia` |

Compose `2.20.3` is the minimum because the pinned top-level AMC file uses
`include`. Ubuntu 24.04 supplies a newer Compose v2 package.

Provisioning performs, in order:

1. **Install base packages**: run `apt-get update` only when packages are
   missing, then install the missing set with `--no-install-recommends`.
2. **Configure the NVIDIA repository**: install its dearmored key under
   `/usr/share/keyrings/` and its signed source under `/etc/apt/sources.list.d/`.
3. **Install the toolkit**: install `nvidia-container-toolkit`.
4. **Configure Docker**: run `nvidia-ctk runtime configure --runtime=docker`
   when the runtime is absent, then restart Docker.
5. **Enable operator access**: add the invoking user to group `docker`.
6. **Validate the GPU path**: run `nvidia-smi -L` in `ubuntu:24.04` with
   `--gpus all`, matching the NVIDIA Container Toolkit validation pattern.

Any failed package, daemon, Compose-version, runtime, or GPU-container check is
fatal. The transcript carries command evidence without secrets.

---

## 4. Configuration and ports

### 4.1 Persistent installer configuration

| Key | Default or source | Consumer |
|---|---|---|
| `AMC_ROOT` | `$HOME/auto-magic-calib` | Step 3 clone and Step 4 result context |
| `HOST_IP` | detected primary IPv4, fallback `127.0.0.1` | AMC UI API URL |
| `AUTO_MAGIC_CALIB_UI_PORT` | first free port from `5000` through `5099` | browser/UI |
| `AUTO_MAGIC_CALIB_MS_PORT` | first free port from `8000` through `8099` | REST API |
| `AUTO_MAGIC_CALIB_MS_API_URL` | unset | optional AMC UI override |
| `LOCATION_ID` | prompted once, required for non-interactive use | site identity for Steps 4–7 |
| `PROJECT_NAME` | prompted once, default `LOCATION_ID` | AMC project display name |
| `AMC_PROJECT_ID` | AMC `create_project` response | Step 4 API path |

**LOCKED:** an occupied configured port does not produce a late Compose error.
Step 3 scans upward deterministically, persists the first free port, and writes
that value into `compose/.env`. Failure to find a free port in the 100-port
window is fatal.

`LOCATION_ID` and `PROJECT_NAME` accept 3–50 characters: letters, numbers,
period, underscore, and hyphen, starting with a letter or number. Interactive
runs ask only for missing values. Non-interactive runs require a persisted
`LOCATION_ID` or `--location-id`; `PROJECT_NAME` then defaults to it unless
`--project` or persisted configuration supplies another value.

### 4.2 Pinned Compose environment

The generated `<AMC_ROOT>/compose/.env` contains the pinned upstream keys:

```dotenv
HOST_IP=127.0.0.1
AUTO_MAGIC_CALIB_MS_PORT=8000
AUTO_MAGIC_CALIB_UI_PORT=5000
PROJECT_DIR=/home/operator/auto-magic-calib/projects
MODEL_DIR=/home/operator/auto-magic-calib/models
```

`AUTO_MAGIC_CALIB_MS_API_URL` is appended only when configured. The file is
written atomically and only when content changes. `PROJECT_NAME` and
`NVIDIA_VISIBLE_DEVICES` are not injected into the Compose environment because
AMC 3.2.1 does not consume them there.

---

## 5. Authentication and Compose bring-up

**REQUIRED:** NGC login is not advisory. AMC images require access to
`nvcr.io`. Step 3 sources `<install_dir>/secrets/ngc.env` inside an
invoking-user child shell and pipes `NGC_API_KEY` to `docker login nvcr.io
--password-stdin`. The key never appears in argv or the transcript.

Bring-up runs in this exact order:

1. **Validate configuration**: `docker compose config --quiet`.
2. **Authenticate**: required NGC login.
3. **Reuse a live stack**: if both pinned Compose services are already running,
   read their actual host ports with `docker compose port`, repair any stale
   persisted values, and skip pull/start so an interrupted installer can
   safely resume.
4. **Pull images**: `docker compose pull`, unless `--skip-pull` is explicit or
   the live stack is reused.
5. **Start services**: `docker compose up -d` unless the stack is already live.

Every command checks its return code. A failure is fatal and includes bounded
command output. The installer never reports Step 3 complete after a failed
clone, login, Compose parse, image pull, or container start.

---

## 6. Readiness and project identity

### 6.1 Service readiness

**REQUIRED:** after `up -d`, poll
`http://localhost:<MS_PORT>/v1/ready` for up to 1800 seconds. A
transport-success response is insufficient; parsed JSON must contain
`"code": 0`. Then require HTTP `200` from `http://localhost:<UI_PORT>`.
Expected connection failures during the container's first-run model downloads
and parser build are retained in the transcript but hidden from the live
terminal. The spinner is ticked during silent retries so elapsed time remains
live. Ctrl-C returns a clean cancellation failure and tears down a stack started
by the interrupted attempt.

**RESOLVED — first-launch wait.** The upstream AMC setup skill's 120-second
readiness budget assumes models are already present. On a first launch the
`auto-magic-calib-ms` container downloads several GB of models (geocalib, vggt,
moge, and others) before `/v1/ready` answers, which routinely exceeds 120
seconds; a field run failed mid-download and the teardown discarded the
in-progress downloads. The wait is therefore bounded at 1800 seconds and, every
10 seconds, the poll also:

1. **Checks liveness**: if `auto-magic-calib-ms` is no longer listed by
   `docker compose ps --status running --services`, the wait ends immediately
   as a readiness failure — there is nothing left to wait for.
2. **Echoes progress**: the service's latest non-blank log line
   (`docker compose logs --no-log-prefix --tail 5 auto-magic-calib-ms`) is fed
   to the live window when it differs from the last one shown, so model
   downloads read as progress rather than a hang.

> **VGGT weights are not required for readiness.** The gated
> `facebook/VGGT-1B-Commercial` download fails with HTTP `401` without a
> Hugging Face login; AMC logs a warning and continues. Step 3 does not treat
> it as a failure (see [§10](#10-out-of-scope)).

Readiness failure is fatal. Capture bounded output from both:

```bash
docker compose ps
docker compose logs --tail 80
```

Containers started by the failed attempt are torn down. If cleanup also fails,
report both failures.

### 6.2 AMC project contract

**RESOLVED:** AMC 3.2.1 documents creation and project lookup endpoints but no
list-projects endpoint. Step 3 therefore never invents a name search.

1. **Reuse persisted identity**: when `AMC_PROJECT_ID` exists, call
   `GET /v1/get_project_info/<AMC_PROJECT_ID>`. Failure is fatal. If a returned
   `project_name` conflicts with configured `PROJECT_NAME`, fail rather than
   silently selecting the wrong project.
2. **Create once**: when no id is persisted, call `POST /v1/create_project`
   with form field `project_name`, require a JSON `project_id`, and persist it as
   `AMC_PROJECT_ID`.
3. **Reject ambiguity**: a conflict/error response or a response without
   `project_id` is fatal. The installer does not guess among existing projects.

This produces the exact Step 4 API contract:

| Key | Meaning |
|---|---|
| `LOCATION_ID` | deployment/node identity |
| `PROJECT_NAME` | operator-facing AMC project name |
| `AMC_PROJECT_ID` | opaque AMC API and storage identity |
| `AUTO_MAGIC_CALIB_MS_PORT` | local API port |
| `AMC_ROOT` | AMC checkout and persistent projects root |

Step 4 retrieves current AMC output from:

```text
GET http://localhost:<AUTO_MAGIC_CALIB_MS_PORT>/v1/result/<AMC_PROJECT_ID>/mv3dt_result?result_type=amc
```

The ZIP contains `transforms.yml`; Step 4 owns download, validation, and ingest.

---

## 7. Browser lifetime and teardown

**LOCKED:** a dedicated browser process maps window lifetime to AMC lifetime.
Chromium-family browsers use app mode and a temporary profile; Firefox uses a
new instance and temporary profile. The profile is handed to the invoking user,
and the browser command is launched through:

```bash
sudo -u <invoking-user> -H env <desktop-environment> <browser> ...
```

The browser must never run as root. `DISPLAY`, `WAYLAND_DISPLAY`, `XAUTHORITY`,
and `DBUS_SESSION_BUS_ADDRESS` are propagated when present.

For standalone launches, normal window close, `SIGINT`, `SIGTERM`, and process
exit converge on a run-once `docker compose down` guard. `--keep-up`,
`--no-open`, and the installer lifecycle handoff to Step 4 intentionally leave
services running and do not arm an exit teardown. Headless interactive runs
print the URL and wait for Enter; headless non-interactive runs print the URL
and leave AMC running.

---

## 8. Launcher and verification

`run()` installs a stable copy of the frozen binary at
`<install_dir>/bin/mv3dt-installer` and writes an executable wrapper:

```bash
#!/usr/bin/env bash
exec "/opt/mv3dt/bin/mv3dt-installer" amc "$@"
```

The `amc` subcommand accepts:

| Flag | Behavior |
|---|---|
| `--project <name>` | explicit `PROJECT_NAME` |
| `--location-id <id>` | explicit `LOCATION_ID` |
| `--skip-pull` | skip explicit image pull |
| `--keep-up` | keep containers after browser close |
| `--down` | tear down without bring-up |
| `--host-ip <ip>` | override UI-to-MS host |
| `--no-open` | start without browser/hold |
| `--non-interactive` | never prompt |

`verify()` requires the wrapper, frozen binary, Compose minimum, NVIDIA Docker
runtime, and all Step 3 configuration keys except `AMC_PROJECT_ID`, which is
created only after an actual launch.

---

## 9. Developer quick-reference

- [ ] Ubuntu 24.04 installs `docker.io` and `docker-compose-v2` idempotently.
- [ ] Compose below `2.20.3` is rejected; legacy `docker-compose` is ignored.
- [ ] NVIDIA repository, toolkit, runtime, and GPU-container checks pass.
- [ ] Every Git, login, Compose, readiness, and project API failure is fatal.
- [ ] The NGC key is absent from argv, logs, and error messages.
- [ ] Tracked source edits are preserved and diagnosed; untracked AMC runtime
      data under `projects/` and `models/` is rerun-safe.
- [ ] UI/MS port collisions select and persist deterministic alternatives.
- [ ] A live AMC stack recovers its Docker-published ports, repairs stale
      configuration, and skips pull/start.
- [ ] Backend returns `code: 0`; UI returns HTTP `200`.
- [ ] A first launch that spends minutes downloading models reaches readiness
      without a timeout; an `auto-magic-calib-ms` container that stops ends the
      wait early.
- [ ] Browser process runs as the invoking user.
- [ ] `LOCATION_ID`, `PROJECT_NAME`, and `AMC_PROJECT_ID` survive reruns.
- [ ] Focused and full installer suites pass apart from documented host-only
      failures.

---

## 10. Out of scope

- Browser-driven upload, alignment, execution, and result review.
- Step 4 result download, ZIP validation, and DeepStream rendering.
- Step 5 multi-project registry and lifecycle management.
- AMC systemd supervision.
- Automatic VGGT commercial-model setup.

---

## 11. Implementation decomposition

| Unit | Branch | Files touched | Depends on | Wave |
|---|---|---|---|---|
| Step 3 AMC hardening | `feat/installer-step3-amc-hardening` | `installer/mv3dt_installer/steps/step3_amc_launcher.py`, `installer/tests/test_step3_amc_launcher.py`, `installer/plan/STEP-3-AMC-LAUNCHER.md` | None | 1 |

Step 4 must merge after this unit because its result API consumes the persisted
`AMC_PROJECT_ID`, `PROJECT_NAME`, `LOCATION_ID`, `AUTO_MAGIC_CALIB_MS_PORT`, and
`AMC_ROOT` contract defined in §6.2.

---

## References

NVIDIA's AMC 3.2.1 repository and official runtime documentation are the
authority for the repository pin, Compose layout, image authentication,
readiness response, and project/result API endpoints.

- [AutoMagicCalib 3.2.1 repository](https://github.com/NVIDIA-AI-IOT/auto-magic-calib/tree/0cfd2b790fd77598b0543340a65c2a0e1d192327) — pinned source and Compose files.
- [AMC setup skill](https://github.com/NVIDIA-AI-IOT/auto-magic-calib/blob/0cfd2b790fd77598b0543340a65c2a0e1d192327/skills/amc-setup-calibration-stack/SKILL.md) — Compose, NGC, and HTTP checks; its 120-second readiness budget is extended for first launch in §6.1.
- [AMC video-calibration skill](https://github.com/NVIDIA-AI-IOT/auto-magic-calib/blob/0cfd2b790fd77598b0543340a65c2a0e1d192327/skills/amc-run-video-calibration/SKILL.md) — project creation, project lookup, and MV3DT result endpoint.
- [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) — repository and Docker runtime configuration.

Repo files referenced:

- [`installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md) — shared framework, reporting, privilege, configuration, and step contracts.
- [`installer/plan/STEP-4-CALIB-OUTPUT-WIRING.md`](STEP-4-CALIB-OUTPUT-WIRING.md) — downstream calibration-result consumer.
- [`laptop/scripts/30_start_amc.sh`](../../laptop/scripts/30_start_amc.sh) — superseded developer harness launcher.
