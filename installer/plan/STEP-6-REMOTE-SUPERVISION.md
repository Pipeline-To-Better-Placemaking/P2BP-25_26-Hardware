# STEP 6 — 24/7 supervised pipeline + remote MQTT control (owner: DevD)

Status: NEW step, added after
[`STEP-5-PER-PROJECT-EXES.md`](STEP-5-PER-PROJECT-EXES.md). It specifies the
`step6_remote_supervision` module. It builds strictly on the contracts in
[`00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md) — the
step-module interface (§12), `Context` services, install-location config
(§11), the reporting/verify strings (§8.3–8.4), and the USER-ACTION display
(§9.3). Those contracts are **not** restated here; only Step 6's own scope is.

Step 5 leaves each project as a foreground `pipeline-<slug>` exe that runs
`deepstream-app` in the operator's shell (dies with the session; no restart;
no remote control). Step 5 §9 explicitly flagged supervised/always-on
pipelines as future work. **Step 6 is that future work, promoted to a real
step.** It turns each project pipeline into a boot-enabled systemd service and
adds a long-running control agent that lets a cloud webapp run / stop /
restart pipelines remotely over the existing Mosquitto/MQTT stack.

Scope, from the product owner (LOCKED):

1. The DeepStream pipeline runs **24/7** and is remotely **run / stopped /
   restarted**.
2. Remote control uses **JSON command packages** exchanged between the cloud
   webapp and the on-prem desktop.
3. Transport = **MQTT command topics** over the existing Mosquitto stack
   (TCP `1883` + WebSocket `9001`; the pipeline already publishes telemetry to
   `mv3dt/#`). The desktop makes an **outbound** connection to the broker,
   **subscribes** to command topics, and **publishes** status back. There is
   **no inbound-to-desktop API** (firewall-friendly for on-prem).
4. Supervision = **systemd**, mirroring the P2BP camera-node `services/`
   pattern (`Restart=on-failure`, `WantedBy=multi-user.target`, journald
   logging). Those units are being removed from this fork
   ([`DELETION-REVIEW` §3](DELETION-REVIEW.md#3-deletions-gated-on-the-harvest-the-jetson-tree)),
   so every convention borrowed from them is written out in full below.
5. Same framework contracts as Steps 1–5 (Step protocol, `Context`,
   `StepStatus`, the exact reporting strings, `install_dir` default
   `/opt/mv3dt`).
6. Anything beyond this — the webapp implementation, cloud auth
   infrastructure, credential issuance — is flagged for the human, not built
   here ([§8](#8-out-of-scope--flag-for-human)).

> **Control plane vs data plane.** Step 6 is the **control plane**: MQTT
> commands that run, stop, and restart pipelines. The **data plane** — the
> desktop's HTTP connection to the web app for registration/status, artifact
> upload, and web-app-initiated one-shot operations — is
> [`STEP-7`](STEP-7-WEBAPP-INTEGRATION.md), a separate opt-in step over a
> separate transport. The two are independent: either may be enabled alone, and
> neither reads the other's state. They overlap only in the per-unit
> `{Active, Sub}` field names, which
> [§C.5](#c5-status--heartbeat-json-schema-desktop--cloud) and
> [`STEP-7` §D.1](STEP-7-WEBAPP-INTEGRATION.md#d1-the-status-payload) must keep
> identical. See [`STEP-7` §H.3](STEP-7-WEBAPP-INTEGRATION.md#h3-relationship-to-step-6)
> for the full comparison.

> systemd unit fields, the enable/`daemon-reload` flow, `Restart=`,
> `WantedBy=`, journald `StandardOutput/Error`, and `systemctl is-active`
> status parsing are **OS-level conventions mirrored from the repo's existing
> P2BP camera-node `services/` tree and its `install.sh`** —
> **not** DeepStream facts. Only DeepStream-specific facts (the
> `deepstream-app` entry point, MV3DT `mv3dt/<LOCATION_ID>/*` telemetry, and
> the Gst-nvmsgbroker MQTT transport) are cited to the DS 9.1 docs. See
> [References](#9-references).

---

## 1. Inputs consumed and outputs produced

### 1.1 Inputs (from prior steps, via `Context`)

- `ctx.install_dir` / `ctx.conf` — install root (default `/opt/mv3dt`),
  resolved through `config.load()` (framework §11). Never hardcoded.
- **Step 5's project registry** at `<install_dir>/projects/registry.json`
  (Step 5 §4). Step 6 supervises exactly what Step 5 registered: each
  `slug` → one templated systemd instance `mv3dt-pipeline@<slug>.service`.
- Each project's `rendered_config` and `location_id` from the registry
  (Step 5 §4.2), used to build the per-instance `ExecStart`.
- The Step 5 `pipeline` subcommand of the frozen `mv3dt-installer` binary
  (Step 5 §3.3), refactored here into a systemctl controller ([§A.2](#a2-step-5-handoff--the-pipeline-slug-exe-becomes-a-systemctl-controller)).
- The Mosquitto broker installed by
  [`STEP-1` §3.2](STEP-1-PREREQUISITES.md#32-mosquitto-broker) from the bundled
  [`mv3dt.conf`](../../laptop/mosquitto/mv3dt.conf), TCP
  `1883` + WebSocket `9001`.

### 1.2 Outputs (what Step 6 writes)

- `/etc/systemd/system/mv3dt-pipeline@.service` — the templated instance unit
  ([§A.1](#a1-the-templated-per-project-instance-unit)).
- `/etc/systemd/system/mv3dt-agent.service` — the control-agent unit
  ([§B.1](#b1-the-agent-systemd-unit)).
- The control agent program + its env/config under `<install_dir>/` and
  `<install_dir>/agent/` ([§B](#b-the-control-agent-remote)).
- `/etc/polkit-1/rules.d/49-mv3dt-agent.rules` — the scoped authorization rule
  letting the agent start/stop/restart pipeline instances without root
  ([§B.1.1](#b11-the-polkit-rule-locked-form)).
- The remote-mode broker config drop-in (`password_file` / `acl_file` /
  bound + TLS listeners) written into `/etc/mosquitto/conf.d/`
  ([§D](#d-security-remote-control-must-be-authenticated)).
- No new `state.json` writes — Step 6 returns a `StepResult`; the framework
  owns all state persistence (framework §12.2).

All operator-facing files are `chown`ed to the invoking user/group; unit
files and broker config are root-owned (framework §9.2). Everything requiring
root goes through `ctx.run_root(...)` (framework §12.3).

---

## A. systemd supervision (local)

### A.1 The templated per-project instance unit

One template supervises every project. The systemd instance specifier
(`%i`) is the Step 5 `slug`, so `mv3dt-pipeline@north-lobby-2.service`
supervises the `north-lobby-2` project. This mirrors the single-template,
many-instances model and the field conventions of
the camera-node `tracker.service` — journald logging, `EnvironmentFile`,
`WantedBy=multi-user.target` — adapted for a long-running, auto-restarting
DeepStream pipeline. The full unit is given below, so nothing depends on that
file remaining in the tree.

`ExecStart` / `ExecStop` are **rendered per instance** because
`deepstream-app` needs the project's absolute config path. The installer
writes the template with `%i`-parameterized paths; the per-instance config is
resolved from the registry at install time (or by the pipeline subcommand at
`ExecStart` time — see [§A.2](#a2-step-5-handoff--the-pipeline-slug-exe-becomes-a-systemctl-controller)).

File `/etc/systemd/system/mv3dt-pipeline@.service`. This is a genuine systemd
**template** unit — `%i` is systemd's own instance specifier, resolved
natively at `systemctl start mv3dt-pipeline@<slug>` time, never touched by
the installer. `User=`, below, is a different kind of substitution: there is
no systemd specifier for "the invoking user of the installer," so that one
field is filled in once at install time via `systemd.render_unit()`
(framework module, [`00` §4.2](00-FRAMEWORK-AND-BOOTSTRAP.md#42-locating-and-staging-bundled-assets-at-runtime)
staging + `@MARKER@`-style substitution, same mechanism
[`STEP-4` §4.5](STEP-4-CALIB-OUTPUT-WIRING.md#45-re-ingest-on-later-recalibrations-systemd-path-unit)
uses for the ingest units) — the bundled template on disk is
`mv3dt-pipeline@.service.in` with an `@USER@` marker, not the literal `%i`
unit shown rendered below:

```ini
[Unit]
Description=MV3DT DeepStream pipeline (%i)
After=network-online.target mosquitto.service
Wants=network-online.target
Requires=mosquitto.service

[Service]
Type=simple
User=@USER@                             ; rendered to the invoking user (framework §9.2)
WorkingDirectory=/opt/mv3dt/deepstream/%i

# ExecStart ports 50_start_pipeline.sh: ensure mosquitto up, source the DS env,
# then exec deepstream-app -c <rendered per-project config>. Implemented once,
# in the pipeline subcommand, invoked in a non-interactive service context.
ExecStart=/opt/mv3dt/bin/mv3dt-installer pipeline --project-slug %i --service-exec
ExecStop=/opt/mv3dt/bin/mv3dt-installer pipeline --project-slug %i --stop

# 24/7 crash recovery. on-failure (not always) per the LOCKED decision, so a
# clean --stop / systemctl stop does not trigger a restart loop.
Restart=on-failure
RestartSec=5
StartLimitIntervalSec=60
StartLimitBurst=5

# Graceful teardown mirrors 99_stop_all.sh: SIGTERM, then SIGKILL after grace.
KillSignal=SIGTERM
TimeoutStopSec=10

EnvironmentFile=/opt/mv3dt/installer.conf
Environment=PYTHONUNBUFFERED=1

StandardOutput=journal
StandardError=journal
LogRateLimitIntervalSec=30s
LogRateLimitBurst=200

[Install]
WantedBy=multi-user.target
```

Notes:

- **`ExecStart` = ported `50_start_pipeline.sh`.** The service does not run
  the bash script directly; it invokes the Step 5 `pipeline` subcommand in a
  new `--service-exec` mode that runs the same steps
  [`50_start_pipeline.sh`](../../laptop/scripts/50_start_pipeline.sh) does —
  `ensure_mosquitto()` (lines 112–132), `. /etc/profile.d/deepstream.sh`
  (lines 201–210), then `exec deepstream-app -c <rendered_config>` (lines
  374–401) — but **non-interactively**: no ping-sweep gating, no
  validation-helper banner (those are for the operator's foreground TTY), and
  logging to journald instead. `deepstream-app -c <config>` is the DS 9.1
  entry point (see [References](#9-references)).
- **`ExecStop` = ported `99_stop_all.sh`.** SIGTERM → wait → SIGKILL
  ([`99_stop_all.sh`](../../laptop/scripts/99_stop_all.sh) lines 62–78);
  systemd's `KillSignal=SIGTERM` + `TimeoutStopSec=10` gives the same
  graceful-then-forced behavior, so `--stop` only needs to stop the one
  pipeline instance (it must NOT stop the shared broker — that is
  `mosquitto.service`, left running for other instances, matching Step 5
  §3.4's default `--stop`).
- **`Requires=mosquitto.service`** guarantees the broker the pipeline
  publishes MV3DT telemetry to is up before the pipeline starts (the
  service-context equivalent of `ensure_mosquitto()`).
- `User=` is rendered to the invoking user (framework §9.2) via
  `systemd.render_unit()`'s `@USER@` marker, so `deepstream-app`, the DS
  env, and GPU access run as the same user the rest of the install used, not
  root. This is a one-time install-time substitution, independent of `%i`.
- `WorkingDirectory=/opt/mv3dt/deepstream/%i` matches the Step 5 registry
  layout (`<install_dir>/deepstream/<slug>/`).
- Hardening directives from `tracker.service` (`ProtectSystem`,
  `PrivateTmp`, etc.) are **intentionally omitted by default** because
  `deepstream-app` needs broad GPU/device/driver access; add them only after
  validating they don't break CUDA/NVENC. Flagged, not applied.

### A.2 Step 5 handoff — the `pipeline-<slug>` exe becomes a systemctl controller

This is the crux of the Step 5 → Step 6 handoff. **Step 5 §3 must change** so
the per-project exe stops running `deepstream-app` in the foreground and
instead drives systemd.

What changes in Step 5 §3.3 ("What the exe does at runtime"):

- **`start` (default):** instead of `exec deepstream-app -c <config>`, the
  exe/subcommand runs `systemctl start mv3dt-pipeline@<slug>` (via
  `ctx.run_root`). The pipeline is now a supervised service, not a child of
  the operator's shell.
- **`--stop`:** instead of `pkill deepstream-app`, run
  `systemctl stop mv3dt-pipeline@<slug>` (the unit's `ExecStop` does the
  graceful SIGTERM→SIGKILL). `--stop-all` still additionally stops the AMC
  stack / broker as in Step 5 §3.4.
- **New internal `--service-exec` mode:** the actual
  `ensure-mosquitto → source env → exec deepstream-app` logic (formerly the
  whole of `start`) moves behind `--service-exec`, which is what the unit's
  `ExecStart` calls **inside** the service. Operators never call it directly.
- **`--foreground` escape hatch (retained for debugging):** runs the old
  Step 5 behavior — `deepstream-app` in the current TTY, with the
  ping-sweep and validation banner — bypassing systemd entirely. `--dry-run`
  still just prints the resolved `deepstream-app -c <config>` command
  (Step 5 §7.3). These keep the debugging story intact when the service
  won't start.

Net effect: the thin wrapper `<install_dir>/bin/pipeline-<slug>` (Step 5 §3.2)
is **unchanged on disk** — it still `exec`s
`mv3dt-installer pipeline --project "<NAME>" "$@"`. Only the subcommand's
behavior changes: `start`→`systemctl start`, `--stop`→`systemctl stop`,
foreground kept behind `--foreground`. Step 5's `preflight`/`verify` gain a
note that supervised mode is active once Step 6 is COMPLETE.

Step 5 §9's "systemd supervision is out of scope" paragraph must be updated to
point at this step (it is no longer out of scope).

### A.3 Boot-enable, per-project lifecycle, and reconciliation

- **Enable for 24/7 / boot:** for each registered project, Step 6 runs
  `systemctl enable mv3dt-pipeline@<slug>` so the instance is `WantedBy`
  `multi-user.target` and comes back after a reboot. `enable --now` both
  enables and starts it. This is the "runs 24/7" guarantee.
- **On project add (Step 5 new-project flow, §5.1):** after Step 5 `upsert`s
  the registry and generates the exe, Step 6's hook enables + starts the new
  instance (only when Step 6 is COMPLETE / opt-in active — see
  [§E.2](#e2-gating-opt-in)).
- **On project removal (reconciles Step 5 §5.4):** Step 5's
  `reconcile_registry` removes install-side artifacts when a project is
  deleted in the AMC GUI. Step 6 adds one reconciliation action to that path:
  before deleting the exe/config, run
  `systemctl disable --now mv3dt-pipeline@<slug>` to stop + boot-disable the
  instance. The instance template file itself
  (`mv3dt-pipeline@.service`) is shared and stays; only the per-slug
  enablement symlink under `multi-user.target.wants/` goes away
  (`disable` removes it). If the removed project was the last one, the
  template may remain installed harmlessly.

### A.4 Installer integration (mirror `install.sh`)

`run()` mirrors the unit-install flow of the P2BP camera-node `install.sh`
(copy units → `daemon-reload` → per-unit `enable`), reproduced here in full
since that script is being removed
([`DELETION-REVIEW` §3](DELETION-REVIEW.md#3-deletions-gated-on-the-harvest-the-jetson-tree)):

1. Render `mv3dt-pipeline@.service.in` and `mv3dt-agent.service.in`
   (bundled templates staged via `ctx.asset_path`, framework §4.2) through
   `systemd.render_unit()` — substituting `@USER@` with the invoking user —
   then install the rendered text into `/etc/systemd/system/` via
   `systemd.install_unit()` (content-idempotent, root-owned `0644`; see the
   callout below). `systemd.py` today only implements the STEP-4-specific
   `render_ingest_units()` wrapper around the generic `render_unit()`
   primitive; Step 6 calls `render_unit()` directly (or adds its own thin
   wrapper) since `mv3dt-pipeline@.service.in` and `mv3dt-agent.service.in`
   are a different template pair. Neither `.in` template exists yet under
   `installer/mv3dt_installer/assets/systemd/` — creating them is part of
   implementing this step.
2. `systemctl daemon-reload` — **required** before any `enable`, or systemd
   acts on a stale view of the unit files.
3. For each project in `registry.json`:
   `systemctl enable --now mv3dt-pipeline@<slug>`, keyed on registry slugs.
   Boot-enablement is the installer's job alone: the polkit rule in
   [§B.1.1](#b11-the-polkit-rule-locked-form) deliberately withholds `enable`
   from the agent, so a pipeline persists across reboots only because this
   step enabled it.
4. `systemctl enable --now mv3dt-agent.service` ([§B](#b-the-control-agent-remote)).
5. Report each touched unit via the framework reporters ([§E](#e-framework-integration)).

> **REQUIRED — go through the shared `systemd.py` helper.** Unit rendering,
> installation, `daemon-reload`, and `enable` are **not** open-coded
> `subprocess` calls in this step. They use the framework's `systemd.py`, which
> installs content-idempotently (read-compare-write, root-owned `0644`) and
> returns whether the file changed, so the caller picks `report_installed` vs
> `report_already_installed` with the exact framework §8.3 strings; refuses to
> render a unit that still carries an unsubstituted placeholder; refuses to
> enable a unit whose `ExecStart` binary is absent; and routes every
> `systemctl` invocation through an injected runner, which production callers
> set to `ctx.run_root`. [`STEP-4` §4.5](STEP-4-CALIB-OUTPUT-WIRING.md#45-re-ingest-on-later-recalibrations-systemd-path-unit)
> uses the same helper for its ingest path units, so the two steps cannot
> drift on unit-install semantics.

**Sudo/root:** all of the above require root; Step 6 uses `ctx.run_root(...)`
(framework §12.3). The installer already runs as root under plain `sudo`
(framework §5.1, §9.1 — `-E` is optional, only needed to pass a pre-set
`NGC_API_KEY` through the environment), so no extra privilege prompt is
needed.

---

## B. The control agent (remote)

A long-running, supervised daemon — `mv3dt-agent` — that connects **outbound**
to the MQTT broker, subscribes to a command topic, validates + executes JSON
commands by calling `systemctl` on the `mv3dt-pipeline@<slug>` instances, and
publishes acks + periodic status. It is the on-prem half of the remote-control
loop; the cloud webapp is the other half (out of scope, [§8](#8-out-of-scope--flag-for-human)).

### B.0 Language + delivery

- **Python**, consistent with the installer (framework §2) and the existing
  P2BP camera-node agents. MQTT via `paho-mqtt` (add to
  `installer/pyproject.toml`; keep the dep list minimal per framework §4.1).
- Ships **as part of the installer binary / assets**: the agent is a
  subcommand of the frozen binary — `mv3dt-installer agent` — so it needs no
  separate packaging, exactly like the `pipeline` subcommand (Step 5 §3.2).
  The unit's `ExecStart` calls `/opt/mv3dt/bin/mv3dt-installer agent`.

### B.1 The agent systemd unit

`/etc/systemd/system/mv3dt-agent.service` (mirrors `heartbeat.service`
conventions — `Restart`, `RestartSec`, `EnvironmentFile`, journald, boot
enable):

```ini
[Unit]
Description=MV3DT remote control agent (MQTT command consumer)
After=network-online.target mosquitto.service
Wants=network-online.target mosquitto.service

[Service]
Type=simple
User=@USER@                             ; rendered to the invoking user, via systemd.render_unit()
ExecStart=/opt/mv3dt/bin/mv3dt-installer agent
WorkingDirectory=/opt/mv3dt

Restart=on-failure
RestartSec=5
StartLimitIntervalSec=60
StartLimitBurst=5

EnvironmentFile=/opt/mv3dt/installer.conf
EnvironmentFile=/opt/mv3dt/agent/agent.env      ; broker creds/endpoint (chmod 600, §D)
Environment=PYTHONUNBUFFERED=1

StandardOutput=journal
StandardError=journal
LogRateLimitIntervalSec=30s
LogRateLimitBurst=200

[Install]
WantedBy=multi-user.target
```

#### Granting the agent authority over the units (LOCKED: scoped polkit rule)

The agent runs as the invoking user but must cause `systemctl
start/stop/restart` on the `mv3dt-pipeline@*` instances. Two approaches were
weighed; **Option 1 is LOCKED, in its polkit form specifically**
([§B.1.1](#b11-the-polkit-rule-locked-form)). Option 2 is retained below
because its trade-offs are the reason for the choice, not because it remains
open.

**Option 1 — scoped authorization drop-in (CHOSEN).** The installer writes a
root-owned rule restricted to the verbs `start`/`stop`/`restart` on
`mv3dt-pipeline@*`, and the agent calls `systemctl` directly.

- *Pros:* direct and obvious; one small greppable file; the call is
  **synchronous**, so the exit status tells the agent whether the action
  actually succeeded and `ok` in §C.4 can be reported truthfully with no extra
  logic; no additional unit files.
- *Cons:* grants a non-root, network-facing process real `systemctl`
  authority — and this process consumes commands from a broker; the rule is
  easy to widen by accident when someone adds a unit (`mv3dt-*` instead of
  `mv3dt-pipeline@*` is one careless edit away); needs care to pin both the
  verbs and the unit pattern.

**Option 2 — flag file plus a `.path` unit (REJECTED, see the rationale after
this block).** The agent writes
`/run/mv3dt/pipeline-<slug>.enabled`; a `.path` unit watches that file and
triggers a `oneshot` toggle service that does the privileged work. This is the
pattern the P2BP camera-node tree already uses for exactly this problem, ported
here (its unit files are inlined below because that tree is being removed —
[`DELETION-REVIEW` §3](DELETION-REVIEW.md#3-deletions-gated-on-the-harvest-the-jetson-tree)):

```ini
# /etc/systemd/system/mv3dt-pipeline@.path
[Unit]
Description=Enable MV3DT pipeline (%i) when its flag exists

[Path]
PathExists=/run/mv3dt/pipeline-%i.enabled
PathChanged=/run/mv3dt/pipeline-%i.enabled
Unit=mv3dt-pipeline-toggle@%i.service

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/mv3dt-pipeline-toggle@.service
[Unit]
Description=Start/stop the MV3DT pipeline (%i) based on its flag
StartLimitIntervalSec=0

[Service]
Type=oneshot
ExecStart=/bin/sh -ec 'if [ -e /run/mv3dt/pipeline-%i.enabled ]; then \
  /usr/bin/systemctl start mv3dt-pipeline@%i.service; else \
  /usr/bin/systemctl stop mv3dt-pipeline@%i.service; fi'
```

Under this option `mv3dt-pipeline@.service` also gains
`ConditionPathExists=/run/mv3dt/pipeline-%i.enabled` as a second guard, so a
manual `systemctl start` of a disabled project is a no-op rather than a
surprise.

- *Pros:* the agent needs **zero** privilege — only write access to its own
  runtime directory, which is the strongest property on offer for a process
  that takes instructions from a network; the pattern is already proven in this
  repo; state is trivially inspectable (`ls /run/mv3dt/`); the extra
  `ConditionPathExists` guard comes free.
- *Cons:* the trigger is **asynchronous** — the agent cannot observe the
  outcome directly and must settle-and-poll `is-active` before publishing
  §C.4's `ok` and `state` (see [§C.6](#c6-action--systemctl-mapping-idempotency-unknown-projects));
  it adds two unit templates; and `restart` has no natural flag-file
  expression, needing either a toggle-off/toggle-on with a settle in between or
  a separate `.restart` path unit.

> **Carve-out required under Option 2.** Units driven by a `.path` must **not**
> themselves be `systemctl enable`d — the `.path` unit owns their lifecycle,
> and enabling both makes the boot-time state race the flag file. The
> camera-node `install.sh` handles exactly this by skipping such units in its
> enable loop.

**Why Option 2 was rejected.** `/run` is a **tmpfs**: the flag files do not
survive a reboot. Option 2's boot-enable story therefore requires a *fourth*
unit — a boot-time reconciler that reads `registry.json` and re-creates one
flag per enabled project — before
[§A.3](#a3-boot-enable-per-project-lifecycle-and-reconciliation)'s "runs 24/7"
guarantee holds at all. That is three unit templates plus a reconciler to avoid
a single authorization rule, and it buys an *asynchronous* trigger that makes
truthful acking harder ([§C.4](#c4-result--ack-json-schema-desktop--cloud)) and
has no natural expression for `restart`. The zero-privilege property is real
and attractive, but it is not worth four moving parts and a weaker status
contract.

### B.1.1 The polkit rule (LOCKED form)

**REQUIRED — polkit, not sudoers.** A sudoers entry is the obvious way to write
Option 1 and the wrong one: sudoers wildcards match **across argument
boundaries**, so a rule ending in `mv3dt-pipeline@*` also authorizes
`systemctl start mv3dt-pipeline@a.service unrelated.service` — extra arguments
slip past the glob. Polkit receives the unit name and the verb as *structured*
values, so both can be matched exactly.

Because the agent runs as a non-root user, its plain `systemctl` call already
goes through polkit over D-Bus — no `sudo`, no wrapper, and the call stays
**synchronous**, which is what preserves the truthful `ok` in
[§C.4](#c4-result--ack-json-schema-desktop--cloud).

File `/etc/polkit-1/rules.d/49-mv3dt-agent.rules` (root-owned, `0644`; the
`49-` prefix sorts it before the distribution defaults in `50-*.rules`):

```javascript
polkit.addRule(function (action, subject) {
    if (action.id !== "org.freedesktop.systemd1.manage-units") {
        return polkit.Result.NOT_HANDLED;
    }
    if (subject.user !== "<INVOKING_USER>") {      // rendered, framework §9.2
        return polkit.Result.NOT_HANDLED;
    }
    var unit = action.lookup("unit");
    var verb = action.lookup("verb");
    // Anchored: no prefix or suffix may be appended to the instance name.
    // The charset mirrors the Step 5 slug rules (§3.1 there): [a-z0-9-], <=64.
    if (/^mv3dt-pipeline@[a-z0-9-]{1,64}\.service$/.test(unit) &&
        (verb === "start" || verb === "stop" || verb === "restart")) {
        return polkit.Result.YES;
    }
    return polkit.Result.NOT_HANDLED;
});
```

Four properties, each load-bearing:

- **Anchored regex** (`^…$`) — the failure mode that sinks the sudoers form
  cannot occur; no other unit name matches, however it is decorated.
- **Explicit verb allowlist** — `enable`, `disable`, and `mask` are *not*
  granted. Boot-enablement stays the installer's job
  ([§A.3](#a3-boot-enable-per-project-lifecycle-and-reconciliation)), so a
  compromised agent cannot make a pipeline persist across reboots.
- **`Result.YES`, not `AUTH_ADMIN`** — the agent runs headless under systemd
  with no polkit authentication agent available; anything requiring interaction
  would simply fail.
- **`NOT_HANDLED` on every other path** — the rule only ever *adds* a narrow
  permission; normal administrative behavior for every other unit and user is
  untouched.

`verify()` ([§E.1](#e1-lifecycle)) must assert the rule file exists and that a
negative case is actually denied — `systemctl start mosquitto` as the agent
user must fail. A rule that grants too much still passes a positive-only test,
which is precisely the bug this form exists to prevent.

> **Portability note (flagged).** This is the polkit `.rules` JavaScript
> format, which requires polkit ≥ 0.106; Ubuntu 24.04 ships polkit 124, so it
> applies. On a host where the rule cannot be installed, the fallback is a
> sudoers entry that enumerates the registry slugs **explicitly** — one full
> command line per project per verb, with no wildcard — regenerated whenever
> the registry changes. It is more churn, and it is the only safe way to write
> the sudoers form.

### B.2 Broker location + connection topology

The agent must reach a broker that the cloud webapp can also reach. Two
topologies, both described; the installer picks a sensible default and flags
the cloud endpoint as human-provided:

- **(1) Local broker + MQTT bridge to cloud (DEFAULT, recommended).** The
  agent connects to the **local** Mosquitto (`127.0.0.1:1883`) — the same
  broker the DeepStream pipeline already publishes MV3DT telemetry to — and
  Mosquitto is configured with an **MQTT bridge** that forwards the
  `mv3dt/<HOST_ID>/#` command/status topics to/from the cloud broker. This is
  the clean way to reach a cloud webapp: the pipeline, the agent, and local
  `mosquitto_sub` inspectors all keep talking to localhost, and exactly one
  outbound TLS connection (the bridge) crosses the firewall. It is
  firewall-friendly (outbound-only, LOCKED decision 3) and keeps the local
  telemetry bus unchanged.

  Bridge drop-in (`/etc/mosquitto/conf.d/mv3dt-bridge.conf`, written by the
  installer in remote mode; endpoint from config, not hardcoded):

  ```conf
  connection cloud-bridge
  address ${MV3DT_CLOUD_BROKER_HOST}:${MV3DT_CLOUD_BROKER_PORT}
  bridge_protocol_version mqttv311
  # Outbound: our status/telemetry up to cloud. Inbound: commands down to us.
  topic mv3dt/${HOST_ID}/status  out 1
  topic mv3dt/${HOST_ID}/cmd/result out 1
  topic mv3dt/${HOST_ID}/cmd     in  1
  bridge_cafile /etc/mosquitto/certs/cloud-ca.crt
  remote_username ${MV3DT_CLOUD_BROKER_USER}
  remote_password ${MV3DT_CLOUD_BROKER_PASS}
  ```

- **(2) Agent connects directly to the cloud broker.** The agent opens an
  outbound TLS MQTT connection straight to the cloud broker and subscribes to
  its command topic there; no local bridge. Simpler config, but the local
  telemetry bus and the command bus are then split across two brokers, and
  every consumer that wants both must connect to both. Use this only when
  there is no local broker to bridge (not our case — the repo already runs
  Mosquitto).

**Default: topology (1)** — reuse the existing local Mosquitto and add a
bridge, because the repo already standardizes on a local broker for MV3DT
telemetry. The **cloud broker endpoint + credentials are human-provided
config**, never hardcoded: they live in
[`laptop/config/laptop.env`](../../laptop/config/laptop.env.example) /
`<install_dir>/installer.conf` as `MV3DT_CLOUD_BROKER_HOST`,
`MV3DT_CLOUD_BROKER_PORT`, `MV3DT_CLOUD_BROKER_USER`,
`MV3DT_CLOUD_BROKER_PASS` (secret → `agent.env`, `chmod 600`, §D). If they are
absent, Step 6 stays in **local-only mode** (agent runs against localhost, no
bridge) and flags the missing cloud endpoint as a USER-ACTION item.

> The DS pipeline's own MQTT publish path (Gst-nvmsgbroker →
> `libnvds_mqtt_proto.so`, `conn-str = localhost;1883`) is unchanged by
> Step 6; the agent is an independent MQTT client on the same broker. See
> [References](#9-references).

---

## C. JSON command + status schema (the crux)

### C.1 `<HOST_ID>` derivation

`<HOST_ID>` identifies this desktop on the shared bus and namespaces its
topics. Derivation, in priority order:

1. `MV3DT_HOST_ID` from `installer.conf` if the operator set one (stable,
   human-readable, recommended for fleets).
2. else the systemd machine id (`/etc/machine-id`), which is stable across
   reboots and unique per install.
3. else the hostname (`socket.gethostname()`), sanitized with the same slug
   rules as Step 5 §3.1.

`<HOST_ID>` is resolved once at agent start, logged, and included in every
status payload. It is distinct from a project `slug` (one host supervises many
project slugs).

### C.2 Topic layout

All command/control topics are namespaced under the host so one broker can
serve a fleet, and the ACL (§D) can pin the agent to its own subtree:

| Topic | Direction | Payload |
| --- | --- | --- |
| `mv3dt/<HOST_ID>/cmd` | cloud → desktop (agent **subscribes**) | Command JSON (§C.3) |
| `mv3dt/<HOST_ID>/cmd/result` | desktop → cloud (agent **publishes**) | Result/ack JSON (§C.4) |
| `mv3dt/<HOST_ID>/status` | desktop → cloud (agent **publishes**, periodic + on-change) | Status JSON (§C.5) |

This is deliberately parallel to the DS 9.1 MV3DT telemetry topics
(`mv3dt/<LOCATION_ID>/sv3d`, `mv3dt/<LOCATION_ID>/fused`; see
[References](#9-references)) — same `mv3dt/<id>/<channel>` shape — but keyed on
`<HOST_ID>` (control plane) rather than `<LOCATION_ID>` (telemetry plane), so
the two planes never collide. Status is published **retained** so a newly
connected webapp immediately sees the last-known host state.

### C.3 Command JSON schema (cloud → desktop)

```json
{
  "action": "restart",
  "project": "North Lobby #2",
  "request_id": "6f1c2b8e-...-a3",
  "ts": "2026-07-01T22:41:12Z",
  "args": {}
}
```

- `action` — one of `run` | `stop` | `restart` | `status` | `list`.
  - `run` / `stop` / `restart` map to the pipeline instance for `project`.
  - `status` returns the current status payload (§C.5) as a result.
  - `list` returns the set of known projects (registry slugs + states).
- `project` — the original `PROJECT_NAME` **or** its `slug` (the agent
  resolves either via the registry, Step 5 §4.2). Required for `run`/`stop`/
  `restart`; ignored for `list`; optional for `status` (omit = whole host).
- `request_id` — client-generated unique id (UUID); echoed in the result for
  correlation. Required.
- `ts` — ISO-8601 UTC command timestamp. Required.
- `args` — optional action-specific object (reserved; e.g. a future
  `{"preview": true}`). Unknown keys are ignored.

### C.4 Result / ack JSON schema (desktop → cloud)

Published to `mv3dt/<HOST_ID>/cmd/result` once per handled command:

```json
{
  "request_id": "6f1c2b8e-...-a3",
  "ok": true,
  "action": "restart",
  "project": "north-lobby-2",
  "state": "active",
  "enabled": "enabled",
  "error": null,
  "ts": "2026-07-01T22:41:13Z"
}
```

- `request_id` — echoes the command (§C.3).
- `ok` — `true` if the systemctl call succeeded (and, for `run`/`restart`,
  the instance reached `active`).
- `state` — post-action `systemctl is-active` value:
  `active` | `activating` | `inactive` | `failed` | `unknown`.
- `enabled` — `systemctl is-enabled` value: `enabled` | `disabled` |
  `static`.
- `error` — `null` on success, else a short human string (e.g.
  `unknown project`, `systemctl start failed: <rc>`).
- `ts` — result timestamp (UTC).

> **Truthfulness of `ok`.** Under the LOCKED polkit design
> ([§B.1.1](#b11-the-polkit-rule-locked-form)) the `systemctl` call is
> **synchronous**, so a non-zero exit is an unambiguous failure and `ok` can
> report it directly — one of the main reasons that design was chosen. A zero
> exit is still not the whole story for `run`/`restart`, because systemd
> returns as soon as the job is *queued*: `ok` additionally requires the
> instance to have reached `active`, which is what the settle-and-poll in
> [§C.6](#c6-action--systemctl-mapping-idempotency-unknown-projects)
> establishes.

### C.5 Status / heartbeat JSON schema (desktop → cloud)

Published periodically (default 15 s, configurable) and on state change to
`mv3dt/<HOST_ID>/status`.

**REQUIRED — field names are shared with the HTTP plane.** The per-unit state
map (`{Active, Sub}`) and the GPU/memory block use the same field names as
[`STEP-7` §D.1](STEP-7-WEBAPP-INTEGRATION.md#d1-the-status-payload), so the web
app can consume either plane with one model. Changing a name here without
changing it there silently forks the contract. The payload is **not** a blind
copy of the HTTP one: it is keyed by project slug and adds the
enable/uptime/exit fields that only supervision needs.

```json
{
  "host_id": "desk-lab-01",
  "agent_version": "6.0.0",
  "ts": "2026-07-01T22:41:12Z",
  "pipelines": {
    "north-lobby-2": {
      "active": "active",
      "sub": "running",
      "enabled": "enabled",
      "uptime_s": 84213,
      "last_exit_code": 0,
      "restarts": 0
    },
    "west-dock": {
      "active": "failed",
      "sub": "failed",
      "enabled": "enabled",
      "uptime_s": 0,
      "last_exit_code": 1,
      "restarts": 5
    }
  },
  "gpu": {
    "utilization_pct": 61,
    "memory_used_mb": 4210,
    "temperature_c": 57
  }
}
```

- `pipelines.<slug>.active` / `sub` — parsed line-wise from

  ```bash
  systemctl show mv3dt-pipeline@<slug> -p ActiveState -p SubState
  ```

  `active` ∈ `active` | `activating` | `inactive` | `failed`.
  **REQUIRED:** any failure of that call — non-zero exit, a ~5 s timeout, or a
  unit that does not exist — yields `{"Active": "unknown", "Sub": "unknown"}`
  rather than raising. One unreadable unit must never prevent the status
  payload from being published. Same rule as
  [`STEP-7` §D.1](STEP-7-WEBAPP-INTEGRATION.md#d1-the-status-payload).
- `enabled` — `systemctl is-enabled mv3dt-pipeline@<slug>`.
- `uptime_s` / `last_exit_code` / `restarts` — from `systemctl show`
  properties (`ActiveEnterTimestamp`, `ExecMainStatus`, `NRestarts`).
- `gpu` — optional, best-effort from
  `nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu
  --format=csv,noheader,nounits` (the same query the Step 5 validation banner
  prints). Omitted/`null` if `nvidia-smi` fails.
- `agent_version` — the installer/step version string.

The set of projects is enumerated from `registry.json` at each publish, so
projects added/removed via Step 5 are reflected without restarting the agent.

### C.6 Action → systemctl mapping, idempotency, unknown projects

| `action` | effective systemctl call (invoked per the [§B.1](#b1-the-agent-systemd-unit) decision) | Idempotent? |
| --- | --- | --- |
| `run` | `systemctl start mv3dt-pipeline@<slug>` | yes — starting an already-`active` unit is a no-op |
| `stop` | `systemctl stop mv3dt-pipeline@<slug>` | yes — stopping an `inactive` unit is a no-op |
| `restart` | `systemctl restart mv3dt-pipeline@<slug>` | yes — starts if stopped, restarts if running |
| `status` | `systemctl show`/`is-active`/`is-enabled` (read-only) | yes |
| `list` | read `registry.json` + `is-active` per slug (read-only) | yes |

- **Idempotency:** every action is safe to repeat. The agent keys de-dup on
  `request_id` within a short TTL window so a redelivered MQTT message (QoS 1
  at-least-once) does not double-execute; the cached result is re-published.
- **Unknown project:** if `project` resolves to no registry slug, the agent
  does **not** call systemctl; it publishes a result with `ok: false`,
  `error: "unknown project"`, and `state: "unknown"`. It never creates units
  on the fly (unit creation is Step 5's registry-driven install path only).
- **`run`/`restart` boot-persistence:** `run` starts the instance but does
  not implicitly `enable` it; boot-enable is owned by the installer
  ([§A.3](#a3-boot-enable-per-project-lifecycle-and-reconciliation)). A future
  `args:{"enable": true}` could expose this, flagged not built.
- **Malformed command:** JSON that fails schema validation (missing
  `action`/`request_id`/`ts`, unknown `action`) yields a result with
  `ok: false`, `error: "invalid command: <reason>"`, and no systemctl call.
- **Settle-and-poll (REQUIRED for `run` and `restart`).** `systemctl start`
  returns once the job is queued, not once the unit is up, and a DeepStream
  pipeline takes seconds to initialize CUDA and open eight RTSP streams. After
  issuing the call, poll `systemctl is-active` until it leaves the transitional
  `activating` / `deactivating` states or a timeout (~10 s, matching the unit's
  `TimeoutStopSec`) elapses, and only then publish the §C.4 result. A unit
  still `activating` at the deadline is reported as-is with `ok: false` and
  `error: "timed out waiting for active"` — never optimistically as `active`.
  `stop` needs the same treatment for the symmetric reason.

---

## D. Security (remote control MUST be authenticated)

The committed [`mv3dt.conf`](../../laptop/mosquitto/mv3dt.conf) uses
`allow_anonymous true` on `1883 0.0.0.0` + `9001` — the deliberate
"simple-testing" posture (its own header says so). For **remote control that
can start/stop production pipelines, this is insufficient**: anyone reaching
the broker could command the desktop. Step 6 requires the hardened posture
from [`DEEPSTREAM-SETUP.md`](../../laptop/docs/DEEPSTREAM-SETUP.md) §6
"Production hardening" (`password_file`, `acl_file`, bound listeners) whenever
remote mode is enabled.

- **Authenticated broker creds for the agent.** In remote mode the installer
  writes a remote-mode drop-in
  (`/etc/mosquitto/conf.d/mv3dt-remote.conf`) setting `allow_anonymous false`
  with a `password_file` and `acl_file`, exactly as DEEPSTREAM-SETUP.md §6
  shows:

  ```conf
  listener 1883 127.0.0.1
  password_file /etc/mosquitto/passwd
  acl_file /etc/mosquitto/aclfile
  allow_anonymous false
  ```

  The agent authenticates with a dedicated username/password stored in
  `<install_dir>/agent/agent.env` (`chmod 600`, chowned to the invoking user,
  gitignored like `laptop/config/laptop.env` per framework §10.1).

- **ACL restricting the agent to its own subtree.** Extend the minimal ACL
  from DEEPSTREAM-SETUP.md §6 to pin the agent to only `mv3dt/<HOST_ID>/#`:

  ```conf
  user mv3dt-agent
  topic readwrite mv3dt/<HOST_ID>/#
  ```

  So a compromised or misconfigured agent cannot command other hosts, and the
  cloud creds (used by the bridge / webapp) are separately scoped.

- **TLS for any non-localhost / cloud hop.** The local listener is bound to
  `127.0.0.1` (localhost stays plaintext, matching the hardened §6 snippet);
  **any hop that leaves the machine — the MQTT bridge to the cloud broker, or
  a direct agent→cloud connection — MUST use TLS** (`bridge_cafile` /
  client TLS to the cloud broker). Cloud creds + CA cert are human-provided
  (§B.2), never committed.

- **This is a broker config change the installer makes for remote mode.**
  Enabling auth means rewriting/adding the Mosquitto drop-in and
  `systemctl restart mosquitto` — the installer does this only when remote
  mode is opted in ([§E.2](#e2-gating-opt-in)), because it breaks the
  anonymous local inspectors the testing harness relies on. In local-only
  mode the broker is left at the simple-testing posture.

- **Credential provisioning is a human decision (flagged).** Who issues the
  agent's broker password, the cloud broker account, and the TLS certificates
  (and their rotation) is **not** decided here — the installer can generate a
  local `mosquitto_passwd` entry for the agent, but the cloud-side account and
  CA trust are provisioned by whoever operates the cloud webapp. Surfaced as a
  USER-ACTION block ([§E](#e-framework-integration)).

---

## E. Framework integration

Implements the `Step` protocol (framework §12.1).
`id = "step6_remote_supervision"`,
`title = "Remote supervision"`, `order = 6`.

### E.1 Lifecycle

#### `preflight(ctx)`

- Confirm **Step 5 is `COMPLETE`** and `registry.json` exists with **at least
  one registered project** — nothing to supervise otherwise
  (`USER_ACTION_REQUIRED` pointing back at Step 5 to create/calibrate a
  project).
- Confirm **Mosquitto is reachable** — `systemctl is-active mosquitto` (the
  `ensure_mosquitto` precondition), or a `mosquitto_sub`/connect probe to
  `127.0.0.1:1883`. Unreachable → `FAILED`, with the remediation pointing at
  [`STEP-1` §3.2](STEP-1-PREREQUISITES.md#32-mosquitto-broker), which owns the
  broker daemon and its `/etc/mosquitto/conf.d/mv3dt.conf` drop-in. The
  remediation is "re-run the installer so Step 1 reinstalls the broker", not
  "run a laptop setup script by hand" — the operator never invokes a script
  directly.
- Confirm `systemctl` is present and the installer is root (framework §9.1).
- If remote mode is opted in ([§E.2](#e2-gating-opt-in)) but the cloud broker
  endpoint/creds are missing, return `USER_ACTION_REQUIRED` listing the
  `MV3DT_CLOUD_BROKER_*` values to set (still allow local-only if not opted
  in).
- All good → `COMPLETE` (ok to run).

#### `run(ctx)`

- Install the two unit files, `daemon-reload`, enable + start per-project
  instances and the agent ([§A.4](#a4-installer-integration-mirror-installsh)).
- Write the scoped polkit rule for the agent
  ([§B.1.1](#b11-the-polkit-rule-locked-form)), root-owned `0644`, then confirm
  a **negative** case is denied before reporting success.
- In remote mode: write the broker remote-mode drop-in + agent creds, add the
  bridge drop-in, `systemctl restart mosquitto`
  ([§D](#d-security-remote-control-must-be-authenticated), [§B.2](#b2-broker-location--connection-topology)).
- Report every dependency/unit touched via the framework reporters
  (§8.3), using the **exact strings**:
  - `installed mv3dt-agent.service version 6.0.0`
  - `installed mv3dt-pipeline@north-lobby-2.service version 6.0.0`
  - `already installed mv3dt-agent.service version 6.0.0` on a
    re-run where the unit is present + enabled (idempotent).
- May return `USER_ACTION_REQUIRED` for human credential provisioning
  ([§D](#d-security-remote-control-must-be-authenticated)); else `COMPLETE`.

#### `verify(ctx)`

Idempotent post-checks (framework §8.4 `verify_pinned` / reporters);
`COMPLETE` only when all pass:

- `systemctl is-enabled mv3dt-agent.service` == `enabled` **and**
  `is-active` == `active`.
- For each registry slug: `systemctl is-enabled mv3dt-pipeline@<slug>` ==
  `enabled` and `is-active` ∈ {`active`, `activating`} (a `failed` instance
  surfaces a `verify` failure with the `journalctl -u` hint).
- The two unit files exist under `/etc/systemd/system/` and
  `daemon-reload` reports them loaded.
- **Authorization scope — positive AND negative
  ([§B.1.1](#b11-the-polkit-rule-locked-form)).** As the agent user, assert
  that `systemctl start mv3dt-pipeline@<slug>` is permitted **and** that
  `systemctl start mosquitto` and `systemctl enable mv3dt-pipeline@<slug>` are
  **denied**. A positive-only check passes just as happily against a rule that
  grants everything, which is the exact failure this design exists to prevent —
  so the negative assertions are not optional.
- **Round-trip `status` check:** publish a `{"action":"status"}` command to
  `mv3dt/<HOST_ID>/cmd` and assert a result returns on
  `mv3dt/<HOST_ID>/cmd/result` within a timeout (the end-to-end remote-control
  proof). Under `--non-interactive` / no reachable cloud, fall back to the
  **dry-run equivalent**: invoke the agent's command handler in-process
  against a local `mosquitto_sub`/`pub` loopback and assert the same result
  shape — so verify never depends on the cloud webapp existing.

#### `report(ctx)`

Prints the human-facing summary (no side effects): the enabled units
(agent + per-project instances with their `is-active`/`is-enabled`), the topic
layout (`mv3dt/<HOST_ID>/cmd`, `.../cmd/result`, `.../status`), the resolved
`<HOST_ID>`, the broker mode (local-only vs remote/bridged), and the
`journalctl -u mv3dt-agent` / `journalctl -u mv3dt-pipeline@<slug>` commands
for logs.

### E.2 Gating (opt-in)

Step 6 is **optional / opt-in**: a workstation used only for local calibration
+ ad-hoc pipeline runs does not need 24/7 supervision or remote control and
can skip it. Gating:

- A framework CLI flag / `installer.conf` key **`MV3DT_REMOTE_SUPERVISION`**
  (`off` default). When `off`, the dispatch loop treats Step 6 as
  auto-`COMPLETE` (skipped) with a one-line log — the same skip discipline as
  a completed step (framework §3.2). The Step 5 exes keep working in plain
  foreground mode.
- `MV3DT_REMOTE_SUPERVISION=local` enables systemd supervision +
  boot-enable + the agent against the **local** broker only (no cloud bridge,
  simple-testing broker posture retained).
- `MV3DT_REMOTE_SUPERVISION=remote` additionally requires the cloud broker
  config + enables the hardened broker posture
  ([§D](#d-security-remote-control-must-be-authenticated)) and the bridge
  ([§B.2](#b2-broker-location--connection-topology)).
- Under `--non-interactive`, an unset value stays `off` (no long-running
  services enabled in an unattended run), matching Step 5 §2.1's
  "default to Close" discipline.

---

## 8. Out of scope / flag for human

- **The cloud webapp itself** — the other end of the command/status loop (UI,
  the MQTT publisher/subscriber, per-host dashboards, command authorization).
  Step 6 only defines the desktop agent, the topics, and the JSON contract it
  honors.
- **The desktop's HTTP connection to the web app** — registration/status over
  HTTPS, signed-URL artifact upload, and web-app-initiated one-shot
  operations. This is **not** unspecified: it is
  [`STEP-7`](STEP-7-WEBAPP-INTEGRATION.md), a separate opt-in step over a
  separate transport. Out of scope *here*, owned *there*.
- **Cloud auth infrastructure + credential issuance** — who provisions the
  cloud broker account, the agent's broker password rotation, and the TLS
  CA/certs ([§D](#d-security-remote-control-must-be-authenticated)). The
  installer can create the *local* `mosquitto_passwd` entry; everything
  cloud-side is human-provisioned.
- **A cloud broker deployment** — Step 6 assumes the cloud broker endpoint is
  given; standing it up is out of scope.
- **Fleet orchestration** (rolling restarts across many hosts, host
  discovery, RBAC per operator) — the `<HOST_ID>` namespacing makes it
  possible, but the orchestration lives in the webapp.
- **Extra systemd hardening** for the pipeline unit
  (`ProtectSystem`/`PrivateTmp`/seccomp) — omitted by default because
  `deepstream-app` needs broad GPU/device access
  ([§A.1](#a1-the-templated-per-project-instance-unit)); revisit after
  validating it doesn't break CUDA/NVENC.

---

## 9. References

DeepStream 9.1 official documentation only — cited **only** for
DeepStream-specific facts (the supervised entry point, MV3DT telemetry
topics, and the MQTT transport the pipeline uses).

- DS 9.1 `deepstream-app` reference (the `deepstream-app -c <config>` entry
  point the `mv3dt-pipeline@.service` `ExecStart` supervises):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html>
- DS 9.1 MV3DT (multi-view 3D tracking; MQTT-based communication via
  `communicatorType`; the `mv3dt/<LOCATION_ID>/*` telemetry topics whose
  `mv3dt/<id>/<channel>` shape the control topics parallel):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_MV3DT.html>
- DS 9.1 Gst-nvmsgbroker (the MQTT protocol adapter
  `libnvds_mqtt_proto.so`, `conn-str = <host>;<port>`, `config` file for
  auth, `nvds_msgapi_subscribe()` for consuming — the transport the pipeline
  already uses and the agent shares):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvmsgbroker.html>
- DS 9.1 IoT / Edge-to-Cloud Messaging (bidirectional device↔cloud messaging
  via Gst-nvmsgbroker — context for the remote command/status loop):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_IoT.html>

**Repo files referenced** (these define the OS-level systemd conventions and
the logic Step 6 supervises/ports — **not** DeepStream docs):

- [`00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md) — Step
  protocol, `Context`, `StepStatus`, reporting strings, `install_dir`.
- [`STEP-5-PER-PROJECT-EXES.md`](STEP-5-PER-PROJECT-EXES.md) — the
  `pipeline-<slug>` exe, `pipeline` subcommand, and `registry.json` Step 6
  supervises; §3/§5.4/§9 are the handoff points that must change.
- [`laptop/scripts/50_start_pipeline.sh`](../../laptop/scripts/50_start_pipeline.sh)
  — start logic ported into the unit `ExecStart` (`--service-exec`).
- [`laptop/scripts/99_stop_all.sh`](../../laptop/scripts/99_stop_all.sh)
  — SIGTERM→SIGKILL teardown mirrored by `ExecStop` /
  `KillSignal`+`TimeoutStopSec`.
- [`laptop/mosquitto/mv3dt.conf`](../../laptop/mosquitto/mv3dt.conf) — the
  simple-testing broker posture (`allow_anonymous true`, `1883`+`9001`) the
  agent connects to; hardened for remote mode.
- [`laptop/docs/DEEPSTREAM-SETUP.md`](../../laptop/docs/DEEPSTREAM-SETUP.md)
  §6 — Mosquitto "Production hardening" (`password_file`, `acl_file`, bound
  listeners) required for remote mode ([§D](#d-security-remote-control-must-be-authenticated)).
- [`STEP-7-WEBAPP-INTEGRATION.md`](STEP-7-WEBAPP-INTEGRATION.md) — the HTTP
  data plane; §D.1 there and [§C.5](#c5-status--heartbeat-json-schema-desktop--cloud)
  here share field names by contract.
- [`DELETION-REVIEW.md`](DELETION-REVIEW.md) — records the camera-node files
  the systemd conventions above were mirrored from, and why they are removed.

> **Attribution — sources no longer in this fork.** The systemd conventions in
> [§A.1](#a1-the-templated-per-project-instance-unit) and
> [§B.1](#b1-the-agent-systemd-unit) (`Restart=`, `RestartSec`, journald
> `StandardOutput/Error`, `EnvironmentFile`, `WantedBy=multi-user.target`), the
> flag-file/`.path`-unit precedent documented (and rejected) as §B.1 Option 2,
> the
> `systemctl show -p ActiveState -p SubState` parsing in §C.5, the status
> payload shape, and the unit-install flow in
> [§A.4](#a4-installer-integration-mirror-installsh) were ported from the P2BP
> camera-node tree (`services/tracker.service`, `services/tracker.path`,
> `services/tracker-toggle.service`, `services/heartbeat.service`,
> `scripts/heartbeat.py`, `scripts/json_models/heartbeat_payload.py`,
> `scripts/systemd_services.py`, `scripts/signals.py`, and `install.sh`). Those
> files are removed per
> [`DELETION-REVIEW` §3](DELETION-REVIEW.md#3-deletions-gated-on-the-harvest-the-jetson-tree)
> and remain in this repository's git history. Every convention borrowed from them is
> written out in full above; they are named here as provenance only —
> deliberately not linked, so the links cannot rot.
