# STEP 8 — Tracking history recorder and read-only tailnet API (owner: DevD)

Status: step spec, **specified, not implemented, never run on the
workstation**. It specifies the `step8_tracking_history` module and the
bundled `mv3dt_tracking` service package it installs. It builds strictly on
the contracts in [`00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md)
— the step-module interface
([`00` section 12](00-FRAMEWORK-AND-BOOTSTRAP.md#12-step-module-interface-the-contract-for-steps-15)),
install-location config
([`00` section 11](00-FRAMEWORK-AND-BOOTSTRAP.md#11-install-location-config)),
the opt-in gate mechanism
([`00` section 3.4](00-FRAMEWORK-AND-BOOTSTRAP.md#34-opt-in-step-gates)),
bundled-asset staging
([`00` section 4.2](00-FRAMEWORK-AND-BOOTSTRAP.md#42-locating-and-staging-bundled-assets-at-runtime))
and the reporting strings
([`00` section 8.3](00-FRAMEWORK-AND-BOOTSTRAP.md#83-reporting-format-for-dependencies-required-exact-strings)).
Those contracts are **not** restated here; only Step 8's own scope is.

The DeepStream pipeline publishes MV3DT tracker output to the local broker on
`mv3dt/${LOCATION_ID}/sv3d` as protobuf `nv.Frame` messages built by
`nvmsgconv_mega`, at roughly 22 fps per camera. Nothing stores it, and
tracker ids churn, so one person fragments into many ids over a walk. **Step 8
records that stream into a local database on the workstation** and serves the
history read-only to the web app's Floor Plan tab, which replays and exports
it. The workstation is the database: nothing is copied to the web app's
storage.

Step 8 is the first installer step that **listens on a network port**. That
amends the "no inbound-to-desktop API" rule of
[`STEP-6`](STEP-6-REMOTE-SUPERVISION.md) and the "nothing listens" rule of
[`STEP-7`](STEP-7-WEBAPP-INTEGRATION.md) for this one service, under the
controls in [section 3](#3-network-exposure-locked). The field mapping in
[section 5](#5-decoding-nvframe) is taken from DeepStream's
`sources/libs/nvmsgconv_mega/deepstream_schema/schema.proto` (Apache-2.0).

---

## 1. Module identity and scope

Module: `installer/mv3dt_installer/steps/step8_tracking_history.py`,
registered in `STEP_REGISTRY` with `order = 8`;
`id = "step8_tracking_history"`; `title = "Tracking history"`.

The service package lives at
`installer/mv3dt_installer/assets/tracking/mv3dt_tracking/` and is bundled
into the binary with the rest of `assets/`. It is **not** imported by the
installer: Step 8 copies it out and runs it from its own virtual environment,
which keeps FastAPI, uvicorn and paho-mqtt out of the PyInstaller binary.

In scope:

1. Install the package, its venv, its data directory and its secrets file
   ([section 4](#4-install-layout-and-pinned-values)).
2. Record every `nv.Frame` into per-day SQLite files
   ([section 5](#5-decoding-nvframe), [section 6](#6-storage)).
3. Serve a read-only, ticket-authenticated HTTP API on localhost, exposed to
   the tailnet only through `tailscale serve`
   ([section 8](#8-http-api)).
4. Enforce one data-access lease per project
   ([section 9](#9-data-access-lock-required)).
5. Provide the local `tracking stats` and `tracking wipe` subcommands
   ([section 10](#10-cli)).

Out of scope for this step, but adjacent:

- The web app's ticket route and Floor Plan replay UI. Both are owned by the
  web app repository; [section 3.2](#32-tickets-required) states only the
  ticket contract this service verifies.
- Remote control of pipelines ([`STEP-6`](STEP-6-REMOTE-SUPERVISION.md)) and
  artifact upload ([`STEP-7`](STEP-7-WEBAPP-INTEGRATION.md)). Step 8 reads
  neither's state.

---

## 2. Inputs consumed and outputs produced

### 2.1 Inputs (via `Context`)

- `ctx.install_dir` / `ctx.conf` — install root (default `/opt/mv3dt`),
  resolved via `config.load()`
  ([`00` section 11](00-FRAMEWORK-AND-BOOTSTRAP.md#11-install-location-config)).
- `LOCATION_ID` from `installer.conf`, which names the subscribed topic and
  selects the calibration files below.
- The local Mosquitto broker set up by
  [`STEP-1`](STEP-1-PREREQUISITES.md), and the pipeline publishing to it
  ([`STEP-6`](STEP-6-REMOTE-SUPERVISION.md) supervision or a foreground
  Step 5 run).
- The Step 4 calibration output for that `LOCATION_ID` — `map.png` and
  `transforms.yml` in `calibration/<LOCATION_ID>/`, and
  `camera_models/camInfo_*.yml`, both under `<install_dir>/deepstream/`
  ([`STEP-4` section 5](STEP-4-CALIB-OUTPUT-WIRING.md#5-calibration-destination),
  [`STEP-4` section 6.4](STEP-4-CALIB-OUTPUT-WIRING.md#64-camera-models)) —
  read by `GET /tracking/meta` and `GET /tracking/floorplan.png`.

**RESOLVED — one project per workstation in v1.** `TRACKING_PROJECT_ID` is
the web app project id; the floor plan and camera files always come from the
`LOCATION_ID` in `installer.conf`, so there is no mapping between web app
projects and local calibrations to maintain.

### 2.2 Outputs

| Path | Contents |
|---|---|
| `<install_dir>/tracking/` | the copied `mv3dt_tracking` package |
| `<install_dir>/tracking/venv/` | the service venv with pinned dependencies |
| `/var/lib/mv3dt-tracking/YYYY-MM-DD.sqlite` | one SQLite database per UTC day, WAL mode |
| `<install_dir>/secrets/tracking.env` | ticket secret, project id, allowed origins; mode `600` |
| `<install_dir>/tracking/cleaning.json` | strict cleaning seed, written once; mode `644` ([section 4.5](#45-cleaningjson-and-offsetsjson)) |
| `<install_dir>/tracking/offsets.json` | never written by Step 8; only `learn-offsets --save` creates it |
| `/etc/systemd/system/mv3dt-tracking.service` | the recorder and API service |
| `tailscale serve` mapping | HTTPS `8443` on the tailnet to `http://127.0.0.1:8090` |

No `state.json` writes — Step 8 returns a `StepResult` and the framework owns
persistence
([`00` section 12.2](00-FRAMEWORK-AND-BOOTSTRAP.md#122-stepresult-and-status-recorded-by-the-state-machine)).

---

## 3. Network exposure (LOCKED)

**LOCKED: a tailnet-only, read-only HTTP listener exists on the
workstation.** FastAPI (under uvicorn) binds to `127.0.0.1:8090` and nothing
else. The only path in from another machine is:

```bash
tailscale serve --bg --https=8443 http://127.0.0.1:8090
```

Rationale: the browser reaches the workstation directly over Tailscale. The
web app's Express backend runs on Vercel, which cannot reach the tailnet, so
Express only mints short-lived tickets and never proxies data. That keeps all
data traffic off the web app's free-tier hosting and database.

This **amends** [`STEP-6`](STEP-6-REMOTE-SUPERVISION.md) ("no
inbound-to-desktop API") and
[`STEP-7`](STEP-7-WEBAPP-INTEGRATION.md) ("nothing listens") for
`mv3dt-tracking.service` only. Every other unit keeps the outbound-only rule.

### 3.1 Security controls (REQUIRED)

| Control | Rule |
|---|---|
| Bind address | `127.0.0.1:8090` only; never `0.0.0.0` or a LAN address |
| Exposure | `tailscale serve` on HTTPS `8443`, tailnet only; never `tailscale funnel` |
| Authentication | every `/tracking/*` route requires a valid ticket ([section 3.2](#32-tickets-required)); only `GET /health` is open |
| CORS | allowed origins are exactly `ALLOWED_ORIGINS` from `tracking.env` |
| Network verbs | read-only; no route writes or deletes recorded data. The lease routes in [section 9](#9-data-access-lock-required) change only in-memory lease state |
| Wipe | local CLI only ([section 10](#10-cli)); never reachable over HTTP |
| Repo hygiene | no hostnames, tailnet names, IPs, project ids or secrets committed; the secret is generated on the workstation |

### 3.2 Tickets (REQUIRED)

The web app's Express route
`POST /api/workstation-tracking/project/:projectId/ticket` mints the ticket;
the browser sends it as `Authorization: Bearer <ticket>`. The service accepts
a ticket only when all of these hold:

| Claim / property | Required value |
|---|---|
| Algorithm | `HS256`, keyed by `WORKSTATION_TICKET_SECRET` |
| `aud` | `p2bp-workstation` |
| `scope` | `tracking` |
| `pid` | equal to `TRACKING_PROJECT_ID` |
| `exp` | present and in the future; tickets live 5 minutes |
| `name` | display name of the holder, shown in lock refusals ([section 9](#9-data-access-lock-required)) |

**RESOLVED — refusal codes and issuer.**

| Failure | Status |
|---|---|
| ticket missing, malformed, bad signature, or expired | `401` |
| valid ticket with wrong `scope` or wrong `pid` | `403` |

`iss` is checked only when `TICKET_ISSUER` is set in `tracking.env`; when
set, it must equal the web app's `JWT_ISSUER`.

---

## 4. Install layout and pinned values

### 4.1 Gate

Gated by `MV3DT_TRACKING_HISTORY` in `installer.conf` with the CLI flag
`--tracking-history`, following the mechanism and seeding tiers in
[`00` section 3.4](00-FRAMEWORK-AND-BOOTSTRAP.md#34-opt-in-step-gates). H5
adds the key to `config.py`, the flag to the `app.py` parser, and a row to
the `00` gate table ([section 13](#13-decomposition)):

| Value | Behavior |
|---|---|
| `off` (default) | dispatch treats Step 8 as auto-`COMPLETE`; no files, no unit, no listener |
| `on` | full install per this section |

Under `--non-interactive` an unset value stays `off`. An unattended run must
never open a listener the operator did not ask for.

### 4.2 Pinned values

| Item | Value |
|---|---|
| Package source (bundled) | `assets/tracking/mv3dt_tracking/` |
| Package destination | `<install_dir>/tracking/` (default `/opt/mv3dt/tracking/`) |
| Venv | `<install_dir>/tracking/venv` |
| Dependencies | `fastapi`, `uvicorn`, `paho-mqtt`, `pyjwt`, `pyyaml`, each pinned in `assets/tracking/requirements.txt` |
| Data directory | `/var/lib/mv3dt-tracking/` |
| Database file | `YYYY-MM-DD.sqlite`, one per UTC day, `journal_mode=WAL` |
| Secrets file | `<install_dir>/secrets/tracking.env`, mode `600` |
| Cleaning file | `<install_dir>/tracking/cleaning.json` (`CLEANING_FILE`) |
| Offsets file | `<install_dir>/tracking/offsets.json` (`OFFSETS_FILE`) |
| Unit | `mv3dt-tracking.service` |
| Listen address | `127.0.0.1:8090` |
| Tailnet exposure | HTTPS `8443` via `tailscale serve` |
| Subscribed topic | `mv3dt/${LOCATION_ID}/sv3d` |

### 4.3 `tracking.env`

| Key | Source |
|---|---|
| `WORKSTATION_TICKET_SECRET` | generated once on first `run`; never regenerated on a re-run |
| `TRACKING_PROJECT_ID` | the web app project whose tickets are accepted; captured from the operator |
| `ALLOWED_ORIGINS` | comma-separated web app origins for CORS; captured from the operator |
| `TICKET_ISSUER` | optional; when set, tickets must carry this `iss` ([section 3.2](#32-tickets-required)) |

The file is mode `600` and owned by the invoking user.

**RESOLVED — a blank issuer is asked once.** A blank answer to the
interactive `TICKET_ISSUER` prompt is written as `TICKET_ISSUER=`, which the
package reads as unset, so a later run does not ask again. Under
`--non-interactive` with no `TICKET_ISSUER` in the environment the key is
left out of the file.

**RESOLVED — the secret is never printed.** It is written only to
`tracking.env`. `report` tells the operator to read it once with
`sudo cat <install_dir>/secrets/tracking.env` and copy the value into the
web app's `WORKSTATION_TICKET_SECRET`. The redaction rule
([`00` section 14.4](00-FRAMEWORK-AND-BOOTSTRAP.md#144-redaction-required))
applies to every log line, including the transcript
([`00` section 8.2](00-FRAMEWORK-AND-BOOTSTRAP.md#82-transcript-log-file)).

### 4.4 The unit

`assets/systemd/mv3dt-tracking.service.in` follows the pattern of
`mv3dt-uploader.service.in` and the conventions in
[`STEP-7` section H.1](STEP-7-WEBAPP-INTEGRATION.md#h1-lifecycle):
`Type=simple`, `Restart=on-failure`, `RestartSec=5`, `StartLimitBurst=5`,
journald output with `LogRateLimitIntervalSec=30s`, `EnvironmentFile=` for
both `installer.conf` and `secrets/tracking.env`,
`WantedBy=multi-user.target`. Like the uploader it runs as the invoking
non-root user (`User=@USER@`), who also owns `<install_dir>/tracking/` and
`/var/lib/mv3dt-tracking/`. Unlike the uploader it **does** depend on the
broker: `After=mosquitto.service` and `Wants=mosquitto.service`. `ExecStart`
runs the package's `serve` command from the venv
([section 10](#10-cli)). One process runs both the recorder and the API, so
the in-memory lease state in [section 9](#9-data-access-lock-required) has a
single owner.

**REQUIRED — one environment list.** The template carries a single
`@SERVICE_ENVIRONMENT@` marker in `[Service]`. `render_tracking_unit()`
expands it to one `Environment="KEY=value"` line per entry of
`service_environment()` in `steps/step8_tracking_history.py`, in this order:

| Variable | Value |
|---|---|
| `PYTHONPATH` | `<install_dir>/tracking` |
| `DATA_DIR` | `/var/lib/mv3dt-tracking` |
| `CALIB_DIR` | `<install_dir>/deepstream/calibration/<LOCATION_ID>`, or the persisted `CALIBRATION_DIR` |
| `CAMERA_MODELS_DIR` | `<install_dir>/deepstream/camera_models` |
| `CLEANING_FILE` | `<install_dir>/tracking/cleaning.json` |
| `OFFSETS_FILE` | `<install_dir>/tracking/offsets.json` |
| `BIND` | `127.0.0.1:8090` |
| `PYTHONUNBUFFERED` | `1` |

Values are checked with the same quoting rule as every other marker and
have `%` doubled for systemd. The `CLEANING_FILE` and `OFFSETS_FILE` paths
match the package's own defaults in `settings.py`. The `tracking`
passthrough ([section 10](#10-cli)) passes the same list, so the two can
never drift.

### 4.5 `cleaning.json` and `offsets.json`

Both files are operator data, read by the package as described in
[`TRACKING-FUSION` section 4.1](TRACKING-FUSION-AND-CLEANING.md#41-file-and-reload-required) and
[`TRACKING-FUSION` section 6.2](TRACKING-FUSION-AND-CLEANING.md#62-offsetsjson-schema).
Neither is a managed template, so the edit protection in
[`LIVE-BASELINE-AND-EDIT-PROTECTION.md`](LIVE-BASELINE-AND-EDIT-PROTECTION.md)
does not apply to them.

| File | Written by | Owner, mode | Re-run and upgrade |
|---|---|---|---|
| `cleaning.json` | Step 8 `run`, only when the path does not exist | invoking user, `644` | never touched again |
| `offsets.json` | `mv3dt-installer tracking learn-offsets --save` only | invoking user | never touched by Step 8 |

**RESOLVED — the seed comes from the package.** The seed is
`{"version", "preset", "zones"}` with the strict values. Step 8 runs a short
program in the venv interpreter (`python3 -I -c ...`, as the invoking user)
that reads `cleaning.STRICT` and `cleaning.PRESETS`, checks the result with
`parse_config()` and prints it, so the installer restates no default. The
file is created under a temporary name with mode `644` regardless of the
umask, chowned to the invoking user, then hard-linked into place: it
appears complete or not at all, and an existing file or symlink, dangling or
not, is never replaced. A failed seed is a warning, not a failure, because a
missing file already means strict defaults.

The invoking user is the unit's `User=`, and it owns
`<install_dir>/tracking/`, so `learn-offsets --save` can write
`offsets.json` there and the service can read it.

---

## 5. Decoding nv.Frame

**LOCKED: a small built-in protobuf wire-format reader, no protobuf
dependency.** It handles varint (wire type 0), 64-bit (1), length-delimited
(2) and 32-bit (5) fields, and skips every field number it does not use, so
schema additions upstream do not break it. Only these fields are read:

| Message | Field | Number | Type | Use |
|---|---|---|---|---|
| `Frame` | `id` | 2 | string | not stored |
| `Frame` | `timestamp` | 3 | `google.protobuf.Timestamp` | `ts_ms` |
| `Frame` | `sensorId` | 4 | string | `sensor`, via the `sensors` table |
| `Frame` | `objects` | 5 | repeated `Object` | one row each |
| `Object` | `id` | 1 | string (numeric tracker id) | `track_id`; a non-numeric id skips the object |
| `Object` | `confidence` | 4 | float | `conf` |
| `Object` | `info` | 5 | `map<string,string>` | key `footLocation3D`, value `"x,y"` in AMC world units (metres) |
| `Object` | `bbox3d` | 14 | `Bbox3d` | `z`, `w`, `l`, `h`; fallback `x`, `y` |
| `Timestamp` | `seconds` | 1 | int64 | `ts_ms = seconds * 1000 + nanos // 1_000_000` |
| `Timestamp` | `nanos` | 2 | int32 | as above |
| `Bbox3d` | `coordinates` | 1 | packed doubles | order `x, y, z, width, length, height, pitch, roll, yaw, vx, vy, vz` |

A `map` entry is a length-delimited message with `key` = 1 and `value` = 2.

**Floor position** is `footLocation3D`; when it is absent or unparsable, the
`x, y` of `bbox3d` is used instead. The recorder uses the message timestamp
and falls back to receive time when `Frame.timestamp` is absent. A payload
that fails to decode is counted and logged and never stops the recorder. An
object whose `id` does not parse as an integer is skipped and counted the
same way.

**RESOLVED — world units are metres.** `footLocation3D` is in AMC world
units, verified live as metres: recorded positions fall inside the
7.41 x 5.02 m floor plan at 100 px per metre. `dist` in
[section 7](#7-stitching) is therefore in metres.

The decode test fixture, `installer/tests/tracking/fixtures/frame_camera1.hex`,
is one real payload captured from the workstation's broker. Its expected
values were produced **once**, by running `protoc --decode=nv.Frame` on the
live output, and are committed as static files beside it. Neither the build
nor the tests run or require `protoc` or a protobuf library.

The recorder does not write to SQLite itself: it calls an injected
`sink(rows)` callable with plain `dict` rows and never imports `store.py`.
`cli.py serve` ([section 10](#10-cli)) wires the recorder to the store.

---

## 6. Storage

One SQLite file per UTC day in `/var/lib/mv3dt-tracking/`, in WAL mode. A
row lands in the file for the UTC day of its `ts_ms`.

```sql
CREATE TABLE detections (
  ts_ms    INTEGER,
  sensor   INTEGER,
  track_id INTEGER,
  x REAL, y REAL, z REAL,
  w REAL, l REAL, h REAL,
  conf REAL
);
CREATE INDEX detections_ts ON detections (ts_ms);
CREATE INDEX detections_track_ts ON detections (track_id, ts_ms);

CREATE TABLE sensors (
  id   INTEGER PRIMARY KEY,
  name TEXT UNIQUE
);
```

- **Writes** are batched inserts.
- **Range reads** span day files and downsample in SQL by grouping on
  `track_id`, `sensor` and `ts_ms / bucket`, where the bucket width follows
  the requested `hz`. Target: under 300 ms for a 20-minute range.
- **Retention (LOCKED): keep everything.** Nothing is pruned automatically;
  `tracking wipe` ([section 10](#10-cli)) is the only deletion path.

---

## 7. Stitching

**LOCKED: stitching is query-time only; raw rows are never modified.** It
runs when `mode=stitched` and is skipped for `mode=raw`, where each tracker id
is its own person.

1. **Shared ids**: tracker ids are already shared across cameras, so one id
   seen by both cameras is one person.
2. **Continuation**: a new tracker id that first appears within `gap`
   seconds and `dist` metres of where a person was last seen, on either
   camera, continues that person. Its id joins that person's `trackIds`.
3. **Fusion**: within each time bucket, the cameras' samples for one person
   are averaged into a single `x, y`.

| Parameter | Default |
|---|---|
| `gap` | `3.0` s |
| `dist` | `1.2` m |

---

## 8. HTTP API

`from` and `to` are unix milliseconds. Every `/tracking` route requires a
ticket ([section 3.2](#32-tickets-required)).

| Route | Auth | Purpose |
|---|---|---|
| `GET /health` | none | liveness, `200` when up |
| `GET /tracking/meta` | ticket | location, recorded data span, floor-plan size, the world-to-pixel `T_gt2px` from `transforms.yml`, camera positions from the camera models |
| `GET /tracking/floorplan.png` | ticket | AMC `map.png` |
| `POST /tracking/session` | ticket | acquire the lease ([section 9](#9-data-access-lock-required)) |
| `POST /tracking/session/{id}/heartbeat` | ticket | keep the lease alive |
| `DELETE /tracking/session/{id}` | ticket | release the lease |
| `GET /tracking/range?from&to&hz=5&mode=stitched\|raw` | ticket + lease | columnar history, max 6 h per request |
| `GET /tracking/export?from&to&mode&format=csv\|jsonl` | ticket + lease | streamed export, max 24 h per request |

`range` returns columnar JSON:

```json
{
  "people": [
    {"id": 1, "trackIds": [17, 23], "t": [0, 200], "x": [1.2, 1.3], "y": [0.4, 0.5]}
  ]
}
```

`hz` defaults to `5`; `mode` is `stitched` or `raw`
([section 7](#7-stitching)). A `range` request wider than 6 hours, or an
`export` wider than 24 hours, is refused.

---

## 9. Data-access lock (REQUIRED)

User requirement: while anyone has the replay or export view open, nobody
else can pull data. This prevents request spamming and protects the
free-tier services behind the web app. The lease lives on the workstation,
the one place all data traffic reaches.

1. **Acquire**: `POST /tracking/session` returns `{SessionId, ExpiresAt}`,
   or `423 Locked` with `{HeldBy, Until}` when another holder has it.
   `HeldBy` is the holder ticket's `name` claim.
2. **Heartbeat**: the holder calls
   `POST /tracking/session/{id}/heartbeat` every 15 s while the view is
   open.
3. **Release**: `DELETE /tracking/session/{id}` on view close or page
   unload.
4. **Data requests**: `range` and `export` require the header
   `X-Tracking-Session: <id>` matching the live lease, else `423`.

| Rule | Value |
|---|---|
| Leases per project | 1 |
| Heartbeat interval | 15 s |
| Expiry after last heartbeat | 45 s |
| Hard cap per lease | 15 min; re-acquire to continue |
| In-flight requests per lease | 1; a second concurrent request gets `429` |
| `range` requests per lease | 30 per minute; beyond that `429` with `Retry-After` |
| Concurrent exports | 1 |
| Session kind | `replay` \| `export`, recorded on the lease |
| State | in memory; a service restart clears it |

The session kind is recorded so that a future upload flow (the Floor Plan
tab's Upload panel) acquires this same lease rather than adding a second
lock.

---

## 10. CLI

The package's `cli.py` provides `serve | stats | wipe --yes [--before DATE]`
and the cleaning commands `zones-suggest | learn-offsets | clean-report`
([`TRACKING-FUSION` section 9](TRACKING-FUSION-AND-CLEANING.md#9-cli-and-installer-passthrough)).
`serve` is the unit's `ExecStart`. Step 8 registers the installer subcommand:

```bash
mv3dt-installer tracking stats
mv3dt-installer tracking wipe --yes [--before DATE]
mv3dt-installer tracking zones-suggest [--from T] [--to T] [--hours N] [--json]
mv3dt-installer tracking learn-offsets [--from T] [--to T] [--hours N] [--model translation|similarity] [--save]
mv3dt-installer tracking clean-report --from T --to T
```

`stats` reports what is recorded (row counts per day file, data span).
**RESOLVED — wipe sequence.** `wipe --yes` stops `mv3dt-tracking`, deletes
the day files (all of them, or only days before `DATE`), and starts the
service again, so recording resumes on a clean state. Without `--yes` it
refuses. If the unit is still active after the stop, it refuses and leaves
the files and the service alone; a workstation without the unit can still
be wiped.
Wipe is local only ([section 3.1](#31-security-controls-required)).

**REQUIRED — the cleaning passthrough.** The three cleaning commands are
forwarded to the package unchanged, `--help` included; only the framework
flags (`--install-dir`, `--log-dir`, `--non-interactive`, `--verbose`) are
removed. Every package command, `stats` and `wipe` too, runs as follows:

| Item | Value |
|---|---|
| Interpreter | `<install_dir>/tracking/venv/bin/python3 -m mv3dt_tracking <command> <args>` |
| User | the invoking user via `ctx.run_as_user`, which is the unit's `User=` |
| Working directory | `<install_dir>/tracking`, the unit's `WorkingDirectory=` |
| Environment | `service_environment()` ([section 4.4](#44-the-unit)), passed through `env` |
| Not passed | the unit's `EnvironmentFile=` sources: `secrets/tracking.env` (the ticket secret never leaves its file) and `installer.conf` (no cleaning command reads its keys) |
| Exit code | the package's, returned unchanged |
| Gate `off` (cleaning commands) | exits `1` with `tracking history is off in installer.conf ...`; nothing runs |
| Not installed | exits `1` with `tracking history is not installed ...`; nothing runs |

The passthrough never prompts. Running as the invoking user keeps a saved
`offsets.json` owned by the user the service runs as.

---

## 11. Framework integration

Implements the `Step` protocol
([`00` section 12.1](00-FRAMEWORK-AND-BOOTSTRAP.md#121-protocol)).

### 11.1 `preflight(ctx)`

- Gate `off` ([section 4.1](#41-gate)) → `COMPLETE` immediately.
- Otherwise confirm the inputs in [section 2.1](#21-inputs-via-context)
  are present.
- **RESOLVED — Tailscale is not a blocker.** If `tailscale` is missing or
  not logged in, the step still installs the service; the expose sub-phase
  of `run` is reported as a warning that prints the manual
  `tailscale serve` command. `tailscale funnel` is never used.
- **RESOLVED — missing calibration output is a warning.** Recording needs
  only `LOCATION_ID` and the broker, so a missing `map.png`,
  `transforms.yml` or camera model is logged as a warning and the step
  continues. Only `GET /tracking/meta` and `GET /tracking/floorplan.png`
  are affected: they return no floor plan or camera positions until Step 4
  output exists. A missing `LOCATION_ID` still stops the step.

### 11.2 `run(ctx)`

1. **Package**: copy `assets/tracking/mv3dt_tracking/` to
   `<install_dir>/tracking/`.
2. **Venv**: build `<install_dir>/tracking/venv` and install the pinned
   requirements.
3. **Data directory**: create `/var/lib/mv3dt-tracking/`, then seed
   `cleaning.json` once per [section 4.5](#45-cleaningjson-and-offsetsjson).
   `offsets.json` is never created.
4. **Secrets**: write `<install_dir>/secrets/tracking.env` with mode `600`
   per [section 4.3](#43-trackingenv), generating the ticket secret only if
   absent.
5. **Unit**: render `mv3dt-tracking.service.in` with the environment list
   of [section 4.4](#44-the-unit), `systemctl daemon-reload`,
   `systemctl enable --now mv3dt-tracking.service`, mirroring
   [`STEP-6` section A.4](STEP-6-REMOTE-SUPERVISION.md#a4-installer-integration-mirror-installsh).
6. **Expose**: `tailscale serve --bg --https=8443 http://127.0.0.1:8090`,
   or a warning carrying that command when Tailscale is unavailable.
7. **Report the unit** with the exact string format of
   [`00` section 8.3](00-FRAMEWORK-AND-BOOTSTRAP.md#83-reporting-format-for-dependencies-required-exact-strings),
   using the package `__version__` as the version:

   ```
   installed mv3dt-tracking.service version 0.1.0
   ```

**RESOLVED — upgrade.** A re-run, including one after an installer upgrade,
repeats steps 1 to 5: it re-syncs the package (only the `mv3dt_tracking/`
directory is replaced), re-renders the unit, and restarts the service when
the package, venv, `tracking.env` or unit changed. With nothing changed it
reports the unit as already installed. `cleaning.json` and `offsets.json`
stay as they are. A completed step re-runs these on its own after an
upgrade, without `--reset-step 8`; see [section 11.5](#115-upgrade-check).

### 11.3 `verify(ctx)`

`COMPLETE` only when all pass:

- [ ] `mv3dt-tracking.service` is enabled and active.
- [ ] `secrets/tracking.env` exists with mode `600` and the three required
      keys set: `WORKSTATION_TICKET_SECRET`, `TRACKING_PROJECT_ID`,
      `ALLOWED_ORIGINS`. `TICKET_ISSUER` is optional.
- [ ] `GET http://127.0.0.1:8090/health` returns `200`.
- [ ] The detection row count increases between two reads while the
      pipeline is publishing.

### 11.4 `report(ctx)`

Prints, with no side effects: the unit state, the data directory, today's
row count, the `journalctl -u mv3dt-tracking` command, and the instruction
to read the ticket secret once with `sudo cat` and copy it into the web app
([section 4.3](#43-trackingenv)). Never prints the secret itself.

**RESOLVED — the report names the tailnet URL.** It also prints the value
for the web app's `WORKSTATION_TRACKING_URL`, built from the machine's
Tailscale DNS name as `https://<name>:8443`, or with a placeholder for the
name when Tailscale is not logged in. That is the one other setting the
operator copies into the web app's environment.

### 11.5 Upgrade check

**RESOLVED — a completed step updates itself.** Up to 0.4.19 the
dispatch loop
([`00` section 3.2](00-FRAMEWORK-AND-BOOTSTRAP.md#32-entrypoint-and-dispatch))
logged `step8_tracking_history: already complete` after an upgrade and left
the installed package on the previous release's code until the operator
ran `--reset-step 8`. The package `__version__` is not bumped every
release, so it cannot tell the two copies apart. The step now defines the
optional `pending_update(ctx)` hook, which the dispatch loop asks for every
`COMPLETE` step that has it:

1. **Gate**: gate `off` is skipped before the hook is asked, as before, and
   the hook itself returns no reason when the gate is off.
2. **Compare**: `update_reason(ctx)` checks, read only, in this order:

   | Check | Installed | Compared with |
   |---|---|---|
   | Package | `<install_dir>/tracking/mv3dt_tracking/` | bundled `mv3dt_tracking/` |
   | Requirements | `<install_dir>/tracking/requirements.txt` | bundled `requirements.txt` |
   | Unit | `/etc/systemd/system/mv3dt-tracking.service` | the unit this release renders |

   The package digest maps each relative path to its SHA-256 and skips
   `__pycache__/` and `*.pyc`, the same set `sync_package` copies. File
   order and mtimes never change it.
3. **Record**: on a reason the loop logs one line, sets the step back to
   `PENDING` and records the reason in `state.json` as the step's
   `update_pending`. Only `mark_complete` clears it.
4. **Re-run as an update**: the loop runs `preflight`, `run`, `verify` with
   `ctx.update_pending` set to the recorded reason. `run` stays idempotent
   ([section 11.2](#112-runctx)), so pip runs only when the requirements
   changed, with two differences for an update:
   - **REQUIRED — it never prompts.** `tracking.env` keys already in the
     file are kept and an absent optional key stays unset. A missing
     required key stops the step with `USER_ACTION_REQUIRED` naming the key
     and the file, never with a prompt.
   - **REQUIRED — it always restarts the unit.** An earlier attempt may
     have replaced the files and stopped before its restart.
5. **Resume**: an update that stopped part way (a `USER_ACTION_REQUIRED`, a
   failure, Ctrl-C or SIGTERM) leaves the step `PENDING` with the reason
   still recorded. The next run logs that it is resuming the update and
   runs it as one, restart included, although the files now match.
6. **No reason**: the step stays `COMPLETE`, nothing runs, nothing restarts.

```
step8_tracking_history: bundled tracking package changed; updating
```

```
step8_tracking_history: resuming an update that did not finish (bundled tracking package changed)
```

> **Never touched by an update:** `cleaning.json`, `offsets.json` and
> `/var/lib/mv3dt-tracking/`. `secrets/tracking.env` keeps every key it
> has; it is rewritten only to add a key from the environment or a missing
> ticket secret ([section 4.3](#43-trackingenv)). Neither the check nor
> the update run prompts, so both behave the same with and without
> `--non-interactive`. A hook that raises is logged as a warning and
> treated as no update.

`--status` asks the same hook without an invoking user, so it compares the
package and requirements only, and also shows an update that did not
finish:

```
step8_tracking_history: COMPLETE, update pending (bundled tracking package changed): runs on next install run
step8_tracking_history: PENDING, update did not finish (bundled tracking package changed): runs on next install run
```

---

## 12. Out of scope

Settled exclusions:

- Live view and 3D rendering — v1 is history replay and export only.
- Copying history to the web app's storage — the workstation is the
  database.
- NVIDIA's VSS Behavior Analytics, which can consume `nv.Frame` but needs
  Kafka or Redis plus Elasticsearch and has no floor-plan replay. Not
  adopted; worth revisiting for zone and tripwire analytics.
- The web app's ticket route, its per-user mint limit, and the replay UI —
  owned by the web app repository.

Open decisions: none. Each question raised by the first draft is now
recorded in place as **RESOLVED**.

---

## 13. Decomposition

| Unit | Branch | Files touched | Depends on | Wave |
|---|---|---|---|---|
| H1 Spec | `feat/installer-step8-spec` | `installer/plan/STEP-8-TRACKING-HISTORY.md`, `installer/plan/STEP-7-WEBAPP-INTEGRATION.md`, `installer/plan/STEP-6-REMOTE-SUPERVISION.md`, `installer/plan/DEVELOPMENT-STATUS.md` | none | 1 |
| H2 Store + stitch | `feat/installer-tracking-store` | `assets/tracking/mv3dt_tracking/__init__.py`, `store.py`, `stitch.py`, `tests/tracking/test_store.py`, `tests/tracking/test_stitch.py` | none | 1 |
| H3 Decode + recorder | `feat/installer-tracking-recorder` | `assets/tracking/mv3dt_tracking/decode.py`, `recorder.py`, `tests/tracking/test_decode.py`, `tests/tracking/test_recorder.py`, `tests/tracking/fixtures/frame_camera1.hex` | none | 1 |
| H4 Auth + lock + API + CLI | `feat/installer-tracking-api` | `assets/tracking/mv3dt_tracking/auth.py`, `lock.py`, `api.py`, `cli.py`, `assets/tracking/requirements.txt`, `tests/tracking/test_api.py`, `tests/tracking/test_auth.py`, `tests/tracking/test_lock.py` | H2, H3 | 2 |
| H5 STEP-8 step | `feat/installer-step8-tracking` | `steps/step8_tracking_history.py`, `__main__.py`, `config.py`, `app.py`, `assets/systemd/mv3dt-tracking.service.in`, `state.py`, `shellout.py`, `installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md`, `tests/test_step8_tracking_history.py`, `tests/test_main.py`, `tests/test_config.py`, `tests/test_shellout.py` | H1, H3, H4 | 3 |

Package paths are relative to `installer/mv3dt_installer/`; test paths are
relative to `installer/`.

Serialization points:

- H2 and H3 both add files under `assets/tracking/mv3dt_tracking/` but share
  none; `__init__.py` belongs to H2 alone, so H3 must not create it.
- H3 is independent of H2: `recorder.py` only calls an injected
  `sink(rows)` callable with plain `dict` rows and never imports
  `store.py` ([section 5](#5-decoding-nvframe)).
- H4 imports `store` and `stitch`, and its `cli.py serve` wires
  `recorder.py` into `store.py`, so it depends on both H2 and H3 and
  branches once both merge.
- H5 implements this spec, bundles the whole package, registers the step
  in `__main__.py`, and owns the gate plumbing (`config.py`, the `app.py`
  flag, the `00` gate table), so it lands last. It edits
  `00-FRAMEWORK-AND-BOOTSTRAP.md`, which H1 does not touch. The reported
  version comes from `mv3dt_tracking.__version__`, which H2 defines.
  `installer.spec` already bundles all of `assets/` recursively, so no row
  touches it.
- Recommended merge order: H1, H2, H3, H4, H5, then a release.

---

## References

DeepStream facts here (the `nv.Frame` message, its field numbers, and the
`nvmsgconv_mega` producer) come from the schema shipped in the DeepStream SDK
sources. Everything else is OS, Tailscale, or library convention.

- DeepStream SDK `sources/libs/nvmsgconv_mega/deepstream_schema/schema.proto`
  (Apache-2.0) — **backs the field table in [section 5](#5-decoding-nvframe)**.
- [Gst-nvmsgconv](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvmsgconv.html)
  — the message converter family that produces the protobuf payload.
- [Protocol Buffers encoding](https://protobuf.dev/programming-guides/encoding/)
  — **backs the wire-format reader**: wire types, packed repeated fields,
  map entries.
- [Tailscale serve](https://tailscale.com/kb/1312/serve) — **backs the
  exposure rule in [section 3](#3-network-exposure-locked)**: tailnet-only
  HTTPS proxying to a localhost port.
- [SQLite write-ahead logging](https://www.sqlite.org/wal.html) — concurrent
  reads during recorder writes.
- [RFC 7519](https://datatracker.ietf.org/doc/html/rfc7519) — JWT claims
  (`aud`, `exp`) used by the ticket.
- [FastAPI CORS](https://fastapi.tiangolo.com/tutorial/cors/) — the origin
  allowlist.

Repo files referenced:

- [`00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md) — sections
  3.4 gates, 4.2 bundled assets, 8.2 transcript, 8.3 reporting strings, 11
  install layout, 12 the Step protocol, 14.4 redaction.
- [`TRACKING-FUSION-AND-CLEANING.md`](TRACKING-FUSION-AND-CLEANING.md) —
  sections 4.1 and 6.2, the two cleaning files, and section 9, the cleaning
  commands the passthrough forwards.
- [`LIVE-BASELINE-AND-EDIT-PROTECTION.md`](LIVE-BASELINE-AND-EDIT-PROTECTION.md)
  — the managed-file edit protection that the cleaning files are outside of.
- [`STEP-1-PREREQUISITES.md`](STEP-1-PREREQUISITES.md) — installs the
  Mosquitto broker the recorder subscribes to.
- [`STEP-4-CALIB-OUTPUT-WIRING.md`](STEP-4-CALIB-OUTPUT-WIRING.md) — sections
  5 and 6.4, the calibration output read by `meta` and `floorplan.png`.
- [`STEP-6-REMOTE-SUPERVISION.md`](STEP-6-REMOTE-SUPERVISION.md) — section
  A.4 unit-install flow; the outbound-only rule amended here.
- [`STEP-7-WEBAPP-INTEGRATION.md`](STEP-7-WEBAPP-INTEGRATION.md) — section
  H.1 unit conventions; the "nothing listens" rule amended here.
- [`DEVELOPMENT-STATUS.md`](DEVELOPMENT-STATUS.md) — the dated workstation
  record, which carries this step's status row.
- `installer/mv3dt_installer/config.py` and `installer/mv3dt_installer/app.py`
  — the existing gate keys and CLI flags the `MV3DT_TRACKING_HISTORY` gate
  joins.
- `installer/mv3dt_installer/assets/systemd/mv3dt-uploader.service.in` — the
  unit template `mv3dt-tracking.service.in` follows.
- `installer/installer.spec` — bundles `assets/` into the binary.
