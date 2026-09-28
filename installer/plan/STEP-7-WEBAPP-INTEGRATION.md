# STEP 7 — Web-app integration over HTTP (owner: DevD)

Status: NEW step, added after
[`STEP-6-REMOTE-SUPERVISION.md`](STEP-6-REMOTE-SUPERVISION.md). It specifies the
`step7_webapp_integration` module. It builds strictly on the contracts in
[`00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md) — the
step-module interface ([`00` §12](00-FRAMEWORK-AND-BOOTSTRAP.md#12-step-module-interface-the-contract-for-steps-15)),
install-location config ([`00` §11](00-FRAMEWORK-AND-BOOTSTRAP.md#11-install-location-config)),
the reporting/verify strings ([`00` §8.3–8.4](00-FRAMEWORK-AND-BOOTSTRAP.md#83-reporting-format-for-dependencies-required-exact-strings)),
the USER-ACTION display ([`00` §9.3](00-FRAMEWORK-AND-BOOTSTRAP.md#93-user-action-display-contract)),
and above all the web-app credential contract
([`00` §14](00-FRAMEWORK-AND-BOOTSTRAP.md#14-web-app-connection-contract)).
Those contracts are **not** restated here; only Step 7's own scope is.

[`STEP-6`](STEP-6-REMOTE-SUPERVISION.md) gives the desktop a **control plane**:
MQTT command topics that run, stop, and restart supervised pipelines. It says
nothing about how the desktop authenticates to the cloud web app, how any
artifact — calibration exports, MV3DT track records, floor-plan renders,
recorded clips — reaches it, or how the web app asks the desktop to perform a
one-shot operation. **Step 7 is that data plane**, and it is deliberately a
separate transport: plain HTTP with a bearer API key, outbound only.

Scope, from the product owner (LOCKED):

1. Transport is **HTTP** with `Authorization: Bearer <API_KEY>`, using the
   credential contract in [`00` §14](00-FRAMEWORK-AND-BOOTSTRAP.md#14-web-app-connection-contract).
   All connections are **outbound**; nothing listens.
2. File transfer uses a **three-phase signed-URL flow** — ask the backend for a
   pre-signed URL, `PUT` the bytes directly to storage, then confirm. Bytes
   never transit the API server.
3. The desktop reuses the **same backend** as the existing P2BP camera nodes,
   with its own routes ([§G](#g-route-table-open-decision)).
4. The web app can trigger **one-shot operations** on the desktop and read
   their results ([§F](#f-web-app-initiated-operations)).
5. The step is **opt-in** and defaults to off ([§H.2](#h2-gating-opt-in)); an
   air-gapped lab install is unaffected.
6. Anything beyond this — the web app itself, its API implementation, key
   issuance — is flagged for the human ([§I](#i-out-of-scope--flag-for-human)).

> **Provenance (REQUIRED reading for the implementer).** Every mechanism below
> is ported from the P2BP Jetson camera-node agent, which has run this exact
> contract against this backend in production. The source files are being
> removed from this fork — see
> [`DELETION-REVIEW` §3](DELETION-REVIEW.md#3-deletions-gated-on-the-harvest-the-jetson-tree)
> — so each mechanism is specified here **in full**, including the non-obvious
> failure handling, rather than by reference to code that will not exist. Where
> a behavior looks arbitrary, it is not: it is the residue of a real failure
> mode, called out in place.

---

## 1. Module identity and scope

Module: `installer/mv3dt_installer/steps/step7_webapp_integration.py`,
registered in `STEP_REGISTRY` with `order = 7`;
`id = "step7_webapp_integration"`; `title = "Web-app integration"`.

In scope:

1. Resolve and validate the web-app credential
   ([§A](#a-credentials-and-endpoint)).
2. Install and enable the artifact upload daemon
   ([§E](#e-artifact-upload-daemon)) and the status reporter
   ([§D](#d-registration-and-status)) as supervised units.
3. Provide the shared upload client ([§B](#b-signed-url-upload)) and DTO layer
   ([§C](#c-the-dto-contract)) that other steps call.
4. Wire the web-app-initiated operation loop
   ([§F](#f-web-app-initiated-operations)).

Out of scope for this step, but adjacent:

- Remote **control** of pipelines (run/stop/restart) —
  [`STEP-6`](STEP-6-REMOTE-SUPERVISION.md). The two planes are independent and
  either may be enabled alone ([§H.3](#h3-relationship-to-step-6)).
- MV3DT telemetry publishing — the DeepStream pipeline publishes that to the
  local broker itself and Step 7 does not touch it.

---

## 2. Inputs consumed and outputs produced

### 2.1 Inputs (via `Context`)

- `ctx.install_dir` / `ctx.conf` — install root (default `/opt/mv3dt`),
  resolved via `config.load()`
  ([`00` §11](00-FRAMEWORK-AND-BOOTSTRAP.md#11-install-location-config)).
  Never hardcoded.
- `ctx.webapp` — the credential handle
  ([`00` §14.3](00-FRAMEWORK-AND-BOOTSTRAP.md#143-capture--handoff-api-webapppy)):
  `load_credentials()`, `enabled()`.
- **Step 5's project registry** at `<install_dir>/projects/registry.json`
  ([`STEP-5` §4](STEP-5-PER-PROJECT-EXES.md#4-the-project-registry)) — the set
  of projects whose artifacts are uploaded and whose state is reported.
- The artifact producers listed in [§E.1](#e1-what-gets-uploaded).

### 2.2 Outputs

| Path | Contents |
|---|---|
| `/etc/systemd/system/mv3dt-uploader.service` | the artifact upload daemon ([§E](#e-artifact-upload-daemon)) |
| `/etc/systemd/system/mv3dt-reporter.service` | the registration/status reporter ([§D](#d-registration-and-status)) |
| `<install_dir>/webapp/uploaded_state.json` | upload dedupe state ([§E.2](#e2-dedupe-and-upload-state)) |
| `<install_dir>/webapp/config.json` | last config received from the web app ([§D.2](#d2-the-response-is-the-config)) |
| `<install_dir>/run/*.json` | per-worker state files, fanned in by the reporter ([§F.2](#f2-state-file-fan-in)) |

Operator-facing files are `chown`ed to the invoking user/group; unit files are
root-owned. Everything requiring root goes through `ctx.run_root(...)`
([`00` §12.3](00-FRAMEWORK-AND-BOOTSTRAP.md#123-context-object-passed-to-every-step)).
No `state.json` writes — Step 7 returns a `StepResult` and the framework owns
persistence ([`00` §12.2](00-FRAMEWORK-AND-BOOTSTRAP.md#122-stepresult-and-status-recorded-by-the-state-machine)).

---

## A. Credentials and endpoint

Fully specified by
[`00` §14](00-FRAMEWORK-AND-BOOTSTRAP.md#14-web-app-connection-contract) —
`<install_dir>/secrets/webapp.env`, `chmod 600`, `API_KEY` + normalized
`ENDPOINT`, the normalization table, and the redaction rules. Step 7 adds only:

- Both long-running units load the credential via `EnvironmentFile=` pointing
  at `secrets/webapp.env`. Because that file is `chmod 600` and owned by the
  invoking user, the units must run as that same user
  ([`00` §9.2](00-FRAMEWORK-AND-BOOTSTRAP.md#92-resolving-the-invoking-user--home)),
  not as `root` and not as `nobody`.
- Every request carries exactly two headers:

  ```
  Authorization: Bearer <API_KEY>
  Content-Type: application/json
  ```

  The **one exception** is the signed-URL `PUT`
  ([§B.1](#b1-the-three-phases)), which carries neither — see the warning
  there.
- A missing or unparseable credential is **never** a crash. `preflight`
  surfaces `USER_ACTION_REQUIRED` ([§H.1](#h1-lifecycle)); the running daemons
  log once and idle rather than exiting in a restart loop.

---

## B. Signed-URL upload

The file-transfer mechanism. The backend never proxies bytes: it mints a
pre-signed storage URL, the desktop `PUT`s directly to storage, then tells the
backend the object landed. This keeps large MV3DT exports and MP4 clips off the
API server entirely.

### B.1 The three phases

1. **Request an upload URL.**

   ```
   POST <endpoint>/api/files/request-upload
   Authorization: Bearer <API_KEY>
   Content-Type: application/json

   {"PathFromRoot": "/vision/tracks-raw", "FileName": "tracks_events-20260810",
    "Extension": "jsonl", "SizeBytes": 184320}
   ```

   Response is an `UploadUrlResponseDto` ([§C](#c-the-dto-contract)). An empty
   `SignedUrl` in an otherwise-`200` response is a **failure**, not a success —
   check it explicitly and raise, because the backend has been observed to
   return `200` with an empty body field under load.

2. **`PUT` the bytes to the signed URL.**

   ```
   PUT <SignedUrl>
   Content-Length: <size_bytes>
   Content-Type: application/octet-stream
   ```

   > **REQUIRED — no `Authorization` header on this request.** The signature in
   > the query string *is* the credential. Sending a bearer token alongside it
   > causes some object stores to reject the request as over-specified. Send
   > the file object as a streaming body; do not read it into memory.
   > Timeouts are asymmetric: ~10 s to connect, ~300 s to read, because a large
   > clip legitimately takes minutes.

3. **Confirm the upload.**

   ```
   POST <endpoint>/api/files/confirm-upload
   {"PathFromRoot": "/vision/tracks-raw", "FileName": "tracks_events-20260810",
    "Extension": "jsonl"}
   ```

   Returns a `MediaRecordResponseDto` whose `Id` is the backend's handle for
   the object. **Until this call succeeds the upload does not exist** as far as
   the web app is concerned — which is why it retries hardest
   ([§B.3](#b3-retry-policy-required)).

### B.2 Remote path shape

Callers pass a single POSIX-style `remote_path` such as
`/vision/tracks-raw/tracks_events-20260810.jsonl`. The client splits it into
the three DTO fields:

| Component | Value | Rule |
|---|---|---|
| `PathFromRoot` | `/vision/tracks-raw` | directory part; `/` if the file is at the root |
| `FileName` | `tracks_events-20260810` | basename **without** the extension |
| `Extension` | `jsonl` | extension **without** the leading dot |

Normalization before splitting: backslashes become forward slashes, surrounding
whitespace is stripped, and a leading `/` is added if absent. A path with no
filename or no extension is a `ValueError` — it is a caller bug, not a
transient failure, so it must not enter the retry path.

### B.3 Retry policy (REQUIRED)

The single most important rule, and the easiest to get wrong:

> **Retry only `429` and `5xx`. Every other `4xx` fails immediately.**

A `400`/`401`/`403`/`404` means the request itself is wrong — a bad key, a
malformed path, a route that does not exist. Retrying it burns the backoff
budget, floods the logs, and delays the real error reaching the operator by
minutes. Retrying a `429` or `5xx`, by contrast, is exactly right.

| Phase | Max attempts | Backoff base / cap | On non-retryable |
|---|---|---|---|
| `request-upload` | 5 | 0.5 s / 30 s | raise immediately |
| `PUT` to signed URL | 1 | — | raise; the outer daemon retries the whole file |
| `confirm-upload` | 8 | 1.0 s / 60 s | raise immediately |

Backoff is exponential **with jitter**:
`delay = min(cap, base * 2^(attempt-1))`, plus a random `0 … 25%` of that
value. The jitter matters when several artifacts fail at once — without it,
every retry re-collides on the same schedule.

`confirm-upload` gets the longest budget because a confirmed-but-unrecorded
object is the worst outcome: the bytes are in storage, paid for, and invisible.

### B.4 Preconditions and refusals

- **Refuse to upload a zero-byte file.** A `0`-length artifact is a producer
  that crashed mid-write, and uploading it overwrites a good previous copy with
  an empty one.
- **Refuse if the local file is missing** — raise `FileNotFoundError` rather
  than reporting a spurious success.
- Failure logging includes the HTTP status and a **truncated** response body
  (~2000 chars, with a `… <truncated N chars>` marker). Untruncated HTML error
  pages from a proxy will otherwise flood journald.
- Every logged URL passes through the redaction helper in
  [`00` §14.4](00-FRAMEWORK-AND-BOOTSTRAP.md#144-redaction-required).

---

## C. The DTO contract

**RESOLVED — field names are fixed by the existing backend.** Per the "same
backend, new endpoints" decision ([§G](#g-route-table-open-decision)), the
routes are open but the payload shapes are not: these are the wire types the
backend already serializes, in PascalCase. Renaming a field to a more
Pythonic form breaks deserialization server-side. They are inlined here in full
because their source file is being removed.

```python
@dataclass(frozen=True)
class RequestUploadUrlDto:
    PathFromRoot: str
    FileName: str
    Extension: str
    SizeBytes: int

@dataclass(frozen=True)
class UploadUrlResponseDto:
    PathFromRoot: str
    SignedUrl: str
    ExpiresAt: Optional[str] = None      # ISO 8601, informational

@dataclass(frozen=True)
class ConfirmUploadedMediaDto:
    PathFromRoot: str
    FileName: str
    Extension: str

@dataclass(frozen=True)
class MediaRecordResponseDto:
    Id: str
    Name: str
    PathFromRoot: str
    Extension: str
```

**REQUIRED — defensive deserialization.** Every `from_dict` coerces rather than
trusting the payload: `str(data.get("Field") or "")` for strings,
`int(... or 0)` for integers. A backend that omits a field, or sends `null`
where a string is expected, must yield an empty value and a usable object — not
a `KeyError` or a `TypeError` inside a retry loop, where the traceback would be
attributed to the wrong cause.

A matching download pair (`RequestDownloadUrlDto` /
`DownloadUrlResponseDto`, same shape with `PathFromRoot` and `SignedUrl`)
exists in the backend contract and should be carried over when a step needs to
*fetch* an artifact; nothing in Step 7 currently does.

---

## D. Registration and status

A long-running reporter — `mv3dt-reporter` — that periodically posts the
desktop's health to the web app and applies whatever configuration comes back.
It is the HTTP sibling of the MQTT status publisher in
[`STEP-6` §C.5](STEP-6-REMOTE-SUPERVISION.md#c5-status--heartbeat-json-schema-desktop--cloud),
and the two **must** keep identical field names ([§D.1](#d1-the-status-payload)).

### D.1 The status payload

```json
{
  "Timestamp": 1786500072,
  "Services": {
    "mv3dt-pipeline@north-lobby-2": { "Active": "active",   "Sub": "running" },
    "mv3dt-agent":                  { "Active": "active",   "Sub": "running" },
    "mosquitto":                    { "Active": "active",   "Sub": "running" }
  },
  "System": {
    "Gpu":    { "UtilizationPct": 61, "FrequencyMhz": -1 },
    "Memory": { "UsedMb": 4210, "TotalMb": 32768 },
    "Disk":   [ { "Path": "/opt/mv3dt", "TotalMb": 940000, "UsedMb": 612000,
                  "FreeMb": 328000, "UsePct": 65, "Status": "ok",
                  "DeletedFiles": 0 } ]
  }
}
```

- `Timestamp` — integer Unix seconds, not ISO-8601. (The MQTT plane uses
  ISO-8601 `ts`; this plane predates it and the backend parses integers. Do not
  "fix" one to match the other.)
- `Services` — one entry per supervised unit, keyed by unit name with the
  `.service` suffix **stripped**. Values come from:

  ```bash
  systemctl show <unit> -p ActiveState -p SubState
  ```

  parsed line-wise into `{"Active": ..., "Sub": ...}`. **Any failure — timeout,
  missing unit, non-zero exit — yields `{"Active": "unknown", "Sub":
  "unknown"}`, never an exception.** One unreadable unit must not cost the
  whole status report. Use a short subprocess timeout (~5 s). The unit set is
  enumerated from the Step 5 registry plus the fixed `mv3dt-agent` and
  `mosquitto`, so projects added or removed are reflected without a restart.
- `System.Gpu` / `System.Memory` — collected from:

  ```bash
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
             --format=csv,noheader,nounits
  ```

  `-1` is the sentinel for "unavailable" in every numeric field, so the web app
  can distinguish "no data" from a genuine zero. `FrequencyMhz` has no
  `nvidia-smi` equivalent in this query and stays `-1`.
- `System.Disk` — a list, fanned in from the disk worker's state file
  ([§F.2](#f2-state-file-fan-in)); an empty list when that worker has not run.

> **Drift note from the Jetson original:** the GPU/memory collector there
> shelled out to `tegrastats` and scraped `GR3D_FREQ` / `RAM` with regexes.
> That is Jetson-only. The **payload shape carries over unchanged**; only the
> collector is replaced. This is the one place where a like-for-like port would
> be wrong.

### D.2 The response is the config

**REQUIRED — the defining property of this loop.** The status POST is not
fire-and-forget: the response body **is** the desktop's new configuration
document. The reporter:

1. Loads the current config from `<install_dir>/webapp/config.json`.
2. Posts the status payload.
3. Takes the JSON response as the new config.
4. Applies it (see [§F](#f-web-app-initiated-operations)) — this happens on
   **every** successful response, even when the document is byte-identical, so
   a desired-state change is never missed because of a comparison bug.
5. Writes it to disk **only if it differs** from the loaded copy, using the
   atomic helper in
   [`00` §6.3](00-FRAMEWORK-AND-BOOTSTRAP.md#63-api-statepy).

Steps 4 and 5 are deliberately asymmetric: *apply always, persist on change.*
Persisting unconditionally would rewrite the file every few seconds and make
the mtime edge-trigger in [§F.1](#f1-command-by-config-with-an-mtime-edge-trigger)
fire continuously.

A response that is not valid JSON is an error, logged and retried on the next
cycle; the previous config stays in force.

### D.3 Cadence

| Condition | Interval |
|---|---|
| Steady state | `HeartbeatInterval` from the config, default **10 s** |
| Floor (any configured value) | **2 s** — clamps a bad config out of a busy loop |
| An operation is active ([§F.1](#f1-command-by-config-with-an-mtime-edge-trigger)) | **2 s** |
| Within 15 s of any config change | **2 s** |

The two fast-mode triggers exist so an operator watching the web app sees a
one-shot operation progress at a usable rate, then the loop settles back. A
configured interval of `0` or a non-numeric value falls back to the default
rather than raising.

### D.4 Log discipline (REQUIRED)

This loop runs every few seconds forever, under journald, on a machine that
also runs DeepStream. Unbounded logging here has filled `/var/log` in practice.

- **Never log the full payload at `INFO`.** `DEBUG` only.
- Emit a compact `status OK (services=N pipelines=M payload=NNNB mem=X/YMB)`
  line **at most every 300 s**, not every cycle.
- Log a received config **only when it changes**, identified by the first 12
  hex characters of a SHA-256 over its stable JSON encoding
  (`sort_keys=True, separators=(",", ":"), ensure_ascii=False`). Stable
  encoding matters: without `sort_keys` an unchanged document hashes
  differently between runs and every cycle looks like a change.
- Rate-limit errors by `(type, first 200 chars of message)` — log a repeat of
  the same error at most once per 60 s.

---

## E. Artifact upload daemon

A long-running daemon — `mv3dt-uploader` — that watches for finished artifacts
and ships them via [§B](#b-signed-url-upload). It is what makes the
workstation's output visible in the web app without an operator copying files.

### E.1 What gets uploaded

| Producer | Artifact | Watch directory | Suggested remote prefix |
|---|---|---|---|
| the bundled `60_record_tracking.sh`, run through `<install_dir>/bin/record-<slug>` ([`STEP-5` §1.2](STEP-5-PER-PROJECT-EXES.md#12-outputs-what-step-5-writes)) | `tracks.jsonl`, `tracks.csv`, `summary.json` | `<install_dir>/tracking_exports/` | `/vision/tracks-raw` |
| [`STEP-4`](STEP-4-CALIB-OUTPUT-WIRING.md) | the ingested AMC calibration export | the chosen calibration dir | `/vision/calibration/<LOCATION_ID>` |

Watch directories and their remote prefixes are configuration, not constants —
read them from `installer.conf` so a deployment can retarget without a rebuild.

> **Visualizations are the web app's job (product decision).** Floor-plan
> trajectory plots and per-camera MP4 clips are **not** produced by the
> installer and have no upload rows: `70_plot_floorplan.py` and
> `record_cameras_mp4.sh` are not bundled into the binary, and nothing on the
> workstation renders them. The web app renders its own visualizations from the
> uploaded `tracks.jsonl` / `tracks.csv`, which keeps a matplotlib dependency
> out of the release build. Both scripts remain in `laptop/scripts/` as
> developer tools.

### E.2 Dedupe and upload state

State lives at `<install_dir>/webapp/uploaded_state.json`, written with the
atomic helper ([`00` §6.3](00-FRAMEWORK-AND-BOOTSTRAP.md#63-api-statepy)) after
**every** state transition — not once per scan, so a crash mid-scan cannot lose
the record of a completed upload:

```json
{
  "tracks_events-20260810.jsonl": {
    "size": 184320.0,
    "mtime": 1786500072.4,
    "uploaded_at": 1786500080.1,
    "failed_attempts": 0,
    "last_failed_at": null
  }
}
```

**REQUIRED — the fingerprint rule.** A file is (re)uploaded when it is new to
the state, or when `size != previous size` **or**
`mtime > previous mtime + 1e-6`. The epsilon absorbs float round-tripping
through JSON; without it a file can re-upload forever on filesystems with
sub-microsecond timestamps. Content hashing is deliberately **not** used —
these artifacts reach hundreds of megabytes and hashing every one every scan
would saturate the disk the pipeline is writing to.

Two further guards:

- **Minimum age** — skip files modified within the last *N* seconds
  (configurable, default `0` for append-only JSONL, but set it non-zero for
  MP4s so a still-recording clip is not uploaded half-written).
- **Legacy state hydration** — an entry lacking `size`/`mtime` is filled in
  from the current file and **skipped this cycle** rather than re-uploaded. The
  Jetson original stored a bare timestamp per file; this preserves the "already
  uploaded" fact across the format change. Keep it: it costs four lines and
  prevents a mass re-upload on first run after an upgrade.

### E.3 Failure handling and cooldown

Each failure increments `failed_attempts` and stamps `last_failed_at`. Once
`failed_attempts >= max_attempts` (default 5), the file enters a **cooldown**
(default 1800 s) during which it is skipped entirely. After the cooldown
elapses it is retried and, on success, the counter resets to zero.

This is the outer loop that makes [§B.3](#b3-retry-policy-required)'s
fail-fast policy safe: a permanently-bad file (wrong extension, rejected by the
backend) is attempted 5 times, then once every 30 minutes, instead of on every
scan forever. Log the cooldown expiry as a wall-clock time — an operator
reading journald needs to know when it will next try, not how many seconds
remain.

The scan interval itself defaults to 240 s. Uploading is not latency-sensitive;
the cost of scanning aggressively is contention with the pipeline's own writes.

### E.4 Disk-pressure remediation

A 24/7 workstation writing MV3DT tracks and MP4 clips fills its disk. Nothing
else in the installer addresses this, and a full disk takes down the pipeline,
the broker, and journald together.

The uploader carries a monitor pass on each cycle:

| Partition | Threshold | Action |
|---|---|---|
| the artifact directories | free `< critical` (default 512 MB) | delete oldest **already-uploaded** artifacts until free space recovers |
| `/var/log` | free `< critical` | `journalctl --vacuum-size=<target>` (default 512 MB) |
| `/` | free `< warning` (default 2 GB) | report only — never auto-delete on the root filesystem |

**REQUIRED — never delete an artifact that has not been confirmed uploaded**
([§B.1](#b1-the-three-phases) phase 3), and never delete a file younger than
the minimum age. The deletion pass and the upload pass share the same state
file, which is what makes "already uploaded" answerable locally.

Every cycle writes a state file consumed by the reporter
([§F.2](#f2-state-file-fan-in)) so disk status reaches the web app as
`System.Disk` ([§D.1](#d1-the-status-payload)), with per-partition `Status` of
`ok` | `warning` | `critical` and a `DeletedFiles` count. All thresholds are
environment-overridable; the defaults above are the starting point, not a pin.

---

## F. Web-app-initiated operations

The complement to [`STEP-6`](STEP-6-REMOTE-SUPERVISION.md)'s MQTT commands.
Where Step 6 answers "start/stop/restart this pipeline," this answers "perform
this one-shot operation and tell me when it finished" — a calibration re-run, a
recording session, a diagnostic capture. It rides the config document already
being exchanged in [§D.2](#d2-the-response-is-the-config), so it needs no
second connection and no inbound port.

The full loop: the web app sets a flag in the config → the reporter writes the
config to disk → a worker notices and acts → the worker publishes its state →
the reporter folds that state into the next status POST → the worker clears the
flag.

### F.1 Command-by-config with an mtime edge-trigger

**REQUIRED — this is the non-obvious part.** A worker must **not** act merely
because a flag is `true`. It acts only when the config file has been *freshly
written* **and** the flag is `true`. Without the freshness test, a flag left
`true` in the backend's stored document re-triggers the operation on every poll
cycle, forever.

The worker keeps two values and polls the config file's `mtime`:

```python
last_seen_mtime = None        # every write we have observed
last_fired_mtime = None       # the write that actually triggered us

while True:
    current = stat_mtime(config_path)          # None if absent
    if current is not None and current != last_seen_mtime:
        begin = read_flag(config_path)         # e.g. Calibration.BeginRun
        last_seen_mtime = current
        if begin and current != last_fired_mtime:
            last_fired_mtime = current
            run_operation()                    # exactly once per fresh write
    sleep(poll_interval)
```

Three properties fall out, and all three are load-bearing:

- **Edge, not level** — a stale `true` cannot re-fire, because its write has
  already been recorded in `last_fired_mtime`.
- **No writes from the poller** — the worker never touches the config to
  "consume" the flag, so it cannot race the reporter, which owns that file.
- **Absent file is not an error** — `stat` failure yields `None` and the loop
  simply waits.

Note the interaction with [§D.2](#d2-the-response-is-the-config): the reporter
persists the config **only on change**, which is precisely what keeps `mtime`
meaningful. If that rule is broken, this one silently breaks too.

### F.2 State-file fan-in

Workers do not talk to the web app. Each writes a small JSON file under
`<install_dir>/run/`, and the reporter folds them into the next status POST:

| Writer | File | Folded into |
|---|---|---|
| uploader / disk monitor ([§E.4](#e4-disk-pressure-remediation)) | `run/disk_state.json` | `System.Disk` |
| a one-shot operation worker | `run/<operation>_state.json` | a top-level key named for the operation |

**REQUIRED — the reader is total.** Missing file, unreadable file, malformed
JSON: every case yields the empty default (`[]` or `{}`), never an exception.
A crashed worker must degrade one field of the status report, not stop the
report from being sent. The corresponding rule for writers is the atomic helper
([`00` §6.3](00-FRAMEWORK-AND-BOOTSTRAP.md#63-api-statepy)) — a reader must
never observe a half-written file.

The payoff is decoupling: a new worker starts reporting to the web app by
dropping a file in a directory, with no change to the reporter, no shared
process, and no IPC.

### F.3 Acknowledgement by config write-back

When a worker finishes a one-shot operation it **clears the flag that requested
it** — writing the config back with, for example,
`Calibration.BeginRun = false`. That single write is the acknowledgement, and
it is the only config write a worker ever performs.

Combined with [§F.1](#f1-command-by-config-with-an-mtime-edge-trigger) this
closes the loop: the web app sets the flag and watches it return to `false`,
which means "the desktop received it and finished." The operation's *result*
travels separately, through [§F.2](#f2-state-file-fan-in) — keeping "did it
happen" and "what came of it" on separate channels, so a large result payload
never blocks the acknowledgement.

> **Interaction to respect:** a worker's write-back changes the config `mtime`,
> which every other worker observes as a fresh write. That is harmless because
> *their* flags are `false`, but it is why the freshness test and the flag test
> are separate conditions in [§F.1](#f1-command-by-config-with-an-mtime-edge-trigger)
> rather than one combined check.

---

## G. Route table (open decision)

Per the product owner: **same backend, new endpoints.** The auth scheme, the
DTO field names ([§C](#c-the-dto-contract)), the three-phase upload flow, and
the normalization rules are all settled and carry over unchanged. Only the
route *names* for the workstation are open.

| Concern | Existing camera-node route | Desktop route | State |
|---|---|---|---|
| Registration / status | `POST /api/Device/heartbeat` | `POST /api/Workstation/heartbeat` | **proposed — confirm with the backend owner** |
| Upload URL request | `POST /api/files/request-upload` | unchanged | settled — the file service is generic |
| Upload confirm | `POST /api/files/confirm-upload` | unchanged | settled |

> **Flagged for human.** The `/api/Workstation/*` name is a working proposal,
> not a decision. It must be confirmed against the deployed API before
> implementation, and this table updated with a **RESOLVED** marker naming who
> confirmed it. Until then `verify()` must not treat a `404` on the
> registration route as an installation defect — it surfaces
> `USER_ACTION_REQUIRED` pointing here ([§H.1](#h1-lifecycle)).

Routes are always constructed as `<normalized ENDPOINT> + <route>`, never
hardcoded with a host, and never assembled with `urljoin`
([`00` §14.2](00-FRAMEWORK-AND-BOOTSTRAP.md#142-endpoint-normalization-required)).

---

## H. Framework integration

Implements the `Step` protocol
([`00` §12.1](00-FRAMEWORK-AND-BOOTSTRAP.md#121-protocol)).

### H.1 Lifecycle

#### `preflight(ctx)`

- If the gate is `off` ([§H.2](#h2-gating-opt-in)), return `COMPLETE`
  immediately — nothing to do.
- Confirm `ctx.webapp.load_credentials()` returns both values. Missing →
  `USER_ACTION_REQUIRED` listing the two keys to set in
  `<install_dir>/secrets/webapp.env`.
- Confirm **Step 5 is `COMPLETE`** and `registry.json` exists — the status
  payload and the upload prefixes are keyed by project.
- Confirm outbound reachability with a single cheap request to the
  registration route. A connection failure → `FAILED` with the endpoint shown
  (redacted per [`00` §14.4](00-FRAMEWORK-AND-BOOTSTRAP.md#144-redaction-required)).
  A `401`/`403` → `USER_ACTION_REQUIRED` ("the API key was rejected"). A `404`
  → `USER_ACTION_REQUIRED` pointing at the open route decision
  ([§G](#g-route-table-open-decision)).
- All good → `COMPLETE`.

#### `run(ctx)`

1. Create `<install_dir>/webapp/` and `<install_dir>/run/`, chowned to the
   invoking user.
2. Install `mv3dt-uploader.service` and `mv3dt-reporter.service` from bundled
   assets ([`00` §4.2](00-FRAMEWORK-AND-BOOTSTRAP.md#42-locating-and-staging-bundled-assets-at-runtime)),
   `systemctl daemon-reload`, then `systemctl enable --now` both — mirroring
   the flow in [`STEP-6` §A.4](STEP-6-REMOTE-SUPERVISION.md#a4-installer-integration-mirror-installsh).
3. Report each unit via the framework reporters
   ([`00` §8.3](00-FRAMEWORK-AND-BOOTSTRAP.md#83-reporting-format-for-dependencies-required-exact-strings)),
   using the **exact strings**:

   ```
   installed mv3dt-reporter.service version 7.0.0
   installed mv3dt-uploader.service version 7.0.0
   already installed mv3dt-reporter.service version 7.0.0
   ```

Both units mirror the conventions in
[`STEP-6` §B.1](STEP-6-REMOTE-SUPERVISION.md#b1-the-agent-systemd-unit):
`Type=simple`, `Restart=on-failure`, `RestartSec=5`, `StartLimitBurst=5`,
journald output with `LogRateLimitIntervalSec=30s`, `EnvironmentFile=` for both
`installer.conf` and `secrets/webapp.env`, `WantedBy=multi-user.target`.
Neither requires `mosquitto` — this plane is independent of the broker.

#### `verify(ctx)`

Idempotent; `COMPLETE` only when all pass:

- [ ] Both units are `is-enabled == enabled` and `is-active == active`.
- [ ] `secrets/webapp.env` exists with mode `0600` and both keys set.
- [ ] A round-trip status POST returns valid JSON within a timeout — the
      end-to-end proof that credentials, endpoint, and route all agree.
- [ ] `<install_dir>/webapp/config.json` exists after that round trip.
- [ ] A **dry-run upload** resolves a real artifact to its `PathFromRoot` /
      `FileName` / `Extension` triple and prints the target without
      transferring bytes ([§B.2](#b2-remote-path-shape)).

Under `--non-interactive` with no reachable backend, the round-trip checks
degrade to "units loaded and credentials present" and the step reports the
network checks as skipped rather than failing.

#### `report(ctx)`

Prints, with no side effects: the resolved endpoint (host only, never the key),
the two enabled units with their states, the watch directories and their remote
prefixes, the resolved routes, and the `journalctl -u mv3dt-reporter` /
`journalctl -u mv3dt-uploader` commands.

### H.2 Gating (opt-in)

Gated by `MV3DT_WEBAPP_INTEGRATION` in `installer.conf`
([`00` §3.4](00-FRAMEWORK-AND-BOOTSTRAP.md#34-opt-in-step-gates)):

| Value | Behavior |
|---|---|
| `off` (default) | dispatch treats Step 7 as auto-`COMPLETE`; no units, no outbound connections |
| `on` | full integration; requires the §A credential |

Under `--non-interactive` an unset value stays `off`. An unattended install
must never start phoning a cloud endpoint the operator did not configure.

### H.3 Relationship to STEP-6

| | [`STEP-6`](STEP-6-REMOTE-SUPERVISION.md) | STEP-7 (this doc) |
|---|---|---|
| Plane | control | data |
| Transport | MQTT over the local broker (+ optional cloud bridge) | HTTPS, outbound |
| Verbs | `run` / `stop` / `restart` / `status` / `list` | status + config pull, artifact upload, one-shot operations |
| Identity | `<HOST_ID>` topic namespace | API key + endpoint |
| Requires the broker | yes | no |

The two are **independent**: either may be enabled alone, and neither reads the
other's state. They overlap only in that both report per-unit `{Active, Sub}`
state — which is exactly why
[`STEP-6` §C.5](STEP-6-REMOTE-SUPERVISION.md#c5-status--heartbeat-json-schema-desktop--cloud)
and [§D.1](#d1-the-status-payload) must keep identical field names, so the web
app can consume either without a second model.

---

## I. Out of scope / flag for human

- **The web app itself** — its UI, its API implementation, per-host dashboards,
  and how it authorizes an operator to trigger an operation. Step 7 defines
  only the desktop half and the contract it honors.
- **Route confirmation** — the `/api/Workstation/*` names in
  [§G](#g-route-table-open-decision) are proposed, not agreed.
- **API-key issuance, scoping, and rotation** — the framework stores a key
  ([`00` §14](00-FRAMEWORK-AND-BOOTSTRAP.md#14-web-app-connection-contract))
  but never provisions one. Mirrors the treatment of broker credentials in
  [`STEP-6` §8](STEP-6-REMOTE-SUPERVISION.md#8-out-of-scope--flag-for-human).
- **Download/restore flows** — the DTO pair exists in the backend contract
  ([§C](#c-the-dto-contract)) but no Step 7 path fetches artifacts back.
- **Retention policy on the backend** — [§E.4](#e4-disk-pressure-remediation)
  manages *local* disk only. How long the web app keeps an uploaded artifact is
  a cloud-side decision.
- **Multi-tenant / fleet identity** — one key, one workstation. Fleet
  orchestration lives in the web app, as it does for
  [`STEP-6` §8](STEP-6-REMOTE-SUPERVISION.md#8-out-of-scope--flag-for-human).

---

## References

No DeepStream 9.1 documentation is cited: nothing in this step is a DeepStream
fact. The authority for the transport, the DTO field names, and the failure
handling is the deployed P2BP backend and the camera-node agent that has been
speaking to it — captured here in full because those sources are leaving this
fork.

Repo files referenced:

- [`00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md) — §3.4 the
  opt-in gate, §6.3 the atomic-write helper, §8.3 reporting strings, §9.2/§9.3
  privilege and USER-ACTION display, §11 install layout, §12 the Step protocol,
  and §14 the credential/endpoint/redaction contract this step consumes.
- [`STEP-4-CALIB-OUTPUT-WIRING.md`](STEP-4-CALIB-OUTPUT-WIRING.md) — produces
  the calibration export listed in [§E.1](#e1-what-gets-uploaded).
- [`STEP-5-PER-PROJECT-EXES.md`](STEP-5-PER-PROJECT-EXES.md) — §4 the project
  registry that keys the status payload and the upload prefixes.
- [`STEP-6-REMOTE-SUPERVISION.md`](STEP-6-REMOTE-SUPERVISION.md) — §A.4 the
  unit-install flow mirrored by [§H.1](#h1-lifecycle), §B.1 the unit
  conventions, §C.5 the status payload that must stay field-identical, and §8
  the precedent for flagging credential provisioning.
- [`DELETION-REVIEW.md`](DELETION-REVIEW.md) — §3.2 lists the five behaviors
  this document must capture before the source files are removed.
- [`laptop/scripts/60_record_tracking.sh`](../../laptop/scripts/60_record_tracking.sh)
  — the sole script-based artifact producer in
  [§E.1](#e1-what-gets-uploaded); bundled into the installer binary and driven
  by the generated `record-<slug>` exe
  ([`STEP-5` §1.2](STEP-5-PER-PROJECT-EXES.md#12-outputs-what-step-5-writes)),
  writing into `<install_dir>/tracking_exports/`.

> **Attribution — sources no longer in this fork.** [§B](#b-signed-url-upload)
> ports `scripts/cloud_storage_media.py`; [§C](#c-the-dto-contract) ports
> `scripts/json_models/cloud_storage.py`; [§D](#d-registration-and-status)
> ports `scripts/heartbeat.py`, `scripts/json_models/heartbeat_payload.py`,
> `scripts/systemd_services.py`, and `scripts/system_stats.py`;
> [§E](#e-artifact-upload-daemon) ports `scripts/tracking_uploader.py` and
> `scripts/disk_monitor.py`; [§F](#f-web-app-initiated-operations) ports the
> polling loops in `scripts/aruco_scanner.py` and
> `scripts/intrinsics_calibrator.py` and the write-back in
> `scripts/homography.py`. All are removed from this fork per
> [`DELETION-REVIEW` §3](DELETION-REVIEW.md#3-deletions-gated-on-the-harvest-the-jetson-tree)
> and remain in this repository's git history. Named as provenance only — deliberately
> not linked, so the links cannot rot.
