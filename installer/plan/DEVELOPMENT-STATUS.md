# DEVELOPMENT STATUS — where the installer stands (owner: shared)

Status: standing record and handoff artifact, not a spec. It states what is
built, what is not, and what a new contributor has to know before touching
the tree. Every contract it mentions is **defined elsewhere** — in
[`00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md), the seven
`STEP-*` docs, or [`08-PROGRESS-AND-OBSERVABILITY.md`](08-PROGRESS-AND-OBSERVABILITY.md)
— and is linked rather than restated. When this doc and a spec disagree, the
spec wins and this doc is stale.

This document exists because the repo has reached a state that is easy to
misread: **all seven steps and the progress subsystem are implemented, and
the apt and download adapters are wired, but failure evidence is not yet
connected across the step modules.** A reader who finds `report.py` can
reasonably conclude the diagnostic work is done. It is not.
[Section 5](#5-what-remains) is the honest list.

Current release: **v0.4.9**. Current version string:
`installer/mv3dt_installer/__init__.py` `__version__ = "0.4.9"`.

---

## 1. What this repo produces

One artifact: a **single self-contained `mv3dt-installer` binary**, built by
PyInstaller and published as a GitHub Release asset. An operator downloads
it, verifies the checksum, and runs it with `sudo` on a bare Ubuntu 24.04 /
x86_64 workstation.

**Nothing clones this repository onto a workstation.** No step, script, or
doc may assume a checkout is present at run time
([`00` §4.1](00-FRAMEWORK-AND-BOOTSTRAP.md#41-what-builds-the-binary)). Two
`laptop/scripts/` entries are bundled *into* the binary as assets; the rest
of `laptop/` is a developer harness exercised from a clone and is not how
anything installs.

The build and release path is `.github/workflows/`: pushing a `v*` tag builds
the binary and attaches it plus a `.sha256` to the release. The version in
`__init__.py` **must equal the pushed tag** or the build fails by design.

---

## 2. Current state, in one table

| Area | State | Where it lives |
|---|---|---|
| Framework (state machine, config, privilege, reboot, logging) | Implemented | `mv3dt_installer/*.py` |
| Steps 1-7 lifecycle (`preflight`/`run`/`verify`/`report`) | Implemented | `mv3dt_installer/steps/step*.py` |
| Progress renderer, bars, spinner, window | Implemented | `mv3dt_installer/progress.py` |
| Streaming command output through `run_root` | Implemented and wired | `app.py`, `shellout.py` |
| Step and phase banner | Implemented and wired | `app.py`, all seven steps |
| Download byte adapter (`follow_download`) | Implemented and wired | `progress.py`, `progress_exec.py`, Steps 1-2 |
| apt percentage adapter (`follow_apt`) | Implemented and wired | `progress.py`, `progress_exec.py`, Steps 1-2 |
| Desktop-safe NVIDIA driver handoff | Implemented and wired | Step 1, `assets/systemd/mv3dt-driver-handoff.*` |
| Authenticated NGC Catalog API download for PeopleNet | Implemented and wired | Step 2 |
| Failure context block, refusal evidence | Implemented, **call sites not retrofitted** | `report.py` |
| Docker pull progress | Not started, open decision | — |

"Implemented" means the code exists, has tests, and is merged to `main`.
"Wired" means a step actually calls it during a real install. The gap between
those two columns is [section 5](#5-what-remains).

### 2.1 Test suite

From `installer/`:

```bash
python3 -m pytest tests/ -q
```

Current result on an **arm64 macOS** development machine:

```
1 failed, 1312 passed, 7 skipped
```

**The single failure is environmental, not a regression (REQUIRED to know
before you start).** The same suite passes fully in CI on x86_64 — the
`installer-tests` checks for PR #73 are green. The failure is:

| Test | Why it fails on this host |
|---|---|
| `test_config.py::test_persist_value_preserves_existing_keys_and_overwrites_its_own` | The macOS temporary path contains `folders`, while the assertion rejects the substring `old` anywhere in the generated configuration |

Treat **any second failure** as a regression you introduced.

---

## 3. What has been developed

### 3.1 The seven steps

All seven step modules implement the lifecycle contract in
[`00` §12](00-FRAMEWORK-AND-BOOTSTRAP.md#12-step-module-interface-the-contract-for-steps-15)
and register into `STEP_REGISTRY` at import.

| Step | Module | Lines | Tests | Spec |
|---|---|---|---|---|
| 1 | `step1_prerequisites.py` | 1536 | 76 | [`STEP-1`](STEP-1-PREREQUISITES.md) |
| 2 | `step2_deepstream_sdk.py` | 1259 | 63 | [`STEP-2`](STEP-2-DEEPSTREAM-SDK.md) |
| 3 | `step3_amc_launcher.py` | 1739 | 98 | [`STEP-3`](STEP-3-AMC-LAUNCHER.md) |
| 4 | `step4_calib_output_wiring.py` | 963 | 43 | [`STEP-4`](STEP-4-CALIB-OUTPUT-WIRING.md) |
| 5 | `step5_per_project_exes.py` | 1578 | 69 | [`STEP-5`](STEP-5-PER-PROJECT-EXES.md) |
| 6 | `step6_remote_supervision.py` | 1217 | 57 | [`STEP-6`](STEP-6-REMOTE-SUPERVISION.md) |
| 7 | `step7_webapp_integration.py` | 1632 | 79 | [`STEP-7`](STEP-7-WEBAPP-INTEGRATION.md) |

Steps 6 and 7 are **gated off by default**
([`00` §3.4](00-FRAMEWORK-AND-BOOTSTRAP.md#34-opt-in-step-gates)) and
complete immediately when their gate is `off`.

### 3.2 Step 1's two-launch structure (REQUIRED to understand)

Step 1 is the only step that spans a reboot, and it is the step most likely
to be edited next, so its shape is worth stating.

`run()` routes on an idempotent probe, `_driver_loaded(ctx)`, which shells
`nvidia-smi --query-gpu=driver_version`:

1. **Launch A** (driver not loaded) — base packages, CUDA repo and toolkit,
   nouveau and distro-driver cleanup, then the NVIDIA driver `.run`. A
   desktop launch hands the disruptive work to a persistent root-owned
   systemd worker, which records whether desktop recovery succeeds after a
   failure or reboots automatically after driver success. TTY and SSH
   launches remain synchronous and also recover the display manager when the
   runfile fails.
2. **Launch B** (driver loaded) — TensorRT, cuDNN, Mosquitto.

Both reboot points return `USER_ACTION_REQUIRED`, **never**
`REBOOT_REQUIRED`. This is deliberate and is documented at the call site: the
merged `reboot.reconcile()` marks the *requesting* step `COMPLETE` as soon as
it confirms the reboot, and `_dispatch()` skips a `COMPLETE` step without
re-running its lifecycle. A `REBOOT_REQUIRED` there would let the framework
auto-complete Step 1 before the driver `.run` — let alone TensorRT — ever
ran. Do not "simplify" this.

The version pins are equality pins, not minimums: DS 9.1 refuses older and
newer minors alike. They live in
[`STEP-1` §2](STEP-1-PREREQUISITES.md#2-the-ds-91-dgpu-prerequisite-pins-equality) and in
`laptop/docs/DEEPSTREAM-SETUP.md`.

### 3.3 The progress and observability subsystem

Specified by [`08`](08-PROGRESS-AND-OBSERVABILITY.md), which was written
against **five real failures observed on the workstation during the 0.1.2
through 0.1.9 install runs** ([`08` §2](08-PROGRESS-AND-OBSERVABILITY.md#2-observed-failures-this-doc-exists-to-fix)),
not against a general wish for nicer output. Two of those five caused the
operator to interrupt work that was proceeding normally; one of those
interruptions broke an apt transaction and required `dpkg --configure -a` to
recover. **Silence is not cosmetic here: it caused operators to break working
installs.**

It was built as 13 units across 8 dependency-ordered waves. All are merged:

| Unit | What it added | Commit |
|---|---|---|
| U1 | Renderer core: bars, spinner, tty and non-tty modes | `55a9cc3` |
| U2 | Optional `phases` declaration on the step interface | `3d0deb1` |
| U3 | Tee runner: stream, capture and redact in one pass | `4a55a2e` |
| U4 | `ctx.progress` handle and streaming `run_root` | `57ac5a6` |
| U5a | Step and phase banner in the dispatch loop | `d4156d8` |
| U5b | The four rendering defects in [`08` §12.2](08-PROGRESS-AND-OBSERVABILITY.md#122-known-defects-u5-carries) | `5775a9e` |
| U6 | Download byte-progress adapter | `ffa4eaf` |
| U7 | apt `Status-Fd` percentage adapter | `9596819` |
| U8/U9 | Phase declarations and task naming, all seven steps | `c7d93ec` |
| U10 | Failure context block and inferred-refusal evidence | `f796085` |
| U11 | `--verbose` and the resolved AUTO streaming rule | `7a9577f` |
| U12 | Transcript-only sink | `60da882` |
| U13 | Parsed `--non-interactive` settles the colour question | `a3ba33a` |

The post-v0.3.0 integration commit `5b94392` connects the apt and download
adapters through the shared `progress_exec.py` orchestration seam. It keeps
the renderer single-writer, retains captured output and the transcript, and
preserves Step 2's invoking-user download contract.

### 3.4 Design decisions worth not re-litigating

Each of these was contested during review and settled with a measurement.
They are recorded here because the obvious alternative is wrong in a way
that is not obvious.

- **Never render a bar without a true denominator (REQUIRED,
  [`08` §5.3](08-PROGRESS-AND-OBSERVABILITY.md#53-no-fabricated-progress-required)).**
  An operation whose duration cannot be known reports elapsed time and its
  last output line. A fabricated bar is worse than none, because the operator
  cannot tell a wrong denominator from a right one. The same reasoning makes
  `ctx.progress.phase(n)` raise on an out-of-range index rather than render.
- **The live region must not be opened for a step that declares no phases —
  and must not be skipped either.** Opening a region only once a step
  declares a phase was proposed in review and is measurably wrong: with no
  region open, `Progress.line()` appends to the window, `_draw()` finds
  `_phase_started is None`, and the terminal receives **zero bytes**. Every
  streamed line would vanish on a tty. The fix taken instead routes foreign
  writes through the renderer's `_erase` / write / `_draw` sandwich.
- **AUTO streaming is not tty-gated (RESOLVED,
  [`00` §8.5](00-FRAMEWORK-AND-BOOTSTRAP.md#85-verbosity-and-live-command-output-required)).**
  `stream=None` streams whenever the tee can serve the call — captured output
  and text mode — and nothing in the decision reads `isatty`. Note that the
  example first used to justify this, `mv3dt-installer | tee install.log`,
  is **wrong**: a pipe redirects stdout only, so stderr stays on the terminal.
  The real cases are `2>&1 | tee`, redirection to a file, CI, and the
  journald-backed STEP-6 units.
- **There is no `--quiet` (RESOLVED, [`08` §11](08-PROGRESS-AND-OBSERVABILITY.md#11-out-of-scope--open-decisions)).**
  It has no consumer. The STEP-6 systemd units pass no `--non-interactive`;
  what covers them is `StandardError=journal`, which is not a tty.
- **The transcript may never show less than the screen (REQUIRED,
  [`08` §7.1](08-PROGRESS-AND-OBSERVABILITY.md#71-live-rendering-must-never-cost-the-transcript-required)).**
  Any line suppressed for rendering reasons still reaches the record via
  `logs.transcript`. The screen may show less; the record may not.
- **`logs` carries two process-global mutables** — the colour override and
  the live-writer registration. Both are reset after every test by
  `installer/tests/conftest.py`. Without it,
  `test_dispatch_banner_degrades_to_a_plain_line_when_non_interactive` passes
  in a full run and fails in isolation. If you add global state to `logs`,
  reset it there too.

### 3.5 Steps 2 through 4 workstation path

Release v0.4.0 hardens the path from an installed DeepStream SDK to a wired
calibration result:

- Step 2 verifies the pinned prerequisite stack, installed DeepStream SDK and
  version, host profile/post-install wiring, Docker runtime when selected, and
  the PeopleNet artifact. The camera-free sample pipeline test was removed
  after repeated false failures on the target RTX PRO 4500 workstation; Step
  2 no longer downloads sample media, counts frames, waits on a runtime
  timeout, or writes an installation-test marker. Real frame-flow validation
  belongs to the calibrated production pipeline, not the installer gate. An
  actual PeopleNet download retains authenticated byte, rate and ETA progress
  when NVIDIA declares the archive size.
- Step 3 installs Docker Engine, Compose v2 and the NVIDIA Container Toolkit
  when needed, checks out the pinned AutoMagicCalib 3.2.1 commit, validates
  container readiness, and persists the AMC location, project and API
  identity needed by Step 4. Reruns accept AMC-generated untracked data under
  `projects/` and `models/` while still rejecting tracked source edits, and
  reuse an already-running stack by recovering its actual Docker-published
  ports and repairing stale configuration. Expected cold-start readiness
  failures stay out of the live terminal while the spinner remains active,
  and cancellation is clean. The guided installer keeps AMC running after the browser closes; the
  standalone `amc` command retains close-to-stop behavior.
- Step 4 polls the persisted AMC project through its API, reports calibration
  errors, downloads the MV3DT result, rejects unsafe or malformed archives,
  requires a root `transforms.yml`, and atomically replaces the installed
  calibration. Camera credentials are captured once through onboarding and
  stored outside `installer.conf`; a missing inventory triggers the initial
  camera scan and position-binding flow automatically. Timer-driven re-ingest
  uses the same API path.

The remaining acceptance milestone is a live Ubuntu 24.04 workstation run
through Steps 2, 3 and 4: complete a real AMC calibration in the launched UI
and verify that Step 4 installs the
exported `transforms.yml`. Unit tests and frozen-build CI cover the known
failure branches, but do not replace that GPU and browser-backed run.

---

## 4. How to work in this repo

- **Commits carry no authorship trailer of any kind** — no
  `Co-Authored-By`, no session link, no agent footer. Enforced by the
  `git-commits` skill and stated in `CLAUDE.md`. The repo is public and
  Kevin is the sole author of record.
- **Prose style for commits, PRs and docs**: no emoji, no section-symbol
  character in commit or PR text (write "section 8.1"), no double hyphen as
  punctuation, no mention of AI or agent authorship.
- **Markdown** follows the `markdown-docs` skill, whose reference is
  [`00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md).
- **Non-trivial development** uses the `parallel-worktree-dev` skill:
  dependency-ordered waves, one PR per unit, fresh-context reviewers posting
  plain `LGTM` or `REQUEST_CHANGES` comments. Reviewers must **never** use
  `gh pr review` — every agent shares one `gh` identity with the PR author
  and GitHub rejects self-approval with HTTP 422.
- **Merging** uses the `merge-prs` command, which retargets every stacked PR
  to `main` via the REST API before merging it. `gh pr edit --base` fails on
  this repo with a GraphQL error about deprecated Projects Classic fields.
- **Squash-merge creates add/add conflicts for stacked branches.** A
  squash-merged file lands on `main` as a new commit with no shared
  ancestry, so a branch stacked on it reports a conflict on a file neither
  side actually changed. Resolve by file ownership, and verify the result is
  byte-identical or purely additive rather than trusting the merge.

---

## 5. What remains

### 5.1 Retrofit the failure-context call sites

U10 built `report.FailureContext`, `report.failure()`, `report.refusal()`
and `report.with_evidence()`, but **no step calls them yet**. Today a step
that fails still discards the `CompletedProcess` that would explain why.

[`08` §6](08-PROGRESS-AND-OBSERVABILITY.md#6-failure-and-refusal-messages-carry-their-evidence)
is REQUIRED and generalises across steps 1-7: every `FAILED` and
`USER_ACTION_REQUIRED` result whose message is the product of an inference
must name the inputs that inference was drawn from. The already-correct
precedent is Step 1's `_session_hazard`, which returns an evidence string
rather than a bare verdict, and `report.with_evidence` accepts that exact
shape unchanged.

### 5.2 Open decision for the human

**Docker pull progress** ([`08` §11](08-PROGRESS-AND-OBSERVABILITY.md#11-out-of-scope--open-decisions),
item 3). Step 2 resolves its install method to `deb`, `tar` or `docker` at
run time. The doc deliberately leaves this open: wire a bar only for
whichever path Step 2 actually takes in practice. This needs a decision
before it needs code.

### 5.3 Known residuals

- **`report._secrets` reads `os.environ`**, not the child process's
  environment, so a secret passed only to a child is not redacted in a
  failure block. Judged acceptable to ship as documented. The clean fix is
  an environment field on `FailureContext`, owned by whichever unit builds
  contexts from `ctx.run_root`.
- **`Progress.task` still calls `log.info`.** This is genuine scrollback
  rather than a duplicate of a drawn row, and it is safe on a tty because of
  U5b's foreign-write fix. Left deliberately.
- **Three re-reviews never posted.** PRs #60, #62 and #63 each had fixes
  pushed whose reviewers were terminated by a session rate limit before
  posting a verdict. Each fix was verified by measurement when made, and
  #63's earlier round had already confirmed the underlying mechanism, but
  those three landed without a second opinion on the final delta. U8 and U9
  likewise merged without a fresh-context review, by explicit decision under
  a budget constraint.

---

## 6. Verification checklist

For the current state to be what this document claims:

- [x] `python3 -m pytest tests/ -q` from `installer/` gives 1322 passed,
      7 skipped and exactly the one environmental failure in
      [section 2.1](#21-test-suite) on arm64 macOS.
- [ ] `grep` for `follow_apt` and `follow_download` finds the Step 1 and
      Step 2 call sites routed through `progress_exec.py`.
- [ ] Every step declares `phases` and calls exactly `phase(1)` through
      `phase(len(phases))` — pinned by
      `tests/test_steps_protocol.py::test_every_step_declares_phases_that_match_the_indices_it_uses`.
- [ ] `__version__` equals the most recent `v*` tag.
- [x] `gh pr list --state open` is empty at the v0.4.9 release cut.

---

## 7. Out of scope

Settled exclusions, carried from [`08` §11](08-PROGRESS-AND-OBSERVABILITY.md#11-out-of-scope--open-decisions):

- No change to step install logic, the state-machine schema, or the
  [`00` §8.3](00-FRAMEWORK-AND-BOOTSTRAP.md#83-reporting-format-for-dependencies-required-exact-strings) reporting
  strings as part of progress work.
- No TUI framework, no curses full-screen mode, no third-party dependency.
  The binary is built by PyInstaller from a dependency-light tree and stays
  that way.
- No progress persisted to `state.json`. Progress is a property of a running
  process, not of installed state.
- The Jetson camera-node stack is **removed**, not deferred. Read
  [`DELETION-REVIEW.md`](DELETION-REVIEW.md) before deleting or resurrecting
  anything.

---

## References

Facts in this document are drawn from the repository through release
`v0.4.9` and from the workstation install runs of `mv3dt-installer`
0.1.2 through 0.1.9, which are the source of the observed-failure inventory
in [`08` §2](08-PROGRESS-AND-OBSERVABILITY.md#2-observed-failures-this-doc-exists-to-fix).
Test counts and the arm64 failure list were produced by running the suite,
not estimated.

- [apt `APT::Status-Fd`](https://manpages.debian.org/bookworm/apt/apt.conf.5.en.html)
  — **backs the apt integration described in [section 3.3](#33-the-progress-and-observability-subsystem)**:
  the `pmstatus:` percentage the wired apt adapter parses.
- [`curl` `--write-out`](https://curl.se/docs/manpage.html) — **backs the
  download integration described in section 3.3**: the byte count used to
  confirm a completed transfer.
- [NVIDIA driver runfile index](https://download.nvidia.com/XFree86/Linux-x86_64/)
  — **backs [section 3.2](#32-step-1s-two-launch-structure-required-to-understand)**:
  the pinned runfile Step 1 downloads and version-verifies.

Repo files referenced:

- [`00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md) — the
  shared contract every step builds on; section 8.5 carries the resolved
  AUTO streaming matrix.
- [`08-PROGRESS-AND-OBSERVABILITY.md`](08-PROGRESS-AND-OBSERVABILITY.md) —
  the authoritative spec for everything in sections 3.3, 3.4 and 5.1 here,
  including the unit decomposition and the four known defects.
- [`STEP-1-PREREQUISITES.md`](STEP-1-PREREQUISITES.md) — the two-launch
  structure and the equality pins.
- [`DELETION-REVIEW.md`](DELETION-REVIEW.md) — what was removed from this
  fork, why, and where each pattern landed.
- `installer/mv3dt_installer/progress.py` — the renderer and apt/download
  adapters.
- `installer/mv3dt_installer/progress_exec.py` — shared adapter/process
  orchestration used by Steps 1 and 2.
- `installer/mv3dt_installer/report.py` — the unretrofitted failure-context
  and evidence surface.
- `installer/tests/conftest.py` — the global-state reset that keeps `logs`
  from leaking across tests.
