# AI Agent Skills — development tooling assessment (owner: shared)

Status: assessment doc, not a step spec. It depends on no framework
contracts and does **not** modify [`00`](00-FRAMEWORK-AND-BOOTSTRAP.md) or
any `STEP-*` doc — it is a sibling reference for whoever implements
`installer/plan/`, evaluating whether NVIDIA's DS 9.1 **Agentic Skills** are
worth adopting for that work, and if so, how to set them up before
development starts.

Scope is deliberately narrow: this doc covers **only** the DS 9.1 Agentic
Skills (installable per-IDE skill packages). The separate **Inference
Builder MCP server** documented alongside them
(`DS_AI_Agent_MCP.html`) is explicitly **out of scope** — see
[§6](#6-out-of-scope) — because this repo has no custom
inference-microservice work that would need it.

---

## 1. Verdict (LOCKED recommendation)

**Adopt the skills, scoped to the five named in [§3](#3-relevance-mapped-to-this-repos-open-work).**
The `installer/plan/` docs were built by hand-verifying facts against
NVIDIA's DS 9.1 documentation section by section — exactly the kind of
verification and pipeline-authoring work several of these skills automate.
Do **not** install the sixth pipeline skill (`deepstream-dev`) as a
priority, and do **not** stand up the Inference Builder MCP server —
neither matches this repo's actual workflow (config-file-driven
`deepstream-app`, no custom pyservicemaker code, no custom inference
microservices).

---

## 2. What the DS 9.1 Agentic Skills are

Per the [DS 9.1 AI Agent page](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_AI_Agent.html),
DeepStream 9.1 ships 9 skill packages in two groups, installed as
plain directories copied into a coding assistant's skills folder
(`~/.claude/skills/`, `~/.cursor/skills/`, or `~/.codex/skills/`). No DS SDK
or GPU is required to *use* a pipeline-generation skill — only to run the
code it produces.

### 2.1 Pipeline / application skills (6)

| Skill | Function |
|-------|----------|
| `deepstream-dev` | Hand-author or refine a pyservicemaker / GStreamer pipeline; the agent consults condensed SDK references. |
| `deepstream-generate-pipeline` | Generate a ready-to-run `gst-launch-1.0` pipeline via questionnaire; retrieves from 270+ verified pipelines. |
| `deepstream-profile-pipeline` | Profile a pipeline with Nsight Systems; derive optimal configs from the measured inference-plateau batch size and hardware ceiling. |
| `deepstream-import-vision-model` | Onboard a HuggingFace or NGC object-detection model: ONNX export, TensorRT build, multi-stream benchmark, PDF report. |
| `deepstream-run-mv3dt` | Run the DeepStream Multi-View 3D Tracking reference app on shipped samples or synchronized MP4 datasets. |
| `deepstream-sop` | Build/deploy/evaluate a GPU-accelerated operator step-sequence-compliance microservice via GEBD + VLM. |

### 2.2 AutoMagicCalib skills (3)

| Skill | Function |
|-------|----------|
| `amc-setup-calibration-stack` | Launch the AutoMagicCalib microservice and web UI from release images via Docker Compose. |
| `amc-run-sample-calibration` | Verify a running AMC stack with the bundled synthetic sample dataset. |
| `amc-run-video-calibration` | Calibrate a camera rig from user-provided pre-recorded MP4 files via REST API. |

---

## 3. Relevance mapped to this repo's open work

| Skill | Relevant to | Why |
|-------|-------------|-----|
| `deepstream-generate-pipeline` | [`laptop/deepstream/deepstream_app_config.txt`](../../laptop/deepstream/deepstream_app_config.txt) | Validate/regenerate this repo's committed pipeline template against NVIDIA's 270+ verified examples rather than hand-editing it against docs alone. |
| `deepstream-run-mv3dt` | [`STEP-4`](STEP-4-CALIB-OUTPUT-WIRING.md), [`STEP-5`](STEP-5-PER-PROJECT-EXES.md) | Directly exercises the MV3DT reference app these steps wire the installer's own pipeline around. |
| `amc-setup-calibration-stack`, `amc-run-sample-calibration`, `amc-run-video-calibration` | [`STEP-3`](STEP-3-AMC-LAUNCHER.md) | Cover exactly what Step 3 automates by hand today (AMC bring-up, sample verification, MP4-based calibration) — useful both to sanity-check Step 3's design and to unblock a developer before the installer exists. |
| `deepstream-import-vision-model` | [`STEP-2` §5.4](STEP-2-DEEPSTREAM-SDK.md#54-peoplenet-model-acquisition-resolved), [`STEP-4` §6.2](STEP-4-CALIB-OUTPUT-WIRING.md#62-application-config) | The PeopleNet acquisition gap this row once flagged is closed: `STEP-2` §5.4 specifies and implements the NGC download. Still useful when a second detector is imported (`yolo11n` is reserved for future work). |
| `deepstream-profile-pipeline` | future work | Relevant once the 8-camera batch pipeline needs perf tuning; not urgent for the current install-time scope. |
| `deepstream-dev` | low priority | This installer drives `deepstream-app` entirely through rendered config files (`STEP-4` §6) — there is no custom pyservicemaker code for this skill to assist with. |
| `deepstream-sop` | not applicable | No operator step-sequence-compliance work exists in this repo. |

---

## 4. How to use it

Skills work in **agent mode**: the coding assistant selects the relevant
skill automatically from a natural-language request — no manual file
referencing needed. Example prompts written against this repo's actual
files:

1. `"Use deepstream-generate-pipeline to check laptop/deepstream/deepstream_app_config.txt against the verified pipeline catalog and flag anything non-idiomatic."`
2. `"Use deepstream-run-mv3dt to sanity-check config_tracker_NvMOT.yml's SV3DT/MV3DT blocks against the DS 9.1 MV3DT reference app."`
3. `"Use amc-run-video-calibration to walk through calibrating a test rig from the sample MP4s, so I can compare its flow against STEP-3-AMC-LAUNCHER.md's design."`

---

## 5. Setup before development starts

**REQUIRED steps**, per the DS 9.1 AI Agent page:

1. **Clone the DeepStream repo** (the skills live in-tree, not in a
   separate repo):

   ```bash
   git clone https://github.com/NVIDIA/DeepStream
   ```

2. **Copy the relevant skill directories** into the assistant's skills
   folder — for Claude Code:

   ```bash
   cp -r DeepStream/skills/deepstream-generate-pipeline ~/.claude/skills/
   cp -r DeepStream/skills/deepstream-run-mv3dt ~/.claude/skills/
   cp -r DeepStream/skills/deepstream-import-vision-model ~/.claude/skills/
   cp -r DeepStream/skills/amc-setup-calibration-stack ~/.claude/skills/
   cp -r DeepStream/skills/amc-run-sample-calibration ~/.claude/skills/
   cp -r DeepStream/skills/amc-run-video-calibration ~/.claude/skills/
   ```

   (Swap `~/.claude/skills/` for `~/.cursor/skills/` or `~/.codex/skills/`
   depending on the tool in use.) A full repo clone is required even to
   copy one skill, since the skills are not distributed as standalone
   downloads.

3. **Prerequisite split** — no extra setup is needed for the pipeline
   skills beyond the coding assistant itself. The three AMC skills need
   GPU + Docker + NVIDIA Container Toolkit + registry access — already
   satisfied once [`STEP-1`](STEP-1-PREREQUISITES.md) and
   [`STEP-3`](STEP-3-AMC-LAUNCHER.md) are implemented and run once on the
   target workstation.

4. **Caveat (do not skip):** generated code is a development starting
   point, not a merge-ready artifact — it still requires the normal review
   this repo already applies (code review, testing). The skills
   *supplement* the direct-citation verification discipline used throughout
   `installer/plan/`; they do not replace it.

---

## 6. Out of scope

- **The Inference Builder MCP server** (`DS_AI_Agent_MCP.html`) — a
  separate running MCP server for natural-language inference-pipeline
  generation and Docker image building. Not assessed here per the user's
  explicit direction: this repo has no custom inference-microservice work
  that would need it.
- **`deepstream-dev` and `deepstream-sop`** — included in [§2](#2-what-the-ds-91-agentic-skills-are)
  for completeness but not recommended for adoption now; neither matches a
  current gap in `installer/plan/`.
- **Automating the skills' invocation inside the installer itself** — these
  are development-time aids for whoever builds `installer/plan/`'s Python
  steps; they are not a runtime dependency of the shipped `mv3dt-installer`
  binary.

---

## References

DeepStream 9.1 official documentation.

- DS 9.1 AI Agent (skill catalog, install commands, prerequisite split):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_AI_Agent.html>
- NVIDIA/DeepStream GitHub repo (skills live at `skills/<skill-name>`
  in-tree):
  <https://github.com/NVIDIA/DeepStream>

Repo files referenced:

- [`installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md`](00-FRAMEWORK-AND-BOOTSTRAP.md)
  — links to this doc as a sibling tooling reference.
- [`installer/plan/STEP-2-DEEPSTREAM-SDK.md`](STEP-2-DEEPSTREAM-SDK.md) —
  the PeopleNet acquisition gap `deepstream-import-vision-model` could help
  close.
- [`installer/plan/STEP-3-AMC-LAUNCHER.md`](STEP-3-AMC-LAUNCHER.md) — what
  the three AMC skills parallel.
- [`installer/plan/STEP-4-CALIB-OUTPUT-WIRING.md`](STEP-4-CALIB-OUTPUT-WIRING.md)
  and [`STEP-5-PER-PROJECT-EXES.md`](STEP-5-PER-PROJECT-EXES.md) — the
  MV3DT pipeline `deepstream-run-mv3dt` exercises.
- [`laptop/deepstream/deepstream_app_config.txt`](../../laptop/deepstream/deepstream_app_config.txt)
  — the committed pipeline template `deepstream-generate-pipeline` could
  validate.
