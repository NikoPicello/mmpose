# RTMO 2D Pose Extraction — Pipeline Documentation

This document describes the **custom integration layer** built on top of the vendored
[mmpose](https://github.com/open-mmlab/mmpose) framework in this directory — i.e. the code
that is actually ours, not the upstream library. It is written at a level of detail intended
for direct adaptation into a technical report (Overleaf/LaTeX), in the same spirit as
`smplifyx-skeleton/DOCUMENTATION.md`. Every claim below was verified against the source in
this repository as of the state checked; file names, function names and constants are quoted
verbatim so they can be traced back to the code (`file.py:function_or_line`).

For the upstream framework itself (installation, model zoo, training), see `mmpose/README.md`
and `mmpose/docs/` — those are OpenMMLab's own docs and are not duplicated here.

## Contents
1. [Overview](#1-overview)
2. [What's ours vs. vendored](#2-whats-ours-vs-vendored)
3. [The RTMO model](#3-the-rtmo-model)
4. [`rtmo_pipeline.py` — the extraction runner](#4-demortmo_pipelinepy--the-extraction-runner)
5. [Orchestration — three ways this runner gets invoked](#5-orchestration--three-ways-this-runner-gets-invoked)
6. [Data flow and downstream consumers](#6-data-flow-and-downstream-consumers)
7. [Shared conventions across the project](#7-shared-conventions-across-the-project)
8. [Known inconsistencies and open items](#8-known-inconsistencies-and-open-items)
9. [Acknowledgements](#9-acknowledgements)

---

## 1. Overview

This module produces **per-camera, per-frame 2D body keypoints**, the first stage of the
body-pose branch of a larger multi-view 3D reconstruction pipeline (see the sibling pipelines
under `../WiLoR`, `../3DDFA-V3`, `../MKER`, `../mamma`, and the final fitting stage in
`../smplifyx-skeleton`). Concretely: for every camera video of a recording session, it runs
RTMO — a one-stage, bottom-up, multi-person pose estimator — and saves each detected person's
COCO-17 keypoints with per-keypoint confidence, in a fixed calibration pixel space, ready for
multi-view triangulation.

This script only performs *extraction*; triangulating the per-view 2D detections into 3D is a
separate downstream step (`MKER/body_triangulation.py`, a different conda environment —
Section 6).

**Role in the wider pipeline.** Per the top-level orchestrator `../run_session.py`, RTMO is one
of five pipelines run per session (alongside 3DDFA, WiLoR, SMPLer-X/MKER, and SAM). It is
grouped on a shared GPU with `sam`, since both are documented as light on VRAM (~450 MB per
RTMO session, per `run_packed_sessions.py`'s own docstring).

---

## 2. What's ours vs. vendored

```
mmpose/
├── mmpose/, configs/, tools/, docs/, projects/, tests/, ...   # vendored OpenMMLab framework (upstream)
├── README.md, README_CN.md                                     # upstream docs — NOT this file
├── rtmo_pipeline.py            # OURS — the per-session/camera extraction runner (§4)
├── run_packed_sessions.py      # OURS — multi-session, GPU-packing scheduler around rtmo_pipeline.py (§5.3)
├── run_logs/                    # OURS — logs written by the scripts above
└── output/                      # ad hoc local test output (FC1.mp4/.png), not part of the pipeline proper
```

Everything under `mmpose/mmpose/`, `configs/`, `tools/`, `docs/`, `projects/`, is the
unmodified upstream `open-mmlab/mmpose` repository, vendored so that the two custom scripts
above can resolve its package, configs and model aliases locally without an installed
dependency. Only `rtmo_pipeline.py` and `run_packed_sessions.py` (plus this file) are
specific to this project.

---

## 3. The RTMO model

[RTMO](https://arxiv.org/abs/2312.07526) ("Towards High-Performance One-Stage Real-Time
Multi-Person Pose Estimation") is a **one-stage, bottom-up** pose estimator: unlike a
top-down pipeline (detect each person with a separate detector, then run pose estimation per
crop), RTMO estimates every person's keypoints in a single forward pass over the whole frame,
by combining coordinate classification (dual 1-D heatmaps) with a YOLO-style dense-prediction
architecture. The upstream project documents two practical advantages that motivate its use
here: no dependency on an auxiliary human detector (simpler pipeline, one model to run instead
of two), and faster inference than top-down alternatives specifically as the number of people
in frame grows — advantageous for a fixed multi-person (here, two-person) scene repeated over
many sessions/activities/cameras/frames.

**Exact model used.** `rtmo_pipeline.py` requests the model alias `'rtmo'`
(`RTMO_MODEL` constant, `rtmo_pipeline.py:106`), which — confirmed directly against
`configs/body_2d_keypoint/rtmo/body7/rtmo_body7.yml:58-64` — resolves to
`rtmo-l_16xb16-600e_body7-640x640`: the **large** RTMO variant, trained on **body7** (a
combined dataset of AI Challenger, COCO, CrowdPose, Halpe, MPII, PoseTrack18 and sub-JHMDB),
at 640×640 input resolution, 16×16 batch/GPU for 600 epochs. Reported benchmark performance for
this exact checkpoint (COCO val2017, ONNXRuntime backend on a V100): **AP 0.748** (AP50 0.911,
AP75 0.813), **AR 0.786** (AR50 0.939), 19.1 ms latency. Note the latency figure is from
upstream's ONNXRuntime benchmark; this pipeline runs the PyTorch checkpoint through
`MMPoseInferencer` directly, so actual wall-clock throughput here will differ.

**Inference filter settings.** `rtmo_pipeline.py` hardcodes `bbox_thr=0.1`, `nms_thr=0.65`,
`pose_based_nms=True` (`RTMO_BBOX_THR`/`RTMO_NMS_THR`/`RTMO_POSE_NMS`, `rtmo_pipeline.py:107-109`).
These are not ad hoc — they are copied from upstream's own RTMO-specific recommended defaults
in upstream's `demo/inferencer_demo.py` (`POSE2D_SPECIFIC_ARGS['rtmo']`, since removed from this tree), confirmed identical.

**Output convention.** RTMO emits the standard COCO-17 keypoint set (no hand or face detail —
those come from WiLoR and 3DDFA respectively, fused downstream):

| # | Keypoint | # | Keypoint | # | Keypoint |
|---|---|---|---|---|---|
| 0 | nose | 6 | right_shoulder | 12 | right_hip |
| 1 | left_eye | 7 | left_elbow | 13 | left_knee |
| 2 | right_eye | 8 | right_elbow | 14 | right_knee |
| 3 | left_ear | 9 | left_wrist | 15 | left_ankle |
| 4 | right_ear | 10 | right_wrist | 16 | right_ankle |
| 5 | left_shoulder | 11 | left_hip | | |

(`COCO_KEYPOINTS`, `rtmo_pipeline.py:92-97`; the corresponding 19-edge skeleton for
visualization is `COCO_SKELETON`, `rtmo_pipeline.py:99-103`.)

---

## 4. `rtmo_pipeline.py` — the extraction runner

### 4.1 Why it lives in the `mmpose/` root

So that mmpose's package, configs and model aliases all resolve locally without needing mmpose
installed as a dependency elsewhere. It locates the project's shared `resources/` directory by
walking up the filesystem tree from its own location until it finds a `resources/` that
contains a `sessions/` subdirectory (`find_resources_dir`, `rtmo_pipeline.py:50-64`) — the
`sessions/` check distinguishes the real project resources from any other `resources/` folder
(e.g. upstream's `demo/resources/`, if present).

### 4.2 Input modes

Two mutually exclusive input sources, selected by `--use_video`:
- **Default** (image mode): reads pre-extracted per-camera frame folders
  (`000000.jpeg`, `000001.jpeg`, ...) produced by the top-level `../extract_frames.py`.
- **`--use_video`**: reads `*.mp4` camera files directly from the session folder, skipping the
  auxiliary, non-calibrated `E1.mp4`/`E2.mp4` streams (`rtmo_pipeline.py:387-389`) — mirrors
  `smpler_pipeline.py`'s own handling of those streams.

Every frame is resized to `1280×720` if it isn't already (`FRAME_W, FRAME_H`,
`rtmo_pipeline.py:113`) — the pixel space the camera calibration was computed for, so the
saved 2D keypoints are directly usable by the triangulator without any further rescaling.

### 4.3 Per-frame inference and instance parsing

Each frame is passed through a single `MMPoseInferencer` call
(`extract_video`, `rtmo_pipeline.py:269-271`); being bottom-up, one forward pass returns every
detected person in the frame. `parse_instances()` (`rtmo_pipeline.py:133-155`) flattens the
inferencer's raw result into one dict per person: `keypoints` (17,2), `keypoint_scores` (17,),
`bbox` (4,), `bbox_score`. Two defensive details worth noting: if the result has no `bbox`
(bottom-up models may omit it), a tight box is derived from the min/max of the predicted
joints instead; if `bbox_score` is absent, it falls back to the mean of the per-keypoint
scores.

### 4.4 Person assignment (`SPATIAL_REGIONS`)

RTMO detects *instances*, not identities — a separate step assigns each detected instance to
one of the study's two fixed `person_id`s (0 or 1) via a **per-camera spatial region** lookup:
a detection is assigned to whichever region (a normalized `[x_min, x_max, y_min, y_max]` box)
its bbox centre falls inside (`assign_center_to_person`, `rtmo_pipeline.py:121-130`; driven by
the `SPATIAL_REGIONS` table, `rtmo_pipeline.py:78-86`). The two overview cameras (`GF`/`GB`)
see both people and split the frame left/right (with the two person_ids swapped between them,
since they face each other); the four close-up cameras (`FC1`, `FC2`, `HA1`, `HA2`) each see
one fixed person in a central horizontal band. If two detections land in the same region, the
higher `bbox_score` wins (`assign_instances`, `rtmo_pipeline.py:176-200`).

This table — along with `cam_map` (raw camera serial → logical camera name,
`rtmo_pipeline.py:71-73`) — is **copied verbatim** from `MKER/smpler_pipeline.py` (confirmed
identical byte-for-byte modulo formatting) specifically so that `person_id` assignment is
consistent across every pipeline in the project (Section 7).

### 4.5 Background/intruder rejection

Two optional heuristics reject a detected background person before they can win a region over
the real subject:
- **`--top-margin`**: blacks out the top fraction of the frame *before* inference (not a crop —
  coordinates are unaffected) and drops any detection centred in that band, to suppress an
  intruder who only appears in the upper part of the frame.
- **`--upper-thr`**: drops a detection whose `upper_body_score()` — 75% shoulder confidence +
  25% max head-joint confidence (`rtmo_pipeline.py:170-173`) — falls below the threshold. The
  weighting is deliberate: a seated subject looking down at the table loses face keypoints but
  keeps their shoulders, while a genuine background intruder tends to lose both, so weighting
  toward the shoulders discriminates the two cases instead of penalizing a subject who is just
  looking down.

### 4.6 Confidence handling and output schema

`--kpt-thr` (default `0.0`, i.e. off) zeroes out unreliable joints by setting their
*coordinates* to `NaN` while leaving their confidence score untouched, so a downstream
consumer can still see how confident (and wrong) a dropped detection was. Final per-camera
output (`<cam>_rtmo.npy`) is a length-`n_frames` object array; element `fidx`:

```python
{ 'fidx': int,
  <pid:int>: { 'keypoints':       (17, 2) float32,   # x,y in 1280x720 px (NaN if < kpt_thr)
               'keypoint_scores': (17,)  float32,    # per-keypoint confidence in [0, 1]
               'bbox':            (4,)   float32,    # x1,y1,x2,y2
               'bbox_score':      float },           # instance confidence
  ... }
```

A sidecar `resources/rtmo_results/rtmo_meta.npy` is written once per run
(`rtmo_pipeline.py:364-368`) holding `model`, `keypoint_names`, `skeleton`, `frame_size`,
`num_keypoints`, `cam_map` — so the triangulator (or any other consumer) is self-describing
about keypoint order and doesn't need to hardcode the COCO-17 convention.

### 4.7 Visualization / QA

`--vis-every N` writes an annotated JPEG (skeleton overlay + `person_id` + bbox score, via
`draw_pose`, `rtmo_pipeline.py:203-214`) every `N` frames to
`resources/rtmo_results/log/<session>/<activity>/<cam>/`, for spot-checking detection and
person-assignment quality without re-running inference.

### 4.8 CLI reference

| Flag | Default | Meaning |
|---|---|---|
| `--sid` | `None` (all sessions) | Session id **substring** filter |
| `--activities` | all 5 activities | Activities to process |
| `--max-frames` | `-1` (whole video) | Cap frames processed per camera |
| `--use_video` | off | Read `*.mp4` directly instead of extracted frame folders |
| `--kpt-thr` | `0.0` (keep all) | Confidence below which a keypoint's coords are set to NaN |
| `--top-margin` | `0.0` (off) | Fraction of frame height to black out at the top |
| `--upper-thr` | `0.1` | Upper-body confidence floor for keeping a detection (see §8) |
| `--model` | `rtmo` | RTMO model alias/config |
| `--device` | `None` (auto) | `cuda` / `mps` / `cpu`, auto-picked if unset |
| `--vis-every` | `0` (off) | Save an annotated frame every N frames |
| `--skip-existing` | off | Skip a camera if its `_rtmo.npy` already exists |

Example:
```bash
python rtmo_pipeline.py --sid 005013 --activities lego_task
```

---

## 5. Orchestration — three ways this runner gets invoked

### 5.1 Standalone (one session)

```bash
python rtmo_pipeline.py --sid <sid> --activities <activities>
```
Run directly from inside the `mmpose` conda env (see the important caveat in Section 8 about
which env name is actually correct).

### 5.2 `../run_session.py` — one session, all 5 pipelines, whole GPU per pipeline

The project's top-level orchestrator runs RTMO as one of five pipelines for a *single* session,
each pipeline on its own dedicated GPU (RTMO shares its GPU with `sam` only, since both are
light on VRAM — `PIPELINE_GROUPS`, `run_session.py:104-109`). This mode optimizes one session's
wall-clock time, not overall throughput, and is meant for processing a short, explicit list of
sessions with GPUs to spare.

### 5.3 `run_packed_sessions.py` — many sessions, GPU packing

The other axis: instead of claiming one whole GPU per session, this scheduler fits up to
`--max-per-gpu` (default 4) of RTMO's *own* sessions concurrently onto each GPU that has no
foreign process on it, gated by a `--min-free-mib` (default 5000) floor on reported free VRAM
per additional session — viable specifically because RTMO is light on VRAM (~450 MB/session
observed). Key mechanics, all implemented in this one script:

- **Staging.** Each session's full folder is copied from `resources/all_sessions/<sid>` into
  local scratch `resources/sessions/<sid>` atomically (copy to a `.tmp` sibling, then
  `os.replace`), so a killed/interrupted copy can never be mistaken for a complete one on a
  later run. Successfully completed sessions are removed from scratch; failed ones are left
  staged for a retry without re-copying.
- **Foreign-GPU detection.** `nvidia-smi --query-compute-apps` is cross-referenced against the
  PIDs of subprocesses this script itself launched (`gpu_status`, `run_packed_sessions.py:150-183`);
  a GPU is entirely off-limits the moment any *other* process appears on it — packing only ever
  adds more of this script's own work to a GPU, never shares one with a stranger's job.
- **Launch stagger.** A `--launch-stagger` delay (default 15 s) between consecutive launches
  avoids several jobs simultaneously triggering a cold GPU's driver initialization (persistence
  mode is off on the target node), which was observed to make concurrent first-touches contend
  and some jobs silently fall back to CPU.
- **Discovery modes.** With explicit session ids on the command line, the candidate list is
  fixed and the process exits once they're done. With none, it auto-discovers every session
  under `resources/all_sessions` not yet present in `resources/rtmo_results`, and keeps
  re-scanning forever (long-lived daemon; Ctrl-C to stop).
- **Argument forwarding gotcha.** `--rtmo-args` consumes every token after it
  (`argparse.REMAINDER`) and forwards them verbatim to `rtmo_pipeline.py` — it **must** be the
  last thing on the command line, or it will swallow session ids meant for this script instead.

```bash
python run_packed_sessions.py                                   # watch all_sessions forever
python run_packed_sessions.py 000000 004096                     # just these, then exit
python run_packed_sessions.py --max-per-gpu 6 --min-free-mib 3000
python run_packed_sessions.py 000000 --rtmo-args --activities lego_task --skip-existing
```

**5.2 vs. 5.3, in short:** `run_session.py` is per-session-first (finish one session's every
pipeline fast, across dedicated GPUs); `run_packed_sessions.py` is per-pipeline-first (run
RTMO alone across many sessions, packing several onto each GPU for throughput). They are not
meant to run against the same GPUs at the same time.

---

## 6. Data flow and downstream consumers

```
resources/sessions/<sid>/<activity>/<cam>/*.jpeg  (or <cam>.mp4)
                    │
                    ▼  rtmo_pipeline.py  (env: mmpose — see §8)
                    │
resources/rtmo_results/<sid>/<activity>/<cam>_rtmo.npy
resources/rtmo_results/rtmo_meta.npy
                    │
                    ▼  MKER/body_triangulation.py  (env: mka)
                    │
resources/triangulation_results/<sid>/<activity>/body_rtmo/p<pid>_triangulated.npy
                    │
                    ▼  smplifyx-skeleton  (fitter_pipeline.py / temporal_window.py)
```

Confirmed downstream readers of `rtmo_results`/`rtmo_meta` (via repository-wide search):

- **`MKER/body_triangulation.py`** — triangulates each person's per-view 2D detections into 3D
  (`pycalib.robust.triangulate_consensus`, confidence-gated and refined — documented in its own
  module once that pipeline is written up).
- **`smplifyx-skeleton/fitter_pipeline.py`, `run_parallel_sessions.py`,
  `visualization/vis_fit_on_video.py`** — RTMO's per-camera 2D detections are used as an
  optional camera-reprojection term inside the fitter's static-root solve
  (`DOCUMENTATION.md` §5.2 in that package) and as an overlay in the video-comparison
  visualizer.
- Pipeline completion is additionally recorded to `resources/pipeline_status.csv` via the
  shared `../pipeline_status.py` helper, but only when run through `run_session.py`
  (`ps.set_status(sid, 'rtmo', ...)`, `run_session.py:253`/`275`) — a direct or
  `run_packed_sessions.py` invocation does not update that CSV.

---

## 7. Shared conventions across the project

`cam_map` and `SPATIAL_REGIONS` (Section 4.4) encode two project-wide, cross-pipeline
assumptions: the mapping from raw camera serials to logical camera names, and where each of
the study's two seated subjects appears in each camera's frame. These constants are duplicated
verbatim in at least `MKER/smpler_pipeline.py` (the documented baseline) and
`mmpose/rtmo_pipeline.py`. **This is a copy, not a shared import** — if the physical
camera rig, seating arrangement, or camera naming ever changes, every copy needs updating
together, or `person_id` assignment will silently disagree between pipelines that are supposed
to describe the same two people.

---

## 8. Known inconsistencies and open items

Recorded for completeness/traceability, verified against the current source:

- **`--upper-thr` default doesn't match its own help text.** The flag's default is `0.1`
  (`rtmo_pipeline.py:341`), but its help string says `"default: 0.0 = off"`
  (`rtmo_pipeline.py:343-345`) — i.e. the upper-body rejection heuristic (Section 4.5) is *on*
  by default, not off as the help text implies.
---

## 9. Acknowledgements

Built on [mmpose](https://github.com/open-mmlab/mmpose) (OpenMMLab Pose Estimation Toolbox)
and the [RTMO](https://arxiv.org/abs/2312.07526) model (Lu et al., "RTMO: Towards
High-Performance One-Stage Real-Time Multi-Person Pose Estimation").
