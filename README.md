# RTMO — body 2D keypoint stage of the UBInteract 3D pipeline

This repository is a fork of **[MMPose](https://github.com/open-mmlab/mmpose)** (OpenMMLab Pose Estimation Toolbox, v1.3.2), used to run **[RTMO](https://arxiv.org/abs/2312.07526)** (Lu et al., *"RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation"*, CVPR 2024) as the **body 2D keypoint stage** of the UBInteract multi-camera 3D reconstruction pipeline.

For every session, activity and camera it:

1. runs RTMO, a one-stage, bottom-up, multi-person pose estimator (no separate person detector), on each frame,
2. discards background people whose upper body is not in frame,
3. assigns each remaining detection to one of the two participants (`person_id` 0 / 1) from where it appears in the frame,
4. saves, for each participant, the 17 COCO keypoints with their per-keypoint confidence (the input of the body triangulation stage) to one `.npy` per camera,
5. optionally saves annotated frames for quality control.

Two entry points are provided:

| Script | Purpose |
|---|---|
| [`rtmo_pipeline.py`](rtmo_pipeline.py) | Processes one or more sessions on a single device. |
| [`run_packed_sessions.py`](run_packed_sessions.py) | Orchestrator: stages sessions from the dataset pool and runs `rtmo_pipeline.py` on each one, packing several sessions onto each GPU. |

The MMPose library code (`mmpose/`, `configs/`, ...) is unmodified upstream code, kept here so that the package, configs and model aliases resolve locally.

---

## 1. Installation

```bash
git clone git@github.com:NikoPicello/mmpose.git
cd mmpose
```

A conda environment is recommended (`run_session.py` expects it to be called `rtmo`):

```bash
conda create --name rtmo python=3.10
conda activate rtmo

pip install torch torchvision          # pick the build matching your CUDA version
pip install -U openmim
mim install "mmengine>=0.4.0,<1.0.0"
mim install "mmcv>=2.0.0,<3.0.0"
mim install "mmdet>=3.0.0,<3.3.0"
pip install -r requirements.txt
pip install -v -e .
```

### Pretrained model

No manual download is needed. `rtmo_pipeline.py` requests the model alias `rtmo`, which resolves to `rtmo-l_16xb16-600e_body7-640x640` (the large RTMO model trained on the Body7 dataset collection, 640×640 input). `MMPoseInferencer` downloads the checkpoint on first use and caches it.

---

## 2. Data layout

Paths are resolved relative to the repository location: the scripts walk up the directory tree from their own location until they find a `resources/` folder that contains `all_sessions/`.

```text
3D_reconstruction/
├── resources/
│   ├── all_sessions/                 # dataset pool, read by rtmo_pipeline.py
│   │   └── <sid>/
│   │       └── <activity>/
│   │           ├── <camera>.mp4                  # video mode (--use_video)
│   │           └── <camera>/000000.jpeg, ...     # image mode (default)
│   ├── sessions/                     # staging area used by run_packed_sessions.py
│   └── rtmo_results/                 # output (see §4)
└── scripts/
    ├── extract_frames.py             # mp4 → per-camera JPEG folders
    ├── run_session.py                # runs all pipelines for one session
    └── mmpose/                       # this repository
```

- **`<sid>`** is a 6-digit session id (e.g. `004096`).
- **`<activity>`** is one of `animals_task`, `gaze_task`, `ghost_task`, `lego_task`, `talk_task`.
- **Cameras.** `GB`, `GF`, `FC1`, `FC2`, `HA1`, `HA2`. A camera that has no entry in the person-assignment table (§4.1) is skipped; in video mode, the `E1`/`E2` cameras are skipped as well. The camera calibration is **not** read by this stage; the mapping from calibration names to camera names (`cam_map`) is only written to the metadata file (§4.2) for the triangulation stage:

  | Calibration file | Video / frame folder |
  |---|---|
  | `GC.yml` | `GB` |
  | `HC.yml` | `GF` |
  | `Z1.yml` | `FC1` |
  | `Z2.yml` | `FC2` |
  | `N1.yml` | `HA1` |
  | `N2.yml` | `HA2` |

### Input modes

- **Image mode (default):** reads JPEG frames from `<sid>/<activity>/<camera>/*.jpeg`, produced by `../extract_frames.py`. Frames are processed in filename order, so the frame index in the output matches the frame index in the source video. Unreadable frames are skipped with a warning.
- **Video mode (`--use_video`):** decodes `<sid>/<activity>/*.mp4` directly with OpenCV.

In both modes, any frame that is not already 1280×720 is resized to it, so the keypoints are in the pixel space of the camera calibration.

---

## 3. Usage

### 3.1 Single run — `rtmo_pipeline.py`

Run from inside the `mmpose/` directory:

```bash
# all activities of session 004096, image mode
python rtmo_pipeline.py --sid 004096

# one activity, first 500 frames only, save an annotated frame every 50 frames
python rtmo_pipeline.py --sid 004096 --activities lego_task --max-frames 500 --vis-every 50

# read mp4 files directly, skip cameras that are already done
python rtmo_pipeline.py --sid 004096 --use_video --skip-existing
```

| Flag | Default | Description |
|---|---|---|
| `--sid` | *all sessions* | Session to process. Matched as a **substring** of the session folder name; if omitted, every folder in `resources/all_sessions/` is processed. |
| `--activities` | all five | One or more activity folders to process. |
| `--max-frames` | `-1` (all) | Process only the first *N* frames of each camera. |
| `--use_video` | off | Read `*.mp4` instead of pre-extracted JPEG folders. |
| `--kpt-thr` | `0.0` (keep all) | Keypoints with confidence below this are saved with NaN coordinates (their confidence is kept). |
| `--top-margin` | `0.0` (off) | Fraction of the frame height to black out at the top before inference; detections centred in that band are dropped too. Use it to hide a background person who only appears at the top of the frame. |
| `--upper-thr` | `0.1` | Drop detections whose upper-body score (§4.1) is below this. `0` disables the filter. |
| `--model` | `rtmo` | RTMO model alias or config. |
| `--device` | auto | `cuda`, `mps` or `cpu`; picked automatically if not set. |
| `--vis-every` | `0` (off) | Save an annotated frame (skeleton, `person_id`, detection confidence) every *N* frames. |
| `--skip-existing` | off | Skip cameras whose `<camera>_rtmo.npy` already exists. |

### 3.2 Multi-GPU batch run — `run_packed_sessions.py`

Runs `rtmo_pipeline.py` over many sessions. RTMO needs little GPU memory (~450 MB per session), so instead of using one whole GPU per session, it runs up to `--max-per-gpu` sessions at the same time on each GPU. Launch it from the already-activated `rtmo` environment; each child process uses the same Python interpreter.

For each session it:

1. copies `resources/all_sessions/<sid>` → `resources/sessions/<sid>` (atomic: copied to `<sid>.tmp` and then renamed, so an interrupted copy is never mistaken for a complete one; an already-staged session is not copied again);
2. runs `rtmo_pipeline.py --sid <sid> [forwarded args]` with `CUDA_VISIBLE_DEVICES` set to one GPU;
3. on success, deletes `resources/sessions/<sid>` to free scratch space. On failure the staged copy is kept so a re-run can reuse it.

A GPU is used only if `nvidia-smi --query-compute-apps` shows no process on it other than our own sessions (from any user or container). On such a GPU, a new session is started only if fewer than `--max-per-gpu` of our sessions are running there and the free memory is at least `--min-free-mib` per extra session. Sessions are tracked in-process, so a session that is still starting up cannot be double-booked.

**Note:** `rtmo_pipeline.py` reads from `resources/all_sessions/`, not from the staged copy in `resources/sessions/`. In the default image mode, the per-camera JPEG folders must therefore exist in `all_sessions/`, so either run `extract_frames.py` first or pass `--use-video`.

```bash
python run_packed_sessions.py                                   # auto-discover from all_sessions, keep watching forever
python run_packed_sessions.py 000000 004096                     # only these sessions, then exit
python run_packed_sessions.py --gpus 0,1,2,3 --max-per-gpu 6    # restrict to a GPU whitelist, pack up to 6 per GPU
python run_packed_sessions.py 004096 --use-video --rtmo-args --activities lego_task --skip-existing
python run_packed_sessions.py --dry-run                         # log the planned actions without executing
```

| Flag | Default | Description |
|---|---|---|
| `sessions` (positional) | *auto-discover* | Explicit session ids. Without them, the script scans `resources/all_sessions/` and keeps re-scanning for new sessions (stop with Ctrl-C). Sessions that already have a non-empty `resources/rtmo_results/<sid>/` are skipped. |
| `--gpus` | all GPUs (or `$GPUS`) | Comma-separated whitelist of GPU indices. GPUs on the list are still skipped while another user's process is on them. |
| `--max-per-gpu` | `4` | Maximum number of our sessions running at the same time on one GPU. |
| `--min-free-mib` | `5000` | Free GPU memory (MiB) required for each extra session on a GPU. |
| `--use-video` | off | Forwards `--use_video` to the pipeline. |
| `--launch-stagger` | `15` s | Delay between consecutive launches. GPU persistence mode is off on the cluster, so simultaneous cold starts can fail CUDA initialisation and silently fall back to CPU. `0` disables the delay. |
| `--poll-interval` | `15` s | Interval between GPU/candidate re-checks. |
| `--dry-run` | off | Log the copy/run/remove actions without executing them. |
| `--rtmo-args ...` | — | Everything after this flag is forwarded to `rtmo_pipeline.py`. **Must be the last flag**: anything after it (including session ids) is forwarded too. |

The first Ctrl-C stops new launches and waits for running sessions to finish. A second Ctrl-C exits immediately, leaving the running jobs and their staged data in place.

**Logs:** `run_logs/<YYYYMMDD_HHMMSS>/<sid>.log` (full stdout/stderr of each session) and `run_logs/<run_id>/summary.tsv` (`<sid>\t<exit_code>` per line). `rtmo_pipeline.py` itself also writes a per-camera status log to `resources/rtmo_results/log/rtmo_log.txt`.

---

## 4. Method and outputs

### 4.1 Processing steps

Each frame of each camera stream is processed independently:

1. **Pose estimation.** RTMO runs on the full frame and returns every person in it, each with a bounding box, a detection confidence and the 17 COCO keypoints with a confidence in `[0, 1]` for each keypoint. The filter settings are RTMO's recommended defaults: detection threshold **0.1**, pose-based NMS with threshold **0.65**.
2. **Background rejection.** Each detection gets an upper-body score `0.75 · mean(shoulder confidences) + 0.25 · max(head confidences)` (head = nose, eyes, ears). Detections with a score below `--upper-thr` are dropped. The score relies mostly on the shoulders, so a participant looking down at the table (weak face keypoints) is kept, while a background person whose upper body is out of frame is dropped. With `--top-margin`, detections centred in the masked top band are dropped as well.
3. **Person assignment.** RTMO detects people but does not identify them. Each detection is assigned to a participant according to where its box centre falls, using a fixed per-camera region table (`SPATIAL_REGIONS`, coordinates normalised to `[0, 1]`):

   | Camera | `person_id` 0 | `person_id` 1 |
   |---|---|---|
   | `GF` | left half | right half |
   | `GB` | right half | left half |
   | `FC1`, `HA1` | central band `x ∈ [0.25, 0.75]` | — |
   | `FC2`, `HA2` | — | central band `x ∈ [0.25, 0.75]` |

   If several detections fall in the same region, the one with the highest detection confidence is kept. Detections outside every region are dropped. The same table is used in `MKER/smpler_pipeline.py`, so `person_id` is consistent across stages.
4. **Confidence gating (optional).** With `--kpt-thr > 0`, keypoints below the threshold get NaN coordinates while keeping their confidence, so the triangulation ignores them. By default every keypoint is kept, and the triangulation weights each one by its confidence.

### 4.2 Output files

```text
resources/rtmo_results/
├── rtmo_meta.npy                                # keypoint convention, written once per run
├── <sid>/<activity>/<camera>_rtmo.npy           # always
└── log/
    ├── rtmo_log.txt                             # per-camera status
    └── <sid>/<activity>/<camera>/f<frame>.jpg   # --vis-every only: annotated frames
```

Each `<camera>_rtmo.npy` is a NumPy object array with **one `dict` per frame**. Each dict holds `fidx` (0-based frame index in the source video) and one entry per assigned participant, keyed by `person_id` (`0` / `1`). A participant that is not detected in a frame has no entry. Each participant entry contains:

| Key | Type / shape | Description |
|---|---|---|
| `keypoints` | `(17, 2)` float32 | COCO-17 keypoint pixel coordinates in the 1280×720 frame (NaN if below `--kpt-thr`). |
| `keypoint_scores` | `(17,)` float32 | Per-keypoint confidence in `[0, 1]`. |
| `bbox` | `(4,)` float32 | Bounding box `x1, y1, x2, y2` (pixels). |
| `bbox_score` | `float` | Detection confidence. |

The keypoints follow the **COCO-17 order**: `0` nose; `1–2` left/right eye; `3–4` left/right ear; `5–6` left/right shoulder; `7–8` left/right elbow; `9–10` left/right wrist; `11–12` left/right hip; `13–14` left/right knee; `15–16` left/right ankle.

`rtmo_meta.npy` stores `model`, `keypoint_names`, `skeleton`, `frame_size`, `num_keypoints` and `cam_map`, so the triangulation stage does not need to hard-code the keypoint order.

Reading the output:

```python
import numpy as np
frames = np.load('resources/rtmo_results/004096/lego_task/GB_rtmo.npy', allow_pickle=True)
p0 = frames[0].get(0)                    # person 0 in the first frame, or None
if p0 is not None:
    left_wrist_px = p0['keypoints'][9]
    left_wrist_conf = p0['keypoint_scores'][9]
```

Downstream, `MKER/body_triangulation.py` triangulates the per-view keypoints into 3D, and `smplifyx-skeleton` uses them as an optional reprojection term in the body fit.

### 4.3 Limitations

- **No tracking.** Each frame is processed independently. Identity comes only from the fixed region table, so it is wrong if a participant leaves their region (e.g. leans across the table).
- **No temporal smoothing.** The outputs can jitter from frame to frame.
- **Fixed scene layout.** `SPATIAL_REGIONS` and `cam_map` are copies of the tables in `MKER/smpler_pipeline.py`, not a shared import. If the camera rig or seating changes, every copy must be updated together.
- **Skipped frames.** In image mode, unreadable frames are skipped, so the position in the output array can differ from the frame index. Use `fidx`.
- **Fixed thresholds.** The detection (0.1) and NMS (0.65) thresholds are hard-coded in `rtmo_pipeline.py`.
- **Help text mismatch.** The `--upper-thr` help text says the default is `0.0 = off`, but the actual default is `0.1`, so background rejection is on by default.

---

## 5. Citation

```bibtex
@inproceedings{lu2024rtmo,
  title     = {{RTMO}: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation},
  author    = {Lu, Peng and Jiang, Tao and Li, Yining and Li, Xiangtai and Chen, Kai and Yang, Wenming},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}

@misc{mmpose2020,
  title        = {OpenMMLab Pose Estimation Toolbox and Benchmark},
  author       = {MMPose Contributors},
  howpublished = {\url{https://github.com/open-mmlab/mmpose}},
  year         = {2020}
}
```
