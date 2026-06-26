"""
rtmo_pipeline.py — per-view 2D body keypoints (RTMO) for multi-view triangulation.

Baseline: ../../MKER/smpler_pipeline.py (session/activity/camera loop, person
assignment via SPATIAL_REGIONS, per-view .npy output).
Reference: ./inferencer_demo.py (MMPoseInferencer usage / RTMO filter args).

This script is the *extraction* half of the pipeline. For every camera video it
runs RTMO (a one-stage, bottom-up COCO-17 pose estimator — no separate detector
needed) and, for each person it can place into a stable person_id, saves the 17
keypoints together with their per-keypoint detection confidences. The actual
multi-view triangulation is done elsewhere (mka env); this script only produces
its input, so the keypoints are stored in the 1280x720 calibration pixel space
and confidences are preserved untouched so the triangulator can weight by them.

It lives inside the mmpose folder so the package, configs and model aliases all
resolve locally; the `resources/` directory is located by walking up the tree,
so the script keeps working regardless of where under pkgs/ it sits.

Output (mirrors the smpler_results layout):
  resources/rtmo_results/rtmo_meta.npy                         # keypoint convention
  resources/rtmo_results/<session>/<activity>/<cam>_rtmo.npy   # per-view keypoints

Each <cam>_rtmo.npy is a list of length n_frames; element fidx is::

  { 'fidx': int,
    <pid:int>: { 'keypoints':       (17, 2) float32,   # x,y in 1280x720 px (NaN if < kpt_thr)
                 'keypoint_scores': (17,)  float32,    # per-keypoint confidence in [0, 1]
                 'bbox':            (4,)   float32,     # x1,y1,x2,y2
                 'bbox_score':      float },            # instance confidence
    ... }

Run (in the `mmpose` env):
  python demo/rtmo_pipeline.py --session 005013 --activities lego
"""

import os
import os.path as osp
import glob
import argparse
from pathlib import Path

import numpy as np
import cv2 as cv
from tqdm import trange

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))


def find_resources_dir(start):
  """Walk up from `start` until the project `resources/` is found.

  The presence of a `sessions/` subdir is required so mmpose's own
  `demo/resources/` (which only holds demo assets) is skipped.
  """
  d = osp.abspath(start)
  while True:
    cand = osp.join(d, 'resources')
    if osp.isdir(osp.join(cand, 'sessions')):
      return cand
    parent = osp.dirname(d)
    if parent == d:
      raise RuntimeError(f"could not locate the project 'resources' directory above {start}")
    d = parent


# ---------------------------------------------------------------------------
# Camera / person conventions (identical to smpler_pipeline.py so the person_id
# assignment matches across the whole reconstruction pipeline).
# ---------------------------------------------------------------------------
cam_map = {
  'GC': 'GB', 'HC': 'GF', 'Z1': 'FC1', 'Z2': 'FC2', 'N1': 'HA1', 'N2': 'HA2',
}

# Per-camera spatial regions (normalized [0,1]) -> person_id. A detection is
# assigned to the person whose region contains its bbox centre. GF/GB see both
# people (split left/right); the close-up cameras each see a single person.
SPATIAL_REGIONS = {
  'GF':  {0: [0., 0.5, 0., 1.], 1: [0.5, 1., 0., 1.]},
  'GB':  {1: [0., 0.5, 0., 1.], 0: [0.5, 1., 0., 1.]},
  'FC1': {0: [0.25, 0.75, 0., 1.]},
  'FC2': {1: [0.25, 0.75, 0., 1.]},
  'HA1': {0: [0.25, 0.75, 0., 1.]},
  'HA2': {1: [0.25, 0.75, 0., 1.]},
  'WA':  {0: [0.25, 0.75, 0., 1.]},
}

# ---------------------------------------------------------------------------
# COCO-17 keypoint convention emitted by RTMO (saved into the meta sidecar so the
# downstream triangulator is self-describing).
# ---------------------------------------------------------------------------
COCO_KEYPOINTS = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]
NUM_KPTS = len(COCO_KEYPOINTS)
COCO_SKELETON = [
  (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
  (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
  (3, 5), (4, 6),
]

# RTMO inference filter args (from inferencer_demo.py POSE2D_SPECIFIC_ARGS).
RTMO_MODEL     = 'rtmo'   # alias -> rtmo-l_16xb16-600e_body7-640x640
RTMO_BBOX_THR  = 0.1      # instance score threshold
RTMO_NMS_THR   = 0.65
RTMO_POSE_NMS  = True

# Frames are resized to the calibration resolution so the saved 2D keypoints are
# already in the pixel space the camera intrinsics were computed for.
FRAME_W, FRAME_H = 1280, 720

PERSON_COLOR = {0: (0, 255, 0), 1: (0, 0, 255)}   # BGR, matches smpler's pid colors


# ---------------------------------------------------------------------------
# Person assignment (centre-of-bbox -> region), mirrors smpler_pipeline.py.
# ---------------------------------------------------------------------------
def assign_center_to_person(x_center, y_center, cam_id):
  """Return the person_id whose region (for cam_id) contains the normalized
  centre (x_center, y_center), or None if it matches no region."""
  regions = SPATIAL_REGIONS.get(cam_id)
  if regions is None:
    return None
  for person_id, (x_min, x_max, y_min, y_max) in regions.items():
    if x_min <= x_center <= x_max and y_min <= y_center <= y_max:
      return person_id
  return None


def parse_instances(result):
  """Flatten one MMPoseInferencer result into a list of per-person dicts.

  split_instances() emits, per detected person:
    keypoints       : list of [x, y]            -> (17, 2) float32
    keypoint_scores : list of float             -> (17,)  float32
    bbox            : ([x1, y1, x2, y2],)        (1-tuple) — present for RTMO
    bbox_score      : float
  """
  instances = result['predictions'][0]
  parsed = []
  for inst in instances:
    kpts   = np.asarray(inst['keypoints'], dtype=np.float32).reshape(-1, 2)
    scores = np.asarray(inst['keypoint_scores'], dtype=np.float32).reshape(-1)
    if 'bbox' in inst and inst['bbox'] is not None:
      bbox = np.asarray(inst['bbox'], dtype=np.float32).reshape(-1)[:4]
    else:  # bottom-up models may omit bbox — derive a tight one from the joints
      bbox = np.array([kpts[:, 0].min(), kpts[:, 1].min(),
                       kpts[:, 0].max(), kpts[:, 1].max()], dtype=np.float32)
    bbox_score = float(inst.get('bbox_score', float(scores.mean())))
    parsed.append({'keypoints': kpts, 'keypoint_scores': scores,
                   'bbox': bbox, 'bbox_score': bbox_score})
  return parsed


# Upper-body keypoints used to reject background / partial detections (e.g. a
# person in the GF background whose upper body is out of frame). Shoulders are
# the stable anchor: a seated subject looking down at the table loses the face
# (nose/eyes) but keeps the shoulders, whereas an intruder whose upper body
# isn't visible loses both — so weighting toward the shoulders drops the
# intruder without false-dropping a looking-down subject.
HEAD_KPTS     = [COCO_KEYPOINTS.index(k)
                 for k in ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear')]
SHOULDER_KPTS = [COCO_KEYPOINTS.index(k)
                 for k in ('left_shoulder', 'right_shoulder')]


def upper_body_score(scores):
  """Confidence that a detection's upper body is genuinely present, weighted
  toward the shoulders with the head joints as a weak bonus."""
  return 0.75 * float(scores[SHOULDER_KPTS].mean()) + 0.25 * float(scores[HEAD_KPTS].max())


def assign_instances(instances, cam_id, img_w, img_h, top_margin=0.0, upper_thr=0.0):
  """Map detected instances to person_ids via SPATIAL_REGIONS. When several
  detections fall in the same region, keep the most confident one.

  top_margin : drop any instance whose bbox centre falls in the top fraction of
               the frame (catches an intruder whose body extends below the
               masked-out band).
  upper_thr  : drop any instance whose upper_body_score() is below this, to
               reject a background person whose head/shoulders are not in frame
               so they can't win a region over the real subject (0.0 = off)."""
  assignments = {}
  for inst in instances:
    x1, y1, x2, y2 = inst['bbox']
    cx = 0.5 * (x1 + x2) / img_w
    cy = 0.5 * (y1 + y2) / img_h
    if cy < top_margin:
      continue
    if upper_thr > 0.0 and upper_body_score(inst['keypoint_scores']) < upper_thr:
      continue
    pid = assign_center_to_person(cx, cy, cam_id)
    if pid is None:
      continue
    if pid not in assignments or inst['bbox_score'] > assignments[pid]['bbox_score']:
      assignments[pid] = inst
  return assignments


def draw_pose(frame, kpts, scores, color, kpt_thr=0.01):
  """Draw the COCO skeleton + joints for one person (joints below kpt_thr skipped)."""
  for i, j in COCO_SKELETON:
    if scores[i] < kpt_thr or scores[j] < kpt_thr:
      continue
    p1 = (int(kpts[i, 0]), int(kpts[i, 1]))
    p2 = (int(kpts[j, 0]), int(kpts[j, 1]))
    cv.line(frame, p1, p2, color, 2)
  for k in range(len(kpts)):
    if scores[k] < kpt_thr or not np.isfinite(kpts[k]).all():
      continue
    cv.circle(frame, (int(kpts[k, 0]), int(kpts[k, 1])), 3, color, -1)


# ---------------------------------------------------------------------------
# Per-video extraction.
# ---------------------------------------------------------------------------
def extract_video(inferencer, vid_path, cam_id, out_npy_path, max_frames,
                  kpt_thr, vis_dir=None, vis_every=0, top_margin=0.0, upper_thr=0.0):
  """Run RTMO over one camera video and save per-person 2D keypoints.

  kpt_thr    : keypoints with confidence below this are written as NaN (their
               score is still kept) so the triangulator drops them; 0.0 keeps all.
  top_margin : fraction of the frame height to black out at the top before
               inference (and exclude in assignment). Use it to suppress a
               background/intruder person who only appears in the upper band so
               RTMO stops detecting them and they don't crowd out the real
               subject in NMS. 0.0 = disabled; coordinates are unchanged because
               the band is masked, not cropped.
  upper_thr  : drop a detection whose upper-body (shoulders + head) confidence is
               below this, to reject a background person whose head/shoulders are
               not in frame (0.0 = off).
  """
  cap = cv.VideoCapture(vid_path)
  total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
  if max_frames is not None:
    total = min(total, max_frames)

  mask_rows = int(round(top_margin * FRAME_H)) if top_margin > 0.0 else 0

  out_results = []
  for fidx in trange(total, desc=cam_id, leave=False):
    ret, frame = cap.read()
    if not ret:
      break
    if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
      frame = cv.resize(frame, (FRAME_W, FRAME_H))
    if mask_rows:
      frame[:mask_rows] = 0   # black out the top band so the intruder isn't seen

    # RTMO is bottom-up: a single forward pass yields every person in the frame,
    # each with COCO-17 keypoints and per-keypoint confidences. BGR is passed
    # straight through (same as mmcv would load from disk).
    result = next(inferencer(
      frame, return_vis=False, bbox_thr=RTMO_BBOX_THR,
      nms_thr=RTMO_NMS_THR, pose_based_nms=RTMO_POSE_NMS))
    instances = parse_instances(result)
    assignments = assign_instances(instances, cam_id, FRAME_W, FRAME_H, top_margin, upper_thr)

    frame_entry = {'fidx': fidx}
    for pid, inst in assignments.items():
      kpts   = inst['keypoints'].copy()
      scores = inst['keypoint_scores']
      if kpt_thr > 0.0:
        kpts[scores < kpt_thr] = np.nan   # weak joints -> NaN, score preserved
      frame_entry[pid] = {
        'keypoints':       kpts,
        'keypoint_scores': scores,
        'bbox':            inst['bbox'],
        'bbox_score':      inst['bbox_score'],
      }
    out_results.append(frame_entry)

    if vis_dir is not None and vis_every and fidx % vis_every == 0:
      canvas = frame.copy()
      for pid, inst in assignments.items():
        draw_pose(canvas, inst['keypoints'], inst['keypoint_scores'],
                  PERSON_COLOR.get(pid, (255, 255, 255)))
        x1, y1 = inst['bbox'][:2]
        cv.putText(canvas, f'p{pid} {inst["bbox_score"]:.2f}',
                   (int(x1), max(0, int(y1) - 6)), cv.FONT_HERSHEY_SIMPLEX,
                   0.6, PERSON_COLOR.get(pid, (255, 255, 255)), 2)
      cv.imwrite(os.path.join(vis_dir, f'f{fidx:06d}.jpg'), canvas)

  cap.release()
  np.save(out_npy_path, np.array(out_results, dtype=object))
  return len(out_results)


def build_inferencer(model, device):
  """Create the RTMO MMPoseInferencer (lazy import keeps --help env-free)."""
  from mmpose.apis.inferencers import MMPoseInferencer
  return MMPoseInferencer(pose2d=model, device=device, show_progress=False)


def pick_device(requested):
  if requested:
    return requested
  import torch
  if torch.cuda.is_available():
    return 'cuda'
  if torch.backends.mps.is_available():
    return 'mps'
  return 'cpu'


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--session', default='005013',
                      help="session id substring to process, or 'all' (default: 005013)")
  parser.add_argument('--activities', nargs='+', default=['lego'],
                      help='activities to process (default: lego)')
  parser.add_argument('--max-frames', type=int, default=50,
                      help='cap frames per video; -1 for the whole video (default: 50)')
  parser.add_argument('--kpt-thr', type=float, default=0.0,
                      help='keypoints below this confidence are saved as NaN (default: 0.0 = keep all)')
  parser.add_argument('--top-margin', type=float, default=0.,
                      help='black out the top fraction of each frame before inference and '
                           'drop detections centred there, to suppress a background person '
                           'in the upper band (e.g. 0.2 for the top 20%%; default: 0.0 = off)')
  parser.add_argument('--upper-thr', type=float, default=0.1,
                      help='drop a detection whose upper-body (shoulders + head) confidence is '
                           'below this, to reject a background person whose head/shoulders are '
                           'not in frame so they cannot win a region over the real subject '
                           '(default: 0.0 = off; try ~0.3)')
  parser.add_argument('--model', default=RTMO_MODEL,
                      help='RTMO model alias / config (default: rtmo)')
  parser.add_argument('--device', default=None, help='cuda / cpu / mps (default: auto)')
  parser.add_argument('--vis-every', type=int, default=0,
                      help='save an annotated frame every N frames (0 = off)')
  parser.add_argument('--skip-existing', action='store_true',
                      help='skip cameras whose _rtmo.npy already exists')
  args = parser.parse_args()

  max_frames = None if args.max_frames is not None and args.max_frames < 0 else args.max_frames

  resources     = find_resources_dir(SCRIPT_DIR)
  sessions_path = osp.join(resources, 'sessions')
  out_root      = osp.join(resources, 'rtmo_results')
  os.makedirs(out_root, exist_ok=True)
  print(f'resources: {resources}')

  # Self-describing sidecar so the triangulation step knows the keypoint order.
  np.save(osp.join(out_root, 'rtmo_meta.npy'), {
    'model': args.model, 'keypoint_names': COCO_KEYPOINTS,
    'skeleton': COCO_SKELETON, 'frame_size': [FRAME_W, FRAME_H],
    'num_keypoints': NUM_KPTS, 'cam_map': cam_map,
  })

  device = pick_device(args.device)
  print(f'using device: {device}')
  inferencer = build_inferencer(args.model, device)

  log_dir = osp.join(out_root, 'log')
  os.makedirs(log_dir, exist_ok=True)
  log_file = open(osp.join(log_dir, 'rtmo_log.txt'), 'w')

  sid_paths = sorted(glob.glob(sessions_path + '/*'))
  for sid_path in sid_paths:
    session_id = Path(sid_path).stem
    if args.session != 'all' and args.session not in session_id:
      continue
    log_file.write(f'{session_id}\n'); log_file.flush()

    for activity in args.activities:
      vid_paths = glob.glob(osp.join(sid_path, activity) + '/*')
      # E1/E2 are auxiliary (non-calibrated) streams — skip like smpler_pipeline.
      vid_paths = [v for v in vid_paths if not ('E1.mp4' in v or 'E2.mp4' in v)]
      vid_paths = sorted(v for v in vid_paths if v.endswith('.mp4'))
      if not vid_paths:
        continue
      print(f'Processing {activity} in session {session_id} ({len(vid_paths)} cams)')
      log_file.write(f'\t{activity}\n'); log_file.flush()

      curr_out = osp.join(out_root, session_id, activity)
      os.makedirs(curr_out, exist_ok=True)

      for vid_path in vid_paths:
        cam_id = Path(vid_path).stem
        if cam_id not in SPATIAL_REGIONS:
          print(f'  [skip] {cam_id}: no spatial region defined')
          continue
        out_npy_path = osp.join(curr_out, f'{cam_id}_rtmo.npy')
        if args.skip_existing and osp.exists(out_npy_path):
          print(f'  [skip] {cam_id}: already extracted')
          continue

        vis_dir = None
        if args.vis_every:
          vis_dir = osp.join(log_dir, session_id, activity, cam_id)
          os.makedirs(vis_dir, exist_ok=True)
        try:
          n = extract_video(inferencer, vid_path, cam_id, out_npy_path,
                            max_frames, args.kpt_thr, vis_dir, args.vis_every,
                            args.top_margin, args.upper_thr)
          print(f'  {cam_id} ---> OK! ({n} frames)')
          log_file.write(f'\t\t{cam_id} ---> OK! ({n} frames)\n'); log_file.flush()
        except Exception as e:
          print(f'  ERROR processing {cam_id}: {e}', flush=True)
          log_file.write(f'\t\t{cam_id} ---> ERROR! {e}\n'); log_file.flush()

  log_file.close()


if __name__ == '__main__':
  main()
