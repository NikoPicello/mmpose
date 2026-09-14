#!/usr/bin/env python3
"""Run demo/rtmo_pipeline.py across many sessions, one per truly-idle GPU.

Lives at the mmpose package root (a sibling of demo/, not inside it) and
invokes demo/rtmo_pipeline.py by relative path from there -- the same
`python demo/rtmo_pipeline.py ...` invocation shown in rtmo_pipeline.py's own
docstring, just orchestrated across sessions/GPUs.

Run this from inside the container, from the already-activated mmpose conda
env (e.g. `python run_parallel_sessions.py ...`) -- each per-session
subprocess is launched with the same interpreter (sys.executable), not a
fresh env activation, so the env this script itself runs under is the env
rtmo_pipeline.py runs under too.

Resource layout (resources/ is located the same way rtmo_pipeline.py finds it
itself -- by walking up from this file until a resources/sessions dir turns
up, so this script keeps working regardless of where under pkgs/ it sits):
    resources/all_sessions/<sid>   the full dataset (big, separately-mounted pool)
    resources/sessions/<sid>       local scratch rtmo_pipeline.py actually reads
                                    from; sessions currently staged/in-progress live
                                    here
    resources/rtmo_results/<sid>   pipeline output; a session with output here
                                    already exists is treated as done and skipped

Per session: copy the *entire* session folder (session_data.txt plus every
activity subfolder -- talk/lego/ghost/animals/gaze) from
resources/all_sessions/<sid> into resources/sessions/<sid>, atomically (a
partial/interrupted copy lands in a .tmp path, never mistaken for a
complete one). Then run `demo/rtmo_pipeline.py --sid <sid>` with
CUDA_VISIBLE_DEVICES pinned to one GPU, and on success remove
resources/sessions/<sid> entirely (freeing local scratch space). A failed run's
staged copy is left in place so a re-run picks it up without re-copying.

rtmo_pipeline.py's own --activities default is every activity (matching
wilor_pipeline.py/smpler_pipeline.py/sam_pipeline.py/3ddfa_pipeline.py); narrow
it via --rtmo-args, e.g. `--rtmo-args --activities lego_task`. Also note --sid is
a substring match (matching rtmo_pipeline.py's own semantics, shared with the
other pipelines) -- session ids passed here should be the full session id so
they can't accidentally match more than one folder under resources/sessions.

The already-processed / discovery check below only looks at whether
resources/rtmo_results/<sid> has *any* output, same coarse granularity as the
sibling run_parallel_sessions.py scripts -- it does not check that every
activity/camera you actually want was extracted. Use rtmo_pipeline.py's own
--skip-existing flag (forward it via --rtmo-args) if you re-run a session that
was only partially processed and want it to fill in gaps instead of redoing
everything.

"Free GPU" means zero processes on it right now (any user, any container --
checked via `nvidia-smi --query-compute-apps`), not just low utilization. The
pool re-checks this continuously: as soon as a GPU has no session of ours *and*
nvidia-smi reports it idle, the next candidate is copied and started on it --
no waiting for the whole batch to finish. GPUs we've already claimed are
tracked in-process so a not-yet-CUDA-initialized subprocess can't be
double-booked during its startup lag.

With explicit session ids on the command line, the candidate list is fixed and
the script exits once they're all done. With no ids, it auto-discovers
candidates from resources/all_sessions and keeps re-scanning for newly
arrived ones, so it runs indefinitely (Ctrl-C to stop; a session already
mid-run finishes before the process exits).

Usage:
    python run_parallel_sessions.py                  # watch all_sessions forever
    python run_parallel_sessions.py 000000 004096     # just these, then exit
    python run_parallel_sessions.py --gpus 0,1,2,3    # restrict the GPU whitelist
    python run_parallel_sessions.py --rtmo-args --activities lego_task
    python run_parallel_sessions.py --dry-run          # log planned actions only

--rtmo-args grabs every token after it (argparse REMAINDER), so it must come
LAST -- session ids or other flags after it are swallowed as rtmo_pipeline.py
args instead. Put session ids first: `... 000000 004096 --rtmo-args --activities lego_task`.

Logs: ./run_logs/<run_id>/<session_id>.log (one per session attempt), plus a
summary.tsv of "<session_id>\t<exit_code>" lines.
"""
from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

RTMO_ROOT = Path(__file__).resolve().parent


def find_resources_dir(start: Path) -> Path:
    """Walk up from `start` until the project `resources/` is found.

    Mirrors rtmo_pipeline.py's own find_resources_dir(): the presence of a
    `sessions/` subdir is required so mmpose's own `demo/resources/` (which
    only holds demo assets, not session data) is skipped.
    """
    d = start.resolve()
    while True:
        cand = d / "resources"
        if (cand / "all_sessions").is_dir():
            return cand
        if d.parent == d:
            raise SystemExit(f"could not locate the project 'resources' directory above {start}")
        d = d.parent


RESOURCES_DIR = find_resources_dir(RTMO_ROOT)
ALL_SESSIONS_DIR = RESOURCES_DIR / "all_sessions"
SESSIONS_DIR = RESOURCES_DIR / "sessions"
RTMO_RESULTS_DIR = RESOURCES_DIR / "rtmo_results"

DEFAULT_POLL_INTERVAL = 15.0
# Gap between consecutive launches. With nvidia-persistenced off (persistence mode
# Disabled), the first client onto a cold GPU triggers a full driver init; several jobs
# doing that at the same instant contend and some fail with CUDA_ERROR_NOT_INITIALIZED,
# silently falling back to CPU. Staggering lets them serialise.
DEFAULT_LAUNCH_STAGGER = 15.0


def free_gpu_indices(whitelist: set[int] | None = None) -> list[int]:
    """GPU indices with zero processes right now (any user), via nvidia-smi."""
    try:
        gpus_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout
        apps_out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"nvidia-smi query failed: {e}") from e

    busy_uuids = {line.strip() for line in apps_out.strip().splitlines() if line.strip()}
    free = []
    for line in gpus_out.strip().splitlines():
        idx_s, uuid = (p.strip() for p in line.split(",", 1))
        idx = int(idx_s)
        if whitelist is not None and idx not in whitelist:
            continue
        if uuid not in busy_uuids:
            free.append(idx)
    return sorted(free)


def already_processed(sid: str) -> bool:
    d = RTMO_RESULTS_DIR / sid
    return d.is_dir() and any(d.iterdir())


def discover_candidates(known: set[str]) -> list[str]:
    if not ALL_SESSIONS_DIR.is_dir():
        return []
    ids = sorted(p.name for p in ALL_SESSIONS_DIR.iterdir() if p.is_dir())
    return [sid for sid in ids if sid not in known and not already_processed(sid)]


class Runner:
    def __init__(self, log_dir: Path, rtmo_args: list[str], dry_run: bool):
        self.log_dir = log_dir
        self.rtmo_args = rtmo_args
        self.dry_run = dry_run

        self.lock = threading.Lock()
        self.active_gpus: set[int] = set()
        self.threads: list[threading.Thread] = []
        self.summary: list[tuple[str, int]] = []

    def _log_path(self, sid: str) -> Path:
        return self.log_dir / f"{sid}.log"

    @staticmethod
    def _copytree_atomic(src: Path, dest: Path) -> None:
        """Copy to a sibling .tmp path then os.replace into dest, so dest only
        ever exists fully-formed or not at all -- an interrupted copy (crash,
        kill, disk pressure) never leaves a partial tree at dest that a later
        `dest.is_dir()` staged-check would mistake for complete."""
        tmp = dest.with_name(dest.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(src, tmp, symlinks=True)
        os.replace(tmp, dest)

    def _stage(self, sid: str, log_fh) -> bool:
        """Copy the whole session folder from all_sessions into sessions/<sid>.
        Idempotent: if dest already exists it's assumed already staged (left
        alone), so a retry after a failed run doesn't re-copy. The copy itself
        is atomic (see _copytree_atomic) so a partial copy from an earlier
        interrupted run is never mistaken for a complete one -- it lands in a
        .tmp path, not the checked destination."""
        src = ALL_SESSIONS_DIR / sid
        if not src.is_dir():
            print(f"[{sid}] ERROR: not found under {ALL_SESSIONS_DIR}", file=log_fh)
            return False

        dest = SESSIONS_DIR / sid
        if dest.is_dir():
            print(f"[{sid}] already staged at {dest}", file=log_fh)
            return True

        print(f"[{sid}] copying {src} -> {dest}", file=log_fh)
        if self.dry_run:
            return True
        try:
            self._copytree_atomic(src, dest)
        except Exception as e:
            print(f"[{sid}] ERROR copying: {e}", file=log_fh)
            return False
        return True

    def _run_pipeline(self, sid: str, gpu: int, log_fh) -> int:
        # Uses the same interpreter this orchestrator is running under (sys.executable),
        # so it must itself already be launched from the right conda env's python --
        # no env activation/wrapping happens here. Invoked as demo/rtmo_pipeline.py
        # (cwd is RTMO_ROOT, the mmpose package root) since this orchestrator lives
        # alongside demo/, not inside it.
        cmd = [sys.executable, "demo/rtmo_pipeline.py", "--sid", sid, *self.rtmo_args]
        print(f"[{sid}] running: {' '.join(cmd)} (CUDA_VISIBLE_DEVICES={gpu})", file=log_fh)
        if self.dry_run:
            print(f"[{sid}] DRY-RUN: skipping actual run", file=log_fh)
            return 0
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_fh.flush()
        proc = subprocess.run(cmd, cwd=str(RTMO_ROOT), env=env,
                              stdout=log_fh, stderr=subprocess.STDOUT)
        return proc.returncode

    def _worker(self, sid: str, gpu: int) -> None:
        print(f"[{sid}] starting on GPU {gpu} (log: {self._log_path(sid)})")
        try:
            with open(self._log_path(sid), "w") as log_fh:
                if self._stage(sid, log_fh):
                    rc = self._run_pipeline(sid, gpu, log_fh)
                else:
                    rc = 1

            with self.lock:
                self.summary.append((sid, rc))

            if rc == 0:
                if self.dry_run:
                    print(f"[{sid}] done (GPU {gpu}) — DRY-RUN, not removing {SESSIONS_DIR / sid}")
                else:
                    try:
                        shutil.rmtree(SESSIONS_DIR / sid)
                        print(f"[{sid}] done (GPU {gpu}) — removed from {SESSIONS_DIR}")
                    except Exception as e:
                        print(f"[{sid}] done (GPU {gpu}) but failed to remove "
                              f"{SESSIONS_DIR / sid}: {e}", file=sys.stderr)
            else:
                print(f"[{sid}] FAILED rc={rc} (GPU {gpu}) — see {self._log_path(sid)}; "
                      f"left staged in {SESSIONS_DIR} for retry", file=sys.stderr)
        finally:
            with self.lock:
                self.active_gpus.discard(gpu)

    def launch(self, sid: str, gpu: int) -> None:
        with self.lock:
            self.active_gpus.add(gpu)
        t = threading.Thread(target=self._worker, args=(sid, gpu), daemon=True)
        t.start()
        self.threads.append(t)

    def active_snapshot(self) -> set[int]:
        with self.lock:
            return set(self.active_gpus)

    def join_all(self) -> None:
        for t in self.threads:
            t.join()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run rtmo_pipeline.py across sessions, one per idle GPU.")
    ap.add_argument("sessions", nargs="*",
                    help="explicit session ids to run (default: auto-discover from "
                         "resources/all_sessions, re-scanning forever)")
    ap.add_argument("--gpus", default=os.environ.get("GPUS"),
                    help="comma-separated GPU index whitelist (default: all GPUs "
                         "reported by nvidia-smi). Still only used when actually idle.")
    ap.add_argument("--rtmo-args", nargs=argparse.REMAINDER, default=[],
                    help="remaining args forwarded to rtmo_pipeline.py, e.g. "
                         "--activities lego_task --skip-existing. Must be LAST on the "
                         "command line -- it swallows everything after it, including "
                         "session ids.")
    ap.add_argument("--no-gpu-check", action="store_true",
                    help="trust --gpus as-is: use exactly those GPUs and never run "
                         "nvidia-smi. Requires --gpus. Use when you've already checked "
                         "they're free (nvitop) and the nvidia-smi probe is stalling "
                         "under load. Note this drops the guard against taking a GPU "
                         "another user grabs mid-run.")
    ap.add_argument("--use-video", action="store_true",
                    help="forward --use_video to rtmo_pipeline.py, reading *.mp4 "
                         "directly instead of the default pre-extracted frame folders "
                         "(see ../../../scripts/extract_frames.py).")
    ap.add_argument("--launch-stagger", type=float, default=DEFAULT_LAUNCH_STAGGER,
                    help=f"seconds to wait between consecutive session launches "
                         f"(default: {DEFAULT_LAUNCH_STAGGER}). Persistence mode is off on "
                         f"this node, so the first job onto a cold GPU pays a slow driver "
                         f"init; launching several at once makes them collide and fall back "
                         f"to CPU. 0 disables.")
    ap.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL,
                    help=f"seconds between nvidia-smi/candidate re-checks (default: "
                         f"{DEFAULT_POLL_INTERVAL})")
    ap.add_argument("--dry-run", action="store_true",
                    help="log planned copy/run/remove actions without doing them")
    args = ap.parse_args()

    whitelist = None
    if args.gpus:
        whitelist = {int(g) for g in args.gpus.replace(",", " ").split()}
    if args.no_gpu_check:
        if not whitelist:
            raise SystemExit("--no-gpu-check requires --gpus (e.g. --gpus 3,4,5,6)")
        print(f"--no-gpu-check: using GPUs {sorted(whitelist)} as given, "
              f"no nvidia-smi probing")
    if args.use_video:
        args.rtmo_args = [*args.rtmo_args, "--use_video"]

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = RTMO_ROOT / "run_logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"logs: {log_dir}/<session_id>.log")

    explicit = bool(args.sessions)
    known: set[str] = set()
    candidates: deque[str] = deque()
    if explicit:
        candidates.extend(args.sessions)
        known.update(args.sessions)
    else:
        new = discover_candidates(known)
        candidates.extend(new)
        known.update(new)
        print(f"auto-discovery mode: watching {ALL_SESSIONS_DIR} forever "
              f"(Ctrl-C to stop)")

    runner = Runner(log_dir, args.rtmo_args, args.dry_run)

    try:
        while candidates or runner.active_snapshot() or not explicit:
            if not explicit:
                new = discover_candidates(known)
                if new:
                    print(f"discovered {len(new)} new session(s): {', '.join(new)}")
                    candidates.extend(new)
                    known.update(new)

            active = runner.active_snapshot()
            pool = sorted(whitelist) if args.no_gpu_check else free_gpu_indices(whitelist)
            usable = [g for g in pool if g not in active]
            for gpu in usable:
                if not candidates:
                    break
                runner.launch(candidates.popleft(), gpu)
                if candidates and args.launch_stagger > 0 and not args.dry_run:
                    time.sleep(args.launch_stagger)

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\ninterrupted — waiting for active sessions to finish "
              "(Ctrl-C again to force)...", file=sys.stderr)

    try:
        runner.join_all()
    except KeyboardInterrupt:
        print("\nforced exit — active sessions were left running in the "
              "background and their staged data was not cleaned up.",
              file=sys.stderr)
        return 130

    summary = runner.summary
    summary_path = log_dir / "summary.tsv"
    with open(summary_path, "w") as f:
        for sid, rc in summary:
            f.write(f"{sid}\t{rc}\n")

    total = len(summary)
    ok = sum(1 for _, rc in summary if rc == 0)
    print(f"\ndone: {ok}/{total} sessions ok")
    if ok < total:
        print("failed sessions:")
        for sid, rc in summary:
            if rc != 0:
                print(f"  {sid} (rc={rc})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
