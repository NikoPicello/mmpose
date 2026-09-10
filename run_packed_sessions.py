#!/usr/bin/env python3
"""Run demo/rtmo_pipeline.py across many sessions, packing several onto each GPU.

Sibling of run_parallel_sessions.py -- same staging/cleanup/logging behavior,
same CLI shape, same resources/ layout. The only difference is the scheduling
model: instead of claiming a whole GPU per session (one session per idle
GPU), this script fits up to --max-per-gpu of *our own* sessions onto each
GPU that has no *other* user's process on it. RTMO is light on VRAM (~450MB
observed per session against a 24GB card), so a GPU that would otherwise sit
idle waiting for one session to finish can run several concurrently.

Lives at the mmpose package root (a sibling of demo/, not inside it) and
invokes demo/rtmo_pipeline.py by relative path from there, same as
run_parallel_sessions.py.

Run this from inside the container, from the already-activated mmpose conda
env (e.g. `python run_packed_sessions.py ...`) -- each per-session subprocess
is launched with the same interpreter (sys.executable), so the env this
script itself runs under is the env rtmo_pipeline.py runs under too.

Resource layout (same as run_parallel_sessions.py; resources/ is located by
walking up from this file until a resources/sessions dir turns up):
    resources/all_sessions/<sid>   the full dataset (big, separately-mounted pool)
    resources/sessions/<sid>       local scratch rtmo_pipeline.py actually reads
                                    from; sessions currently staged/in-progress live
                                    here
    resources/rtmo_results/<sid>   pipeline output; a session with output here
                                    already exists is treated as done and skipped

Per session: copy the *entire* session folder (session_data.txt plus every
activity subfolder -- talk_task/lego_task/ghost_task/animals_task/gaze_task)
from resources/all_sessions/<sid> into resources/sessions/<sid>, atomically (a
partial/interrupted copy lands in a .tmp path, never mistaken for a complete
one). Then run `demo/rtmo_pipeline.py --session <sid>` with CUDA_VISIBLE_DEVICES
pinned to one GPU, and on success remove resources/sessions/<sid> entirely
(freeing local scratch space). A failed run's staged copy is left in place so
a re-run picks it up without re-copying.

Note rtmo_pipeline.py's own --activities default is all five real activity
folder names (animals_task, gaze_task, ghost_task, lego_task, talk_task) --
NOT the shorter 'lego'/'talk'/etc you might guess from the activity name
alone; the folders on disk carry the '_task' suffix. Pass --rtmo-args
--activities explicitly if you only want a subset, e.g. `--rtmo-args
--activities lego_task`. Also note --session is a substring match, shared
with rtmo_pipeline.py/smpler_pipeline.py's own semantics -- session ids
passed here should be the full session id so they can't accidentally match
more than one folder under resources/sessions.

The already-processed / discovery check below only looks at whether
resources/rtmo_results/<sid> has *any* output, same coarse granularity as the
sibling run_parallel_sessions.py script -- it does not check that every
activity/camera you actually want was extracted. Use rtmo_pipeline.py's own
--skip-existing flag (forward it via --rtmo-args) if you re-run a session that
was only partially processed and want it to fill in gaps instead of redoing
everything.

Scheduling model ("packing"):
  A GPU is entirely off-limits the moment ANY process that isn't one of our
  own tracked session subprocesses shows up on it (any user, any container --
  checked via `nvidia-smi --query-compute-apps`, cross-referenced against the
  PIDs of subprocesses we ourselves launched). This mirrors
  run_parallel_sessions.py's "don't touch a GPU someone else is using"
  contract -- packing only ever adds more of *our* work onto a GPU, never
  onto one a stranger is already on.

  Within a GPU that passes that check, up to --max-per-gpu of our own
  sessions may run concurrently on it. A further --min-free-mib floor on
  nvidia-smi's reported free memory gates each additional session, so a GPU
  that's already packed tighter than expected (bigger videos, more people in
  frame, etc.) doesn't get pushed into an OOM.

  GPUs we've already claimed slots on are tracked in-process (a per-GPU
  count, not just a set) so a not-yet-CUDA-initialized subprocess can't be
  over-booked during its startup lag, the same concern
  run_parallel_sessions.py's single-slot version has.

With explicit session ids on the command line, the candidate list is fixed and
the script exits once they're all done. With no ids, it auto-discovers
candidates from resources/all_sessions and keeps re-scanning for newly
arrived ones, so it runs indefinitely (Ctrl-C to stop; sessions already
mid-run finish before the process exits).

Usage:
    python run_packed_sessions.py                        # watch all_sessions forever
    python run_packed_sessions.py 000000 004096           # just these, then exit
    python run_packed_sessions.py --gpus 2,3,4,5           # restrict the GPU whitelist
    python run_packed_sessions.py --max-per-gpu 6           # pack up to 6 sessions/GPU
    python run_packed_sessions.py --min-free-mib 3000        # bigger safety margin
    python run_packed_sessions.py --rtmo-args --activities talk_task lego_task ghost_task animals_task gaze_task
    python run_packed_sessions.py --dry-run                   # log planned actions only

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
DEFAULT_MAX_PER_GPU = 4
DEFAULT_MIN_FREE_MIB = 5000


def gpu_status(whitelist: set[int] | None, own_pids: set[int]) -> dict[int, int]:
    """Free-memory (MiB) per GPU index that is safe for us to add more of our
    own sessions to -- i.e. has zero processes that aren't one of `own_pids`.
    A GPU with any foreign process is omitted entirely (never a candidate)."""
    try:
        gpus_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout
        apps_out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"nvidia-smi query failed: {e}") from e

    foreign_uuids: set[str] = set()
    for line in apps_out.strip().splitlines():
        if not line.strip():
            continue
        pid_s, uuid = (p.strip() for p in line.split(",", 1))
        if int(pid_s) not in own_pids:
            foreign_uuids.add(uuid)

    free_mib: dict[int, int] = {}
    for line in gpus_out.strip().splitlines():
        idx_s, uuid, free_s = (p.strip() for p in line.split(",", 2))
        idx = int(idx_s)
        if whitelist is not None and idx not in whitelist:
            continue
        if uuid in foreign_uuids:
            continue
        free_mib[idx] = int(free_s)
    return free_mib


def already_processed(sid: str) -> bool:
    d = RTMO_RESULTS_DIR / sid
    return d.is_dir() and any(d.iterdir())


def discover_candidates(known: set[str]) -> list[str]:
    if not ALL_SESSIONS_DIR.is_dir():
        return []
    ids = sorted(p.name for p in ALL_SESSIONS_DIR.iterdir() if p.is_dir())
    return [sid for sid in ids if sid not in known and not already_processed(sid)]


class Runner:
    def __init__(self, log_dir: Path, rtmo_args: list[str], dry_run: bool,
                max_per_gpu: int, min_free_mib: int):
        self.log_dir = log_dir
        self.rtmo_args = rtmo_args
        self.dry_run = dry_run
        self.max_per_gpu = max_per_gpu
        self.min_free_mib = min_free_mib

        self.lock = threading.Lock()
        self.gpu_counts: dict[int, int] = {}   # gpu -> count of our active sessions on it
        self.own_pids: set[int] = set()        # PIDs of subprocesses we've launched (still running)
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
        cmd = [sys.executable, "demo/rtmo_pipeline.py", "--session", sid, *self.rtmo_args]
        print(f"[{sid}] running: {' '.join(cmd)} (CUDA_VISIBLE_DEVICES={gpu})", file=log_fh)
        if self.dry_run:
            print(f"[{sid}] DRY-RUN: skipping actual run", file=log_fh)
            return 0
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_fh.flush()
        # Popen (not run) so the PID is known immediately -- it needs to be
        # registered in own_pids right away, before nvidia-smi has any chance
        # to see this process's CUDA context and (mis)classify it as foreign.
        proc = subprocess.Popen(cmd, cwd=str(RTMO_ROOT), env=env,
                                stdout=log_fh, stderr=subprocess.STDOUT)
        with self.lock:
            self.own_pids.add(proc.pid)
        try:
            return proc.wait()
        finally:
            with self.lock:
                self.own_pids.discard(proc.pid)

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
                self.gpu_counts[gpu] -= 1
                if self.gpu_counts[gpu] <= 0:
                    del self.gpu_counts[gpu]

    def launch(self, sid: str, gpu: int) -> None:
        with self.lock:
            self.gpu_counts[gpu] = self.gpu_counts.get(gpu, 0) + 1
        t = threading.Thread(target=self._worker, args=(sid, gpu), daemon=True)
        t.start()
        self.threads.append(t)

    def snapshot(self) -> tuple[dict[int, int], set[int]]:
        with self.lock:
            return dict(self.gpu_counts), set(self.own_pids)

    def slots(self, whitelist: set[int] | None) -> list[int]:
        """GPU indices with a free packing slot right now, one entry per open
        slot (a GPU with 3 of --max-per-gpu 4 used and enough free memory
        appears once; a fully-idle one with plenty of memory could appear
        --max-per-gpu times)."""
        counts, own_pids = self.snapshot()
        free_mib = gpu_status(whitelist, own_pids)
        out: list[int] = []
        for gpu, mib in free_mib.items():
            used = counts.get(gpu, 0)
            open_slots = self.max_per_gpu - used
            if open_slots <= 0:
                continue
            # Each additional session needs its own min_free_mib headroom;
            # nvidia-smi's free memory already reflects sessions that have
            # actually allocated, but not ones still starting up (CUDA init
            # lag) -- gpu_counts (in-process) is what protects against that,
            # this is just an extra floor against under-estimating usage.
            fit_by_mem = mib // self.min_free_mib
            out.extend([gpu] * max(0, min(open_slots, fit_by_mem)))
        return out

    def join_all(self) -> None:
        for t in self.threads:
            t.join()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run rtmo_pipeline.py across sessions, packing several per GPU.")
    ap.add_argument("sessions", nargs="*",
                    help="explicit session ids to run (default: auto-discover from "
                         "resources/all_sessions, re-scanning forever)")
    ap.add_argument("--gpus", default=os.environ.get("GPUS"),
                    help="comma-separated GPU index whitelist (default: all GPUs "
                         "reported by nvidia-smi). Still skipped entirely whenever "
                         "another user's process is on it.")
    ap.add_argument("--max-per-gpu", type=int, default=DEFAULT_MAX_PER_GPU,
                    help=f"max concurrent sessions of ours packed onto one GPU "
                         f"(default: {DEFAULT_MAX_PER_GPU})")
    ap.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB,
                    help=f"required free GPU memory (MiB) per additional packed "
                         f"session (default: {DEFAULT_MIN_FREE_MIB})")
    ap.add_argument("--rtmo-args", nargs=argparse.REMAINDER, default=[],
                    help="remaining args forwarded to rtmo_pipeline.py, e.g. "
                         "--activities talk_task lego_task --skip-existing. Must be "
                         "LAST on the command line -- it swallows everything after "
                         "it, including session ids.")
    ap.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL,
                    help=f"seconds between nvidia-smi/candidate re-checks (default: "
                         f"{DEFAULT_POLL_INTERVAL})")
    ap.add_argument("--dry-run", action="store_true",
                    help="log planned copy/run/remove actions without doing them")
    args = ap.parse_args()

    if args.max_per_gpu < 1:
        raise SystemExit("--max-per-gpu must be >= 1")

    whitelist = None
    if args.gpus:
        whitelist = {int(g) for g in args.gpus.replace(",", " ").split()}

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = RTMO_ROOT / "run_logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"logs: {log_dir}/<session_id>.log")
    print(f"packing: up to {args.max_per_gpu} sessions/GPU, "
          f"{args.min_free_mib} MiB free required per additional session")

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

    runner = Runner(log_dir, args.rtmo_args, args.dry_run,
                    args.max_per_gpu, args.min_free_mib)

    try:
        while candidates or runner.snapshot()[0] or not explicit:
            if not explicit:
                new = discover_candidates(known)
                if new:
                    print(f"discovered {len(new)} new session(s): {', '.join(new)}")
                    candidates.extend(new)
                    known.update(new)

            for gpu in runner.slots(whitelist):
                if not candidates:
                    break
                runner.launch(candidates.popleft(), gpu)

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
