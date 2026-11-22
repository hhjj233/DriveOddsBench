#!/usr/bin/env python3
"""Run one command on a shared server and stop it (its own process group only) if the host or the GPU gets too tight.

  guarded_run.py --log FILE.json [--min-host-avail-gib 4] [--gpu-uuid UUID --min-gpu-free-mib 6144] -- CMD ARGS...

Every second it reads MemAvailable (/proc/meminfo) and, with --gpu-uuid, the free memory of that GPU (nvidia-smi). If
MemAvailable falls below --min-host-avail-gib, or the GPU's free memory below --min-gpu-free-mib, it sends SIGTERM (then
SIGKILL after 20 s) to the command's process group, which this script created; it never signals any other process.
The JSON log records the minima, the child's peak RSS (VmHWM) and the exit status.

Vista copy (models/epona/guarded_run.py plus two additions): --pid-file FILE gets the child's pid (its process group) while
it runs, and SIGTERM / SIGINT / SIGHUP to this script are forwarded to the child's process group (SIGTERM, then SIGKILL
after 20 s) before it exits, so stopping a chain never leaves the child running on the GPU (the child runs in its own
session and does not see the terminal's signals).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def mem_available_gib() -> float:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 2**20
    return float('nan')


def gpu_free_mib(uuid: str) -> float | None:
    try:
        out = subprocess.check_output(['nvidia-smi', '-i', uuid, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=10)
        return float(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def child_hwm_gib(pid: int) -> float:
    """The child's peak resident memory (VmHWM, GiB)."""
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmHWM:'):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log', type=Path, required=True)
    ap.add_argument('--min-host-avail-gib', type=float, default=4.0)
    ap.add_argument('--gpu-uuid', default=None)
    ap.add_argument('--min-gpu-free-mib', type=float, default=6144)
    ap.add_argument('--pid-file', type=Path, default=None)
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == '--' else args.cmd
    if not cmd:
        raise SystemExit('no command')
    rec = dict(cmd=cmd, started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), host_avail_start_gib=round(mem_available_gib(), 2),
               gpu_free_start_mib=gpu_free_mib(args.gpu_uuid) if args.gpu_uuid else None, min_host_avail_gib=None, min_gpu_free_mib=None,
               child_peak_rss_gib=0.0, stopped_by_guard=None)
    proc = subprocess.Popen(cmd, start_new_session=True)
    rec['pid'] = proc.pid
    if args.pid_file:
        args.pid_file.parent.mkdir(parents=True, exist_ok=True)
        args.pid_file.write_text(f'{proc.pid}\n')
    got = {}

    def on_signal(signum, frame):  # forward to the child's process group, wait, then leave the loop
        got['signal'] = signum

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, on_signal)
    min_host, min_gpu, last_gpu_check = float('inf'), float('inf'), 0.0
    reason = None
    while proc.poll() is None:
        h = mem_available_gib()
        min_host = min(min_host, h)
        rec['child_peak_rss_gib'] = max(rec['child_peak_rss_gib'], round(child_hwm_gib(proc.pid), 2))
        if args.gpu_uuid and time.time() - last_gpu_check >= 2:
            g = gpu_free_mib(args.gpu_uuid)
            last_gpu_check = time.time()
            if g is not None:
                min_gpu = min(min_gpu, g)
                if g < args.min_gpu_free_mib:
                    reason = f'GPU free {g:.0f} MiB < {args.min_gpu_free_mib:.0f}'
        if h < args.min_host_avail_gib:
            reason = f'MemAvailable {h:.2f} GiB < {args.min_host_avail_gib}'
        if got.get('signal'):
            reason = f'signal {got["signal"]} to guarded_run'
        if reason:
            rec['stopped_by_guard'] = reason
            print(f'guarded_run: {reason}; stopping process group {proc.pid}', file=sys.stderr, flush=True)
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            break
        time.sleep(1.0)
    rc = proc.wait()
    if args.pid_file:
        args.pid_file.unlink(missing_ok=True)
    rec.update(returncode=rc, finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'), min_host_avail_gib=round(min_host, 2) if min_host < float('inf') else None,
               min_gpu_free_mib=min_gpu if min_gpu < float('inf') else None)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(rec, indent=1) + '\n')
    sys.exit(rc if not reason else 75)


if __name__ == '__main__':
    main()
