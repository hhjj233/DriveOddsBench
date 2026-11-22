#!/usr/bin/env python3
"""PID/birth-bound cleanup of our own session only; no GPU or scheduler calls.

Copied without changes from the reviewed SkyReels DEV OwnedSession primitives.
A caller must create its child with start_new_session=True and retain this object.
"""
import os
from pathlib import Path
import select
import signal
import subprocess
import time

def process(pid):
    try:
        r = Path('/proc') / str(pid)
        t = (r / 'stat').read_text().rsplit(') ', 1)[1].split()
        return dict(pid=pid, state=t[0], ppid=int(t[1]), pgid=int(t[2]), sid=int(t[3]),
                    birth=int(t[19]), argv=[v.decode() for v in (r / 'cmdline').read_bytes().split(b'\0') if v])
    except (FileNotFoundError, ProcessLookupError):
        return None


def session_rows(pgid):
    rows = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            t = path.read_text().rsplit(') ', 1)[1].split()
            if int(t[2]) == pgid and int(t[3]) == pgid and t[0] not in ('Z', 'X'):
                rows.append(dict(pid=int(path.parent.name), state=t[0], ppid=int(t[1]), pgid=int(t[2]), sid=int(t[3]), birth=int(t[19])))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return rows


class OwnedSession:
    def __init__(self, child, parent_pid):
        self.child = child
        self.parent_pid = parent_pid
        self.pgid = child.pid  # Popen(start_new_session=True), not inferred from a partial ledger.
        self.birth = None
        self.handles = {}

    def bind(self):
        row = process(self.child.pid)
        if row is None or row['state'] in ('Z', 'X'):
            return False
        if row['ppid'] != self.parent_pid or row['pgid'] != self.pgid or row['sid'] != self.pgid:
            raise RuntimeError('wrapper direct-child/session proof lost')
        if self.birth is not None and row['birth'] != self.birth:
            raise RuntimeError('wrapper birth changed')
        self.birth = row['birth']
        self._bind_row(row)
        return True

    def _bind_row(self, row):
        key = (row['pid'], row['birth'])
        if key in self.handles:
            return self.handles[key]
        fd = os.pidfd_open(row['pid'], 0)
        try:
            fresh = process(row['pid'])
            if fresh is None or any(fresh[k] != row[k] for k in ('pid', 'birth', 'pgid', 'sid')):
                raise ProcessLookupError('process changed during pidfd binding')
        except BaseException:
            os.close(fd)
            raise
        self.handles[key] = fd
        return fd

    @staticmethod
    def dead(fd):
        poll = select.poll()
        poll.register(fd, select.POLLIN)
        return bool(poll.poll(0))

    def signal_row(self, row, sig):
        fd = self._bind_row(row)
        if not self.dead(fd):
            signal.pidfd_send_signal(fd, sig, None, 0)

    def cleanup(self, grace=5):
        """Returns True only with a fresh empty owned session. Unknown scope remains pending."""
        leader = process(self.child.pid)
        if leader and leader['state'] not in ('Z', 'X'):
            if not self.bind():
                return False
            # Must happen before signalling flock, memory-wait children, or science.
            self.signal_row(leader, signal.SIGSTOP)
            fresh = process(self.child.pid)
            if fresh is None or fresh['birth'] != self.birth or fresh['state'] not in ('T', 't'):
                return False
        elif self.birth is None:
            # Reaped/fast child with no birth proof: never signal an unknown session.
            return not session_rows(self.pgid)
        rows = [r for r in session_rows(self.pgid) if r['pid'] != self.child.pid]
        for row in rows:
            self.signal_row(row, signal.SIGTERM)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not [r for r in session_rows(self.pgid) if r['pid'] != self.child.pid]:
                break
            time.sleep(.05)
        # The leader remains STOPped, so a killed flock cannot cause fallthrough.
        for row in [r for r in session_rows(self.pgid) if r['pid'] != self.child.pid]:
            self.signal_row(row, signal.SIGKILL)
        if leader and leader['state'] not in ('Z', 'X'):
            fresh = process(self.child.pid)
            if fresh and fresh['birth'] == self.birth:
                self.signal_row(fresh, signal.SIGKILL)
        if self.child.poll() is None:
            try:
                self.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                return False
        return not session_rows(self.pgid)

    def close(self):
        for fd in self.handles.values():
            os.close(fd)
        self.handles.clear()
