#!/usr/bin/env python3
"""eval_run_log.py — UPGRADE I: per-run console log tee.

Split out of flow_eval_node.py unchanged. See RunLogTee's own docstring for the
full rationale (fd-level redirection is the only way to also capture rclpy's
get_logger() lines, which the C rcutils layer writes straight to the process
stdout/stderr).
"""
import os
import sys
import threading
import time as _time


class RunLogTee:
    """UPGRADE I: mirror the ENTIRE console output of this run into a text file.
    Enabled with -p log_to_file:=true. Works at the FILE-DESCRIPTOR level (fd 1/2
    are redirected through pipes and pumped to both the original console and the
    file), which is the only way to also capture rclpy's get_logger() lines —
    those are written by the C rcutils layer directly to the process's stdout/
    stderr and would be invisible to a Python-level sys.stdout wrapper.
    * One file per run, timestamped: <log_dir>/flow_eval_YYYYmmdd_HHMMSS.txt
    * Written incrementally (unbuffered), so Ctrl+C at ANY moment loses nothing:
      the file is already complete on disk; close() just restores the fds.
    """
    def __init__(self, log_dir):
        log_dir = os.path.expanduser(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        stamp = _time.strftime('%Y%m%d_%H%M%S')
        self.path = os.path.join(log_dir, f'flow_eval_{stamp}.txt')
        self._file = open(self.path, 'ab', buffering=0)
        self._file.write(
            f'# flow_eval_node run log — started {_time.strftime("%Y-%m-%d %H:%M:%S")}\n'
            f'# argv: {" ".join(sys.argv)}\n'.encode())
        # Flush anything Python has buffered BEFORE swapping the fds out from
        # under it, or those bytes would appear out of order.
        sys.stdout.flush()
        sys.stderr.flush()
        self._saved = [os.dup(1), os.dup(2)]
        self._readers = []
        self._threads = []
        for fd, orig in ((1, self._saved[0]), (2, self._saved[1])):
            r, w = os.pipe()
            os.dup2(w, fd)
            os.close(w)
            t = threading.Thread(target=self._pump, args=(r, orig), daemon=True)
            t.start()
            self._readers.append(r)
            self._threads.append(t)
        # fd 1 now points at a pipe (not a tty), so Python would switch stdout to
        # block buffering and the console would lag ~4 kB behind. Force line mode.
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass
    def _pump(self, r, orig):
        while True:
            try:
                data = os.read(r, 65536)
            except OSError:
                break
            if not data:
                break
            os.write(orig, data)          # still show on the console
            self._file.write(data)        # and persist immediately
    def close(self):
        sys.stdout.flush()
        sys.stderr.flush()
        # Restoring the fds closes the last write-ends of the pipes -> the pump
        # threads see EOF and finish after draining everything still in flight.
        os.dup2(self._saved[0], 1)
        os.dup2(self._saved[1], 2)
        for t in self._threads:
            t.join(timeout=2.0)
        for r in self._readers:
            try:
                os.close(r)
            except OSError:
                pass
        for s in self._saved:
            os.close(s)
        self._file.write(
            f'# run ended {_time.strftime("%Y-%m-%d %H:%M:%S")}\n'.encode())
        self._file.close()
