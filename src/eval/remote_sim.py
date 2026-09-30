"""
Parent-side client for the habitat worker (runs under `latentpilot`, py3.10).

Presents an ordinary simulator-like object while the real habitat process lives
behind a socket in the `habitat_render` environment. See habitat_worker.py for
why the split exists.
"""

import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]

import pathlib
import socket
import subprocess

import os

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_ROOT / "src"))

from eval.ipc import recv_msg, send_msg  # noqa: E402

# Python of the habitat_render env (environment/habitat_render.yml); override with HABITAT_PYTHON.
DEFAULT_WORKER_PY = pathlib.Path(os.environ.get("HABITAT_PYTHON", "python"))
WORKER_SCRIPT = _ROOT / "src" / "eval" / "habitat_worker.py"


class RemoteSimError(RuntimeError):
    """The worker reported a failure, with its traceback attached."""


class RemoteSim:
    """habitat-sim driven over a socket, from a different Python environment."""

    def __init__(self, worker_python=DEFAULT_WORKER_PY, quiet=True):
        worker_python = pathlib.Path(worker_python)
        if not worker_python.exists():
            raise FileNotFoundError(
                f"worker interpreter not found: {worker_python}. Pass "
                f"--worker-python pointing at the habitat_render env."
            )

        parent, child = socket.socketpair()
        # Habitat logs heavily; discard it so it cannot interleave with output.
        sink = subprocess.DEVNULL if quiet else None
        self.proc = subprocess.Popen(
            [str(worker_python), str(WORKER_SCRIPT), str(child.fileno())],
            pass_fds=(child.fileno(),),
            stdout=sink,
            stderr=sink,
        )
        child.close()                  # only the worker keeps its end
        self.sock = parent

    # -- protocol -----------------------------------------------------------
    def _call(self, **req):
        send_msg(self.sock, req)
        resp, raw = recv_msg(self.sock)
        if not resp.get("ok"):
            raise RemoteSimError(
                f"{req.get('cmd')} failed in worker: {resp.get('error')}\n"
                f"{resp.get('traceback', '')}"
            )
        return resp, raw

    @staticmethod
    def _frame(resp, raw):
        return np.frombuffer(raw, dtype=np.uint8).reshape(resp["shape"])

    # -- simulator surface --------------------------------------------------
    def load_scene(self, glb):
        self._call(cmd="load_scene", glb=str(glb))

    def reset(self, position, rotation):
        """Place the agent and return (frame, position)."""
        resp, raw = self._call(cmd="reset", position=list(position),
                               rotation=list(rotation))
        return self._frame(resp, raw), resp["position"]

    def step(self, action_name):
        """Execute one primitive and return (frame, position)."""
        resp, raw = self._call(cmd="step", action=action_name)
        return self._frame(resp, raw), resp["position"]

    def geodesic(self, a, b) -> float:
        resp, _ = self._call(cmd="geo", a=list(a), b=list(b))
        return resp["dist"]

    def geodesic_fn(self):
        """A distance_fn for src/eval/metrics.py."""
        return lambda a, b: self.geodesic(a, b)

    def close(self):
        try:
            self._call(cmd="close")
        except Exception:                                  # noqa: BLE001
            pass
        finally:
            self.sock.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
