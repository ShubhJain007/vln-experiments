"""
Habitat worker process — runs under `habitat_render` (Python 3.9).

Owns the simulator and nothing else. Receives commands over a socket inherited
from the parent, returns observations and geodesic distances. The parent
(`latentpilot`, Python 3.10) owns the trained model.

WHY THE SPLIT: no habitat-sim build supports Python 3.10, while
transformers/peft/accelerate at the versions this project verified all REQUIRE
>=3.10. Installing older ML packages into the 3.9 env would mean evaluating
with a different transformers than training used -- and the APIs this code
depends on (`get_image_features().pooler_output` as a tuple,
`get_vision_position_ids`, the `compute_3d_position_ids` gating behind the
mRoPE fix) were all verified against 5.16.1 specifically. Splitting the process
keeps both environments exactly as verified, with zero version drift.

Not launched directly -- `scripts/eval_stage0.py` spawns it.
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import pathlib
import socket
import traceback

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from eval.ipc import recv_msg, send_msg  # noqa: E402
from model.action_space import habitat_agent_action_config  # noqa: E402


class SimHost:
    """Holds at most one open simulator, swapping scenes on demand."""

    def __init__(self, resolution=(448, 448)):
        self.resolution = tuple(resolution)
        self.sim = None
        self.scene = None

    def load_scene(self, glb):
        import habitat_sim

        if self.scene == glb and self.sim is not None:
            return
        self.close()

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(glb)
        sim_cfg.enable_physics = False

        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = "color_sensor"
        rgb.sensor_type = habitat_sim.SensorType.COLOR
        rgb.resolution = list(self.resolution)

        agent_cfg = habitat_sim.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb]
        # Explicit primitives: habitat's default turn is 10 deg, the paper's
        # is 15 (README gotcha #5). Never inherit these.
        agent_cfg.action_space = {
            name: habitat_sim.agent.ActionSpec(
                name, habitat_sim.agent.ActuationSpec(amount=spec["amount"])
            )
            for name, spec in habitat_agent_action_config().items()
        }

        self.sim = habitat_sim.Simulator(
            habitat_sim.Configuration(sim_cfg, [agent_cfg])
        )
        self.scene = glb

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None
            self.scene = None

    def set_state(self, position, rotation):
        import habitat_sim
        import quaternion  # noqa: F401  (registers the numpy quaternion dtype)

        st = habitat_sim.AgentState()
        st.position = np.array(position, dtype=np.float32)
        x, y, z, w = rotation          # R2R-CE stores xyzw; habitat wants wxyz
        st.rotation = np.quaternion(w, x, y, z)
        self.sim.get_agent(0).set_state(st)

    def observe(self):
        """Current RGB frame and agent position.

        Alpha is dropped here: habitat's color sensor emits RGBA, Eq. 1 wants
        C=3 (README gotcha #4). Doing it at the source also cuts the bytes
        crossing the socket by a quarter.
        """
        obs = self.sim.get_sensor_observations()
        frame = np.ascontiguousarray(
            np.asarray(obs["color_sensor"])[..., :3], dtype=np.uint8
        )
        pos = self.sim.get_agent(0).get_state().position.tolist()
        return frame, pos

    def step(self, action_name):
        self.sim.step(action_name)
        return self.observe()

    def geodesic(self, a, b):
        import habitat_sim

        p = habitat_sim.ShortestPath()
        p.requested_start = np.array(a, dtype=np.float32)
        p.requested_end = np.array(b, dtype=np.float32)
        return p.geodesic_distance if self.sim.pathfinder.find_path(p) else float("inf")


def main():
    fd = int(sys.argv[1])
    sock = socket.socket(fileno=fd)
    host = SimHost()

    try:
        while True:
            try:
                req, _ = recv_msg(sock)
            except (ConnectionError, OSError):
                break

            cmd = req.get("cmd")
            try:
                if cmd == "load_scene":
                    host.load_scene(req["glb"])
                    send_msg(sock, {"ok": True})

                elif cmd == "reset":
                    host.set_state(req["position"], req["rotation"])
                    frame, pos = host.observe()
                    send_msg(sock, {"ok": True, "position": pos,
                                    "shape": list(frame.shape)}, frame.tobytes())

                elif cmd == "step":
                    frame, pos = host.step(req["action"])
                    send_msg(sock, {"ok": True, "position": pos,
                                    "shape": list(frame.shape)}, frame.tobytes())

                elif cmd == "geo":
                    send_msg(sock, {"ok": True,
                                    "dist": host.geodesic(req["a"], req["b"])})

                elif cmd == "close":
                    send_msg(sock, {"ok": True})
                    break

                else:
                    send_msg(sock, {"ok": False, "error": f"unknown cmd {cmd!r}"})

            except Exception as exc:                       # noqa: BLE001
                # Report the failure back rather than dying silently -- a dead
                # worker would surface in the parent only as a confusing
                # ConnectionError several steps later.
                send_msg(sock, {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc()})
    finally:
        host.close()
        sock.close()


if __name__ == "__main__":
    main()
