#!/usr/bin/env python3
"""
verify_habitat.py — habitat-sim scene-load check (AGENTS.md Sec. 1.5, split out).

Split from verify_env.py because no habitat-sim build on conda-forge/aihabitat
supports Python 3.10 (max is 3.9), while the backbone stack (torch/transformers)
is on 3.10. Run this under the separate `habitat_render` (Python 3.9) env;
run verify_env.py under `latentpilot` for the backbone checks.

Run:
  conda activate habitat_render
  python scripts/verify_habitat.py --scene <path-to-.glb>
"""

import argparse
import pathlib
import sys

SEPARATOR = "-" * 62


def main() -> int:
    ap = argparse.ArgumentParser(description="habitat-sim scene-load smoke test")
    ap.add_argument("--scene", required=True, help="Path to a .glb scene file")
    ap.add_argument("--resolution", type=int, nargs=2, default=[448, 448],
                    metavar=("H", "W"))
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    scene_path = pathlib.Path(args.scene).expanduser().resolve()
    if not scene_path.exists():
        print(f"ERROR: scene file not found: {scene_path}")
        return 1

    print(f"{SEPARATOR}\n  habitat-sim scene-load check\n{SEPARATOR}")

    try:
        import habitat_sim
    except ModuleNotFoundError:
        print("ERROR: habitat_sim not importable. Are you in the habitat_render env?")
        print("  conda activate habitat_render")
        return 1

    print(f"  habitat_sim import: OK  (version: {getattr(habitat_sim, '__version__', 'unknown')})")
    print(f"  scene: {scene_path}")

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_path)
    sim_cfg.enable_physics = False

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = list(args.resolution)

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    try:
        agent = sim.initialize_agent(0)
        start = agent.get_state().position
        print(f"  start position: {start}")

        actions = ["move_forward", "turn_left", "turn_right"]
        obs = None
        for step in range(args.steps):
            obs = sim.step(actions[step % len(actions)])

        end = agent.get_state().position
        # obs values are numpy arrays; never use `or` between them (ambiguous truth value).
        frame = obs["color_sensor"] if "color_sensor" in obs else next(iter(obs.values()))

        print(f"  end position  : {end}")
        print(f"  frame shape   : {frame.shape}")
        print(f"  {args.steps} steps completed without error")
        print(f"\n  PASS: habitat scene-load + agent stepping works.")
        return 0
    finally:
        sim.close()


if __name__ == "__main__":
    raise SystemExit(main())
