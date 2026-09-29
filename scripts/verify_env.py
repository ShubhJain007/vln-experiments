#!/usr/bin/env python3
"""
verify_env.py — LatentPilot environment verification (AGENTS.md Sec. 1.5).

One script, one run, one report. Checks:
  1. PyTorch version, CUDA availability, GPU name and VRAM
  2. habitat-sim import + scene load + 10 agent steps (skipped if not installed)
  3. Cosmos-Reason2-2B backbone: hidden size d, visual tokens per frame N_v,
     vision encoder output dim, and VRAM after a bf16 load + forward pass.

Run:  python scripts/verify_env.py
"""

# --- ROS guard: strip /opt/ros/* from sys.path before any other import. ------
# ROS Humble injects its site-packages via PYTHONPATH and shadows/breaks
# unrelated imports (e.g. `launch` -> missing `lark`). Must run first.
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
# ----------------------------------------------------------------------------

import pathlib
import traceback

SEPARATOR = "-" * 62

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_PATH = str(_REPO_ROOT / "models" / "cosmos-reason2-2b")

TEST_IMAGE_SIZE = (448, 448)   # (H, W) -> expected N_v = 196
HABITAT_SCENE = None           # set to a .glb path once MP3D scenes are present


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# 1. PyTorch / CUDA
# ---------------------------------------------------------------------------
section("1. PyTorch / CUDA")
cuda_ok = False
try:
    import torch

    print(f"  torch version : {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"  cuda available: {cuda_ok}")
    if cuda_ok:
        print(f"  cuda version  : {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}  |  VRAM: {props.total_memory / 1024**3:.2f} GB")
    else:
        print("  WARNING: CUDA not available — training will not work.")
except Exception:
    print("  ERROR: torch import failed.")
    traceback.print_exc()
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# 2. habitat-sim
# ---------------------------------------------------------------------------
section("2. habitat-sim")
try:
    import habitat_sim  # noqa: F401

    print("  habitat_sim import: OK")

    if HABITAT_SCENE is None:
        print("  HABITAT_SCENE not set — skipping scene-load test.")
        print("  (Set HABITAT_SCENE at the top of this script once MP3D is downloaded.)")
    else:
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = HABITAT_SCENE          # must be set BEFORE Configuration()
        sim_cfg.enable_physics = False

        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "color_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = list(TEST_IMAGE_SIZE)

        agent_cfg = habitat_sim.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_spec]

        sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        try:
            agent = sim.initialize_agent(0)
            start = agent.get_state().position
            print(f"  scene loaded   : {HABITAT_SCENE}")
            print(f"  start position : {start}")

            actions = ["move_forward", "turn_left", "turn_right"]
            obs = None
            for step in range(10):
                obs = sim.step(actions[step % len(actions)])

            end = agent.get_state().position
            # NB: obs values are numpy arrays — never use `or` to pick a key,
            # `bool(ndarray)` raises "truth value is ambiguous".
            frame = obs["color_sensor"] if "color_sensor" in obs else next(iter(obs.values()))
            print(f"  end position   : {end}")
            print(f"  frame shape    : {frame.shape}")
            print("  habitat scene-load test: PASS")
        finally:
            sim.close()

except ModuleNotFoundError:
    print("  habitat_sim not installed — skipping (expected until Sec. 1.3 is done).")
except Exception:
    print("  habitat_sim error:")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# 3. Cosmos-Reason2-2B backbone
# ---------------------------------------------------------------------------
section("3. Cosmos-Reason2-2B backbone")
try:
    import numpy as np
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"  model path: {MODEL_PATH}")
    cfg = AutoConfig.from_pretrained(MODEL_PATH)

    d = cfg.text_config.hidden_size
    vision_internal = cfg.vision_config.hidden_size
    vision_out = cfg.vision_config.out_hidden_size
    patch = cfg.vision_config.patch_size
    merge = cfg.vision_config.spatial_merge_size
    H, W = TEST_IMAGE_SIZE

    n_v_formula = (H // patch) * (W // patch) // (merge ** 2)

    print(f"  hidden size d                  : {d}")
    print(f"  vision encoder internal dim    : {vision_internal}")
    print(f"  vision encoder out_hidden_size : {vision_out}")
    print(f"  patch_size / spatial_merge     : {patch} / {merge}")
    print(f"  test resolution                : {H} x {W}")
    print(f"  N_v (formula)                  : {n_v_formula}")

    if vision_out == d:
        print(f"  vision_out == d == {d}: OK (no projection mismatch)")
    else:
        print(f"  WARNING: vision_out ({vision_out}) != d ({d}) — check the merger.")

    print("\n  loading model weights (bf16) ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda" if cuda_ok else "cpu",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    img = Image.fromarray(np.random.randint(0, 255, (H, W, 3), dtype=np.uint8))
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": img},
                    {"type": "text", "text": "What do you see?"}],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt")
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    # N_v measured two independent ways.
    n_v_tokens = int((inputs["input_ids"][0] == cfg.image_token_id).sum().item())

    with torch.no_grad():
        # Eq. 4: v_t = E_phi(o_t) in R^{N_v x d}.
        # The vision tower returns BaseModelOutputWithDeepstackFeatures, not a
        # tensor; the merged tokens are in .pooler_output (one entry per image).
        vis_out = model.model.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"]
        )
        v_t = vis_out.pooler_output[0]
        _ = model(**inputs)          # full forward, for the VRAM figure

    print(f"  N_v (image_token count)        : {n_v_tokens}")
    print(f"  E_phi output shape (N_v, d)    : {tuple(v_t.shape)}")

    checks = {
        "N_v formula == token count": n_v_formula == n_v_tokens,
        "N_v == encoder output rows": n_v_tokens == v_t.shape[0],
        "encoder output dim == d": v_t.shape[1] == d,
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")

    if cuda_ok:
        print("\n  VRAM after bf16 load + forward:")
        print(f"    allocated : {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"    reserved  : {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"    peak      : {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    del model, inputs
    if cuda_ok:
        torch.cuda.empty_cache()

except Exception:
    print("  ERROR during backbone check:")
    traceback.print_exc()


# ---------------------------------------------------------------------------
section("SUMMARY — pass criteria (AGENTS.md Sec. 1.5)")
print("""  - torch.cuda.is_available() = True
  - GPU VRAM >= 12 GB (16 GB target)
  - d = 2048
  - vision_out = 2048 (matches d, no projection mismatch)
  - N_v consistent across formula, token count, and encoder output
  - VRAM after load reported (decides LoRA vs QLoRA)
""")
