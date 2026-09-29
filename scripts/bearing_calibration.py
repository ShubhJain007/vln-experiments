#!/usr/bin/env python3
"""Is the turn miss a REGRESSION-SHRINKAGE artifact?

Failure analysis showed: expert LEFT -> predicted FWD 38.8%, expert RIGHT ->
predicted FWD 42.8%, but LEFT<->RIGHT confusion only 1.5-5.9%. The model knows
which way to turn and fails to commit. That signature is what an L2 bearing
regression produces when 62% of training steps have near-zero bearing: the
prediction shrinks toward the mean and falls under the 7.5 deg decision boundary.

Dumps raw (pred_theta, true_theta) so we can (a) measure the shrinkage slope and
(b) sweep the decision threshold post-hoc -- a fix needing no retraining.
Multiple scans, unlike the single-scan first pass.
"""
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
import json, math, pathlib, collections
import numpy as np, torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from peft import LoraConfig, get_peft_model
from data.pointing_dataset import PointingDataset, load_frame
from model.backbone import Backbone
from model.input_sequence import InputSequenceBuilder
from model.pointing_head import PointingHead, split_outputs
from model.vision_encoder import VisionEncoder

ck = pathlib.Path("checkpoints/pointing_fusion/step120000")
cfg = json.loads((ck / "config.json").read_text())
K, S = cfg["history_frames"], cfg["history_stride"]
FL = tuple(cfg["fusion_layers"]) if cfg.get("fusion_layers") else None

enc = VisionEncoder()
sd = torch.load(ck / "adapter.pt", map_location="cpu", weights_only=True)
r = sd[[k for k in sd if k.endswith("lora_A.default.weight")][0]].shape[0]
get_peft_model(enc.model, LoraConfig(r=r, lora_alpha=2*r, lora_dropout=0.0, bias="none",
    task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj"]))
enc.model.load_state_dict(sd, strict=False)
head = PointingHead(enc.d, dtype=torch.float32, device=enc.model.device, layers=FL)
head.load_state_dict(torch.load(ck/"head.pt", map_location="cpu", weights_only=True))
head.to(enc.model.device).eval(); enc.model.eval()
b = InputSequenceBuilder(enc.model, enc.processor.tokenizer)
bb = Backbone(enc.model)

ds = PointingDataset("val_unseen", max_episodes=400, history_frames=K, history_stride=S)
steps = [s for s in ds.steps if not s.is_stop]
# spread across scans instead of taking a contiguous prefix (first pass hit one scan)
by_scan = collections.defaultdict(list)
for s in steps: by_scan[s.scan].append(s)
per = max(1, 3000 // max(len(by_scan), 1))
sel = [s for sc in sorted(by_scan) for s in by_scan[sc][:per]]
print(f"{len(sel)} steps across {len(by_scan)} scans", flush=True)

rows = []
with torch.no_grad():
    for i, s in enumerate(sel):
        v = enc.encode(load_frame(s.frame_path))
        hist = [enc.encode(load_frame(h)) for h in s.history_paths] if K else None
        seq = b.build(s.instruction, v, history_visual=hist)
        if FL:
            lm = enc.model.model.language_model(inputs_embeds=seq.inputs_embeds,
                attention_mask=seq.attention_mask, position_ids=bb.build_position_ids(seq),
                use_cache=False, output_hidden_states=True)
            feats = head.fuse(lm.hidden_states,
                torch.tensor([seq.action_index], device=enc.model.device), FL)
        else:
            H = bb.forward(seq, position_ids=bb.build_position_ids(seq))
            feats = H[seq.action_index].float().unsqueeze(0)
        o = split_outputs(head(feats))
        rows.append((math.atan2(o["dir_x"].item(), o["dir_y"].item()),
                     math.atan2(s.dx, s.dy), s.scan))
        if (i+1) % 500 == 0: print(f"  {i+1}/{len(sel)}", flush=True)

P = np.array([r_[0] for r_ in rows]); T = np.array([r_[1] for r_ in rows])
np.save("logs/bearings.npy", np.stack([P, T]))

print("\nSHRINKAGE  (slope<1 => predictions pulled toward straight-ahead)")
slope = float(np.linalg.lstsq(T.reshape(-1,1), P, rcond=None)[0][0])
print(f"  pred = {slope:.3f} x true      (1.0 = calibrated)")
turn = np.abs(T) > math.radians(7.5)
print(f"  mean |theta| on expert turns:  true {np.degrees(np.abs(T[turn])).mean():5.1f}d"
      f"   pred {np.degrees(np.abs(P[turn])).mean():5.1f}d")
print(f"  mean |theta| on expert straight: true {np.degrees(np.abs(T[~turn])).mean():5.1f}d"
      f"   pred {np.degrees(np.abs(P[~turn])).mean():5.1f}d")

def act(th, thr): return 2 if th > thr else 1 if th < -thr else 0
print("\nDECISION-THRESHOLD SWEEP (applied to predictions only; no retraining)")
print(f"  {'thr':>6} {'overall':>8} {'FWD':>7} {'LEFT':>7} {'RIGHT':>7}")
gt = np.array([act(t, math.radians(7.5)) for t in T])
for d in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0):
    pr = np.array([act(p, math.radians(d)) for p in P])
    ok = (pr == gt)
    cells = [f"{ok[gt==c].mean():>7.3f}" if (gt==c).sum() else "      -" for c in (0,1,2)]
    print(f"  {d:>5.1f}d {ok.mean():>8.3f} " + " ".join(cells))
