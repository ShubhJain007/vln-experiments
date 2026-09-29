#!/usr/bin/env python3
"""
Where does the action-query actually look?

Reports attention mass from the ACTION position over each segment of the
input sequence, per layer and pooled.

IMPORTANT: raw attention MASS is misleading here. The sequence is ~588 visual
tokens against ~25 instruction tokens, so vision wins on sheer count even if
every individual text token is attended far more strongly. Both totals and
PER-TOKEN means are reported; the per-token number is what says which modality
the model weights more heavily.

Attention is also not the same thing as functional dependence -- the
instruction-swap ablation is the causal test. This is the descriptive one.
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import argparse, json, pathlib
import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--samples", type=int, default=24)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from data.pointing_dataset import PointingDataset, load_frame
    from model.input_sequence import InputSequenceBuilder
    from model.backbone import Backbone
    from model.vision_encoder import VisionEncoder

    ck = pathlib.Path(args.checkpoint)
    cfg = json.loads((ck/"config.json").read_text()) if (ck/"config.json").exists() else {}
    K, S = cfg.get("history_frames", 0), cfg.get("history_stride", 1)

    # SDPA does not return attention weights; eager does. Loaded explicitly
    # here rather than changing VisionEncoder, so training/eval keep the faster
    # kernel and only this diagnostic pays the cost.
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    mp = str(_ROOT.parent / "models" / "cosmos-reason2-2b")
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        mp, dtype=torch.bfloat16, device_map="cuda",
        tie_word_embeddings=False, attn_implementation="eager")
    enc = VisionEncoder(model=base, processor=AutoProcessor.from_pretrained(mp))
    sd = torch.load(ck/"adapter.pt", map_location="cpu", weights_only=True)
    r = sd[[k for k in sd if k.endswith("lora_A.default.weight")][0]].shape[0]
    get_peft_model(enc.model, LoraConfig(r=r, lora_alpha=2*r, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj"]))
    enc.model.load_state_dict(sd, strict=False)
    enc.model.eval()
    b = InputSequenceBuilder(enc.model, enc.processor.tokenizer)
    bb = Backbone(enc.model)

    ds = PointingDataset(args.split, max_episodes=20, history_frames=K, history_stride=S)
    steps = ds.steps[:args.samples]
    print(f"{args.checkpoint}   history K={K} stride={S}   samples={len(steps)}\n")

    per_layer_v, per_layer_t = [], []
    tot = {"visual": [], "instr": [], "other": []}
    ptok = {"visual": [], "instr": []}

    with torch.no_grad():
        for s in steps:
            v_t = enc.encode(load_frame(s.frame_path))
            hist = [enc.encode(load_frame(h)) for h in s.history_paths] if K else None
            seq = b.build(s.instruction, v_t, history_visual=hist)
            out = enc.model.model.language_model(
                inputs_embeds=seq.inputs_embeds,
                attention_mask=seq.attention_mask,
                position_ids=bb.build_position_ids(seq),
                use_cache=False, output_attentions=True)
            att = out.attentions                      # tuple(L) of (1,H,N,N)
            q = seq.action_index
            vs, ins = seq.visual_slice, seq.instr_slice
            nv = vs.stop - vs.start
            ni = ins.stop - ins.start
            lv, lt = [], []
            for a in att:
                row = a[0, :, q, :].mean(0)           # mean over heads
                row = row / row.sum().clamp(min=1e-9)
                v_mass = row[vs].sum().item()
                i_mass = row[ins].sum().item()
                lv.append(v_mass); lt.append(i_mass)
            per_layer_v.append(lv); per_layer_t.append(lt)
            vm, im = float(np.mean(lv)), float(np.mean(lt))
            tot["visual"].append(vm); tot["instr"].append(im)
            tot["other"].append(1.0 - vm - im)
            ptok["visual"].append(vm/nv); ptok["instr"].append(im/ni)

    V = np.array(per_layer_v); T = np.array(per_layer_t)
    print(f"sequence: {nv} visual tokens, {ni} instruction tokens\n")
    print("TOTAL attention mass from the ACTION position (mean over heads+layers):")
    for k in ("visual","instr","other"):
        print(f"   {k:8s} {np.mean(tot[k]):.4f}")
    print("\nPER-TOKEN attention (mass / #tokens) -- corrects for there being "
          f"{nv//max(ni,1)}x more visual tokens:")
    print(f"   visual  {np.mean(ptok['visual']):.6f}")
    print(f"   instr   {np.mean(ptok['instr']):.6f}")
    ratio = np.mean(ptok['instr'])/max(np.mean(ptok['visual']),1e-12)
    print(f"   => each instruction token gets {ratio:.1f}x the attention of a visual token")

    print("\nPER-LAYER (visual / instruction mass):")
    L = V.shape[1]
    for i in range(0, L, max(1, L//10)):
        print(f"   layer {i:2d}:  visual {V[:,i].mean():.3f}   instr {T[:,i].mean():.3f}")
    print(f"   layer {L-1:2d}:  visual {V[:,L-1].mean():.3f}   instr {T[:,L-1].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
