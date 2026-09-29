#!/usr/bin/env python3
"""Option B: SFT the Cosmos3-Edge REASONER as a direct-answer R2R policy.

Zero-shot, the reasoner picks the right action 0.375 of the time against a
0.573 constant baseline, and finds "stop" 0.08 of the time. This teaches the
one join that is broken -- (video, instruction) -> command -- and leaves the
parts that already work (its perception, the diffusion action head) alone.

  input    the SAME prompt the closed loop uses (EMBODIMENT + HIER_OVERALL +
           ANSWER_FORMAT); 120 source frames (8 s of the robot's own view)
           sampled by the processor at 1 FPS -> 8 frames; thinking OFF via the
           chat template (assistant turn opens "<think></think>")
  target   the command string, e.g. "turn left", then <|im_end|>
  loss     causal-LM cross-entropy on the target tokens only
  trained  LoRA on the language model's q/k/v/o AND the vision->language
           projector; SigLIP2 vision tower frozen -- the same split as NVIDIA's
           own Edge reasoner SFT recipe (which does full FT of projector + LM at
           lr 1e-6 on 8 GPUs; we use LoRA on one 16 GB card)
  data     decision points sampled UNIFORMLY along expert trajectories, so the
           label prior is the one a rollout meets (~66% forward). Over-sampling
           turn onsets taught a turn-heavy prior that scored 0.656 offline and
           SR 0.000 in closed loop.

Batch size 1 with gradient accumulation: video token counts vary per sample
and padding multimodal batches is where silent bugs live.

Direct-answer (no <think>) on purpose: after SFT a direct policy is the normal
thing, and it runs ~60x faster than thinking, which also fixes the 300 s per
episode problem in closed loop.
"""
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
import argparse, collections, json, pathlib, time

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
CK = "checkpoints/Cosmos3-Edge"


def encode(processor, tokenizer, clip, instruction, label, device, fps=15.0):
    from nav.cosmos_prompts import SYSTEM, reasoner_prompt, video_kwargs
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "video", "video": clip},
                                         {"type": "text", "text": reasoner_prompt(instruction, think=False)}]}]
    enc = processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        enable_thinking=False, **video_kwargs(len(clip)))
    ans = tokenizer.encode(label, add_special_tokens=False) + [tokenizer.convert_tokens_to_ids("<|im_end|>")]
    ans = torch.tensor([ans], dtype=enc["input_ids"].dtype)
    n_prompt = enc["input_ids"].shape[1]
    enc["input_ids"] = torch.cat([enc["input_ids"], ans], 1)
    enc["attention_mask"] = torch.cat([enc["attention_mask"], torch.ones_like(ans)], 1)
    if "mm_token_type_ids" in enc:            # 0 = text token
        enc["mm_token_type_ids"] = torch.cat([enc["mm_token_type_ids"], torch.zeros_like(ans)], 1)
    labels = enc["input_ids"].clone()
    labels[:, :n_prompt] = -100
    enc["labels"] = labels
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = whole index")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--out", default="checkpoints/reasoner_sft")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Cosmos3EdgeForConditionalGeneration
    from data.ego_pose_9d import rot_from_6d
    from nav.reasoner_sft_data import build_index, ReasonerSFTDataset, COMMANDS

    device = "cuda"
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    idx_file = out / "index_train.json"
    if idx_file.exists():
        items = json.loads(idx_file.read_text())
    else:
        items = build_index(_ROOT / "data/video", "train", rot_from_6d)
        idx_file.write_text(json.dumps(items))
    if args.max_samples:
        items = items[:args.max_samples]
    dist = collections.Counter(it["label"] for it in items)
    print(f"{len(items):,} samples  " + "  ".join(f"{k} {v/len(items)*100:.0f}%" for k, v in dist.items()), flush=True)

    processor = AutoProcessor.from_pretrained(CK)
    tok = processor.tokenizer
    model = Cosmos3EdgeForConditionalGeneration.from_pretrained(CK, dtype=torch.bfloat16, device_map=device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
        # language q/k/v/o + the vision->language PROJECTOR. NVIDIA's own Edge
        # reasoner SFT trains "projector + LM" and freezes the SigLIP2 tower;
        # we mirror that split (LoRA instead of full FT: one 16 GB card).
        target_modules=r".*(language_model.*\.(q_proj|k_proj|v_proj|o_proj)|projector\.linear_fc[12])$"))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)

    ds = ReasonerSFTDataset(items)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=True, num_workers=args.workers,
                                         persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)

    if args.smoke:
        torch.cuda.reset_peak_memory_stats()
        s = ds[0]
        enc = encode(processor, tok, s["clip"], s["instruction"], s["label"], device)
        n_tgt = int((enc["labels"] != -100).sum())
        print(f"clip {tuple(s['clip'].shape)} -> video_grid_thw {enc['video_grid_thw'].tolist()}  "
              f"prompt+target tokens {enc['input_ids'].shape[1]}  target tokens {n_tgt}  "
              f"target text {tok.decode(enc['input_ids'][0, -n_tgt:])!r}")
        t0 = time.time(); model.train()
        loss = model(**enc).loss; loss.backward(); torch.cuda.synchronize()
        print(f"SMOKE OK  loss {loss.item():.4f}  fwd+bwd {time.time()-t0:.2f}s  "
              f"peak {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        model.eval()
        with torch.no_grad():
            p = {k: v for k, v in enc.items() if k != "labels"}
            p["input_ids"] = p["input_ids"][:, :-n_tgt]; p["attention_mask"] = p["attention_mask"][:, :-n_tgt]
            if "mm_token_type_ids" in p: p["mm_token_type_ids"] = p["mm_token_type_ids"][:, :-n_tgt]
            g = model.generate(**p, max_new_tokens=8, do_sample=False)
            print("untrained generation:", repr(tok.decode(g[0, p["input_ids"].shape[1]:], skip_special_tokens=True)))
        return 0

    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    total = int(len(ds) * args.epochs)
    model.train(); run = collections.deque(maxlen=args.log_every * args.accum); t0 = time.time()
    step = seen = 0; opt.zero_grad(set_to_none=True)
    while seen < total:
        for s in loader:
            enc = encode(processor, tok, s["clip"], s["instruction"], s["label"], device)
            loss = model(**enc).loss
            (loss / args.accum).backward(); run.append(loss.item()); seen += 1
            if seen % args.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % args.log_every == 0:
                    el = time.time() - t0
                    print(f"step {step:>5}  samples {seen:>6}/{total}  loss {np.mean(run):.4f}  "
                          f"{el/seen:.2f}s/sample  eta {(total-seen)*el/seen/3600:.1f}h", flush=True)
                if step % args.save_every == 0:
                    model.save_pretrained(out / f"step{step}"); print(f"  saved {out}/step{step}", flush=True)
            if seen >= total:
                break
    model.save_pretrained(out / "final"); print(f"saved {out}/final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
