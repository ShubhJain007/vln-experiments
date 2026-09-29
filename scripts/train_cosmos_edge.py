#!/usr/bin/env python3
"""Post-train Cosmos3-Edge as an R2R navigation policy.

Flow-matching on the action stream only. The scheduler config says
prediction_type="flow_prediction" with predict_x0=True, so x0 = x_t - sigma*v
with x_t = (1-sigma)*x0 + sigma*eps -- i.e. the target velocity is eps - x0.

Trains: LoRA on the transformer's attention projections, plus the `av` rows of
action_proj_in/out (8.4M params total across 32 domains; ours is domain 1).
Everything else frozen. The video stream is left noisy and its prediction
ignored -- we want the policy, not the world model.

Sequence assembly is captured from the pipeline itself (see train/cosmos_pack)
rather than reimplemented.
"""
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
import argparse, json, math, pathlib, time

import numpy as np
import torch
from PIL import Image

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

CK = "checkpoints/Cosmos3-Edge"
RAW_DIM = 9


def build_batch(pipe, ds, idx, device, dtype, generator=None):
    """One flow-matching training example: captured pack + noised action tokens."""
    from train.cosmos_pack import capture_pack

    s = ds[idx]
    kw = capture_pack(pipe, prompt=s["instruction"], image=Image.fromarray(s["image"]),
                      chunk_size=ds.chunk, fps=s["fps"])

    x0 = torch.zeros_like(kw["action_tokens"][0])                 # [T, 64]
    x0[:, :RAW_DIM] = torch.as_tensor(s["actions"], dtype=x0.dtype, device=x0.device)
    eps = torch.randn(x0.shape, generator=generator, dtype=torch.float32).to(device=x0.device, dtype=x0.dtype)
    eps[:, RAW_DIM:] = 0                                          # padding stays exactly zero

    sigma = float(torch.rand(1, generator=generator).item())
    xt = (1.0 - sigma) * x0.float() + sigma * eps.float()
    xt[:, RAW_DIM:] = 0
    target = (eps.float() - x0.float())[:, :RAW_DIM]

    kw = dict(kw)
    kw["action_tokens"] = [xt.to(dtype)]
    kw["action_timesteps"] = torch.full_like(kw["action_timesteps"],
                                             int(sigma * 1000), dtype=kw["action_timesteps"].dtype)
    return kw, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="one fwd/bwd, report memory")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--out", default="checkpoints/edge_r2r")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args()

    from diffusers import Cosmos3OmniPipeline
    from peft import LoraConfig, get_peft_model
    from data.cosmos_dataset import CosmosChunkDataset

    device, dtype = "cuda", torch.bfloat16
    ds = CosmosChunkDataset(split="train", chunk=args.chunk)
    print(f"{len(ds):,} training samples", flush=True)

    pipe = Cosmos3OmniPipeline.from_pretrained(CK, dtype=dtype).to(device)
    tr = pipe.transformer
    tr.requires_grad_(False)

    get_peft_model(tr, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"]))

    # the action projections are what map 9D <-> hidden for our domain
    trainable = []
    for n, p in tr.named_parameters():
        if "lora_" in n or "action_proj" in n or "action_modality_embed" in n:
            p.requires_grad_(True)
            trainable.append((n, p))
    # Params stay bf16 on purpose. ModelMixin.dtype reports the dtype of the
    # first parameter, and the pipeline uses transformer.dtype to build the
    # vision latents -- casting these to fp32 silently makes the whole capture
    # fp32 and it explodes against bf16 proj_in weights.
    n_tr = sum(p.numel() for _, p in trainable)
    print(f"trainable {n_tr/1e6:.2f} M params in {len(trainable)} tensors", flush=True)

    opt = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(0)

    if args.smoke:
        torch.cuda.reset_peak_memory_stats()
        kw, target = build_batch(pipe, ds, 0, device, dtype, gen)
        t0 = time.time()
        _, _, pred_action = tr(**kw)
        p = pred_action[0] if isinstance(pred_action, (list, tuple)) else pred_action
        loss = torch.nn.functional.mse_loss(p[:, :RAW_DIM].float(), target.to(p.device))
        loss.backward()
        torch.cuda.synchronize()
        print(f"\nSMOKE OK")
        print(f"  pred_action {tuple(p.shape)} {p.dtype}   loss {loss.item():.4f}")
        print(f"  fwd+bwd {time.time()-t0:.2f}s")
        print(f"  peak GPU {torch.cuda.max_memory_allocated()/1024**3:.2f} GB "
              f"of {torch.cuda.mem_get_info()[1]/1024**3:.2f} GB")
        g = [p_.grad for _, p_ in trainable if p_.grad is not None]
        print(f"  params with grad: {len(g)}/{len(trainable)}   "
              f"grad norm {torch.norm(torch.stack([x.norm() for x in g])).item():.4f}")
        return 0

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    order = np.random.RandomState(0).permutation(len(ds))
    run, t0 = [], time.time()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        for a in range(args.accum):
            idx = int(order[(step * args.accum + a) % len(order)])
            kw, target = build_batch(pipe, ds, idx, device, dtype, gen)
            _, _, pred_action = tr(**kw)
            p = pred_action[0] if isinstance(pred_action, (list, tuple)) else pred_action
            loss = torch.nn.functional.mse_loss(p[:, :RAW_DIM].float(), target.to(p.device))
            (loss / args.accum).backward()
            run.append(loss.item())
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        opt.step()
        if (step + 1) % args.log_every == 0:
            print(f"step {step+1:>6}  loss {np.mean(run[-args.log_every*args.accum:]):.4f}  "
                  f"{(time.time()-t0)/(step+1):.2f}s/step", flush=True)
        if (step + 1) % args.save_every == 0:
            torch.save({n: p.detach().cpu() for n, p in trainable}, out / f"step{step+1}.pt")
            print(f"  saved {out}/step{step+1}.pt", flush=True)
    torch.save({n: p.detach().cpu() for n, p in trainable}, out / "final.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
