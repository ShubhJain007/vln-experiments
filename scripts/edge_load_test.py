#!/usr/bin/env python3
"""Go/no-go: does Cosmos3-Edge load and fit on our 16 GB card, and what does
its action path actually look like?

Their SFT recipes assume 16-256 GPUs, but that is full SFT of Nano (16B) from
base. Edge is 4B / 9.18 GB of weights. This measures the real footprint and
locates the action-conditioning modules we would LoRA.
"""
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
import torch, json, pathlib

CK = "checkpoints/Cosmos3-Edge"

def gb(x): return x / 1024**3

from diffusers import Cosmos3OmniTransformer, AutoencoderKLWan

print("loading transformer (bf16)...", flush=True)
tr = Cosmos3OmniTransformer.from_pretrained(CK, subfolder="transformer", torch_dtype=torch.bfloat16)
n = sum(p.numel() for p in tr.parameters())
print(f"  params {n/1e9:.3f} B   cpu size {gb(n*2):.2f} GB")

tr = tr.to("cuda")
torch.cuda.synchronize()
print(f"  on GPU: allocated {gb(torch.cuda.memory_allocated()):.2f} GB  "
      f"reserved {gb(torch.cuda.memory_reserved()):.2f} GB")

cfg = tr.config
print(f"\nconfig: hidden {cfg.hidden_size}  layers {cfg.num_hidden_layers}  "
      f"action_dim {cfg.action_dim}  action_gen {cfg.action_gen}")
print(f"        num_embodiment_domains {cfg.num_embodiment_domains}  base_fps {cfg.base_fps}  "
      f"fps_modulation {cfg.enable_fps_modulation}")

print("\n--- modules whose name mentions action / embodiment / domain ---")
for name, m in tr.named_modules():
    if any(k in name.lower() for k in ("action", "embodiment", "domain")):
        p = sum(q.numel() for q in m.parameters(recurse=False))
        if p or not list(m.children()):
            print(f"  {name:<58} {type(m).__name__:<22} {p:>10,}")

print("\n--- action-related parameter shapes ---")
tot = 0
for name, p in tr.named_parameters():
    if any(k in name.lower() for k in ("action", "embodiment", "domain")):
        print(f"  {name:<62} {str(tuple(p.shape)):>20}")
        tot += p.numel()
print(f"  total action-path params: {tot:,}")

print("\nloading VAE...", flush=True)
vae = AutoencoderKLWan.from_pretrained(CK, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")
torch.cuda.synchronize()
print(f"  with VAE: allocated {gb(torch.cuda.memory_allocated()):.2f} GB  "
      f"reserved {gb(torch.cuda.memory_reserved()):.2f} GB")
print(f"  free on device: {gb(torch.cuda.mem_get_info()[0]):.2f} GB of "
      f"{gb(torch.cuda.mem_get_info()[1]):.2f} GB")
