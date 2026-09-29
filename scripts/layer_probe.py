#!/usr/bin/env python3
"""
Which layer carries which signal, and does fusing layers help?

Extracts the ACTION-position hidden state from EVERY decoder layer, then fits
a small probe on each layer independently for each output:

    u, v        pointing            (regression, visible steps only)
    dir_x/dir_y direction           (regression -> action agreement)
    is_stop     stop                (classification -> separation + AUC)

Then tries FUSIONS: concatenations of the best layers, and all-layer pooling.

WHY THIS EXISTS. The deployed head reads ONE hidden state -- the FINAL layer at
one position -- and linearly projects it. The final layer of an LLM is shaped
by next-token prediction, so a signal like "am I at the described destination"
may be present mid-stack and washed out by the end. GR00T/pi-0 style heads
cross-attend over MULTIPLE layers for exactly this reason.

Probes are trained on TRAIN-split frames and scored on held-out val_unseen, so
a layer cannot win by memorising.
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import argparse, json, math, pathlib
import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def collect(enc, b, bb, ds, n, K, desc):
    """Return per-layer action-position hidden states + targets."""
    from data.pointing_dataset import load_frame
    feats, tg = None, {"u": [], "v": [], "dx": [], "dy": [],
                       "stop": [], "vis": []}
    steps = ds.steps[:n]
    with torch.no_grad():
        for i, s in enumerate(steps):
            v_t = enc.encode(load_frame(s.frame_path))
            hist = [enc.encode(load_frame(h)) for h in s.history_paths] if K else None
            seq = b.build(s.instruction, v_t, history_visual=hist)
            out = enc.model.model.language_model(
                inputs_embeds=seq.inputs_embeds,
                attention_mask=seq.attention_mask,
                position_ids=bb.build_position_ids(seq),
                use_cache=False, output_hidden_states=True)
            hs = out.hidden_states                      # tuple(L+1) of (1,N,d)
            row = np.stack([h[0, seq.action_index].float().cpu().numpy() for h in hs])
            if feats is None:
                feats = np.zeros((len(steps), row.shape[0], row.shape[1]), dtype=np.float32)
            feats[i] = row
            tg["u"].append(s.u); tg["v"].append(s.v)
            tg["dx"].append(s.dx); tg["dy"].append(s.dy)
            tg["stop"].append(s.is_stop); tg["vis"].append(s.visible)
            if (i + 1) % 200 == 0:
                print(f"  {desc}: {i+1}/{len(steps)}", flush=True)
    return feats, {k: np.array(v) for k, v in tg.items()}


def ridge(X, y, lam=1.0):
    """Ridge in whichever form is cheaper.

    Fusing 4 layers gives d = 8192 features against n = 2000 samples.
    The primal form solves a d x d system -- 8193^3 ~ 5.5e11 ops, ~30 s
    per fit, which made a 570-combination search take hours. The dual
    form solves an n x n system instead and is exactly equivalent when
    n < d. Measured: 0.15 s vs ~30 s, a 200x speedup.
    """
    X1 = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    n, d = X1.shape
    if n < d:                      # dual: w = X^T (X X^T + lam I)^-1 y
        G = X1 @ X1.T + lam * np.eye(n, dtype=np.float32)
        return X1.T @ np.linalg.solve(G, y)
    A = X1.T @ X1 + lam * np.eye(d, dtype=np.float32)
    return np.linalg.solve(A, X1.T @ y)


def apply_w(X, w):
    return np.concatenate([X, np.ones((len(X), 1), np.float32)], 1) @ w


def auc(scores, labels):
    o = np.argsort(scores); l = labels[o]
    pos, neg = l.sum(), (1 - l).sum()
    if pos == 0 or neg == 0: return float("nan")
    return (np.cumsum(1 - l)[l == 1].sum()) / (pos * neg)


def eval_block(Xtr, Xte, ttr, tte):
    """Fit probes on one feature block; return metrics on held-out."""
    r = {}
    m = ttr["vis"] > 0.5; mt = tte["vis"] > 0.5
    if m.sum() > 20 and mt.sum() > 20:
        pu = apply_w(Xte[mt], ridge(Xtr[m], ttr["u"][m]))
        r["u_corr"] = float(np.corrcoef(pu, tte["u"][mt])[0, 1])
    else:
        r["u_corr"] = float("nan")
    # direction -> action agreement
    wdx, wdy = ridge(Xtr, ttr["dx"]), ridge(Xtr, ttr["dy"])
    pdx, pdy = apply_w(Xte, wdx), apply_w(Xte, wdy)
    th_p = np.arctan2(pdx, pdy); th_t = np.arctan2(tte["dx"], tte["dy"])
    TH = math.radians(7.5)
    ap = np.where(th_p > TH, 2, np.where(th_p < -TH, 1, 0))
    at = np.where(th_t > TH, 2, np.where(th_t < -TH, 1, 0))
    r["act"] = float((ap == at).mean())
    # stop
    ps = apply_w(Xte, ridge(Xtr, ttr["stop"]))
    r["stop_auc"] = float(auc(ps, tte["stop"]))
    hi, lo = ps[tte["stop"] > 0.5], ps[tte["stop"] < 0.5]
    r["stop_sep"] = float(hi.mean() - lo.mean()) if len(hi) else float("nan")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-steps", type=int, default=2000)
    ap.add_argument("--test-steps", type=int, default=1000)
    ap.add_argument("--refresh", action="store_true", help="ignore cached features")
    ap.add_argument("--search", action="store_true",
                    help="search fusion combinations for the best ALL-ROUND set")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from data.pointing_dataset import PointingDataset
    from model.backbone import Backbone
    from model.input_sequence import InputSequenceBuilder
    from model.vision_encoder import VisionEncoder

    ck = pathlib.Path(args.checkpoint)
    cfg = json.loads((ck/"config.json").read_text()) if (ck/"config.json").exists() else {}
    K, S = cfg.get("history_frames", 0), cfg.get("history_stride", 1)

    enc = VisionEncoder()
    sd = torch.load(ck/"adapter.pt", map_location="cpu", weights_only=True)
    r = sd[[k for k in sd if k.endswith("lora_A.default.weight")][0]].shape[0]
    get_peft_model(enc.model, LoraConfig(r=r, lora_alpha=2*r, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj"]))
    enc.model.load_state_dict(sd, strict=False); enc.model.eval()
    b = InputSequenceBuilder(enc.model, enc.processor.tokenizer)
    bb = Backbone(enc.model)
    print(f"{args.checkpoint}  K={K} stride={S}\n")

    cache = _ROOT / "logs" / f"layerfeat_{ck.name}_{args.train_steps}_{args.test_steps}.npz"
    if cache.exists() and not args.refresh:
        print(f"loading cached features from {cache.name}")
        z = np.load(cache, allow_pickle=True)
        Xtr, Xte = z["Xtr"], z["Xte"]
        ttr = {k: z[f"ttr_{k}"] for k in ("u","v","dx","dy","stop","vis")}
        tte = {k: z[f"tte_{k}"] for k in ("u","v","dx","dy","stop","vis")}
    else:
        dtr = PointingDataset("train", max_episodes=140, history_frames=K, history_stride=S)
        dte = PointingDataset("val_unseen", max_episodes=90, history_frames=K, history_stride=S)
        Xtr, ttr = collect(enc, b, bb, dtr, args.train_steps, K, "train")
        Xte, tte = collect(enc, b, bb, dte, args.test_steps, K, "test")
        np.savez_compressed(cache, Xtr=Xtr, Xte=Xte,
                            **{f"ttr_{k}": v for k, v in ttr.items()},
                            **{f"tte_{k}": v for k, v in tte.items()})
        print(f"cached features -> {cache.name}")
    L = Xtr.shape[1]
    print(f"\nlayers (incl. embeddings): {L}   train {len(Xtr)}  test {len(Xte)}")
    print(f"stop base rate: train {ttr['stop'].mean():.4f}  test {tte['stop'].mean():.4f}\n")

    # z-score per layer using TRAIN statistics only
    def norm(X, mu, sd_): return (X - mu) / sd_
    rows = []
    print(f"{'layer':>6} {'u corr':>8} {'act':>7} {'stop AUC':>9} {'stop sep':>9}")
    print("-" * 44)
    for l in range(L):
        mu, sd_ = Xtr[:, l].mean(0), Xtr[:, l].std(0) + 1e-6
        m = eval_block(norm(Xtr[:, l], mu, sd_), norm(Xte[:, l], mu, sd_), ttr, tte)
        rows.append((l, m))
        print(f"{l:>6} {m['u_corr']:>8.3f} {m['act']:>7.3f} "
              f"{m['stop_auc']:>9.3f} {m['stop_sep']:>9.4f}")

    best_u = max(rows, key=lambda x: (x[1]["u_corr"] if x[1]["u_corr"]==x[1]["u_corr"] else -9))
    best_a = max(rows, key=lambda x: x[1]["act"])
    best_s = max(rows, key=lambda x: (x[1]["stop_auc"] if x[1]["stop_auc"]==x[1]["stop_auc"] else -9))
    print(f"\nBEST SINGLE LAYER   u: {best_u[0]} ({best_u[1]['u_corr']:.3f})   "
          f"act: {best_a[0]} ({best_a[1]['act']:.3f})   "
          f"stop: {best_s[0]} (AUC {best_s[1]['stop_auc']:.3f})")
    print(f"FINAL LAYER ({L-1})     u: {rows[-1][1]['u_corr']:.3f}   "
          f"act: {rows[-1][1]['act']:.3f}   stop AUC: {rows[-1][1]['stop_auc']:.3f}")

    # ---- FUSIONS -----------------------------------------------------------
    print("\nFUSION (concatenated layers):")
    def build(idxs):
        mu = Xtr[:, idxs].reshape(len(Xtr), -1).mean(0)
        sd_ = Xtr[:, idxs].reshape(len(Xtr), -1).std(0) + 1e-6
        return ((Xtr[:, idxs].reshape(len(Xtr), -1) - mu) / sd_,
                (Xte[:, idxs].reshape(len(Xte), -1) - mu) / sd_)
    combos = {
        "final only":            [L-1],
        "best-per-task":         sorted({best_u[0], best_a[0], best_s[0]}),
        "quarter/half/3q/final": [L//4, L//2, 3*L//4, L-1],
        "last 4":                [L-4, L-3, L-2, L-1],
        "every 4th":             list(range(0, L, 4)) + [L-1],
    }
    print(f"{'combo':>24} {'layers':>22} {'u corr':>8} {'act':>7} {'stop AUC':>9}")
    print("-" * 74)
    for name, idxs in combos.items():
        A, B = build(idxs)
        m = eval_block(A, B, ttr, tte)
        ss = ",".join(map(str, idxs))
        print(f"{name:>24} {ss[:22]:>22} {m['u_corr']:>8.3f} {m['act']:>7.3f} {m['stop_auc']:>9.3f}")

    if args.search:
        # ONE fused block predicting EVERY output. Ranked by a combined score
        # so no single task dominates the choice: each metric is scaled to its
        # own observed range across candidates before averaging.
        import itertools
        early = [0, 5, 10]
        mid = [14, 15, 16, 21, 23, 24]
        late = [25, 26, 27, 28]
        cands = []
        for ne in (0, 1):
            for nm in (1, 2):
                for nl in (1, 2):
                    for e in itertools.combinations(early, ne):
                        for m_ in itertools.combinations(mid, nm):
                            for l in itertools.combinations(late, nl):
                                idxs = sorted(set(e + m_ + l))
                                if 2 <= len(idxs) <= 4:
                                    cands.append(idxs)
        seen, uniq = set(), []
        for c in cands:
            t = tuple(c)
            if t not in seen:
                seen.add(t); uniq.append(c)
        print(f"\nSEARCHING {len(uniq)} fusion combinations (one block -> all outputs) ...")
        res = []
        for idxs in uniq:
            A, B = build(idxs)
            m = eval_block(A, B, ttr, tte)
            if m["u_corr"] != m["u_corr"]:
                continue
            res.append((idxs, m))
        us = np.array([r[1]["u_corr"] for r in res])
        ac = np.array([r[1]["act"] for r in res])
        st = np.array([r[1]["stop_auc"] for r in res])
        def sc(a): return (a - a.min()) / max(a.max() - a.min(), 1e-9)
        comb = (sc(us) + sc(ac) + sc(st)) / 3.0
        order = np.argsort(-comb)
        print(f"\nTOP 12 ALL-ROUND FUSIONS")
        print(f"{'layers':>20} {'u corr':>8} {'act':>7} {'stop AUC':>9} {'score':>7}")
        print("-" * 56)
        for i in order[:12]:
            idxs, m = res[i]
            print(f"{','.join(map(str,idxs)):>20} {m['u_corr']:>8.3f} {m['act']:>7.3f} "
                  f"{m['stop_auc']:>9.3f} {comb[i]:>7.3f}")
        print(f"\nbaseline final-only(28):"
              f"  u {rows[-1][1]['u_corr']:.3f}  act {rows[-1][1]['act']:.3f} "
              f" stop {rows[-1][1]['stop_auc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
