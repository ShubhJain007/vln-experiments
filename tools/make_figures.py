#!/usr/bin/env python3
"""Paper figures from logs already in the repository (no GPU, no simulator). Writes docs/figures/*.png.

    python tools/make_figures.py

Figures
  fig_loss_balance.png   Stage 1: L_act vs lambda * L_pil over training and their ratio (DECISIONS.md D25)
  fig_stage0prime.png    Stage 0' Design A vs Design B training accuracy (D17 / D19)
  fig_probe_confusion.png  Reasoner SFT next-action probe: confusion matrix (results/probe_sft_final.json)
  fig_navigation.png     OS and strict SR of every evaluated policy vs the paper (docs/figures/navigation_rows.json)
  fig_stop_threshold.png Pointing fusion: OS vs strict SR by STOP threshold, and strict SR over training
  fig_bearing.png        Pointing: predicted vs expert bearing (logs/bearings.npy)
  fig_layer_probe.png    STOP linear-probe AUC per backbone layer (logs/layerprobe_154820.log)
  fig_edge_guidance.png  Cosmos3-Edge action model, zero-shot: guidance vs turn-sign / instruction following
  fig_reasoner_zeroshot.png  Cosmos3-Edge reasoner, zero-shot: predicted waypoints on the probe frames
  fig_qualitative.png    Frames from the demo videos in media/ (LatentPilot, pointing, reasoner)
  fig_lp_failure.png     Why trained LatentPilot fails: training fit, future prediction and navigation, learned vs identity
  fig_pointing_failure.png  How pointing episodes end, and the teacher-forced action confusion
  fig_reasoner_failure.png  How reasoner episodes end, and STOP rate vs training-label mix
  fig_slot_shortcut.png  Held-out action accuracy by Pilot-slot content (results/slot_shortcut*.json)
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
# categorical slots 1-3 of the reference palette (validated all-pairs, light mode); ink and grid neutrals
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LAMBDA = 0.1

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "lines.linewidth": 2,
})


def trainlog(run):
    return json.load(open(ROOT / "checkpoints" / run / "trainlog.json"))


def fig_loss_balance():
    log = trainlog("stage1_learned")
    s = np.array([r["step"] for r in log])
    act = np.array([r["l_act"] for r in log])
    pil = LAMBDA * np.array([r["l_pil"] for r in log])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    a.plot(s, act, color=BLUE)
    a.plot(s, pil, color=ORANGE)
    a.set_yscale("log")
    a.set_xlabel("training step")
    a.set_ylabel("loss (log scale)")
    a.set_title("Stage 1: the two losses converge at different rates")
    a.text(9000, 0.55, "λ · L_pil", color=INK)
    a.text(9000, 0.022, "L_act", color=INK)
    ratio = pil / act
    k = 15
    sm = np.convolve(ratio, np.ones(k) / k, mode="valid")
    b.plot(s, ratio, color=GRID, linewidth=1)
    b.plot(s[k // 2: k // 2 + len(sm)], sm, color=BLUE)
    b.axhline(1.0, color=INK2, linewidth=1, linestyle="--")
    b.set_xlabel("training step")
    b.set_ylabel("λ · L_pil  /  L_act")
    b.set_title("Effective weight of the pilot loss")
    b.text(600, 21, "at step %d: %.1f× (DECISIONS D25)\nmean of the last %d logs: %.1f×" % (s[-1], ratio[-1], k, sm[-1]),
           color=INK)
    b.text(300, 1.6, "balanced (1×)", color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_loss_balance.png", dpi=160)


def smooth(x, k=9):
    return np.convolve(x, np.ones(k) / k, mode="valid")


def fig_stage0prime():
    fig, a = plt.subplots(figsize=(6.4, 3.6))
    for run, color, name in (("stage0prime_A", BLUE, "Design A (pilot = last input)"),
                             ("stage0prime_B", ORANGE, "Design B (action-query slot)")):
        log = trainlog(run)
        s = np.array([r["step"] for r in log])
        acc = smooth(np.array([r["acc"] for r in log]))
        a.plot(s[len(s) - len(acc):], acc, color=color, label=name)
    a.set_xlabel("training step")
    a.set_ylabel("training action accuracy (smoothed)")
    a.set_title("Stage 0′: training accuracy of Design A vs Design B")
    a.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_stage0prime.png", dpi=160)


def fig_probe_confusion():
    d = json.load(open(ROOT / "results" / "probe_sft_final.json"))
    v = d["variants"]["hier_overall_nothink"]
    labels = ["move forward", "turn left", "turn right", "stop"]
    conf = v["confusion"]
    m = np.zeros((4, 4))
    for pair, n in conf.items():  # keys are "expert->predicted"
        exp_, pred = pair.split("->")
        if exp_ in labels and pred in labels:
            m[labels.index(exp_), labels.index(pred)] = n
    row = m / np.maximum(m.sum(1, keepdims=True), 1)
    fig, a = plt.subplots(figsize=(6.2, 4.8))
    a.imshow(row, cmap="Blues", vmin=0, vmax=1)
    a.grid(False)
    a.set_xticks(range(4), labels, rotation=20)
    a.set_yticks(range(4), labels)
    a.set_xlabel("predicted")
    a.set_ylabel("expert")
    for i in range(4):
        for j in range(4):
            a.text(j, i, "%d" % m[i, j], ha="center", va="center", color="white" if row[i, j] > 0.55 else INK)
    a.set_title("Reasoner SFT, next-action probe\naccuracy %.1f%%  (majority baseline %.1f%%, n = %d)"
                % (100 * v["accuracy"], 100 * d["majority_baseline"], d["n"]))
    fig.tight_layout()
    fig.savefig(OUT / "fig_probe_confusion.png", dpi=160)


def fig_navigation(rows):
    """Per policy: OS (light), end-SR (medium) and standard SR (solid), from docs/figures/navigation_rows.json."""
    colors = {"LatentPilot": BLUE, "Pointing": ORANGE, "Cosmos": AQUA, "Paper": INK2}
    fig, a = plt.subplots(figsize=(11.5, 4.6))
    w = 0.27
    for i, r in enumerate(rows):
        c = colors[r["group"]]
        for j, (key, alpha) in enumerate((("os", 0.3), ("sr_end", 0.6), ("sr", 1.0))):
            x = i + (j - 1) * w
            v = r.get(key)
            if isinstance(v, (int, float)):
                a.bar(x, v, w, color=c, alpha=alpha, edgecolor=SURFACE, linewidth=1.5)
                a.text(x, v + 0.8, "%.0f" % v, ha="center", fontsize=7, color=INK if alpha == 1.0 else INK2)
            elif not (r["group"] == "Paper" and key == "sr_end"):
                a.text(x, 1.0, "n/r" if v == "n/r" else "n/m", ha="center", fontsize=6.5, color=INK2, rotation=90)
    a.set_xticks(range(len(rows)), ["%s\n(n=%s)" % (r["label"], r["n"]) for r in rows], fontsize=7.5, rotation=30, ha="right")
    a.set_ylabel("% of episodes")
    a.set_title("R2R-CE val_unseen, three bars per policy: OS (light) · end-SR (medium) · SR (solid)")
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=g) for g, c in colors.items()]
    handles += [Patch(color=INK2, alpha=a_, label=l) for a_, l in ((0.3, "OS: came within 3 m"),
                (0.6, "end-SR: ended within 3 m"), (1.0, "SR: stopped within 3 m"))]
    a.legend(handles=handles, frameon=False, loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.99, 0.01, "n/m = not measured (diagnostic evaluation)   n/r = not recoverable from logs",
             ha="right", fontsize=7.5, color=INK2)
    fig.savefig(OUT / "fig_navigation.png", dpi=160)


def eval_blocks(log):
    """Blocks '######## <ckpt> strict thr=X n=N' followed by 'SR = ..', 'OS = ..', 'model_stop K', 'where it STOPPED ...
    within 3m P%'. SR in the logs is end-SR (final position within 3 m, STOP not required); the standard SR (own STOP
    within 3 m) is model_stop x P / n, available only where 'where it STOPPED' was logged."""
    import re
    out, cur = [], None
    for line in open(ROOT / "logs" / log):
        m = re.match(r"#+ (.*?)thr=([0-9.]+)", line)
        if m:
            n = re.search(r"n=(\d+)", line)
            cur = {"ckpt": m.group(1).strip(), "thr": float(m.group(2)), "n": int(n.group(1)) if n else 150}
            out.append(cur)
        m = re.match(r"\s+(SR|SPL|OS)\s+=\s+([0-9.]+)", line)
        if m and cur is not None and m.group(1) not in cur:
            cur[m.group(1)] = float(m.group(2))
        m = re.match(r"\s+model_stop\s+(\d+)\s+\(", line)
        if m and cur is not None and "stops" not in cur:
            cur["stops"] = int(m.group(1))
        m = re.search(r"where it STOPPED.*within 3m\s+([0-9.]+)%", line)
        if m and cur is not None and "stops" in cur:
            cur["SR_std"] = round(cur["stops"] * float(m.group(1)) / 100) / cur["n"]
    return out


def fig_stop_threshold():
    b = sorted(eval_blocks("eval120k_220002.log"), key=lambda r: r["thr"])
    steps = [(20, "fusesweep_234742.log"), (60, "sweep60k_104019.log"), (80, "test80k_125633.log"),
             (100, "eval100k_185222.log"), (120, "eval120k_220002.log")]
    fig, (a, c) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    t = [r["thr"] for r in b]
    a.plot(t, [100 * r["OS"] for r in b], color=BLUE, marker="o", markersize=5)
    a.plot(t, [100 * r["SR"] for r in b], color=ORANGE, marker="o", markersize=5)
    a.plot(t, [100 * r["SR_std"] for r in b], color=AQUA, marker="o", markersize=5)
    labels = ["OS: came within 3 m", "end-SR: ended within 3 m", "SR: stopped within 3 m"]
    vs_p = ROOT / "results" / "followups" / "summary.json"
    if vs_p.exists():   # SR on val_seen, used to CHOOSE the threshold without touching the test split
        V = json.load(open(vs_p))
        vs = sorted((float(k.split("thr")[1]), v["SR_stop"]) for k, v in V.items() if k.startswith("cl_valseen_thr"))
        if vs:
            a.plot([x for x, _ in vs], [100 * y for _, y in vs], color=AQUA, marker="o", markersize=4, linestyle="--")
            labels.append("SR on val_seen (picks τ)")
            best = max(vs, key=lambda xy: xy[1])[0]
            a.axvline(best, color=INK2, linewidth=1, linestyle=":")
            a.text(best + 0.004, 46, "τ chosen on val_seen", fontsize=8, color=INK2)
    a.legend(labels, frameon=False, fontsize=8, loc="lower right")
    a.set_ylim(8, 48)
    a.set_xlabel("STOP probability threshold")
    a.set_ylabel("% of episodes (n = 150)")
    a.set_title("Fusion, step 120k: the STOP threshold trade-off")
    end, std = [], []
    for k, log in steps:
        rows = [r for r in eval_blocks(log) if abs(r["thr"] - 0.20) < 1e-9 and "fusion" in r["ckpt"]]
        end.append(100 * rows[0]["SR"] if rows else np.nan)
        std.append(100 * rows[0]["SR_std"] if rows and "SR_std" in rows[0] else np.nan)
    ks = [k for k, _ in steps]
    c.plot(ks, end, color=ORANGE, marker="o", markersize=5, label="end-SR (ended within 3 m)")
    c.plot(ks, std, color=AQUA, marker="o", markersize=5, label="SR (stopped within 3 m)")
    for k, v, w in zip(ks, end, std):
        c.text(k, v + 1.2, "%.1f" % v, ha="center", fontsize=8, color=INK)
        if not np.isnan(w):
            c.text(k, w - 3.0, "%.1f" % w, ha="center", fontsize=8, color=INK)
    c.set_xlabel("training step (thousands)")
    c.set_ylabel("% of episodes (thr 0.20, n = 150)")
    c.set_title("Fusion over training (SR not logged at 20k)")
    c.set_ylim(0, 40)
    c.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_stop_threshold.png", dpi=160)


def fig_lp_failure():
    """Why fully trained LatentPilot fails: it fits its 6 training buildings better and predicts the future better,
    yet navigates worse than the identity control."""
    L, I = trainlog("stage1_learned"), trainlog("stage1_identity")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2.4, 1, 1]})
    curves = ((trainlog("stage0"), INK2, "Stage 0: no Pilot slot", "--"),
              (trainlog("stage0prime_B"), AQUA, "Stage 0′: slot fed the true next frame", "-"),
              (L, BLUE, "Stage 1, learned G_ψ", "-"), (I, ORANGE, "Stage 1, identity G_ψ (control)", "-"))
    for log, col, name, ls in curves:
        st = np.array([r["step"] for r in log])
        acc = smooth(np.array([r["acc"] for r in log]))
        a.plot(st[:len(acc)], 100 * acc, color=col, label=name, linestyle=ls)
    a.set_xlabel("training step")
    a.set_ylabel("training action accuracy (%)")
    a.set_title("(a) Training accuracy jumps once the slot sees the next frame")
    a.legend(frameon=False, loc="lower right", fontsize=7.5)
    names = ["learned", "identity"]
    lpil = [L[-1]["l_pil"], I[-1]["l_pil"]]
    b.bar(names, lpil, color=[BLUE, ORANGE], width=0.6, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(lpil):
        b.text(i, v + 0.12, "%.2f" % v, ha="center", color=INK)
    b.set_ylabel("L_pil at end of training")
    b.set_title("(b) Future prediction\n(lower = better)")
    os_ = [10.67, 17.33]  # logs/evaln150_stage1_{learned,identity}_final.log
    c.bar(names, os_, color=[BLUE, ORANGE], width=0.6, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(os_):
        c.text(i, v + 0.4, "%.1f %%" % v, ha="center", color=INK)
    c.set_ylabel("oracle success (%, n = 150)")
    c.set_title("(c) Navigation, unseen\nbuildings (higher = better)")
    c.set_ylim(0, 22)
    fig.tight_layout()
    fig.savefig(OUT / "fig_lp_failure.png", dpi=160)


def fig_pointing_failure():
    """How the best pointing policy's episodes end, and which actions it gets wrong."""
    r = [x for x in eval_blocks("eval120k_220002.log") if abs(x["thr"] - 0.20) < 1e-9][0]
    n = r["n"]
    ok = round(r["SR_std"] * n)
    end_ok = round(r["SR"] * n)
    parts = [("stopped within 3 m\n(success)", ok, AQUA, 1.0),
             ("ran out of steps\nwithin 3 m", end_ok - ok, AQUA, 0.45),
             ("stopped > 3 m away", r["stops"] - ok, ORANGE, 1.0),
             ("ran out of steps\n> 3 m away", n - r["stops"] - (end_ok - ok), "#b9b7b1", 1.0)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
    left = 0
    for name, k, col, al in parts:
        v = 100 * k / n
        a.barh([0], [v], left=left, color=col, alpha=al, edgecolor=SURFACE, linewidth=2, height=0.5,
               label=name.replace("\n", " "))
        a.text(left + v / 2, 0, "%.1f %%" % v, ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold")
        left += v
    a.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    a.set_xlim(0, 100)
    a.set_ylim(-0.4, 0.4)
    a.set_yticks([])
    a.set_xlabel("% of 150 episodes")
    a.set_title("(a) How episodes end (fusion step 120k, STOP threshold 0.20)")
    a.grid(False)
    conf = {"FWD": (0.901, 0.032, 0.067), "LEFT": (0.388, 0.553, 0.059), "RIGHT": (0.428, 0.015, 0.557)}  # failanalysis log
    xs = list(conf)
    bottom = np.zeros(3)
    for j, (lab, col) in enumerate((("predicted FORWARD", "#b9b7b1"), ("predicted LEFT", BLUE), ("predicted RIGHT", ORANGE))):
        vals = np.array([100 * conf[x][j] for x in xs])
        b.bar(xs, vals, bottom=bottom, color=col, edgecolor=SURFACE, linewidth=2, width=0.6, label=lab)
        for i, v in enumerate(vals):
            if v > 8:
                b.text(i, bottom[i] + v / 2, "%.0f" % v, ha="center", va="center", fontsize=8.5, color=INK)
        bottom += vals
    b.set_xlabel("expert's action")
    b.set_ylabel("% of steps (teacher-forced)")
    b.set_title("(b) Missed turns become FORWARD")
    b.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pointing_failure.png", dpi=160)


def fig_reasoner_failure():
    """Reasoner: how closed-loop episodes end, and how often it says STOP vs how often STOP was in its training data."""
    runs = [("v1, step 1,250", "sft_step1250_direct"), ("v1, final", "sft_final_direct"),
            ("v3, final", "v3_final_direct"), ("v3, step 1,500", "v3_step1500_direct")]
    d = {k: json.load(open(ROOT / "results" / "runs" / f"{f}.json")) for k, f in runs}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    cats = [("term_within_radius", "reached 3 m of the goal", AQUA), ("term_model_stop", "said STOP (> 3 m away)", ORANGE),
            ("term_timeout", "ran out of steps", "#b9b7b1")]
    for i, (k, _) in enumerate(runs):
        left = 0
        tot = sum(d[k][c] for c, _, _ in cats)
        for c, name, col in cats:
            v = 100 * d[k][c] / tot
            a.barh(i, v, left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.62, label=name if i == 0 else None)
            if v >= 7:
                a.text(left + v / 2, i, "%.0f" % v, ha="center", va="center", fontsize=8.5, color=INK)
            left += v
    a.set_yticks(range(len(runs)), [k for k, _ in runs])
    a.invert_yaxis()
    a.set_xlim(0, 100)
    a.set_xlabel("% of 40 episodes (diagnostic evaluation)")
    a.set_title("(a) How closed-loop episodes end")
    a.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    a.grid(False)
    train = {"v1": 25.0, "v3": 5.0}  # STOP share of SFT labels (logs/train_reasoner_{sft,v3}.log)
    said = {"v1": 100 * d["v1, final"]["commands"]["stop"], "v3": 100 * d["v3, step 1,500"]["commands"]["stop"]}
    ended = {"v1": 100 * d["v1, final"]["term_model_stop"] / 40, "v3": 100 * d["v3, step 1,500"]["term_model_stop"] / 40}
    x = np.arange(2)
    w = 0.26
    for j, (vals, lab, col) in enumerate(((train, "STOP share of training labels", "#b9b7b1"),
                                          (said, "STOP share of its commands", BLUE),
                                          (ended, "episodes it ended with STOP", ORANGE))):
        v = [vals["v1"], vals["v3"]]
        b.bar(x + (j - 1) * w, v, w, color=col, edgecolor=SURFACE, linewidth=2, label=lab)
        for i, y in enumerate(v):
            b.text(x[i] + (j - 1) * w, y + 1, "%.0f" % y, ha="center", fontsize=8, color=INK)
    b.set_xticks(x, ["SFT v1 (final)", "SFT v3 (step 1,500)"])
    b.set_ylabel("%")
    b.set_ylim(0, 75)
    b.set_title("(b) STOP follows the training mix")
    b.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_reasoner_failure.png", dpi=160)


def fig_bearing():
    b = np.degrees(np.load(ROOT / "logs" / "bearings.npy"))
    pred, true = b[0], b[1]
    slope = float((pred * true).sum() / (true * true).sum())
    fig, a = plt.subplots(figsize=(5.2, 4.6))
    a.scatter(true, pred, s=6, color=BLUE, alpha=0.25, linewidths=0)
    lim = np.percentile(np.abs(true), 99.5)
    x = np.linspace(-lim, lim, 10)
    a.plot(x, x, color=INK2, linewidth=1, linestyle="--")
    a.plot(x, slope * x, color=ORANGE)
    a.text(lim * 0.2, lim * 0.85, "ideal (slope 1)", color=INK2)
    a.text(lim * 0.2, slope * lim * 0.2 - lim * 0.35, "fit: slope %.3f" % slope, color=INK)
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)
    a.set_xlabel("expert bearing (°)")
    a.set_ylabel("predicted bearing (°)")
    a.set_title("Pointing bearing is shrunk towards 0°\n(fusion step 120k, 3,000 steps)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_bearing.png", dpi=160)


def fig_layer_probe():
    import re
    rows = []
    for line in open(ROOT / "logs" / "layerprobe_154820.log"):
        m = re.match(r"\s+(\d+)\s+(-?[0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(-?[0-9.]+)\s*$", line)
        if m:
            rows.append([float(g) for g in m.groups()])
    r = np.array(rows)
    fig, a = plt.subplots(figsize=(8, 3.6))
    a.plot(r[:, 0], r[:, 3], color=BLUE, marker="o", markersize=4)
    a.axhline(0.5, color=INK2, linewidth=1, linestyle="--")
    a.text(5.5, 0.505, "chance (0.5)", color=INK2)
    best = int(r[:, 3].argmax())
    a.text(r[best, 0], r[best, 3] + 0.012, "layer %d: %.3f" % (r[best, 0], r[best, 3]), ha="center", color=INK)
    a.text(r[-1, 0] - 0.4, r[-1, 3] - 0.012, "final layer: %.3f" % r[-1, 3], ha="right", color=INK)
    a.set_ylim(0.41, 0.76)
    a.set_xlabel("backbone layer (0 = embeddings)")
    a.set_ylabel("STOP linear-probe AUC")
    a.set_title("Where STOP information lives (pointing_full3, 1,000 held-out steps)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_layer_probe.png", dpi=160)


def fig_edge():
    """Cosmos3-Edge action model, zero-shot: does guidance make it follow language?"""
    import re
    g, sign = [], []
    for line in open(ROOT / "logs" / "edge_ctrl_cfg.log"):
        m = re.search(r"guidance_scale = ([0-9.]+)", line)
        if m:
            g.append(float(m.group(1)))
        m = re.search(r"sign correct on (\d+)%", line)
        if m:
            sign.append(int(m.group(1)))
    full = {}  # guidance -> (correct, swapped) turn-direction agreement with full R2R instructions
    txt = open(ROOT / "logs" / "edge_policy_zs.log").read()
    full[1.0] = tuple(float(x) for x in re.findall(r"turn direction\s+([0-9.]+)", txt)[:2])
    cur = None
    for line in open(ROOT / "logs" / "edge_policy_cfg.log"):
        m = re.search(r"guidance_scale = ([0-9.]+)", line)
        if m:
            cur = float(m.group(1))
        m = re.search(r"(correct|swapped): .*turn dir ([0-9.]+)", line)
        if m and cur is not None:
            full.setdefault(cur, [None, None])
            full[cur] = list(full[cur])
            full[cur][0 if m.group(1) == "correct" else 1] = float(m.group(2))
    fig, (a, c) = plt.subplots(1, 2, figsize=(10, 3.6))
    a.plot(g, sign, color=BLUE, marker="o", markersize=6)
    for x, y in zip(g, sign):
        a.text(x, y + 1.2, "%d%%" % y, ha="center", color=INK)
    a.axhline(50, color=INK2, linewidth=1, linestyle="--")
    a.text(g[-1], 52, "chance", ha="right", color=INK2)
    a.set_ylim(40, 102)
    a.set_xlabel("classifier-free guidance scale")
    a.set_ylabel("turn sign correct (%)")
    a.set_title("Short commands (\"turn left / right\"), n = 22")
    ks = sorted(full)
    x = np.arange(len(ks))
    cor = [100 * full[k][0] for k in ks]
    swp = [100 * full[k][1] for k in ks]
    c.bar(x - 0.19, cor, 0.36, color=BLUE, edgecolor=SURFACE, linewidth=2, label="matching instruction")
    c.bar(x + 0.19, swp, 0.36, color=ORANGE, edgecolor=SURFACE, linewidth=2, label="another episode's instruction")
    for i in range(len(ks)):
        c.text(i - 0.19, cor[i] + 1, "%.0f" % cor[i], ha="center", fontsize=8.5, color=INK)
        c.text(i + 0.19, swp[i] + 1, "%.0f" % swp[i], ha="center", fontsize=8.5, color=INK)
    c.axhline(50, color=INK2, linewidth=1, linestyle="--")
    n = {1.0: 192}  # guidance 1 comes from edge_policy_zs.log (384 frames); 5 and 7.5 from edge_policy_cfg.log
    c.set_xticks(x, ["guidance %g\n(n = %d)" % (k, n.get(k, 212)) for k in ks])
    c.set_ylim(0, 85)
    c.set_ylabel("turn direction agrees with expert (%)")
    c.set_title("Full R2R instructions")
    c.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.suptitle("Cosmos3-Edge action model, zero-shot: language only reaches the sampler at high guidance",
                 fontweight="bold", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_edge_guidance.png", dpi=160)


def fig_reasoner_zeroshot():
    """First reasoner probe (results/reasoner_probe.md): zero-shot 'navigation trajectory' points on the still frame."""
    import re
    md = open(ROOT / "results" / "reasoner_probe.md").read()
    cases = re.split(r"^## ", md, flags=re.M)[1:]
    fig, axes = plt.subplots(1, len(cases), figsize=(4 * len(cases), 4.6))
    for a, case in zip(np.atleast_1d(axes), cases):
        head = case.splitlines()[0].strip()
        scan, traj = [t.strip() for t in head.split("/")]
        instr = re.search(r"\*\*instruction:\*\* (.*)", case).group(1)
        nav = case.split("### traj_nav", 1)[1].split("###", 1)[0]
        pts = np.array([[int(u), int(v)] for u, v in re.findall(r'"point_2d": \[(\d+), (\d+)\]', nav)]) / 1000.0
        img = plt.imread(ROOT / "logs" / "reasoner_frames" / f"{scan}_{traj.replace('traj', '')}.png")
        h, w = img.shape[:2]
        a.imshow(img)
        a.plot(pts[:, 0] * w, pts[:, 1] * h, color="white", linewidth=4)
        a.plot(pts[:, 0] * w, pts[:, 1] * h, color=BLUE, linewidth=2, marker="o", markersize=7,
               markeredgecolor="white", markeredgewidth=1.5)
        for k, (u, v) in enumerate(pts):
            a.text(u * w + 9, v * h - 6, str(k + 1), color="white", fontsize=9, fontweight="bold")
        wrapped = __import__("textwrap").wrap(instr, 52)
        a.set_title("\n".join(wrapped[:3]) + (" …" if len(wrapped) > 3 else ""), fontsize=8, fontweight="normal",
                    color=INK2)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.suptitle("Cosmos3-Edge reasoner, zero-shot: predicted navigation waypoints (single still frame)",
                 fontweight="bold", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig_reasoner_zeroshot.png", dpi=160)


def frame(src, t=None, last=False):
    """One frame of a published demo video, via ffmpeg (seconds from start, or the last frame)."""
    import subprocess
    import tempfile
    out = pathlib.Path(tempfile.mkdtemp()) / "f.png"
    seek = ["-sseof", "-0.3"] if last else ["-ss", str(t)]
    subprocess.run(["ffmpeg", "-v", "error", "-y", *seek, "-i", str(ROOT / src), "-frames:v", "1", str(out)], check=True)
    return plt.imread(out)


def fig_qualitative():
    """Rollout frames from media/: LatentPilot comparison, pointing failures, reasoner decisions."""
    top = frame("media/comparisons/ep09_x8F5xyUWy9e_expert_vs_stage1.mp4", t=14)
    bottom = [
        (frame("media/rollouts/step100k_failures/ep04_EU6Fwq7SyZv.mp4", t=8),
         "(b) Pointing: 1.3 m from the goal,\np(stop) = 0.02; it never stops"),
        (frame("media/rollouts/step100k_failures/ep16_pLe4wQe7qrG.mp4", last=True),
         "(c) Pointing: stops 5.4 m short\n(it passed within 1.5 m)"),
        (frame("media/previews/reasoner_v1_2azQ1b91cZZ_t225.gif", last=True),
         "(d) Reasoner SFT v1, decision\nprobe at the goal: stop ✓"),
        (frame("media/previews/reasoner_v3_2azQ1b91cZZ_t225.gif", last=True),
         "(e) Reasoner SFT v3, same moment:\nmove forward ✗"),
    ]
    fig = plt.figure(figsize=(11, 8.6))
    g = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.25], hspace=0.12, wspace=0.04)
    a = fig.add_subplot(g[0, :])
    a.imshow(top)
    a.set_title("(a) LatentPilot, same episode: expert | Stage 1 step 10k (reaches 2.8 m) | Stage 1 final (never within 9 m)",
                fontsize=9.5, fontweight="normal", color=INK)
    a.axis("off")
    for k, (img, cap) in enumerate(bottom):
        b = fig.add_subplot(g[1, k])
        b.imshow(img)
        b.set_title(cap, fontsize=8.5, fontweight="normal", color=INK)
        b.axis("off")
    fig.savefig(OUT / "fig_qualitative.png", dpi=150, bbox_inches="tight")


def fig_slot_shortcut():
    """Offline test of the Pilot-slot shortcut (scripts/test_slot_shortcut.py): held-out action accuracy with the slot
    filled with the true next frame (training condition) vs the model's own latent (test condition)."""
    main_p = ROOT / "results" / "slot_shortcut.json"
    over_p = ROOT / "results" / "slot_shortcut_over_training.json"
    if not main_p.exists():
        return
    R = json.load(open(main_p))
    conds = [("next_frame", "true next frame (training input)", BLUE, 1.0),
             ("own_z", "its own latent z (test-time input)", ORANGE, 1.0),
             ("current_frame", "current frame (no future info)", AQUA, 1.0),
             ("other_next_frame", "another episode's next frame", "#b9b7b1", 1.0)]
    models = [(k, lab) for k, lab in (("stage0prime_B", "Stage 0′\n(no L_pil)"), ("stage1_learned", "Stage 1\nlearned G_ψ"),
                                       ("stage1_identity", "Stage 1\nidentity G_ψ"),
                                       ("stage2_p50", "Stage 2\n50 % own z"), ("stage2_p75", "Stage 2\n75 % own z"),
                                       ("stage2_p75_long", "Stage 2, 3× longer\n75 % own z"))
              if k in R]
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.9, 1]})
    x = np.arange(len(models))
    w = 0.2
    for j, (c, lab, col, al) in enumerate(conds):
        v = [100 * R[m]["conditions"][c]["accuracy"] for m, _ in models]
        a.bar(x + (j - 1.5) * w, v, w, color=col, alpha=al, edgecolor=SURFACE, linewidth=1.5, label=lab)
        for i, y in enumerate(v):
            a.text(x[i] + (j - 1.5) * w, y + 1, "%.0f" % y, ha="center", fontsize=7.5, color=INK)
    if "stage0" in R:
        s0 = 100 * R["stage0"]["conditions"]["no_slot"]["accuracy"]
        a.axhline(s0, color=INK2, linestyle="--", linewidth=1, label="Stage 0, no slot (%.0f %%)" % s0)
    counts = R[models[0][0]]["expert_action_counts"]
    maj = 100 * max(counts.values()) / sum(counts.values())
    a.axhline(maj, color=INK2, linestyle=":", linewidth=1, label="always FORWARD (%.0f %%)" % maj)
    a.set_xticks(x, [lab for _, lab in models])
    a.set_ylabel("held-out action accuracy (%)")
    a.set_ylim(0, 112)
    n = R[models[0][0]]["steps"]
    a.set_title("(a) Same model, same steps, only the Pilot slot differs (%d steps)" % n)
    a.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=3)
    if over_p.exists():
        O = json.load(open(over_p))
        ks = sorted(O, key=lambda k: int(k.rsplit("step", 1)[1]))
        st = [int(k.rsplit("step", 1)[1]) / 1000 for k in ks]
        for c, lab, col, _ in conds[:3]:
            b.plot(st, [100 * O[k]["conditions"][c]["accuracy"] for k in ks], color=col, marker="o", markersize=4,
                   label=lab.split(" (")[0])
        b.set_xlabel("training step (thousands), Stage 1 learned G_ψ")
        b.set_ylabel("held-out action accuracy (%)")
        b.set_ylim(0, 100)
        b.set_title("(b) Learned G_ψ: the gap opens by step 4k and stays")
        b.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_slot_shortcut.png", dpi=160)


def main(nav_rows=None):
    OUT.mkdir(parents=True, exist_ok=True)
    fig_loss_balance()
    fig_stage0prime()
    fig_probe_confusion()
    fig_stop_threshold()
    fig_bearing()
    fig_layer_probe()
    fig_edge()
    fig_reasoner_zeroshot()
    fig_qualitative()
    fig_lp_failure()
    fig_pointing_failure()
    fig_reasoner_failure()
    fig_slot_shortcut()
    if nav_rows:
        fig_navigation(nav_rows["rows"])
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


NAV = json.load(open(ROOT / "docs" / "figures" / "navigation_rows.json")) if (
    ROOT / "docs" / "figures" / "navigation_rows.json").exists() else None

if __name__ == "__main__":
    main(NAV)
