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
    """Oracle success (light) and strict self-stop success (solid) per policy, from docs/figures/navigation_rows.json."""
    colors = {"LatentPilot": BLUE, "Pointing": ORANGE, "Cosmos": AQUA, "Paper": INK2}
    fig, a = plt.subplots(figsize=(11, 4.4))
    for i, r in enumerate(rows):
        c = colors[r["group"]]
        a.bar(i - 0.19, r["os"], 0.36, color=c, alpha=0.35, edgecolor=SURFACE, linewidth=2)
        a.text(i - 0.19, r["os"] + 0.8, "%.0f" % r["os"], ha="center", fontsize=7.5, color=INK2)
        if r["sr"] is not None:
            a.bar(i + 0.19, r["sr"], 0.36, color=c, edgecolor=SURFACE, linewidth=2)
            a.text(i + 0.19, r["sr"] + 0.8, "%.0f" % r["sr"], ha="center", fontsize=7.5, color=INK)
        else:
            a.text(i + 0.19, 1.2, "n/m", ha="center", fontsize=7, color=INK2, rotation=90)
    a.set_xticks(range(len(rows)), ["%s\n(n=%s)" % (r["label"], r["n"]) for r in rows], fontsize=7.5, rotation=30, ha="right")
    a.set_ylabel("% of episodes")
    a.set_title("R2R-CE val_unseen: oracle success (light) vs strict self-stop success (solid; n/m = not measured)")
    for g, c in colors.items():
        a.bar(0, 0, color=c, label=g)
    a.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_navigation.png", dpi=160)


def eval_blocks(log):
    """Blocks '######## <ckpt> strict thr=X n=N' followed by 'SR = ..', 'OS = ..' lines."""
    import re
    out, cur = [], None
    for line in open(ROOT / "logs" / log):
        m = re.match(r"#+ (\S+).*?thr=([0-9.]+)", line)
        if m:
            cur = {"ckpt": m.group(1), "thr": float(m.group(2))}
            out.append(cur)
        m = re.match(r"\s+(SR|SPL|OS)\s+=\s+([0-9.]+)", line)
        if m and cur is not None and m.group(1) not in cur:
            cur[m.group(1)] = float(m.group(2))
    return out


def fig_stop_threshold():
    b = sorted(eval_blocks("eval120k_220002.log"), key=lambda r: r["thr"])
    steps = [(20, "fusesweep_234742.log"), (60, "sweep60k_104019.log"), (80, "test80k_125633.log"),
             (100, "eval100k_185222.log"), (120, "eval120k_220002.log")]
    fig, (a, c) = plt.subplots(1, 2, figsize=(10, 3.6))
    t = [r["thr"] for r in b]
    a.plot(t, [100 * r["OS"] for r in b], color=BLUE, marker="o", markersize=5)
    a.plot(t, [100 * r["SR"] for r in b], color=ORANGE, marker="o", markersize=5)
    a.text(t[0], 100 * b[0]["OS"] + 1.0, "oracle success (OS)", color=INK)
    a.text(t[0], 100 * b[0]["SR"] + 1.0, "strict success (SR)", color=INK)
    a.set_xlabel("STOP probability threshold")
    a.set_ylabel("% of episodes (n = 150)")
    a.set_title("Fusion, step 120k: the STOP threshold trade-off")
    sr = []
    for k, log in steps:
        rows = [r for r in eval_blocks(log) if abs(r["thr"] - 0.20) < 1e-9 and "fusion" in r["ckpt"]]
        sr.append(100 * rows[0]["SR"] if rows else np.nan)
    c.plot([k for k, _ in steps], sr, color=ORANGE, marker="o", markersize=5)
    for (k, _), v in zip(steps, sr):
        c.text(k, v + 1.2, "%.1f" % v, ha="center", fontsize=8, color=INK)
    c.set_xlabel("training step (thousands)")
    c.set_ylabel("strict SR % (thr 0.20, n = 150)")
    c.set_title("Fusion: strict SR over training (non-monotonic)")
    c.set_ylim(0, 40)
    fig.tight_layout()
    fig.savefig(OUT / "fig_stop_threshold.png", dpi=160)


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
    if nav_rows:
        fig_navigation(nav_rows["rows"])
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


NAV = json.load(open(ROOT / "docs" / "figures" / "navigation_rows.json")) if (
    ROOT / "docs" / "figures" / "navigation_rows.json").exists() else None

if __name__ == "__main__":
    main(NAV)
