#!/usr/bin/env python3
"""Build the README demo media from frames already on disk (no simulator, no GPU).

Comparisons: for each val_unseen episode recorded by scripts/record_episode.py --per-scan 1 (the first episode of every
scan, sorted by scan id), put three views side by side on the same instruction:

    expert (reference-path follower, data/rollouts/val_unseen/<scan>/<episode_id>/frames)
    | Stage 1 learned Pilot Token, step 10k (recordings/step10000_multiscene)
    | Stage 1 learned Pilot Token, final   (recordings/final_multiscene)

The expert stops when it reaches the goal; its last frame is held so the panels stay aligned in time.

Reasoner clips: the Cosmos3-Edge reasoner's input window at saved probe decision points, with the expert label and the
answers of both SFT checkpoints (results/probe_sft_final.json, results/probe_v3_final.json).

    python tools/make_demo_media.py            # writes media/comparisons/*.mp4 and media/previews/reasoner_*.gif
"""
import gzip
import json
import pathlib
import subprocess
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FPS = 4
W, H = 448, 600  # panel size: the 448 x 448 view + an overlay whose height grows with the instruction (538-553 px); all padded to 600


def first_episode_per_scan(split="val_unseen"):
    eps = json.load(gzip.open(ROOT / "data" / "R2R_VLNCE_v1-3" / split / f"{split}.json.gz", "rt"))["episodes"]
    by_scan = {}
    for e in eps:
        by_scan.setdefault(e["scene_id"].split("/")[1], []).append(e)
    return [(scan, v[0]) for scan, v in sorted(by_scan.items())]


def label(text, x):
    return (f"drawtext=fontfile={FONT}:text='{text}':x={x}+({W}-text_w)/2:y=10:fontsize=20:fontcolor=white")


def comparison(idx, scan, ep, out):
    exp = ROOT / "data" / "rollouts" / "val_unseen" / scan / str(ep["episode_id"]) / "frames"
    a = ROOT / "recordings" / "step10000_multiscene" / f"ep{idx:02d}_{scan}"
    b = ROOT / "recordings" / "final_multiscene" / f"ep{idx:02d}_{scan}"
    if not (exp.is_dir() and a.is_dir() and b.is_dir()):
        return None
    n = max(len(list(a.glob("*.png"))), len(list(b.glob("*.png"))))
    n_exp = len(list(exp.glob("*.jpg")))
    hold = max(0, n - n_exp) / FPS
    # characters with meaning in ffmpeg filter syntax (separators, quoting, expansion) are dropped from captions
    text = " ".join(ep["instruction"]["instruction_text"].translate({ord(c): None for c in "',;:[]%\\"}).split())
    lines = textwrap.wrap(text, 62)[:6]
    cap = "".join(f",drawtext=fontfile={FONT_REG}:text='{l}':x=6:y={456 + 16 * k}:fontsize=12:fontcolor=white"
                  for k, l in enumerate(lines))
    filt = (f"[0]pad={W}:{H}:0:0:black{cap},tpad=stop_mode=clone:stop_duration={hold}[e];"
            f"[1]pad={W}:{H}:0:0:black[p];[2]pad={W}:{H}:0:0:black[q];"
            f"[e][p][q]hstack=inputs=3,pad={3 * W}:{H + 44}:0:44:black,"
            f"{label('Expert (reference path)', 0)},{label('Stage 1 - step 10k', W)},{label('Stage 1 - final', 2 * W)}[v]")
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-framerate", str(FPS), "-i", str(exp / "%04d.jpg"),
           "-framerate", str(FPS), "-i", str(a / "%04d.png"),
           "-framerate", str(FPS), "-i", str(b / "%04d.png"),
           "-filter_complex", filt, "-map", "[v]", "-frames:v", str(n),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30", str(out)]
    subprocess.run(cmd, check=True)
    return out


def gif(src, dst, seconds=18, width=780):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-t", str(seconds), "-i", str(src), "-vf",
                    f"fps={FPS},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=80[p];[b][p]paletteuse",
                    str(dst)], check=True)


# Probe decision points whose transcripts were saved for both reasoner checkpoints (results/probe_*_final.json).
REASONER_CASES = [("2azQ1b91cZZ", "1039", 225), ("8194nk5LbLH", "1141", 34)]


def clean(text):
    return " ".join(text.translate({ord(c): None for c in "',;:[]%\\"}).split())


def reasoner_decision(scan, traj, t, out, width=480, ctx=120, hold=5.0):
    """A decision probe, NOT a closed-loop rollout. Part 1 plays the reasoner's input: the last 8 s of the rendered expert
    walk up to decision frame t (the model sees 8 of these frames, 1 per second). Part 2 freezes on frame t and shows
    the question, the expert's action and what each fine-tuned reasoner answered (results/probe_*_final.json)."""
    import tempfile
    preds = {}
    for name, f in (("v1", "probe_sft_final.json"), ("v3", "probe_v3_final.json")):
        for r in json.load(open(ROOT / "results" / f))["transcripts"]:
            if (r["scan"], str(r["traj"]), int(r["t"])) == (scan, traj, t):
                preds[name], preds["expert"] = r["pred"], r["label"]
    d = ROOT / "data" / "video" / "val_unseen" / scan / f"traj{traj}"
    instruction = json.loads((d / "meta.json").read_text())["instructions"]
    instruction = instruction[0] if isinstance(instruction, list) else instruction.strip("[]'\" ").split("', '")[0]
    tmp = pathlib.Path(tempfile.mkdtemp())

    def txt(x, y, text, size=14, colour="white", bold=False):
        f = tmp / f"t{len(list(tmp.glob('*.txt')))}.txt"
        f.write_text(text)
        return (f",drawtext=fontfile={FONT if bold else FONT_REG}:textfile={f}:expansion=none:x={x}:y={y}"
                f":fontsize={size}:fontcolor={colour}")

    top, lines = 54, textwrap.wrap(" ".join(instruction.split()), 60)[:4]
    bottom = 28 + 17 * len(lines)
    H = top + width + bottom
    frame_y = top
    instr = txt(10, frame_y + width + 8, "Instruction:", 13, "0xbbbbbb", True) + "".join(
        txt(10, frame_y + width + 26 + 17 * k, l, 13) for k, l in enumerate(lines))
    lo = max(0, t - ctx + 1)
    secs = (t - lo + 1) / 15.0
    part1 = (f"[0]trim=start_frame={lo}:end_frame={t + 1},setpts=PTS-STARTPTS,fps=6,scale={width}:{width},"
             f"pad={width}:{H}:0:{top}:black"
             + txt(10, 8, "DECISION PROBE  -  not a rollout", 17, "0xf2c14e", True)
             + txt(10, 31, f"Playing the {secs:.0f} s of camera view the reasoner receives as input", 13)
             + instr + "[a]")
    ok = {k: preds[k] == preds["expert"] for k in ("v1", "v3")}
    y0 = frame_y + 70
    card = (f",drawbox=x=0:y={frame_y}:w={width}:h={width}:color=black@0.72:t=fill"
            + txt(24, y0, "DECISION POINT", 20, "0xf2c14e", True)
            + txt(24, y0 + 30, "What should the robot do now?", 15)
            + txt(24, y0 + 80, "Expert (ground truth)", 14, "0xbbbbbb")
            + txt(24, y0 + 100, preds["expert"].upper(), 22, "white", True)
            + txt(24, y0 + 150, "SFT v1 answered", 14, "0xbbbbbb")
            + txt(24, y0 + 170, preds["v1"].upper() + ("   ✓ correct" if ok["v1"] else "   ✗ wrong"), 22,
                  "0x7ee2b8" if ok["v1"] else "0xff8a65", True)
            + txt(24, y0 + 220, "SFT v3 answered", 14, "0xbbbbbb")
            + txt(24, y0 + 240, preds["v3"].upper() + ("   ✓ correct" if ok["v3"] else "   ✗ wrong"), 22,
                  "0x7ee2b8" if ok["v3"] else "0xff8a65", True)
            + txt(24, y0 + 300, "v1 = first fine-tune: onset-balanced data, ~1 s of video", 12, "0xbbbbbb")
            + txt(24, y0 + 318, "v3 = final fine-tune: uniform data, 8 s of video", 12, "0xbbbbbb"))
    part2 = (f"[0]trim=start_frame={t}:end_frame={t + 1},setpts=PTS-STARTPTS,scale={width}:{width},"
             f"loop=loop={int(hold * 6)}:size=1:start=0,fps=6,pad={width}:{H}:0:{top}:black"
             + txt(10, 8, "DECISION PROBE  -  the models' answers", 17, "0xf2c14e", True)
             + txt(10, 31, "Frozen at the moment the question is asked", 13)
             + instr + card + "[b]")
    graph = (part1 + ";" + part2 + ";[a][b]concat=n=2:v=1,split[c][e];[c]palettegen=max_colors=96[p];[e][p]paletteuse")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(d / "frames.mp4"), "-filter_complex", graph, str(out)],
                   check=True)
    return out

def main():
    for scan, traj, t in REASONER_CASES:
        o = reasoner_decision(scan, traj, t, ROOT / "media" / "previews" / f"reasoner_{scan}_traj{traj}_t{t}.gif")
        print("wrote", o.relative_to(ROOT))
    out_dir = ROOT / "media" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for idx, (scan, ep) in enumerate(first_episode_per_scan()):
        o = comparison(idx, scan, ep, out_dir / f"ep{idx:02d}_{scan}_expert_vs_stage1.mp4")
        if o:
            made.append((idx, scan, ep["episode_id"], ep["info"]["geodesic_distance"], o.name))
            print("wrote", o.relative_to(ROOT))
    (out_dir / "index.json").write_text(json.dumps(
        [{"idx": i, "scan": s, "episode_id": e, "geodesic_m": round(g, 2), "file": f} for i, s, e, g, f in made], indent=1))
    return made


if __name__ == "__main__":
    main()
