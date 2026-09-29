#!/usr/bin/env python3
"""Build the README demo media from frames already on disk (no simulator, no GPU).

Comparisons: for each val_unseen episode recorded by scripts/record_episode.py --per-scan 1 (the first episode of every
scan, sorted by scan id), put three views side by side on the same instruction:

    expert (reference-path follower, data/rollouts/val_unseen/<scan>/<episode_id>/frames)
    | Stage 1 learned Pilot Token, step 10k (recordings/step10000_multiscene)
    | Stage 1 learned Pilot Token, final   (recordings/final_multiscene)

The expert stops when it reaches the goal; its last frame is held so the panels stay aligned in time.

    python tools/make_demo_media.py            # writes media/comparisons/*.mp4 and media/previews/*.gif
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


def main():
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
