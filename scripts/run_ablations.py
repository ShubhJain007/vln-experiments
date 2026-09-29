#!/usr/bin/env python3
"""Ablation runner: declares arms, runs them, writes results to a file.

Every arm is one eval_edge.py run under identical settings, differing only in
the prompt template. Results land in results/ablations.md (a cumulative table)
and results/runs/<stamp>_<arm>.json (the full record), so future ablations
append rather than scroll past in a terminal.

Failures are NOT swallowed: a non-zero exit is recorded in the table with the
log tail. An earlier runner grepped stdout for metric lines only and reported
four crashed evals as "completed".

Add an arm by appending to ARMS. Anything a run needs to be reproducible --
prompt, guidance, chunk, scale, patience -- is written into its JSON.
"""
import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = "/home/kneepolean/miniconda3/envs/latentpilot/bin/python"

# TWO different prompt distributions, from two different components:
#
# AV_ACTION is what NVIDIA's own *diffusers action* example passes as the whole
# prompt for domain "av" -- and the action pipeline puts our text into the JSON
# caption's `description` field, which was trained on short motion captions of
# a ~1.6 s clip ("mouse arrangement" for UMI). That is the distribution we are
# actually feeding.
#
# AV_REASONER is the long paragraph from the Reasoner Prompt Guide's "Drive
# Scene Next Action". It belongs to the TEXT-generation tower, which we are not
# calling here. Included as an explicit arm because mixing the two up is easy
# (I did it) and it is worth measuring rather than assuming.
#
# Both talk about vehicles while our instructions talk about bedrooms and
# bathrooms. That mismatch is unavoidable -- there is no indoor-walker domain,
# and `av` (planar, forward-dominant, ego-pose 9D) is the closest fit. Whether
# the domain framing helps despite the semantic clash is what these arms test.

AV_ACTION = "You are an autonomous vehicle planning system."

AV_REASONER = ("You are an autonomous vehicle planning system. The video depicts "
               "the observation from the vehicle's camera. You need to observe the "
               "critical objects in the environment and reason your next action and "
               "the driving trajectory ahead.")

# Each arm: (name, script, extra CLI args). eval_edge.py arms differ only in the
# prompt template; eval_cosmos_nav.py arms use the two-stage reasoner navigator.
# Listed in the order they are RUN (reasoner arms first, then diffusion prompt
# arms 5 -> 1 so r2r_raw -- already measured at SR 0.075 -- comes last).
ARMS = [
    ("reasoner_diffusion_grounded", "eval_cosmos_nav.py", ["--policy", "diffusion_grounded"]),
    ("reasoner_direct",             "eval_cosmos_nav.py", ["--policy", "direct"]),
    ("reasoner_diffusion_bare",     "eval_cosmos_nav.py", ["--policy", "diffusion"]),
    ("r2r_av_reasoner",             "eval_edge.py", ["--prompt-template", AV_REASONER + " {instruction}"]),
    ("av_action_only",              "eval_edge.py", ["--prompt-template", AV_ACTION]),
    ("r2r_av_action",               "eval_edge.py", ["--prompt-template", AV_ACTION + " {instruction}"]),
    ("fixed_forward",               "eval_edge.py", ["--prompt-template", "The camera moves forward."]),
    ("r2r_raw",                     "eval_edge.py", ["--prompt-template", "{instruction}"]),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--stop-patience", type=int, default=2)
    ap.add_argument("--arms", nargs="*", default=None, help="subset of arm names")
    args = ap.parse_args()

    stamp = dt.datetime.now().strftime("%m%d_%H%M%S")
    runs = _ROOT / "results" / "runs"; runs.mkdir(parents=True, exist_ok=True)
    table = _ROOT / "results" / "ablations.md"
    logs = _ROOT / "logs" / f"ablation_{stamp}"; logs.mkdir(parents=True, exist_ok=True)

    arms = [a for a in ARMS if not args.arms or a[0] in args.arms]   # ARMS is already in run order
    print(f"{len(arms)} arms, {args.limit} episodes each -> {table}", flush=True)

    records = []
    for name, script, extra in arms:
        jf = runs / f"{stamp}_{name}.json"
        lf = logs / f"{name}.log"
        cmd = [PY, "-u", str(_ROOT / "scripts" / script),
               "--limit", str(args.limit), "--max-steps", str(args.max_steps),
               "--chunk", str(args.chunk), "--steps", str(args.steps),
               "--guidance", str(args.guidance),
               "--stop-patience", str(args.stop_patience),
               "--arm-name", name, "--results-json", str(jf)] + extra
        print(f"\n=== {name} ===\n  {script} {' '.join(extra)[:110]}", flush=True)
        with open(lf, "w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                 cwd=str(_ROOT),
                                 env={**__import__("os").environ,
                                      "PYTHONPATH": "", "HF_HUB_OFFLINE": "1"})
        if rc != 0 or not jf.exists():
            tail = "".join(open(lf).readlines()[-12:])
            print(f"  FAILED rc={rc}\n{tail}", flush=True)
            records.append({"arm": name, "failed": True, "rc": rc})
            continue
        rec = json.loads(jf.read_text())
        records.append(rec)
        print(f"  SR {rec['SR']:.4f}  nDTW {rec['nDTW']:.4f}  NE {rec['NE']:.2f}  "
              f"stop {rec['term_model_stop']}/{rec['n_episodes']}", flush=True)
        write_table(table, stamp, args, records)   # rewrite after every arm

    write_table(table, stamp, args, records)
    print(f"\nDONE -> {table}")
    return 0


def write_table(table, stamp, args, records):
    """Append this batch's table, replacing it if the batch is re-written."""
    header = f"## Cosmos3-Edge zero-shot prompt ablation — {stamp}\n"
    body = [header,
            f"\n`n={args.limit}` val_unseen · chunk {args.chunk} · guidance "
            f"{args.guidance} · {args.steps} steps · stop_patience "
            f"{args.stop_patience} · stock weights, no training\n\n",
            "| arm | SR | SPL | nDTW | NE | stop% | within3m | within5m | "
            "calls/ep | fwd m/chunk | <0.25m | speed m/s |\n",
            "|---|---|---|---|---|---|---|---|---|---|---|---|\n"]
    for r in records:
        if r.get("failed"):
            body.append(f"| {r['arm']} | FAILED rc={r['rc']} | | | | | | | | | | |\n")
            continue
        n = r["n_episodes"]
        body.append(
            f"| {r['arm']} | {r['SR']:.4f} | {r['SPL']:.4f} | {r['nDTW']:.4f} | "
            f"{r['NE']:.2f} | {r['term_model_stop']/n*100:.0f}% | "
            f"{r['within_3m_pct']:.1f}% | {r['within_5m_pct']:.1f}% | "
            f"{r['calls_per_episode']:.1f} | {r['chunk_fwd_mean_m']:.3f} | "
            f"{r['chunk_fwd_below_025_pct']:.0f}% | {r['implied_speed_mps']:.2f} |\n"
            .replace("nan", "–"))
    body.append("\n**Prompt templates**\n\n")
    for r in records:
        if not r.get("failed"):
            body.append(f"- `{r['arm']}`: {r['prompt_template'][:200]}\n")
    body.append("\n---\n\n")

    txt = table.read_text() if table.exists() else "# Ablation results\n\n"
    if header in txt:                       # replace this batch's section
        pre, _, rest = txt.partition(header)
        _, _, after = rest.partition("\n---\n\n")
        txt = pre + "".join(body) + after
    else:
        txt = txt + "".join(body)
    table.write_text(txt)


if __name__ == "__main__":
    raise SystemExit(main())
