"""Decision-point dataset for SFT of the Cosmos3-Edge reasoner as an R2R policy.

One sample = (last CTX frames of the robot's view, R2R instruction) -> one
command in habitat's vocabulary. Labels come free from action9d.npy (Cosmos
frame, +yaw = right, translation in metres), at the same 1 s / 20 deg
granularity the offline probe scores at, so train and eval agree on what a
"turn" is.

Label rule (deliberately different from probe v2's in one place):
    t within the last STOP_TAIL frames of the trajectory -> "stop"
    net yaw over the next HORIZON frames >  20 deg       -> "turn right"
    net yaw over the next HORIZON frames < -20 deg       -> "turn left"
    otherwise                                             -> "move forward"
Probe v2 also called low-motion mid-trajectory windows "stop"; that swept slow
manoeuvring into the stop class (24 of 96 labels) and is not what we want the
policy to learn. Here STOP means "the instruction is complete", full stop.

Sampling is class-balanced by construction rather than by loss weighting: per
trajectory we take turn onsets, a few forwards, and the end. The zero-shot
reasoner's worst failures were stop (recall 0.08) and invented turns, so the
stop and turn classes are what the data has to teach.
"""
import json
import math
import pathlib
import random

import numpy as np

from nav.cosmos_prompts import VIDEO_SPAN

CTX, HORIZON, FPS = VIDEO_SPAN, 16, 15.0
STOP_TAIL = 6                 # frames from the end that count as "stop"
TURN_DEG = 20.0
COMMANDS = ("move forward", "turn left", "turn right", "stop")


def _yaw_deg(a9, rot_from_6d):
    R = rot_from_6d(np.asarray(a9[3:], dtype=np.float64))
    return math.degrees(math.atan2(R[0, 2], R[2, 2]))


def label_at(actions, t, rot_from_6d):
    n = len(actions)
    if t >= n - STOP_TAIL:
        return "stop"
    yaw = sum(_yaw_deg(a, rot_from_6d) for a in actions[t:t + HORIZON])
    if yaw > TURN_DEG:
        return "turn right"
    if yaw < -TURN_DEG:
        return "turn left"
    return "move forward"


def build_index(root, split, rot_from_6d, per_traj=8, seed=0, max_traj=None,
                uniform=True, per_traj_turns=4, per_traj_fwd=3, per_traj_stop=1):
    """-> list of dicts {dir, t, label, instr_idx}. Cached by the caller.

    uniform=True samples decision points UNIFORMLY over each trajectory, so the
    label distribution is the natural one the policy meets in a rollout
    (~70% forward). This is not a detail: class-balancing by over-sampling turn
    onsets (uniform=False, the first run) taught a turn-heavy prior that scored
    0.656 offline and SR 0.000 in closed loop -- it spun in place, issuing turns
    71% of the time against a true rate of ~25%. The prior selects WHICH failure
    you get, and a rollout only forgives the natural one.
    """
    rng = random.Random(seed)
    items = []
    dirs = sorted(pathlib.Path(root, split).glob("*/traj*"))
    if max_traj:
        dirs = dirs[:max_traj]
    for d in dirs:
        meta = json.loads((d / "meta.json").read_text())
        acts = np.load(d / "action9d.npy")
        n = len(acts)
        if n < HORIZON + STOP_TAIL + 2:      # short clips are padded, not skipped
            continue
        if uniform:
            # every step is an equally likely decision point, exactly as in a
            # rollout; the trajectory end is included so STOP still appears at
            # its natural rate rather than being engineered in.
            picks = set(rng.sample(range(1, n), min(per_traj, n - 1)))
        else:
            yaws = np.array([abs(_yaw_deg(a, rot_from_6d)) for a in acts])
            onsets = [t for t in range(1, n - STOP_TAIL - HORIZON)
                      if yaws[t:t + 4].sum() > 15 and yaws[max(0, t - 4):t].sum() < 4]
            picks = set(rng.sample(onsets, min(len(onsets), per_traj_turns)))
            fwd_pool = [t for t in range(1, n - STOP_TAIL - HORIZON) if t not in picks]
            picks |= set(rng.sample(fwd_pool, min(len(fwd_pool), per_traj_fwd)))
            picks |= set(rng.sample(range(n - STOP_TAIL, n), min(STOP_TAIL, per_traj_stop)))
        for t in sorted(picks):
            # each of the ~3 instructions per path is a separate sample
            for i in range(len(meta["instructions"])):
                items.append(dict(dir=str(d), t=int(t), label=label_at(acts, t, rot_from_6d),
                                  instr_idx=i))
    rng.shuffle(items)
    return items


def load_clip(d, t):
    """CTX frames ending at t (inclusive); the start is padded by repeating frame 0."""
    import imageio.v3 as iio
    frames = iio.imread(pathlib.Path(d) / "frames.mp4", plugin="pyav")
    lo = max(0, t - CTX + 1)
    clip = frames[lo:t + 1]
    if len(clip) < CTX:
        clip = np.concatenate([np.repeat(clip[:1], CTX - len(clip), 0), clip])
    return np.asarray(clip)


class ReasonerSFTDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        meta = json.loads((pathlib.Path(it["dir"]) / "meta.json").read_text())
        return dict(clip=load_clip(it["dir"], it["t"]),
                    instruction=meta["instructions"][it["instr_idx"]],
                    label=it["label"])
