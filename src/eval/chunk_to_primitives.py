"""Bridge Cosmos 3 action chunks back to habitat's discrete primitives.

Edge emits a chunk of 9D ego-pose deltas at 15 FPS. Habitat executes FWD
(0.25 m), LEFT / RIGHT (15 deg). To keep SR / SPL / nDTW comparable with every
number we have measured, the chunk is replayed into primitives rather than the
harness being changed.

Ordering matters: a chunk that turns then walks is a different trajectory from
one that walks then turns, so we integrate frame by frame and emit a primitive
whenever an accumulator crosses its threshold, instead of summing the whole
chunk and emitting turns-then-forwards.

Translations arrive in the model's own unit (fitted s = 7.47 against metres,
see cosmos_dataset), so they are divided back to metres before comparison with
the 0.25 m step.
"""
import math

import numpy as np

FORWARD_STEP_M = 0.25
TURN_DEG = 15.0


def _yaw_deg(a9, rot_from_6d):
    R = rot_from_6d(np.asarray(a9[3:], dtype=np.float64))
    return math.degrees(math.atan2(R[0, 2], R[2, 2]))


def chunk_to_primitives(chunk, rot_from_6d, translation_scale=7.47,
                        max_primitives=None):
    """(T, 9) chunk -> ordered list of 'move_forward' / 'turn_left' / 'turn_right'.

    Returns (actions, net_forward_m, net_yaw_deg).
    """
    acc_yaw = acc_fwd = 0.0
    net_fwd = net_yaw = 0.0
    out = []
    for a in np.asarray(chunk, dtype=np.float64):
        dy = _yaw_deg(a, rot_from_6d)
        dz = float(a[2]) / translation_scale        # model unit -> metres
        acc_yaw += dy
        acc_fwd += dz
        net_yaw += dy
        net_fwd += dz
        while acc_yaw >= TURN_DEG:
            out.append("turn_right"); acc_yaw -= TURN_DEG      # +yaw = right
        while acc_yaw <= -TURN_DEG:
            out.append("turn_left"); acc_yaw += TURN_DEG
        while acc_fwd >= FORWARD_STEP_M:
            out.append("move_forward"); acc_fwd -= FORWARD_STEP_M
        if max_primitives and len(out) >= max_primitives:
            return out[:max_primitives], net_fwd, net_yaw
    return out, net_fwd, net_yaw
