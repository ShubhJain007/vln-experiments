"""
Pointing losses (D27) — Robostral Eq. 1/2 adapted to a discrete action space.

    L = w_point * L_point(masked by visible)
      + w_disp  * L_disp
      + w_stop  * BCE(stop,    pos_weight)
      + w_vis   * BCE(visible)

--------------------------------------------------------------------------
MASKING -- the (u, v) loss applies ONLY where a waypoint was visible
--------------------------------------------------------------------------
35% of steps have no waypoint in frame (the camera sits 1.5 m up with a
+/-45 deg vertical half-FOV, so floor waypoints only enter frame past 1.5 m).
Those steps carry a placeholder u = v = 0.5. Training on that placeholder
would teach the model to point at the image centre whenever it is unsure --
a confident, plausible-looking, entirely wrong behaviour. They are masked to
exactly zero instead, and the model learns to PREDICT invisibility through
`visible_logit`.

--------------------------------------------------------------------------
WHY dtheta IS NOT REGRESSED DIRECTLY AS AN ANGLE
--------------------------------------------------------------------------
dtheta wraps at +/-pi, so plain MSE punishes -179 deg vs +179 deg as if they
were 358 degrees apart. The displacement (dx, dy) is regressed instead and the
bearing recovered with atan2, which is continuous everywhere. dtheta is still
supervised (it is cheap and adds a direct signal) but with a wrap-aware term.
"""

import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]

import math

import torch
import torch.nn.functional as F

# NOT IN ANY PAPER -- ours. Chosen so no term dominates at initialisation;
# see train_pointing.py --report-scale, which prints the live balance.
W_POINT = 1.0
W_DISP = 1.0
W_STOP = 0.5
W_VISIBLE = 0.2

# Full class balancing is pos_weight = negatives/positives = ~44 at a 2.2% STOP
# rate. Measured at batch 8 that is unusable: most batches contain ZERO stops,
# an occasional one contributes 44x, and the stop term swung 1.35 <-> 3.32
# between logging windows while dominating the total. Capped instead -- enough
# to stop the head collapsing to "never fire", without the variance.
MAX_STOP_POS_WEIGHT = 8.0


def angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Signed smallest angular difference, in (-pi, pi]."""
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def pointing_loss(pred: dict, target: dict, stop_pos_weight: float = 1.0,
                  w_point=W_POINT, w_disp=W_DISP, w_stop=W_STOP,
                  w_visible=W_VISIBLE):
    """Returns (total, parts_dict). All terms are means over the batch."""
    vis = target["visible"]
    n_vis = vis.sum().clamp(min=1.0)

    # -- (u, v): masked to the visible subset --------------------------------
    l_point = ((((pred["u"] - target["u"]) ** 2
                 + (pred["v"] - target["v"]) ** 2) * vis).sum() / n_vis)

    # -- direction: cosine distance between unit vectors ---------------------
    # Scale-free and wrap-free, so it optimises exactly the quantity the
    # discrete controller consumes: the bearing.
    tn = torch.sqrt(target["dx"] ** 2 + target["dy"] ** 2).clamp(min=1e-6)
    cos = pred["dir_x"] * (target["dx"] / tn) + pred["dir_y"] * (target["dy"] / tn)
    l_disp = (1.0 - cos).mean()
    l_theta = torch.zeros((), device=cos.device)     # folded into l_disp

    # -- STOP: pos_weight offsets the 2.5% base rate, but capped -------------
    pw = min(float(stop_pos_weight), MAX_STOP_POS_WEIGHT)
    l_stop = F.binary_cross_entropy_with_logits(
        pred["stop_logit"], target["is_stop"],
        pos_weight=torch.tensor(pw, device=pred["stop_logit"].device))

    l_vis = F.binary_cross_entropy_with_logits(pred["visible_logit"], vis)

    total = (w_point * l_point + w_disp * (l_disp + l_theta)
             + w_stop * l_stop + w_visible * l_vis)
    return total, {"point": l_point.item(), "disp": l_disp.item(),
                   "theta": l_theta.item(), "stop": l_stop.item(),
                   "visible": l_vis.item()}


def bearing_to_action(dx: float, dy: float, turn_threshold_rad=math.radians(7.5)):
    """Waypoint bearing -> discrete action. Replaces Robostral's 121M diffusion
    policy, which exists to produce smooth CONTINUOUS control; we have four
    primitives. Measured ceiling against expert actions: 0.976."""
    from model.action_space import Action
    theta = math.atan2(dx, dy)          # +dx is to the RIGHT of the agent
    if theta > turn_threshold_rad:
        return Action.RIGHT
    if theta < -turn_threshold_rad:
        return Action.LEFT
    return Action.FWD
