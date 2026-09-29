"""
Eq. 13 — Action imitation loss.

    p_theta(.) = Softmax(W_LM h_t^act)

    L_act  =  - sum_{t=1}^{T} log p_theta( a^col_t | x, o_t, v_bar_{t+1} )

Cross-entropy of the collected expert action a^col_t (Eq. 10) under the action
distribution from Eq. 7. In Stage 0 there is no Pilot slot, so the conditioning
is just (x, o_t); the loss itself is unchanged.

--------------------------------------------------------------------------
REDUCTION -- the paper specifies this exactly; do not "improve" it
--------------------------------------------------------------------------
The paper leaves no room here:

    Eq. 13   L_act = - sum_{t=1}^{T}   log p_theta(...)      <- sum, range T
    Eq. 14   L_pil =   sum_{t=1}^{T-2} ||z_t - v_bar||^2     <- sum, range T-2
    Eq. 15   min E_{tau ~ D} [ L_act(tau) + lambda L_pil(tau) ],  lambda = 0.1

So the reduction is: SUM over timesteps WITHIN a trajectory, then MEAN across
trajectories (that is what E_{tau ~ D} denotes). Hence `reduction="sum"` is the
default here -- this function returns one trajectory's L_act(tau), and the
training loop averages those per-trajectory totals over the batch.

Why this is not a free choice: using "mean" within a trajectory divides L_act
by T and L_pil by T-2, which are DIFFERENT. The effective weight silently
becomes lambda * T/(T-2) instead of lambda -- roughly 0.107 at T=30, but 0.15
at T=6, a 50% error in the one hyperparameter the paper actually states.

Summed loss does make gradient magnitude scale with episode length, so longer
trajectories carry more weight. That is the paper's intent (more decisions =
more supervision), not an artefact to normalise away. Absorb the overall scale
into the learning rate instead.

`reduction="mean"` remains available for diagnostics (e.g. reporting a
per-step loss comparable across episodes) but must NOT be used for training.
"""

# --- ROS guard: strip /opt/ros/* from sys.path before any other import. ------
import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]
# ----------------------------------------------------------------------------

import torch
import torch.nn.functional as F


def action_loss(
    action_logits: torch.Tensor,
    target_actions: torch.Tensor,
    reduction: str = "sum",
) -> torch.Tensor:
    """Eq. 13:  L_act(tau) = - sum_{t=1}^{T} log p_theta(a^col_t | .)

    Returns ONE TRAJECTORY's action loss. Eq. 15's E_{tau ~ D} is applied by
    the caller, by averaging these per-trajectory values across the batch --
    see `batch_action_loss`.

    Args:
        action_logits:  (T, 4) or (4,) -- W_LM h_t^act restricted to the action
                        tokens, in canonical ACTIONS order. Pass LOGITS, not
                        probabilities; cross_entropy applies log-softmax itself
                        and doing it twice would silently flatten the loss.
        target_actions: (T,) or scalar -- expert action indices a^col_t.
        reduction:      "sum" (default; the equation as written), "mean"
                        (DIAGNOSTICS ONLY -- see module docstring, it corrupts
                        lambda), or "none" for per-step losses.

    Returns:
        Scalar loss, or (T,) when reduction="none".
    """
    if action_logits.ndim == 1:
        action_logits = action_logits.unsqueeze(0)
    if target_actions.ndim == 0:
        target_actions = target_actions.unsqueeze(0)

    if action_logits.shape[0] != target_actions.shape[0]:
        raise ValueError(
            f"got {action_logits.shape[0]} logit rows but "
            f"{target_actions.shape[0]} targets"
        )

    # float32 for the loss: bf16 log-softmax loses precision precisely where
    # the loss is small, which is where the gradient signal matters most.
    return F.cross_entropy(
        action_logits.float(),
        target_actions.to(action_logits.device).long(),
        reduction=reduction,
    )


def batch_action_loss(
    per_trajectory_logits, per_trajectory_targets
) -> torch.Tensor:
    """Eq. 15's E_{tau ~ D} applied to L_act.

    Sums within each trajectory (Eq. 13) and averages those totals across the
    batch. Longer trajectories therefore contribute more, which is the paper's
    intent -- more decisions means more supervision -- and is exactly what
    per-step normalisation would erase.

    Args:
        per_trajectory_logits:  list of (T_i, 4) tensors, one per trajectory.
        per_trajectory_targets: list of (T_i,) tensors of expert actions.

    Returns:
        Scalar: mean over trajectories of each trajectory's summed loss.
    """
    if len(per_trajectory_logits) != len(per_trajectory_targets):
        raise ValueError(
            f"{len(per_trajectory_logits)} logit trajectories but "
            f"{len(per_trajectory_targets)} target trajectories"
        )
    if not per_trajectory_logits:
        raise ValueError("empty batch")

    totals = torch.stack([
        action_loss(lg, tg, reduction="sum")
        for lg, tg in zip(per_trajectory_logits, per_trajectory_targets)
    ])
    return totals.mean()
