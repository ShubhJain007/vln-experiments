"""
Eq. 14 — Pilot loss (future-privileged supervision), and Eq. 15's combination.

    z_t   = G_psi(h_t^pil)  in  R^d
    L_pil = sum_{t=1}^{T-2} || z_t - v_bar_{t+2} ||^2_2

    L     = L_act + lambda * L_pil,      lambda = 0.1   (Eq. 15, stated)

--------------------------------------------------------------------------
||.||^2_2 IS A SUM OF SQUARES, NOT A MEAN — and lambda depends on it
--------------------------------------------------------------------------
The squared L2 norm sums over all d=2048 dimensions. Using torch's default
`mse_loss`, which MEANS over dimensions, would divide L_pil by 2048 and
silently turn the paper's lambda=0.1 into an effective 0.1/2048 = 4.9e-5 --
the Pilot supervision would be ~20,000x weaker than intended and would look
like "the Pilot Token doesn't help".

This module therefore sums over the feature dimension explicitly, and
`pilot_loss_scale_report` exists so the actual magnitude can be checked against
L_act before committing to a training run. lambda is only meaningful if the two
terms are within a sane ratio of each other.

--------------------------------------------------------------------------
MASKING
--------------------------------------------------------------------------
Eq. 14's sum stops at T-2 because "the last two steps have no t+2". Steps
without a target contribute EXACTLY ZERO -- they are not dropped, not zero-
filled, and never divided into the mean. See PilotRolloutDataset for the
indexing.
"""

import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]

from typing import Optional

import torch

# Eq. 15, stated explicitly in Sec. 3.3: "we set 0.1 in practice".
LAMBDA_PIL = 0.1


def pilot_loss(
    z: torch.Tensor,
    v_bar_target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "sum",
) -> torch.Tensor:
    """Eq. 14:  L_pil = sum_t || z_t - v_bar_{t+2} ||^2_2

    Args:
        z:            (B, d) predicted Pilot Tokens, G_psi(h_t^pil).
        v_bar_target: (B, d) cached v_bar_{t+2}.
        mask:         (B,) 1.0 where a t+2 target exists, 0.0 otherwise.
                      None means every row has a target.
        reduction:    "sum"  -- Eq. 14 as written (one trajectory's L_pil)
                      "mean" -- per-VALID-step average, for diagnostics
                      "none" -- (B,) per-step squared distances

    Returns:
        Scalar, or (B,) when reduction="none".
    """
    if z.shape != v_bar_target.shape:
        raise ValueError(
            f"z {tuple(z.shape)} != target {tuple(v_bar_target.shape)}"
        )
    if z.ndim != 2:
        raise ValueError(f"expected (B, d), got {tuple(z.shape)}")

    # float32: the squared distance is summed over 2048 dims, and bf16 loses
    # precision exactly where the loss is small.
    diff = z.float() - v_bar_target.float()
    per_step = (diff * diff).sum(dim=-1)             # ||.||^2_2, NOT a mean

    if mask is not None:
        if mask.shape != per_step.shape:
            raise ValueError(
                f"mask {tuple(mask.shape)} != batch {tuple(per_step.shape)}"
            )
        per_step = per_step * mask.float()

    if reduction == "none":
        return per_step
    if reduction == "sum":
        return per_step.sum()
    if reduction == "mean":
        denom = mask.float().sum().clamp(min=1.0) if mask is not None \
            else torch.tensor(float(per_step.numel()), device=per_step.device)
        return per_step.sum() / denom
    raise ValueError(f"unknown reduction {reduction!r}")


def combined_loss(
    l_act: torch.Tensor,
    l_pil: torch.Tensor,
    lam: float = LAMBDA_PIL,
) -> torch.Tensor:
    """Eq. 15:  L = L_act + lambda * L_pil

    Both terms must use the SAME reduction (D9). Eq. 13 sums to T and Eq. 14 to
    T-2, so mixing "mean" into one and "sum" into the other rescales lambda by
    T/(T-2) -- 1.5x at T=6.
    """
    return l_act + lam * l_pil


@torch.no_grad()
def pilot_loss_scale_report(z, v_bar_target, mask=None, l_act=None,
                            lam: float = LAMBDA_PIL) -> dict:
    """Diagnostic: is lambda=0.1 actually balancing these two terms?

    Run this BEFORE a training run. If lambda * L_pil dwarfs L_act, the action
    objective is drowned and SR collapses; if it vanishes, the Pilot Token gets
    no gradient and Stage 1 reduces to Stage 0' with extra parameters. Either
    way the run is wasted, and the loss curve alone will not say which.
    """
    per_step = pilot_loss(z, v_bar_target, mask, reduction="none")
    valid = per_step[per_step > 0] if mask is None else per_step[mask.bool()]
    report = {
        "l_pil_mean_per_step": valid.mean().item() if valid.numel() else 0.0,
        "l_pil_weighted": lam * (valid.mean().item() if valid.numel() else 0.0),
        "z_norm": z.float().norm(dim=-1).mean().item(),
        "target_norm": v_bar_target.float().norm(dim=-1).mean().item(),
        "lambda": lam,
    }
    if l_act is not None:
        act = float(l_act)
        report["l_act"] = act
        report["ratio_weighted_pil_to_act"] = (
            report["l_pil_weighted"] / act if act else float("inf")
        )
    return report
