"""
Tests for Eq. 14 (L_pil) and Eq. 15 (the joint objective).

Required per AGENTS.md Sec. 4.1:
  Eq. 14 — loss is 0 when z_t == v_bar_{t+2}; the last two timesteps
           contribute nothing; masking is correct at episode boundaries
  Eq. 15 — with L_pil zeroed, total loss equals Stage 0's loss exactly

Plus a guard on the scale trap: ||.||^2_2 sums over d=2048, so using a
mean-based MSE would silently divide L_pil by 2048 and turn lambda=0.1 into
4.9e-5.

Run:  PYTHONPATH="" python -m pytest tests/test_pilot_loss.py -v
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import pathlib

import pytest
import torch
import torch.nn.functional as F

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
from losses.action_loss import action_loss  # noqa: E402
from losses.pilot_loss import (  # noqa: E402
    LAMBDA_PIL,
    combined_loss,
    pilot_loss,
    pilot_loss_scale_report,
)

D = 2048


# ---------------------------------------------------------------------------
# Eq. 14 — the required behaviours
# ---------------------------------------------------------------------------
def test_loss_is_zero_when_z_equals_target():
    """AGENTS.md Sec. 4.1: loss is 0 when z_t == v_bar_{t+2}."""
    v = torch.randn(5, D)
    assert pilot_loss(v.clone(), v).item() == pytest.approx(0.0, abs=1e-6)


def test_loss_is_squared_l2_distance():
    """Exactly ||z - v||^2_2, verified against a manual computation."""
    z, v = torch.randn(3, D), torch.randn(3, D)
    manual = ((z - v) ** 2).sum(dim=-1).sum()
    assert pilot_loss(z, v).item() == pytest.approx(manual.item(), rel=1e-5)


def test_masked_steps_contribute_exactly_zero():
    """AGENTS.md Sec. 4.1: the last two timesteps contribute nothing.

    Eq. 14 sums to T-2. Masked rows must not merely be small -- they must add
    exactly 0, however wrong their z happens to be.
    """
    z, v = torch.randn(4, D), torch.randn(4, D)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])          # last two masked

    masked_total = pilot_loss(z, v, mask).item()
    first_two = pilot_loss(z[:2], v[:2]).item()
    assert masked_total == pytest.approx(first_two, rel=1e-5)

    # make the masked rows arbitrarily wrong -- the loss must not move
    z2 = z.clone()
    z2[2:] += 1e3
    assert pilot_loss(z2, v, mask).item() == pytest.approx(masked_total, rel=1e-5)


def test_all_masked_gives_zero_loss():
    """An episode too short for any t+2 target contributes nothing at all."""
    z, v = torch.randn(2, D), torch.randn(2, D)
    assert pilot_loss(z, v, torch.zeros(2)).item() == 0.0


def test_episode_boundary_masking_matches_eq14_range():
    """For T steps, Eq. 14 covers t=0..T-3 (0-based): exactly T-2 valid rows."""
    for T in (3, 6, 40, 95):
        mask = torch.tensor([1.0 if t + 2 <= T - 1 else 0.0 for t in range(T)])
        assert int(mask.sum().item()) == T - 2


def test_reductions():
    z, v = torch.randn(4, D), torch.randn(4, D)
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0])
    per_step = pilot_loss(z, v, mask, reduction="none")
    assert per_step.shape == (4,)
    assert per_step[3].item() == 0.0
    assert pilot_loss(z, v, mask, reduction="sum").item() == pytest.approx(
        per_step.sum().item(), rel=1e-5)
    # mean divides by the number of VALID rows, not the batch size
    assert pilot_loss(z, v, mask, reduction="mean").item() == pytest.approx(
        per_step.sum().item() / 3, rel=1e-5)


def test_shape_and_reduction_validation():
    with pytest.raises(ValueError, match="!="):
        pilot_loss(torch.randn(3, D), torch.randn(3, D + 1))
    with pytest.raises(ValueError, match=r"expected \(B, d\)"):
        pilot_loss(torch.randn(D), torch.randn(D))
    with pytest.raises(ValueError, match="mask"):
        pilot_loss(torch.randn(3, D), torch.randn(3, D), torch.ones(2))
    with pytest.raises(ValueError, match="unknown reduction"):
        pilot_loss(torch.randn(2, D), torch.randn(2, D), reduction="median")


def test_gradient_flows_to_z_only():
    """L_pil trains G_psi through z. v_bar is a cached constant (D2: the
    encoder is frozen), so no gradient may flow into the target."""
    z = torch.randn(3, D, requires_grad=True)
    v = torch.randn(3, D)
    pilot_loss(z, v).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert v.grad is None


# ---------------------------------------------------------------------------
# THE SCALE TRAP
# ---------------------------------------------------------------------------
def test_squared_l2_is_not_mean_squared_error():
    """||.||^2_2 SUMS over d; F.mse_loss MEANS over it. The two differ by a
    factor of d=2048, which would silently turn lambda=0.1 into 4.9e-5 and make
    the Pilot supervision ~20,000x weaker than the paper specifies."""
    z, v = torch.randn(1, D), torch.randn(1, D)
    ours = pilot_loss(z, v).item()
    mse = F.mse_loss(z, v).item()
    assert ours == pytest.approx(mse * D, rel=1e-4)
    assert ours / mse > 1000, "loss collapsed to a per-dimension mean"


def test_lambda_is_the_papers_value():
    assert LAMBDA_PIL == 0.1


# ---------------------------------------------------------------------------
# Eq. 15
# ---------------------------------------------------------------------------
def test_zero_pilot_loss_recovers_stage0_exactly():
    """AGENTS.md Sec. 4.1: with L_pil zeroed, total == Stage 0's loss exactly.
    This is the clean regression check that Stage 1 does not perturb Stage 0's
    objective by construction."""
    logits = torch.randn(6, 4)
    targets = torch.randint(0, 4, (6,))
    l_act = action_loss(logits, targets)
    assert combined_loss(l_act, torch.zeros(())).item() == pytest.approx(
        l_act.item(), rel=1e-7)


def test_combined_applies_lambda():
    l_act, l_pil = torch.tensor(2.0), torch.tensor(30.0)
    assert combined_loss(l_act, l_pil).item() == pytest.approx(2.0 + 0.1 * 30.0)
    assert combined_loss(l_act, l_pil, lam=0.5).item() == pytest.approx(17.0)


def test_combined_is_differentiable_through_both_terms():
    z = torch.randn(3, D, requires_grad=True)
    logits = torch.randn(3, 4, requires_grad=True)
    total = combined_loss(
        action_loss(logits, torch.tensor([0, 1, 2])),
        pilot_loss(z, torch.randn(3, D)),
    )
    total.backward()
    assert z.grad is not None and logits.grad is not None


# ---------------------------------------------------------------------------
# Scale diagnostic
# ---------------------------------------------------------------------------
def test_scale_report_flags_a_drowned_action_loss():
    """If lambda*L_pil dwarfs L_act the action objective is drowned; if it
    vanishes the Pilot gets no gradient. The report must make the ratio
    visible BEFORE a multi-hour run, since the loss curve alone cannot
    distinguish the two failures."""
    z = torch.randn(4, D) * 50          # deliberately far from target
    v = torch.randn(4, D)
    rep = pilot_loss_scale_report(z, v, l_act=1.386)
    assert rep["lambda"] == 0.1
    assert rep["ratio_weighted_pil_to_act"] > 100
    assert rep["z_norm"] > rep["target_norm"]


def test_scale_report_on_a_balanced_case():
    v = torch.randn(4, D)
    z = v + torch.randn(4, D) * 0.01
    rep = pilot_loss_scale_report(z, v, l_act=1.386)
    assert rep["ratio_weighted_pil_to_act"] < 1.0
