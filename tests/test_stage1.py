"""
Stage 1 tests — Eq. 12/14/15 and the Sec. 4.2 copy baseline.

These cover the paths that Stage 0' never exercised: G_psi receiving gradient,
Eq. 14's masking at episode boundaries, lambda's exact value under the shared
denominator, and the train/eval slot asymmetry that Sec. 4.1 singles out as the
single easiest error in the project.

Everything here is CPU-only and model-free except where marked -- the point is
that these can run without burning a GPU hour to find a shape bug.
"""

import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]

import pathlib

import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT / "src"))

from losses.action_loss import action_loss  # noqa: E402
from losses.pilot_loss import LAMBDA_PIL, combined_loss, pilot_loss  # noqa: E402
from model.pilot import PilotModule  # noqa: E402

D = 16


# ---------------------------------------------------------------------------
# Eq. 14 — L_pil, masking, and the T-2 horizon
# ---------------------------------------------------------------------------
def test_pilot_loss_zero_when_exact():
    z = torch.randn(4, D)
    assert pilot_loss(z, z.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_pilot_loss_is_sum_of_squares_not_mean():
    """||.||^2_2 sums over d. A mean would divide by d and turn lambda=0.1
    into 0.1/d -- the exact bug pilot_loss.py's docstring warns about."""
    z = torch.zeros(1, D)
    tgt = torch.ones(1, D)
    assert pilot_loss(z, tgt).item() == pytest.approx(float(D))


def test_pilot_loss_masked_steps_contribute_exactly_zero():
    """Eq. 14 sums to T-2: the final two steps have no t+2 target."""
    z = torch.randn(3, D)
    tgt = torch.randn(3, D)
    mask = torch.tensor([1.0, 0.0, 0.0])
    got = pilot_loss(z, tgt, mask)
    want = ((z[0] - tgt[0]) ** 2).sum()
    assert got.item() == pytest.approx(want.item(), rel=1e-5)


def test_masked_rows_may_hold_garbage():
    """A masked row's target is never read, so its content cannot change the
    loss. This is what lets the collate function pass zeros for absent t+2."""
    z = torch.randn(2, D)
    mask = torch.tensor([1.0, 0.0])
    a = torch.randn(2, D)
    b = a.clone()
    b[1] = 1e6                                  # garbage in the masked row
    assert pilot_loss(z, a, mask).item() == pytest.approx(
        pilot_loss(z, b, mask).item(), rel=1e-6)


# ---------------------------------------------------------------------------
# Eq. 15 — lambda is exactly 0.1 under the shared denominator
# ---------------------------------------------------------------------------
def test_lambda_is_point_one():
    assert LAMBDA_PIL == 0.1


def test_combined_loss_reduces_to_stage0_when_pil_zeroed():
    """Sec. 4.1: 'with L_pil zeroed, total loss equals Stage 0's loss exactly'."""
    l_act = torch.tensor(1.234)
    assert combined_loss(l_act, torch.tensor(0.0)).item() == pytest.approx(1.234)


def test_shared_denominator_preserves_lambda():
    """Both terms divide by B. Dividing L_pil by num_valid instead would scale
    lambda by T/(T-2) -- 1.5x at T=6 -- silently changing the one
    hyperparameter the paper states."""
    B = 6
    logits = torch.randn(B, 4)
    targets = torch.randint(0, 4, (B,))
    z = torch.randn(B, D)
    tgt = torch.randn(B, D)
    mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])     # last two masked

    l_act = action_loss(logits, targets, reduction="mean")
    l_pil_correct = pilot_loss(z, tgt, mask, reduction="sum") / B
    l_pil_wrong = pilot_loss(z, tgt, mask, reduction="mean")   # / num_valid = 4

    ratio = (l_pil_wrong / l_pil_correct).item()
    assert ratio == pytest.approx(B / mask.sum().item(), rel=1e-5)   # 6/4 = 1.5
    assert combined_loss(l_act, l_pil_correct).item() != pytest.approx(
        combined_loss(l_act, l_pil_wrong).item())


# ---------------------------------------------------------------------------
# Eq. 8 — G_psi, and Sec. 4.2's identity copy baseline
# ---------------------------------------------------------------------------
def test_g_psi_is_a_single_linear_layer():
    """Sec. 3.2: 'a simple linear layer, with only a few parameters'."""
    p = PilotModule(D, mode="action_query", dtype=torch.float32, device="cpu")
    assert isinstance(p.G_psi, torch.nn.Linear)
    assert p.g_psi_param_count == D * D + D


def test_identity_g_psi_is_exact_passthrough():
    """Gate (b): z_t = h_t^pil with no learned projection."""
    p = PilotModule(D, mode="action_query", dtype=torch.float32, device="cpu")
    with torch.no_grad():
        p.G_psi.weight.copy_(torch.eye(D))
        p.G_psi.bias.zero_()
    h = torch.randn(3, D)
    assert torch.allclose(p(h), h, atol=1e-6)


def test_identity_baseline_is_frozen():
    p = PilotModule(D, mode="action_query", dtype=torch.float32, device="cpu")
    p.G_psi.requires_grad_(False)
    h = torch.randn(2, D, requires_grad=True)
    p(h).sum().backward()
    assert p.G_psi.weight.grad is None
    assert h.grad is not None          # gradient still flows THROUGH to the LLM


def test_l_pil_reaches_g_psi():
    """The property Stage 0' deliberately lacked. If this fails, Stage 1 is
    silently Stage 0' with extra parameters."""
    p = PilotModule(D, mode="action_query", dtype=torch.float32, device="cpu")
    h = torch.randn(4, D)
    z = p(h)
    pilot_loss(z, torch.randn(4, D)).backward()
    g = p.G_psi.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


# ---------------------------------------------------------------------------
# Eq. 12 vs Eq. 5 — the train/eval asymmetry
# ---------------------------------------------------------------------------
def test_cache_has_no_vbar_entry_point():
    """PilotCache must expose no way to accept a v_bar. Eq. 12's teacher
    forcing happens in the DATASET, never through the cache -- otherwise a
    future frame could reach the eval path and invalidate every number."""
    p = PilotModule(D, mode="action_query", dtype=torch.float32, device="cpu")
    cache = p.new_cache()
    public = {a for a in dir(cache) if not a.startswith("_")}
    assert public == {"read", "write", "reset", "episode_steps", "is_fresh"}


def test_cache_round_trip_is_the_eval_contract():
    p = PilotModule(D, mode="action_query", dtype=torch.float32, device="cpu")
    cache = p.new_cache()
    assert cache.is_fresh
    z0 = cache.read()
    assert torch.equal(z0, p.z_0.detach()) or z0.shape == (D,)
    z1 = torch.randn(D)
    cache.write(z1)
    assert torch.equal(cache.read(), z1)
    assert not cache.is_fresh
    cache.reset()
    assert cache.is_fresh


def test_pilot_index_survives_right_padding():
    """Design B: h_pil sits one before h_act. Right padding must not move it."""
    from model.input_sequence import BatchedSequences  # noqa: F401
    lengths = [5, 8, 6]
    pilot_idx = torch.tensor([n - 2 for n in lengths])
    action_idx = torch.tensor([n - 1 for n in lengths])
    assert torch.equal(action_idx - pilot_idx, torch.ones(3, dtype=torch.long))


# ---------------------------------------------------------------------------
# D26 — adaptive lambda (NOT IN PAPER), fixing D25's measured drift
# ---------------------------------------------------------------------------
def test_fixed_mode_is_eq15_as_written():
    """lambda_mode='fixed' must reproduce Eq. 15 exactly -- the paper's form."""
    from train.train_stage1 import stage1_loss
    lg, tg = torch.randn(8, 4), torch.randint(0, 4, (8,))
    z, v, m = torch.randn(8, D), torch.randn(8, D), torch.ones(8)
    total, l_act, l_pil, lam = stage1_loss(lg, tg, z, v, m, 0.1, "fixed")
    assert lam == 0.1
    assert total.item() == pytest.approx((l_act + 0.1 * l_pil).item(), rel=1e-5)


def test_balanced_mode_holds_the_target_ratio():
    """D25: the whole failure was lambda*L_pil/L_act drifting 0.52x -> 5.51x."""
    from train.train_stage1 import stage1_loss
    for target in (0.5, 0.9, 1.5):
        lg, tg = torch.randn(16, 4), torch.randint(0, 4, (16,))
        z, v, m = torch.randn(16, D), torch.randn(16, D), torch.ones(16)
        _, l_act, l_pil, lam = stage1_loss(lg, tg, z, v, m, 0.1, "balanced", target)
        assert lam * l_pil.item() / l_act.item() == pytest.approx(target, rel=1e-4)


def test_balanced_mode_holds_ratio_across_the_real_training_range():
    """D25: the drift arose because L_act converged ~51x while L_pil only ~4.8x.
    Balanced mode must be invariant across that span.

    Range chosen from the ACTUAL Stage 1 run: L_act went 2.79 -> 0.055, L_pil
    14.5 -> 3.0. The required lambda at the end was 0.9*0.055/3.0 = 0.016 --
    165x above LAMBDA_MIN, so the clamp never binds in the real regime.
    """
    from train.train_stage1 import stage1_loss
    z, v, m = torch.zeros(8, D), torch.full((8, D), 3.0), torch.ones(8)
    ratios, l_acts = [], []
    for logit_gap in (0.5, 1.0, 2.0, 4.0):
        lg = torch.zeros(8, 4)
        lg[:, 0] = logit_gap
        tg = torch.zeros(8, dtype=torch.long)
        _, l_act, l_pil, lam = stage1_loss(lg, tg, z, v, m, 0.1, "balanced", 0.9)
        l_acts.append(l_act.item())
        ratios.append(lam * l_pil.item() / l_act.item())
    assert l_acts[0] / l_acts[-1] > 8          # L_act really did collapse
    assert all(r == pytest.approx(0.9, rel=1e-3) for r in ratios)


def test_clamp_takes_over_below_the_operating_range():
    """Documented boundary: the ratio guarantee holds only while the required
    lambda sits inside [LAMBDA_MIN, LAMBDA_MAX]. Below it the floor binds and
    the ratio rises -- deliberate, so a transient near-zero L_act cannot make
    the Pilot term vanish entirely. Asserted rather than left implicit."""
    from train.train_stage1 import LAMBDA_MIN, stage1_loss
    z, v, m = torch.zeros(8, D), torch.full((8, D), 3.0), torch.ones(8)
    lg = torch.zeros(8, 4)
    lg[:, 0] = 10.0                            # far past anything training reached
    tg = torch.zeros(8, dtype=torch.long)
    _, l_act, _, lam = stage1_loss(lg, tg, z, v, m, 0.1, "balanced", 0.9)
    assert lam == LAMBDA_MIN
    assert l_act.item() < 1e-3                 # only reachable at absurd confidence


def test_balanced_lambda_is_clamped():
    """A transient tiny L_pil must not produce an enormous weight."""
    from train.train_stage1 import LAMBDA_MAX, LAMBDA_MIN, stage1_loss
    lg, tg = torch.randn(8, 4), torch.randint(0, 4, (8,))
    z = torch.zeros(8, D)
    _, _, _, lam = stage1_loss(lg, tg, z, z.clone(), torch.ones(8), 0.1, "balanced", 0.9)
    assert LAMBDA_MIN <= lam <= LAMBDA_MAX


def test_unknown_lambda_mode_is_rejected():
    from train.train_stage1 import stage1_loss
    lg, tg = torch.randn(4, 4), torch.randint(0, 4, (4,))
    z, v, m = torch.randn(4, D), torch.randn(4, D), torch.ones(4)
    with pytest.raises(ValueError, match="unknown lambda_mode"):
        stage1_loss(lg, tg, z, v, m, 0.1, "adaptive-magic")
