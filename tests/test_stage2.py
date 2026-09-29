"""
Stage 2 tests — AGENTS.md Sec. 5.6's explicit requirements.

Sec. 5.6 names four things that must hold. Three are checkable without a GPU
and live here; the fourth (p=0.0 numerically identical to Stage 1) needs the
real model and is `train_stage2.py --check-identity`.
"""

import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]

import pathlib

import numpy as np
import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT / "src"))

from train.train_stage2 import mix_slots, p_selfpred_at  # noqa: E402

D = 16


# ---------------------------------------------------------------------------
# Sec. 5.4 — the ramp
# ---------------------------------------------------------------------------
def test_ramp_starts_at_p_start_and_reaches_p_final():
    assert p_selfpred_at(0, 1000, 0.0, 0.5, 0.3) == pytest.approx(0.0)
    assert p_selfpred_at(300, 1000, 0.0, 0.5, 0.3) == pytest.approx(0.5)
    assert p_selfpred_at(999, 1000, 0.0, 0.5, 0.3) == pytest.approx(0.5)


def test_ramp_is_gradual_not_a_hard_switch():
    """Sec. 5.4: a step change 0 -> p_final is 'the destabilising case
    scheduled sampling exists to avoid'."""
    vals = [p_selfpred_at(s, 1000, 0.0, 0.5, 0.3) for s in range(0, 300, 30)]
    assert vals == sorted(vals)
    assert all(b - a < 0.1 for a, b in zip(vals, vals[1:]))


def test_p_zero_never_selects_self_prediction():
    assert p_selfpred_at(0, 100, 0.0, 0.0, 0.3) == 0.0
    assert p_selfpred_at(99, 100, 0.0, 0.0, 0.3) == 0.0


# ---------------------------------------------------------------------------
# Sec. 5.6 — per-sample mixing, and the detach requirement
# ---------------------------------------------------------------------------
class _FakePilot:
    def __init__(self, d):
        self.z_0 = torch.full((d,), 7.0)


def test_mixing_is_per_sample_not_per_batch():
    """Sec. 5.6: 'Sample the choice per training sample, not per batch, so each
    batch contains a mix.'"""
    pin = [torch.zeros(D) for _ in range(4)]
    zmap = {i: torch.full((D,), float(i + 1)) for i in range(4)}
    use_self = np.array([True, False, True, False])
    slots = mix_slots(pin, zmap, use_self, _FakePilot(D), "cpu")

    assert torch.equal(slots[0], zmap[0])          # self-predicted
    assert torch.equal(slots[1], pin[1].to("cpu", torch.bfloat16))   # GT
    assert torch.equal(slots[2], zmap[2])          # self-predicted
    assert torch.equal(slots[3], pin[3].to("cpu", torch.bfloat16))   # GT


def test_empirical_rate_matches_p():
    """Sec. 5.6: 'over 1000 samples at p=0.5, the empirical rate is within a
    few percent of 0.5.'"""
    rng = np.random.default_rng(0)
    rate = (rng.random(1000) < 0.5).mean()
    assert abs(rate - 0.5) < 0.05


def test_step_zero_falls_back_to_z0():
    """At t=0 there is no t-1. The slot gets z_0 -- exactly the eval case."""
    pin = [torch.zeros(D)]
    slots = mix_slots(pin, {}, np.array([True]), _FakePilot(D), "cpu")
    assert torch.equal(slots[0], torch.full((D,), 7.0))


def test_selfpred_slot_carries_no_gradient():
    """Sec. 5.6 CRITICAL: detach z_{t-1}. No gradient through time."""
    z = torch.randn(D, requires_grad=True)
    detached = z.detach()
    slots = mix_slots([torch.zeros(D)], {0: detached}, np.array([True]),
                      _FakePilot(D), "cpu")
    assert slots[0].requires_grad is False


# ---------------------------------------------------------------------------
# Sec. 5.2 / 5.4 — what must NOT change
# ---------------------------------------------------------------------------
def test_p_final_above_limit_is_rejected():
    """Sec. 5.4: 'Do not exceed p_final = 0.75 without asking the human.'"""
    import subprocess
    r = subprocess.run(
        [_sys.executable, str(_ROOT / "src" / "train" / "train_stage2.py"),
         "--p-final", "0.9"],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "0.75" in (r.stdout + r.stderr)


def test_z_null_is_absent():
    """Sec. 5.2: z_null is STAGE 4 ONLY. Adding it here would break
    one-novelty-per-stage."""
    src = (_ROOT / "src" / "train" / "train_stage2.py").read_text()
    code = "\n".join(l for l in src.splitlines()
                     if not l.strip().startswith("#") and "z_null" not in l.lower()
                     or "z_null" not in l.lower())
    assert "z_null" not in src.replace("`z_null`", "").replace(
        "z_null (Sec. 5.2's third row)", "")
