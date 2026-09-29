"""
Tests for the LR schedule (NOT IN PAPER -- AGENTS.md Sec. 8 item 3).

The original schedule was warmup-only and then CONSTANT, so the LR never
decayed and the final checkpoint was taken mid-flight. These tests pin the
corrected shape.

Run:  PYTHONPATH="" python -m pytest tests/test_schedule.py -v
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import pathlib

import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
from train.schedule import (  # noqa: E402
    DEFAULT_MIN_LR_FRAC,
    describe,
    lr_lambda,
    make_scheduler,
)

WARMUP, TOTAL = 100, 8000


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------
def test_warmup_ramps_from_near_zero_to_peak():
    assert lr_lambda(0, WARMUP, TOTAL) == pytest.approx(1 / WARMUP)
    assert lr_lambda(WARMUP - 1, WARMUP, TOTAL) == pytest.approx(1.0)


def test_peak_is_reached_exactly_once_at_warmup_end():
    assert lr_lambda(WARMUP, WARMUP, TOTAL) == pytest.approx(1.0, abs=1e-3)


def test_warmup_is_monotonic():
    vals = [lr_lambda(s, WARMUP, TOTAL) for s in range(WARMUP)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


# ---------------------------------------------------------------------------
# THE BUG THIS MODULE FIXES — decay must actually happen
# ---------------------------------------------------------------------------
def test_lr_actually_decays_after_warmup():
    """The original schedule stayed at peak for ~99% of the run. If this ever
    passes trivially again, the decay has been lost."""
    mid = lr_lambda(TOTAL // 2, WARMUP, TOTAL)
    end = lr_lambda(TOTAL - 1, WARMUP, TOTAL)
    assert mid < 0.95, f"no meaningful decay by mid-run (got {mid:.3f})"
    assert end < 0.1, f"LR did not decay by the end (got {end:.3f})"


def test_decay_is_monotonic_after_warmup():
    vals = [lr_lambda(s, WARMUP, TOTAL) for s in range(WARMUP, TOTAL, 50)]
    assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))


def test_ends_at_the_floor_not_zero():
    """A hard 0 would freeze the last steps; the floor keeps them moving."""
    end = lr_lambda(TOTAL - 1, WARMUP, TOTAL)
    assert end == pytest.approx(DEFAULT_MIN_LR_FRAC, abs=1e-3)


def test_never_exceeds_peak_or_drops_below_floor():
    for s in range(0, TOTAL, 37):
        v = lr_lambda(s, WARMUP, TOTAL)
        assert 0.0 < v <= 1.0 + 1e-9
        if s >= WARMUP:
            assert v >= DEFAULT_MIN_LR_FRAC - 1e-9


# ---------------------------------------------------------------------------
# Schedule variants
# ---------------------------------------------------------------------------
def test_constant_reproduces_the_original_behaviour():
    """Kept so the pre-fix runs remain reproducible for comparison."""
    for s in (WARMUP, TOTAL // 2, TOTAL - 1):
        assert lr_lambda(s, WARMUP, TOTAL, schedule="constant") == 1.0


def test_cosine_and_linear_differ_but_share_endpoints():
    cos_mid = lr_lambda(TOTAL // 2, WARMUP, TOTAL, schedule="cosine")
    lin_mid = lr_lambda(TOTAL // 2, WARMUP, TOTAL, schedule="linear")
    assert cos_mid != pytest.approx(lin_mid, abs=1e-3)
    for sch in ("cosine", "linear"):
        assert lr_lambda(TOTAL - 1, WARMUP, TOTAL, schedule=sch) == pytest.approx(
            DEFAULT_MIN_LR_FRAC, abs=1e-3)


def test_cosine_decays_more_slowly_early():
    """Cosine holds near peak longer than linear -- that is why it is default."""
    q = WARMUP + (TOTAL - WARMUP) // 4
    assert lr_lambda(q, WARMUP, TOTAL, schedule="cosine") > \
           lr_lambda(q, WARMUP, TOTAL, schedule="linear")


def test_unknown_schedule_rejected():
    with pytest.raises(ValueError, match="schedule must be"):
        lr_lambda(0, WARMUP, TOTAL, schedule="exponential")


# ---------------------------------------------------------------------------
# make_scheduler / guards
# ---------------------------------------------------------------------------
def test_scheduler_drives_a_real_optimizer():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1e-4)
    sched = make_scheduler(opt, warmup=10, total_steps=100)
    seen = []
    for _ in range(100):
        seen.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert seen[0] < seen[10]          # warmed up
    assert seen[-1] < seen[10]         # then decayed
    assert max(seen) == pytest.approx(1e-4, rel=1e-6)


def test_total_steps_must_exceed_warmup():
    """Otherwise the run is pure warmup and never decays -- silently."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1e-4)
    with pytest.raises(ValueError, match="must exceed warmup"):
        make_scheduler(opt, warmup=500, total_steps=100)


def test_describe_reports_both_endpoints():
    s = describe(WARMUP, TOTAL, 1e-4)
    assert "1.00e-04" in s and "5.00e-06" in s and "cosine" in s
