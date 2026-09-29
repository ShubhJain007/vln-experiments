"""
Frame-history tests (D28).

These exist because a 7-hour overnight run silently trained the degenerate
configuration: `--history-stride` defaulted to 1, so K=2 history spanned two
actions -- 0.5 m of travel or 30 degrees of turn, roughly 5% of a median
40-step episode. Nothing failed. The loss curves looked normal. The run simply
learned nothing about revisiting, and that cost a night.

Two independent guards are pinned here: the DEFAULT is the useful value, and
the degenerate value is an ERROR rather than a quiet mistake.
"""

import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]

import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT / "src"))

from data.pointing_dataset import (  # noqa: E402
    DEFAULT_HISTORY_STRIDE, MIN_HISTORY_SPAN_STEPS, PointingDataset,
)


# ---------------------------------------------------------------------------
# The two guards
# ---------------------------------------------------------------------------
def test_default_stride_is_the_useful_value():
    """Forgetting the flag must give working memory, not the degenerate case."""
    assert DEFAULT_HISTORY_STRIDE >= 8


def test_default_span_clears_the_minimum():
    assert 2 * DEFAULT_HISTORY_STRIDE >= MIN_HISTORY_SPAN_STEPS


def test_degenerate_span_is_rejected():
    """K=2, stride=1 spans 2 actions. This is the exact configuration that ran
    overnight without complaint; it must now refuse to construct."""
    with pytest.raises(ValueError, match="spans only"):
        PointingDataset("train", max_episodes=2, history_frames=2,
                        history_stride=1)


def test_error_names_a_working_stride():
    """A guard that only says 'no' invites guessing at 2am."""
    with pytest.raises(ValueError, match="use stride"):
        PointingDataset("train", max_episodes=2, history_frames=2,
                        history_stride=1)


def test_memoryless_is_still_allowed():
    """history_frames=0 is a legitimate configuration -- the guard must not
    block the memoryless baseline we compare against."""
    ds = PointingDataset("train", max_episodes=2, history_frames=0)
    assert ds.history_frames == 0
    assert all(s.history_paths == () for s in ds.steps[:5])


# ---------------------------------------------------------------------------
# Strided indexing -- eval must reproduce it exactly or the input distribution
# silently differs from training
# ---------------------------------------------------------------------------
def test_history_is_strided_oldest_to_newest():
    ds = PointingDataset("train", max_episodes=3, history_frames=2,
                         history_stride=8)
    s = next(x for x in ds.steps if x.step_index == 20)
    got = [int(p.stem) for p in s.history_paths]
    assert got == [4, 12]                      # 20-16, 20-8
    assert got == sorted(got), "history must run oldest -> newest"


def test_episode_start_pads_with_the_first_frame():
    """Fewer than K*stride frames exist early on; padding keeps the temporal
    extent fixed so sequences stay batchable."""
    ds = PointingDataset("train", max_episodes=3, history_frames=2,
                         history_stride=8)
    s0 = next(x for x in ds.steps if x.step_index == 0)
    assert [int(p.stem) for p in s0.history_paths] == [0, 0]


def test_eval_buffer_indexing_matches_the_dataset():
    """The eval keeps a rolling buffer and samples every stride-th frame. If
    that drifts from the dataset's indexing, a trained model is evaluated on
    inputs it never saw -- and the numbers look plausible anyway."""
    from collections import deque
    K, stride = 2, 8
    span = K * stride
    buf = deque(maxlen=span)
    for t in range(30):
        if not buf:
            for _ in range(span):
                buf.append(0)                  # episode start: repeat frame 0
        b = list(buf)
        eval_pick = [b[-k * stride] for k in range(K, 0, -1)]
        dataset_pick = [max(0, t - k * stride) for k in range(K, 0, -1)]
        assert eval_pick == dataset_pick, f"diverged at step {t}"
        buf.append(t)
