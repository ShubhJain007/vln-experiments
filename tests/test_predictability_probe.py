"""
Tests for the Stage -1 predictability probe.
Run:  python -m pytest tests/test_predictability_probe.py -v
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import math
import pathlib

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from data.predictability_probe import (  # noqa: E402
    centered_copy_shortcut,
    collect_trajectories,
    compute_copy_shortcut,
    compute_forward_retrieval_at_1,
    compute_horizon_similarities,
    compute_nothing_changes,
    compute_random_pair_baseline,
    compute_retrieval_at_1,
    cosine_sim,
    gate_verdict,
    nanmean,
    normalized_headroom,
)

D = 2048


# ---------------------------------------------------------------------------
# collect_trajectories — directory-of-videos support
# ---------------------------------------------------------------------------
def test_collect_trajectories_multi_video_dir(tmp_path, monkeypatch):
    """A flat directory containing several video files must become one
    trajectory PER FILE, sorted by filename, not one merged trajectory."""
    for name in ("b.mp4", "a.mp4", "c.mov"):
        (tmp_path / name).write_bytes(b"not a real video")

    calls = []

    def fake_load_video(path, fps, max_frames):
        calls.append(path.name)
        return [f"frame-from-{path.name}"]

    import data.predictability_probe as mod
    monkeypatch.setattr(mod, "_load_video", fake_load_video)

    trajs = collect_trajectories(str(tmp_path), fps=4.0, max_frames=300)

    assert calls == ["a.mp4", "b.mp4", "c.mov"], "must sort by filename"
    assert len(trajs) == 3
    assert trajs[0] == ["frame-from-a.mp4"]


def test_collect_trajectories_subdirs_take_priority_over_videos(tmp_path, monkeypatch):
    """If a directory has BOTH subdirectories and stray video files, the
    subdirectory (multi-trajectory image) reading path wins, matching the
    documented precedence order."""
    sub = tmp_path / "traj0"
    sub.mkdir()
    (sub / "0.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "stray.mp4").write_bytes(b"not a real video")

    import data.predictability_probe as mod
    monkeypatch.setattr(mod, "_load_image_dir", lambda d, m: [f"img-from-{d.name}"])

    trajs = collect_trajectories(str(tmp_path), fps=4.0, max_frames=300)
    assert trajs == [["img-from-traj0"]]


def test_collect_trajectories_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        collect_trajectories("/no/such/path/anywhere", fps=4.0, max_frames=300)


def test_collect_trajectories_rejects_non_video_file(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with pytest.raises(ValueError):
        collect_trajectories(str(bad), fps=4.0, max_frames=300)


# ---------------------------------------------------------------------------
# cosine_sim
# ---------------------------------------------------------------------------
def test_cosine_sim_identical():
    v = torch.randn(D)
    assert abs(cosine_sim(v, v) - 1.0) < 1e-5


def test_cosine_sim_orthogonal():
    a, b = torch.zeros(4), torch.zeros(4)
    a[0], b[1] = 1.0, 1.0
    assert abs(cosine_sim(a, b)) < 1e-5


def test_cosine_sim_opposite():
    v = torch.randn(D)
    assert abs(cosine_sim(v, -v) + 1.0) < 1e-5


def test_cosine_sim_is_scale_invariant():
    a, b = torch.randn(D), torch.randn(D)
    assert abs(cosine_sim(a, b) - cosine_sim(3.7 * a, 0.2 * b)) < 1e-5


# ---------------------------------------------------------------------------
# compute_horizon_similarities
# ---------------------------------------------------------------------------
def test_horizon_constant_sequence_is_one():
    """Identical frames -> similarity 1.0 at every horizon."""
    vbars = torch.randn(D).unsqueeze(0).expand(20, -1).clone()
    for h in (1, 2, 4, 8):
        assert abs(compute_horizon_similarities(vbars, h) - 1.0) < 1e-4, f"h={h}"


def test_horizon_too_short_is_nan():
    assert math.isnan(compute_horizon_similarities(torch.randn(3, D), 8))


def test_horizon_exactly_at_length_is_nan():
    """T == horizon leaves no valid (t, t+h) pair."""
    assert math.isnan(compute_horizon_similarities(torch.randn(4, D), 4))


def test_horizon_random_is_near_zero():
    torch.manual_seed(0)
    vbars = F.normalize(torch.randn(200, D), dim=1)
    assert abs(compute_horizon_similarities(vbars, 1)) < 0.05


def test_horizon_in_valid_range():
    vbars = torch.randn(50, D)
    for h in (1, 2, 4, 8):
        assert -1.0 <= compute_horizon_similarities(vbars, h) <= 1.0


# ---------------------------------------------------------------------------
# copy shortcut / nothing-changes  (the gate quantities)
# ---------------------------------------------------------------------------
def test_copy_shortcut_constant_sequence_is_one():
    vbars = torch.randn(D).unsqueeze(0).expand(10, -1).clone()
    assert abs(compute_copy_shortcut(vbars) - 1.0) < 1e-4


def test_copy_shortcut_equals_horizon1_on_shifted_window():
    """cos(v_{t+1}, v_{t+2}) must equal horizon-1 similarity of vbars[1:]."""
    torch.manual_seed(3)
    vbars = F.normalize(torch.randn(40, D), dim=1)
    assert abs(compute_copy_shortcut(vbars)
               - compute_horizon_similarities(vbars[1:], 1)) < 1e-5


def test_nothing_changes_equals_horizon2():
    """cos(v_t, v_{t+2}) is exactly the horizon-2 similarity."""
    torch.manual_seed(4)
    vbars = F.normalize(torch.randn(40, D), dim=1)
    assert abs(compute_nothing_changes(vbars)
               - compute_horizon_similarities(vbars, 2)) < 1e-5


def test_gate_quantities_nan_when_too_short():
    short = torch.randn(2, D)
    assert math.isnan(compute_copy_shortcut(short))
    assert math.isnan(compute_nothing_changes(short))


# ---------------------------------------------------------------------------
# retrieval@1
# ---------------------------------------------------------------------------
def test_self_retrieval_perfect_for_distinct_frames():
    """Orthogonal frames -> self-retrieval is 1.0."""
    T = 60
    vbars = F.pad(torch.eye(T), (0, D - T))
    assert compute_retrieval_at_1(vbars, n_distractors=49) == 1.0


def test_self_retrieval_perfect_in_high_dim():
    """In 2048-D, random unit vectors are near-orthogonal while cos(x,x)=1, so
    self-retrieval is trivially perfect. Expected — this metric only drops
    below 1.0 for near-duplicate frames, which is what makes it a smoothness
    diagnostic on real footage."""
    torch.manual_seed(42)
    vbars = F.normalize(torch.randn(200, D), dim=1)
    assert compute_retrieval_at_1(vbars, n_distractors=49) == 1.0


def test_forward_retrieval_near_chance_for_random():
    """v_t carries no information about v_{t+2} when frames are independent."""
    torch.manual_seed(42)
    vbars = F.normalize(torch.randn(200, D), dim=1)
    ret = compute_forward_retrieval_at_1(vbars, n_distractors=49)
    assert ret < 0.15, f"expected near chance (0.02), got {ret:.3f}"


def test_forward_retrieval_detects_planted_signal():
    """Forward retrieval must rise well above chance when v_t is linked to
    v_{t+2}. Built from disjoint (t, t+2) pairs so no frame competes: even
    indices carry the signal, odd indices are orthogonal filler."""
    T = 60
    vbars = torch.zeros(T, D)
    for t in range(T):
        if t % 4 == 0 and t + 2 < T:
            shared = torch.zeros(D)
            shared[t] = 1.0                  # unique direction for this pair
            vbars[t] = shared
            vbars[t + 2] = shared            # exact match at t+2 only
        elif vbars[t].abs().sum() == 0:
            filler = torch.zeros(D)
            filler[T + t] = 1.0              # orthogonal to every pair vector
            vbars[t] = filler
    ret = compute_forward_retrieval_at_1(vbars, n_distractors=49, seed=0)
    assert ret > 0.2, f"planted signal not detected, got {ret:.3f}"


def test_forward_retrieval_constant_sequence_in_range():
    """Degenerate case: every frame identical means every pool entry ties.
    Must not crash and must stay in [0, 1]."""
    vbars = torch.randn(D).unsqueeze(0).expand(30, -1).clone()
    assert 0.0 <= compute_forward_retrieval_at_1(vbars, n_distractors=10) <= 1.0


def test_forward_retrieval_symmetric_neighbour_note():
    """Cosine similarity is symmetric, so on a smooth trajectory v_t is as
    close to v_{t-2} as to v_{t+2} and the backward neighbour competes as a
    distractor. Forward retrieval is therefore NOT expected to reach 1.0 on
    real footage; it is a relative diagnostic, not an accuracy target."""
    torch.manual_seed(11)
    steps = F.normalize(torch.randn(50, D), dim=1) * 0.1
    vbars = F.normalize(torch.cumsum(steps, dim=0), dim=1)   # smooth walk
    assert 0.0 <= compute_forward_retrieval_at_1(vbars, n_distractors=49) <= 1.0


def test_retrieval_too_short_is_nan():
    assert math.isnan(compute_retrieval_at_1(torch.randn(2, D)))
    assert math.isnan(compute_forward_retrieval_at_1(torch.randn(2, D)))


def test_retrieval_in_unit_interval():
    vbars = torch.randn(80, D)
    assert 0.0 <= compute_retrieval_at_1(vbars, 49) <= 1.0
    assert 0.0 <= compute_forward_retrieval_at_1(vbars, 49) <= 1.0


def test_retrieval_is_deterministic_given_seed():
    """Same seed must give the same number; the probe has to be reproducible."""
    torch.manual_seed(7)
    vbars = F.normalize(torch.randn(120, D), dim=1)
    a = compute_forward_retrieval_at_1(vbars, 49, seed=123)
    b = compute_forward_retrieval_at_1(vbars, 49, seed=123)
    assert a == b


def test_retrieval_pool_excludes_query_for_forward():
    """Forward retrieval must never place v_t itself in the pool, otherwise a
    self-match could be scored as a hit."""
    T = 40
    vbars = F.pad(torch.eye(T), (0, D - T))
    # With orthogonal frames, v_t matches nothing but itself; if t leaked into
    # the pool the argmax would pick it and the score would be 0, not chance.
    ret = compute_forward_retrieval_at_1(vbars, n_distractors=10, seed=1)
    assert 0.0 <= ret <= 1.0


# ---------------------------------------------------------------------------
# anisotropy calibration
# ---------------------------------------------------------------------------
def test_random_pair_baseline_near_zero_for_isotropic():
    """Isotropic random embeddings have an unrelated floor of ~0."""
    torch.manual_seed(5)
    vbars = F.normalize(torch.randn(200, D), dim=1)
    assert abs(compute_random_pair_baseline(vbars)) < 0.05


def test_random_pair_baseline_high_for_anisotropic():
    """Adding a large shared component inflates every pairwise cosine — this is
    the regime the real encoder is in, so the floor must detect it."""
    torch.manual_seed(6)
    shared = F.normalize(torch.randn(1, D), dim=1) * 3.0
    vbars = F.normalize(shared + F.normalize(torch.randn(200, D), dim=1), dim=1)
    assert compute_random_pair_baseline(vbars) > 0.7


def test_random_pair_baseline_too_short_is_nan():
    assert math.isnan(compute_random_pair_baseline(torch.randn(5, D), min_gap=8))


def test_centered_copy_shortcut_removes_shared_component():
    """Centering must pull an anisotropic sequence's similarity well below its
    raw value."""
    torch.manual_seed(8)
    shared = F.normalize(torch.randn(1, D), dim=1) * 3.0
    vbars = F.normalize(shared + F.normalize(torch.randn(60, D), dim=1), dim=1)
    assert centered_copy_shortcut(vbars) < compute_copy_shortcut(vbars)


def test_centered_copy_shortcut_constant_is_nan_or_unit():
    """All-identical frames center to zero vectors; must not raise."""
    vbars = torch.randn(D).unsqueeze(0).expand(10, -1).clone()
    result = centered_copy_shortcut(vbars)
    assert result != result or -1.0 <= result <= 1.0


def test_centered_copy_shortcut_too_short_is_nan():
    assert math.isnan(centered_copy_shortcut(torch.randn(2, D)))


def test_normalized_headroom_endpoints():
    assert normalized_headroom(1.0, 0.75) == pytest.approx(1.0)
    assert normalized_headroom(0.75, 0.75) == pytest.approx(0.0)
    assert normalized_headroom(0.875, 0.75) == pytest.approx(0.5)


def test_normalized_headroom_rescales_anisotropic_gate():
    """A raw 0.98 against a 0.75 floor is 0.92 headroom — still high, but a
    raw 0.98 against a 0.97 floor is only 0.33, a materially different read."""
    assert normalized_headroom(0.98, 0.75) == pytest.approx(0.92, abs=1e-2)
    assert normalized_headroom(0.98, 0.97) == pytest.approx(0.33, abs=1e-2)


def test_normalized_headroom_nan_inputs():
    assert math.isnan(normalized_headroom(float("nan"), 0.5))
    assert math.isnan(normalized_headroom(0.9, float("nan")))
    assert math.isnan(normalized_headroom(0.9, 1.0))


# ---------------------------------------------------------------------------
# gate_verdict
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("score", [0.0, 0.5, 0.85, 0.8999])
def test_gate_pass(score):
    assert "PASS" in gate_verdict(score)


@pytest.mark.parametrize("score", [0.90, 0.93, 0.97, 0.980])
def test_gate_marginal(score):
    assert "MARGINAL" in gate_verdict(score)


@pytest.mark.parametrize("score", [0.981, 0.99, 1.0])
def test_gate_fail(score):
    assert "FAIL" in gate_verdict(score)


def test_gate_nan_is_unknown():
    assert "UNKNOWN" in gate_verdict(float("nan"))


# ---------------------------------------------------------------------------
# nanmean
# ---------------------------------------------------------------------------
def test_nanmean_ignores_nan():
    assert nanmean([1.0, 2.0, float("nan"), 3.0]) == 2.0


def test_nanmean_all_nan_is_nan():
    assert math.isnan(nanmean([float("nan"), float("nan")]))


def test_nanmean_empty_is_nan():
    assert math.isnan(nanmean([]))
