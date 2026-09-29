"""
Tests for the Sec. 4.1 metrics: NE, SR, OS, SPL, nDTW.

THE required test (AGENTS.md Sec. 3.4): on ground-truth reference paths,
SR must be ~1.0 and SPL ~1.0. "If not, the metric implementation is wrong --
fix before evaluating any model." That test runs against REAL R2R-CE episodes,
not synthetic data.

All tests use Euclidean distance. That is legitimate here because it verifies
the metric ARITHMETIC; geodesic distance needs a navmesh (habitat + MP3D) and
only changes the distance values fed in, not the formulas.

Run:  PYTHONPATH="" python -m pytest tests/test_metrics.py -v
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import gzip
import json
import math
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
from eval.metrics import (  # noqa: E402
    SUCCESS_THRESHOLD_M,
    resample_path,
    EpisodeMetrics,
    aggregate,
    dtw_distance,
    euclidean,
    evaluate_episode,
    navigation_error,
    ndtw,
    oracle_navigation_error,
    oracle_success,
    path_length,
    spl,
    success,
)

R2R = _ROOT / "data" / "R2R_VLNCE_v1-3" / "val_unseen" / "val_unseen.json.gz"


# ---------------------------------------------------------------------------
# THE GATE TEST — real R2R-CE reference paths must score ~1.0
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not R2R.exists(), reason="R2R-CE episodes not downloaded")
def test_ground_truth_reference_paths_score_near_perfect():
    """AGENTS.md Sec. 3.4: on ground-truth reference paths SR ~1.0, SPL ~1.0.

    Feeds each episode's own reference_path back in as if the agent had walked
    it exactly. A metric bug shows up here as a score well below 1.0, and
    AGENTS.md is explicit that this must be fixed BEFORE evaluating any model.

    Euclidean distance is used, so SPL is compared against the Euclidean length
    of the reference path rather than the stored geodesic value -- mixing the
    two would test the data, not the metric. See the separate geodesic
    consistency test below.
    """
    episodes = json.load(gzip.open(R2R, "rt"))["episodes"]
    results = []
    for ep in episodes[:300]:
        ref = ep["reference_path"]
        goal = ep["goals"][0]["position"]
        results.append(
            evaluate_episode(
                path=ref,
                goal_position=goal,
                reference_path=ref,
                shortest_path_length=path_length(ref),
                distance_fn=euclidean,
            )
        )

    agg = aggregate(results)
    assert agg["SR"] > 0.99, f"SR on ground truth = {agg['SR']:.4f}, expected ~1.0"
    assert agg["SPL"] > 0.99, f"SPL on ground truth = {agg['SPL']:.4f}, expected ~1.0"
    assert agg["OS"] > 0.99, f"OS on ground truth = {agg['OS']:.4f}, expected ~1.0"
    assert agg["nDTW"] > 0.99, f"nDTW on ground truth = {agg['nDTW']:.4f}, expected 1.0"
    assert agg["NE"] < SUCCESS_THRESHOLD_M


@pytest.mark.skipif(not R2R.exists(), reason="R2R-CE episodes not downloaded")
def test_spl_ceiling_under_the_standard_convention_is_below_one():
    """SPL's `l` is the geodesic START-TO-GOAL distance, NOT the reference path
    length. That is the standard definition (Anderson et al.; habitat and
    VLN-CE both use episode.info["geodesic_distance"]).

    Consequence that matters for the Stage 0 gate: the reference path is not
    the geodesic-shortest route, so even a PERFECT agent reproducing it exactly
    scores SPL ~0.92, not 1.0. AGENTS.md Sec. 3.4's "SPL ~1.0" is therefore
    approximate -- it holds when `l` is the path's own length (the pure
    arithmetic check above), not under the reporting convention.

    Corroboration that the paper uses this same convention: Table 3's NaN row
    gives SR 51.7 / SPL 47.1, a ratio of 0.911 -- essentially this ceiling.
    """
    episodes = json.load(gzip.open(R2R, "rt"))["episodes"]
    results = [
        evaluate_episode(
            path=ep["reference_path"],
            goal_position=ep["goals"][0]["position"],
            reference_path=ep["reference_path"],
            shortest_path_length=ep["info"]["geodesic_distance"],  # the convention
            distance_fn=euclidean,
        )
        for ep in episodes
    ]
    agg = aggregate(results)
    assert agg["SR"] > 0.99, "success itself must still be perfect"
    assert 0.85 < agg["SPL"] < 1.0, (
        f"SPL ceiling on ground truth = {agg['SPL']:.4f}; expected ~0.92. "
        "Outside this range means the SPL denominator convention has changed."
    )


@pytest.mark.skipif(not R2R.exists(), reason="R2R-CE episodes not downloaded")
def test_reference_path_endpoint_is_within_goal_radius():
    """Sanity check on the DATA rather than the metric: a reference path must
    actually end at its goal, otherwise the gate test above is meaningless."""
    episodes = json.load(gzip.open(R2R, "rt"))["episodes"]
    for ep in episodes[:300]:
        d = euclidean(ep["reference_path"][-1], ep["goals"][0]["position"])
        assert d <= SUCCESS_THRESHOLD_M, (
            f"episode {ep['episode_id']} reference path ends {d:.2f} m from goal"
        )


# ---------------------------------------------------------------------------
# NE
# ---------------------------------------------------------------------------
def test_navigation_error_is_zero_at_goal():
    assert navigation_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_navigation_error_is_a_distance():
    assert navigation_error([0, 0, 0], [3, 4, 0]) == pytest.approx(5.0)


def test_navigation_error_uses_injected_distance_fn():
    """Production passes geodesic; the metric must not hardcode Euclidean."""
    called = {}

    def fake(a, b):
        called["yes"] = True
        return 42.0

    assert navigation_error([0, 0, 0], [1, 1, 1], fake) == 42.0
    assert called


# ---------------------------------------------------------------------------
# SR
# ---------------------------------------------------------------------------
def test_success_threshold_is_three_metres():
    assert SUCCESS_THRESHOLD_M == 3.0


@pytest.mark.parametrize("ne,expected", [
    (0.0, True), (2.99, True), (3.0, True), (3.01, False), (10.0, False),
])
def test_success_boundary(ne, expected):
    """Inclusive at exactly 3 m -- 'within 3 meters' (Sec. 4.1)."""
    assert success(ne) is expected


# ---------------------------------------------------------------------------
# OS
# ---------------------------------------------------------------------------
def test_oracle_success_when_passing_near_goal_then_leaving():
    """The defining case: the agent passes the goal but stops far away.
    SR must fail, OS must succeed."""
    goal = [0.0, 0.0, 0.0]
    path = [[10, 0, 0], [1, 0, 0], [20, 0, 0]]     # passes within 1 m, ends 20 m away
    assert not success(navigation_error(path[-1], goal))
    assert oracle_success(path, goal)


def test_oracle_success_false_when_never_close():
    goal = [0.0, 0.0, 0.0]
    path = [[10, 0, 0], [11, 0, 0], [12, 0, 0]]
    assert not oracle_success(path, goal)


def test_oracle_success_is_at_least_success():
    """OS >= SR by construction: the closest point is at least as close as the
    final one. A violation would mean the implementations disagree."""
    goal = [0.0, 0.0, 0.0]
    for path in ([[5, 0, 0], [1, 0, 0]], [[1, 0, 0], [5, 0, 0]], [[0, 0, 0]]):
        sr = success(navigation_error(path[-1], goal))
        assert oracle_success(path, goal) or not sr


def test_oracle_navigation_error_is_the_minimum():
    goal = [0.0, 0.0, 0.0]
    path = [[10, 0, 0], [2, 0, 0], [7, 0, 0]]
    assert oracle_navigation_error(path, goal) == pytest.approx(2.0)


def test_oracle_functions_reject_empty_path():
    with pytest.raises(ValueError):
        oracle_success([], [0, 0, 0])
    with pytest.raises(ValueError):
        oracle_navigation_error([], [0, 0, 0])


# ---------------------------------------------------------------------------
# path length / SPL
# ---------------------------------------------------------------------------
def test_path_length_sums_segments():
    assert path_length([[0, 0, 0], [3, 4, 0], [3, 4, 5]]) == pytest.approx(10.0)


def test_path_length_of_short_paths_is_zero():
    assert path_length([]) == 0.0
    assert path_length([[1, 2, 3]]) == 0.0


def test_spl_is_one_for_perfect_success():
    assert spl(True, 10.0, 10.0) == pytest.approx(1.0)


def test_spl_is_zero_on_failure_however_efficient():
    """Efficiency must not rescue a failed episode."""
    assert spl(False, 10.0, 10.0) == 0.0
    assert spl(False, 10.0, 1.0) == 0.0


def test_spl_penalises_a_longer_route():
    assert spl(True, 10.0, 20.0) == pytest.approx(0.5)
    assert spl(True, 10.0, 40.0) == pytest.approx(0.25)


def test_spl_never_exceeds_one():
    """A shorter-than-geodesic path (possible via goal-radius slack) must not
    score above 1 -- that is what the max() in the denominator prevents."""
    assert spl(True, 10.0, 5.0) == pytest.approx(1.0)


def test_spl_degenerate_zero_length_episode():
    """Agent starts at the goal and never moves: succeeded, wasted nothing."""
    assert spl(True, 0.0, 0.0) == 1.0


# ---------------------------------------------------------------------------
# nDTW
# ---------------------------------------------------------------------------
def test_dtw_of_identical_paths_is_zero():
    p = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    assert dtw_distance(p, p) == pytest.approx(0.0)


def test_ndtw_of_identical_paths_is_one():
    p = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    assert ndtw(p, p) == pytest.approx(1.0)


def test_ndtw_decreases_with_deviation():
    ref = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    near = [[0, 0.1, 0], [1, 0.1, 0], [2, 0.1, 0]]
    far = [[0, 5, 0], [1, 5, 0], [2, 5, 0]]
    assert 1.0 > ndtw(near, ref) > ndtw(far, ref) > 0.0


def test_ndtw_is_in_unit_interval():
    ref = [[0, 0, 0], [1, 0, 0]]
    for q in ([[0, 0, 0]], [[100, 100, 100]], [[0, 0, 0], [1, 0, 0], [2, 0, 0]]):
        assert 0.0 <= ndtw(q, ref) <= 1.0


def test_dtw_handles_different_lengths():
    """DTW must align sequences of unequal length -- an agent rarely takes
    exactly as many steps as the reference has waypoints."""
    ref = [[0, 0, 0], [2, 0, 0]]
    dense = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    # every dense point lies ON the reference segment, so cost stays small
    assert dtw_distance(dense, ref) == pytest.approx(1.0)


def test_dtw_is_order_sensitive():
    """Traversing the reference backwards must cost more than forwards."""
    ref = [[0, 0, 0], [1, 0, 0], [5, 0, 0]]
    assert dtw_distance(list(reversed(ref)), ref) > dtw_distance(ref, ref)


def test_ndtw_rejects_empty_inputs():
    with pytest.raises(ValueError):
        ndtw([], [[0, 0, 0]])
    with pytest.raises(ValueError):
        ndtw([[0, 0, 0]], [])


# ---------------------------------------------------------------------------
# evaluate_episode / aggregate
# ---------------------------------------------------------------------------
def test_evaluate_episode_perfect_run():
    ref = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    m = evaluate_episode(ref, [2, 0, 0], ref, path_length(ref))
    assert m.ne == pytest.approx(0.0)
    assert m.success and m.oracle_success
    assert m.spl == pytest.approx(1.0)
    assert m.ndtw == pytest.approx(1.0)


def test_evaluate_episode_failed_run():
    ref = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    agent = [[0, 0, 0], [0, 10, 0], [0, 20, 0]]
    m = evaluate_episode(agent, [2, 0, 0], ref, path_length(ref))
    assert not m.success
    assert m.spl == 0.0
    assert m.ne > SUCCESS_THRESHOLD_M


def test_evaluate_episode_rejects_empty_path():
    with pytest.raises(ValueError):
        evaluate_episode([], [0, 0, 0], [[0, 0, 0]], 1.0)


def test_aggregate_averages_each_metric():
    a = EpisodeMetrics(2.0, True, True, 1.0, 1.0, 10.0, 2.0)
    b = EpisodeMetrics(4.0, False, True, 0.0, 0.5, 20.0, 1.0)
    agg = aggregate([a, b])
    assert agg["NE"] == pytest.approx(3.0)
    assert agg["SR"] == pytest.approx(0.5)
    assert agg["OS"] == pytest.approx(1.0)
    assert agg["SPL"] == pytest.approx(0.5)
    assert agg["nDTW"] == pytest.approx(0.75)
    assert agg["n_episodes"] == 2


def test_aggregate_rejects_empty():
    with pytest.raises(ValueError):
        aggregate([])


def test_metrics_dict_keys_match_reporting_format():
    """AGENTS.md Sec. 7 reports the table as NE / SR / SPL / OS."""
    m = EpisodeMetrics(1.0, True, True, 1.0, 1.0, 1.0, 1.0).as_dict()
    for key in ("NE", "SR", "OS", "SPL", "nDTW"):
        assert key in m


# ---------------------------------------------------------------------------
# nDTW granularity normalisation
# ---------------------------------------------------------------------------
def test_resample_path_spacing():
    out = resample_path([[0, 0, 0], [1, 0, 0]], 0.25)
    assert len(out) == 5                       # 0.00 0.25 0.50 0.75 1.00
    assert out[0] == (0, 0, 0)
    assert out[-1] == pytest.approx((1.0, 0.0, 0.0))
    for a, b in zip(out, out[1:]):
        assert euclidean(a, b) == pytest.approx(0.25)


def test_resample_path_short_inputs():
    assert resample_path([], 0.25) == []
    assert resample_path([[1, 2, 3]], 0.25) == [(1, 2, 3)]


def test_resample_preserves_endpoints():
    p = [[0, 0, 0], [3, 4, 0], [3, 4, 9]]
    out = resample_path(p, 0.25)
    assert out[0] == (0, 0, 0)
    assert out[-1] == pytest.approx((3.0, 4.0, 9.0))


def test_ndtw_identity_holds_under_resampling():
    """Resampling BOTH sides keeps the identity property: a path compared
    against itself must still score exactly 1.0. Resampling only the reference
    would break this."""
    p = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
    assert ndtw(p, p) == pytest.approx(1.0)


def test_ndtw_fixes_dense_vs_sparse_granularity():
    """THE bug this exists for. A dense agent path that tracks a sparse
    reference closely must score HIGH.

    Measured on a real expert rollout (31 agent points at 0.25 m vs 6 waypoints
    at ~1.7 m, passing within 0.2-1.0 m of every waypoint): the literal formula
    gave 0.35 because DTW summed 31 terms while dividing by only 6; resampling
    both sides gave 0.78.
    """
    ref = [[0, 0, 0], [5, 0, 0], [10, 0, 0]]            # sparse, 5 m apart
    dense = resample_path(ref, 0.25)                     # same route, dense
    literal = ndtw(dense, ref, resample_spacing=0)
    fixed = ndtw(dense, ref)
    assert fixed > 0.95, f"identical route scored {fixed:.3f}"
    assert fixed > literal, "resampling must improve a faithful dense path"


def test_ndtw_still_penalises_a_genuinely_wrong_route():
    """The fix must not make nDTW blind -- a path that ignores the reference
    must still score low."""
    ref = [[0, 0, 0], [5, 0, 0], [10, 0, 0]]
    wrong = resample_path([[0, 0, 0], [0, 10, 0], [0, 20, 0]], 0.25)
    assert ndtw(wrong, ref) < 0.2


def test_ndtw_resample_disabled_reproduces_literal_formula():
    ref = [[0, 0, 0], [1, 0, 0]]
    q = [[0, 0, 0], [1, 0, 0]]
    assert ndtw(q, ref, resample_spacing=0) == pytest.approx(
        math.exp(-dtw_distance(q, ref) / (len(ref) * SUCCESS_THRESHOLD_M))
    )
