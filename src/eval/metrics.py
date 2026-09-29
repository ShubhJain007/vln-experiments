"""
Stage 0 evaluation metrics (Sec. 4.1).

    NE     Navigation Error -- final geodesic distance from the stopping
           location to the goal, in metres.
    SR     Success Rate -- fraction of episodes stopping within 3 m of the goal.
    OS     Oracle Success -- success if the CLOSEST point along the trajectory
           was within 3 m, i.e. treating the best point as the stop point.
    SPL    Success weighted by Path Length -- accounts for success and
           path efficiency.
    nDTW   Normalized Dynamic Time Warping -- trajectory fidelity to the
           reference path (Ilharco et al., paper ref [19]).

--------------------------------------------------------------------------
DISTANCE FUNCTION IS INJECTABLE -- and the choice matters
--------------------------------------------------------------------------
NE, SR and OS are defined on GEODESIC distance, which needs a navmesh
(habitat's `PathFinder.geodesic_distance`). Habitat lives in a separate Python
3.9 env, and MP3D is not downloaded yet, so every function here takes a
`distance_fn`. Production passes the geodesic one; tests pass Euclidean, which
makes the metric arithmetic verifiable today.

Euclidean is NOT a silent stand-in for geodesic: it ignores walls, so it
under-estimates true distance and would inflate SR. Use it for unit tests and
for straight-line synthetic cases only, never to report a gate number.

--------------------------------------------------------------------------
PATH LENGTH
--------------------------------------------------------------------------
The agent's travelled path length is the sum of EUCLIDEAN distances between
consecutive visited positions. That is standard practice: consecutive steps are
0.25 m apart, so locally Euclidean and geodesic agree, and summing geodesic
distances between adjacent points would be both wrong and expensive. The
shortest-path length `l` used by SPL is geodesic, and for R2R-CE it is supplied
directly as `episode["info"]["geodesic_distance"]`.
"""

# --- ROS guard: strip /opt/ros/* from sys.path before any other import. ------
import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]
# ----------------------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import Callable, Sequence

Point = Sequence[float]
DistanceFn = Callable[[Point, Point], float]

# Sec. 4.1: "the agent stops within 3 meters of the goal". R2R-CE episodes
# independently carry goals[0]["radius"] == 3.0, which matches.
SUCCESS_THRESHOLD_M = 3.0


def euclidean(a: Point, b: Point) -> float:
    """Straight-line distance. Test/diagnostic use only -- see module docstring."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# NE
# ---------------------------------------------------------------------------
def navigation_error(final_position: Point, goal_position: Point,
                     distance_fn: DistanceFn = euclidean) -> float:
    """NE -- geodesic distance from the stopping location to the goal (m).

    Lower is better. Reported in metres.
    """
    return distance_fn(final_position, goal_position)


# ---------------------------------------------------------------------------
# SR
# ---------------------------------------------------------------------------
def success(navigation_error_m: float,
            threshold: float = SUCCESS_THRESHOLD_M) -> bool:
    """SR (per episode) -- did the agent STOP within `threshold` of the goal?

    Note this depends on where the agent actually stopped, not on whether it
    ever passed near the goal; that weaker notion is OS below.
    """
    return navigation_error_m <= threshold


# ---------------------------------------------------------------------------
# OS
# ---------------------------------------------------------------------------
def oracle_success(path: Sequence[Point], goal_position: Point,
                   distance_fn: DistanceFn = euclidean,
                   threshold: float = SUCCESS_THRESHOLD_M) -> bool:
    """OS -- success if the CLOSEST point along the trajectory was within
    `threshold` of the goal (Sec. 4.1).

    OS >= SR always: the closest point is at least as close as the final one.
    """
    if not path:
        raise ValueError("path is empty")
    return min(distance_fn(p, goal_position) for p in path) <= threshold


def oracle_navigation_error(path: Sequence[Point], goal_position: Point,
                            distance_fn: DistanceFn = euclidean) -> float:
    """Distance from the goal at the closest point along the trajectory."""
    if not path:
        raise ValueError("path is empty")
    return min(distance_fn(p, goal_position) for p in path)


# ---------------------------------------------------------------------------
# SPL
# ---------------------------------------------------------------------------
def path_length(path: Sequence[Point]) -> float:
    """Total distance travelled: sum of Euclidean steps along the trajectory.

    A path with 0 or 1 points has length 0.
    """
    return sum(euclidean(path[i], path[i + 1]) for i in range(len(path) - 1))


def spl(is_success: bool, shortest_path_length: float,
        travelled_path_length: float) -> float:
    """SPL = S * l / max(l, p)

    S = success indicator, l = geodesic shortest-path length from start to goal,
    p = length the agent actually travelled.

    A failed episode scores 0 regardless of efficiency. A successful episode
    that takes the shortest route scores 1. Taking a longer route than
    necessary scales the score down; `max` prevents a shorter-than-geodesic
    path (possible from goal-radius slack) from scoring above 1.
    """
    if not is_success:
        return 0.0
    denominator = max(shortest_path_length, travelled_path_length)
    if denominator <= 0.0:
        # Degenerate: agent starts at the goal and never moves. Counted as a
        # perfect episode -- it succeeded and travelled no unnecessary distance.
        return 1.0
    return shortest_path_length / denominator


# ---------------------------------------------------------------------------
# nDTW
# ---------------------------------------------------------------------------
def dtw_distance(query: Sequence[Point], reference: Sequence[Point],
                 distance_fn: DistanceFn = euclidean) -> float:
    """Dynamic Time Warping cost between two trajectories.

    Standard DTW with unit step costs: every element of each sequence must be
    matched at least once, and matches must preserve order.
    """
    if not query or not reference:
        raise ValueError("query and reference must both be non-empty")

    n, m = len(query), len(reference)
    inf = float("inf")
    prev = [inf] * (m + 1)
    prev[0] = 0.0

    for i in range(1, n + 1):
        cur = [inf] * (m + 1)
        for j in range(1, m + 1):
            cost = distance_fn(query[i - 1], reference[j - 1])
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[m]


def resample_path(path: Sequence[Point], spacing: float) -> list:
    """Resample a polyline to approximately `spacing` metres between points.

    Straight-line interpolation between consecutive vertices. Used to bring a
    sparse reference path to the same granularity as a dense agent path.
    """
    if len(path) < 2:
        return [tuple(p) for p in path]

    out = [tuple(path[0])]
    for a, b in zip(path, path[1:]):
        seg = euclidean(a, b)
        n = max(1, int(math.ceil(seg / spacing)))
        for i in range(1, n + 1):
            f = i / n
            out.append(tuple(x + (y - x) * f for x, y in zip(a, b)))
    return out


def ndtw(query: Sequence[Point], reference: Sequence[Point],
         distance_fn: DistanceFn = euclidean,
         threshold: float = SUCCESS_THRESHOLD_M,
         resample_spacing: float = 0.25) -> float:
    """nDTW = exp( -DTW(Q, R) / (|R| * d_th) )     (Ilharco et al., ref [19])

    1.0 when the agent reproduces the reference exactly; decays towards 0 as it
    deviates.

    --------------------------------------------------------------------
    GRANULARITY -- why the reference is resampled first
    --------------------------------------------------------------------
    The published formula assumes Q and R are sequences of COMPARABLE
    granularity; in discrete R2R both are viewpoint sequences of similar
    length. In R2R-CE they are not: the agent path is one point per 0.25 m
    step while `reference_path` is sparse nav-graph waypoints ~1.7 m apart,
    roughly 5x coarser.

    DTW must match every one of the ~31 agent points to one of the ~6
    waypoints, so the cost accumulates ~31 terms while the normaliser
    |R| * d_th counts only 6. Measured on a PERFECT expert rollout that passes
    within 0.2-1.0 m of every waypoint: raw DTW 18.76 against a normaliser of
    18.0, giving nDTW 0.19 -- a near-zero score for a faithful trajectory.

    The reference waypoints are samples of a continuous path, so the fix is to
    restore that path: resample to `resample_spacing` (0.25 m, the FWD
    primitive) before comparing. Then |Q| ~ |R| and the formula behaves as
    intended. Pass `resample_spacing=0` to disable and get the literal formula.

    BOTH sequences are resampled, not just the reference. Resampling one side
    alone reintroduces the same asymmetry in reverse and would break the
    identity property -- a path compared against itself must score exactly 1.0.
    """
    if not reference:
        raise ValueError("reference path is empty")
    if not query:
        raise ValueError("query path is empty")

    if resample_spacing and resample_spacing > 0:
        q = resample_path(query, resample_spacing)
        ref = resample_path(reference, resample_spacing)
    else:
        q, ref = query, reference

    cost = dtw_distance(q, ref, distance_fn)
    return math.exp(-cost / (len(ref) * threshold))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EpisodeMetrics:
    """All Sec. 4.1 metrics for one episode."""

    ne: float
    success: bool
    oracle_success: bool
    spl: float
    ndtw: float
    path_length: float
    oracle_ne: float

    def as_dict(self) -> dict:
        return {
            "NE": self.ne,
            "SR": float(self.success),
            "OS": float(self.oracle_success),
            "SPL": self.spl,
            "nDTW": self.ndtw,
            "path_length": self.path_length,
            "oracle_NE": self.oracle_ne,
        }


def evaluate_episode(
    path: Sequence[Point],
    goal_position: Point,
    reference_path: Sequence[Point],
    shortest_path_length: float,
    distance_fn: DistanceFn = euclidean,
    threshold: float = SUCCESS_THRESHOLD_M,
) -> EpisodeMetrics:
    """Compute every Sec. 4.1 metric for a single episode.

    Args:
        path:                 positions the agent visited, in order. The LAST
                              entry is treated as the stopping location.
        goal_position:        the episode's goal.
        reference_path:       expert reference path, for nDTW.
        shortest_path_length: geodesic start-to-goal distance (`l` in SPL). For
                              R2R-CE this is episode["info"]["geodesic_distance"].
        distance_fn:          geodesic in production, Euclidean in tests.
    """
    if not path:
        raise ValueError("path is empty")

    ne = navigation_error(path[-1], goal_position, distance_fn)
    is_success = success(ne, threshold)
    travelled = path_length(path)

    return EpisodeMetrics(
        ne=ne,
        success=is_success,
        oracle_success=oracle_success(path, goal_position, distance_fn, threshold),
        spl=spl(is_success, shortest_path_length, travelled),
        ndtw=ndtw(path, reference_path, distance_fn, threshold),
        path_length=travelled,
        oracle_ne=oracle_navigation_error(path, goal_position, distance_fn),
    )


def aggregate(results: Sequence[EpisodeMetrics]) -> dict:
    """Mean of each metric across episodes -- the Sec. 4.1 reporting format.

    SR and OS become fractions of episodes; NE, SPL and nDTW become means.
    """
    if not results:
        raise ValueError("no episode results to aggregate")
    n = len(results)
    return {
        "NE": sum(r.ne for r in results) / n,
        "SR": sum(r.success for r in results) / n,
        "OS": sum(r.oracle_success for r in results) / n,
        "SPL": sum(r.spl for r in results) / n,
        "nDTW": sum(r.ndtw for r in results) / n,
        "n_episodes": n,
    }
