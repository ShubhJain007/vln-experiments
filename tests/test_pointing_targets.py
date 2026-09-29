"""
Pointing-target geometry tests.

The projection is the foundation of the whole pointing approach: if yaw or the
image axes are wrong, every label is wrong in a way that still LOOKS plausible
(coordinates in range, loss decreasing) and would only surface as a policy that
confidently steers the wrong way. Hence hand-computed cases, not smoke tests.
"""

import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]

import math

import numpy as np
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT / "src"))

from data.pointing_targets import (  # noqa: E402
    agent_frame_delta, focal_px, project_to_image, quat_to_yaw,
    targets_for_episode,
)

RES = (448, 448)
NO_OFFSET = (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# focal length / projection
# ---------------------------------------------------------------------------
def test_focal_length_for_90_degrees():
    """f = (W/2)/tan(hfov/2); at 90 deg the half-angle is 45 deg so f == W/2."""
    assert focal_px(90.0, 448) == pytest.approx(224.0)


def test_point_straight_ahead_projects_to_centre():
    u, v, d = project_to_image([0, 0, -3], [0, 0, 0], 0.0, 90, RES, NO_OFFSET)
    assert (u, v) == pytest.approx((224.0, 224.0))
    assert d == pytest.approx(3.0)


def test_forward_is_negative_z():
    """Habitat's agent frame: -Z forward. +Z must be BEHIND the camera."""
    _, _, d_front = project_to_image([0, 0, -2], [0, 0, 0], 0.0, 90, RES, NO_OFFSET)
    u_back, _, d_back = project_to_image([0, 0, 2], [0, 0, 0], 0.0, 90, RES, NO_OFFSET)
    assert d_front > 0
    assert d_back < 0 and u_back is None


def test_right_of_agent_projects_right_of_centre():
    u, _, _ = project_to_image([1, 0, -3], [0, 0, 0], 0.0, 90, RES, NO_OFFSET)
    assert u > 224.0


def test_above_agent_projects_above_centre():
    """Image v grows DOWNWARD, so a higher world point gets a SMALLER v."""
    _, v, _ = project_to_image([0, 1, -3], [0, 0, 0], 0.0, 90, RES, NO_OFFSET)
    assert v < 224.0


def test_frustum_edge_at_45_degrees():
    """At hfov=90 a point 45 deg off-axis lands exactly on the frame edge."""
    u, _, _ = project_to_image([3, 0, -3], [0, 0, 0], 0.0, 90, RES, NO_OFFSET)
    assert u == pytest.approx(448.0)


def test_camera_height_offset_shifts_projection_down():
    """The camera sits 1.5 m above the agent, so a point at agent height is
    BELOW the optical axis and lands in the lower half of the image."""
    _, v_no, _ = project_to_image([0, 0, -3], [0, 0, 0], 0.0, 90, RES, NO_OFFSET)
    _, v_off, _ = project_to_image([0, 0, -3], [0, 0, 0], 0.0, 90, RES, (0, 1.5, 0))
    assert v_off > v_no == pytest.approx(224.0)


# ---------------------------------------------------------------------------
# yaw handling -- the part reconstruction got wrong 6% of the time
# ---------------------------------------------------------------------------
def test_yaw_rotates_the_world_into_view():
    """A point due +X is invisible facing yaw=0, but centred after turning
    -90 deg to face +X."""
    u, _, d = project_to_image([3, 0, 0], [0, 0, 0], 0.0, 90, RES, NO_OFFSET)
    assert d <= 1e-6 or u is None or not (0 <= u < 448)
    u2, _, d2 = project_to_image([3, 0, 0], [0, 0, 0], -math.pi / 2, 90, RES, NO_OFFSET)
    assert d2 == pytest.approx(3.0)
    assert u2 == pytest.approx(224.0)


def test_quat_to_yaw_identity_and_quarter_turn():
    assert quat_to_yaw([0, 0, 0, 1]) == pytest.approx(0.0)
    q = [0, math.sin(math.pi / 4), 0, math.cos(math.pi / 4)]      # +90 deg about Y
    assert quat_to_yaw(q) == pytest.approx(math.pi / 2)


def test_agent_frame_delta_forward_and_right():
    dx, dy = agent_frame_delta([0, 0, -2], [0, 0, 0], 0.0)
    assert dx == pytest.approx(0.0) and dy == pytest.approx(2.0)   # dead ahead
    dx, dy = agent_frame_delta([2, 0, 0], [0, 0, 0], 0.0)
    assert dx == pytest.approx(2.0) and dy == pytest.approx(0.0)   # due right


# ---------------------------------------------------------------------------
# episode-level target construction
# ---------------------------------------------------------------------------
def _straight_episode(n=6):
    """Agent walks along -Z, facing -Z (yaw 0) the whole way."""
    return {
        "positions": [[0.0, 0.0, -0.25 * i] for i in range(n)],
        "rotations": [[0.0, 0.0, 0.0, 1.0]] * n,
        "camera": {"hfov_deg": 90.0, "resolution": [448, 448],
                   "position_offset": [0.0, 1.5, 0.0]},
    }


def test_last_step_is_stop():
    ts = targets_for_episode(_straight_episode())
    assert ts[-1].is_stop
    assert not any(t.is_stop for t in ts[:-1])


def test_one_target_per_step():
    ep = _straight_episode(7)
    assert len(targets_for_episode(ep)) == 7


def test_waypoints_closer_than_camera_height_are_not_visible():
    """GEOMETRIC FLOOR: the camera is 1.5 m up with a +/-45 deg vertical FOV,
    so a floor-level waypoint only enters frame beyond ~1.5 m. At 0.25 m per
    step that is ~6 steps. Not a bug -- you do not point at your own feet."""
    ts = targets_for_episode(_straight_episode(6))     # max reach 1.25 m
    assert not any(t.visible for t in ts)


def test_straight_path_points_near_horizontal_centre():
    """Walking straight, a waypoint past the 1.5 m floor is dead ahead -> u = 0.5."""
    ts = targets_for_episode(_straight_episode(16))    # reaches 3.75 m
    vis = [t for t in ts if t.visible]
    assert vis, "no visible waypoints -- projection or horizon is wrong"
    for t in vis:
        assert t.u == pytest.approx(0.5, abs=1e-6)


def test_normalised_coords_in_unit_range():
    for t in targets_for_episode(_straight_episode(16)):
        if t.visible:
            assert 0.0 <= t.u <= 1.0 and 0.0 <= t.v <= 1.0


def test_picks_the_furthest_visible_waypoint():
    """Robostral Sec. 2.2 points at the FURTHEST visible waypoint, so on a
    straight corridor the lookahead should reach the horizon cap."""
    ts = targets_for_episode(_straight_episode(16))
    assert ts[0].lookahead == 15         # all remaining steps stay in frame


def test_missing_rotations_raises_clearly():
    ep = _straight_episode()
    del ep["rotations"]
    with pytest.raises(KeyError, match="re-collect"):
        targets_for_episode(ep)


def test_behind_agent_falls_back_to_displacement():
    """Waypoint behind the agent: pointing is impossible, metric fallback must
    still be defined (Robostral Eq. 2)."""
    ep = {
        "positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
        "rotations": [[0.0, 0.0, 0.0, 1.0]] * 3,     # facing -Z, path goes +Z
        "camera": {"hfov_deg": 90.0, "resolution": [448, 448],
                   "position_offset": [0.0, 1.5, 0.0]},
    }
    t0 = targets_for_episode(ep)[0]
    assert not t0.visible and t0.u is None
    assert t0.dy < 0                       # target is behind
    assert abs(t0.dtheta) > math.pi / 2    # requires a large turn


# ---------------------------------------------------------------------------
# Stationary in-place turns -- the bug that silently taught "go straight"
# through every turn (expert-LEFT steps were labelled FWD 28% of the time,
# and RIGHT more often than LEFT). Agreement with the expert's own action
# went 0.689 -> 0.972 once these two cases were handled.
# ---------------------------------------------------------------------------
def _turn_then_walk(n_turn=6, n_fwd=10):
    """Habitat turns IN PLACE: position is unchanged across the turn steps."""
    import math as _m
    positions, rotations = [], []
    for i in range(n_turn):                       # rotating, not moving
        positions.append([0.0, 0.0, 0.0])
        yaw = _m.radians(15 * i)
        rotations.append([0.0, _m.sin(yaw / 2), 0.0, _m.cos(yaw / 2)])
    yaw = _m.radians(15 * n_turn)
    q = [0.0, _m.sin(yaw / 2), 0.0, _m.cos(yaw / 2)]
    fwd = np.array([-_m.sin(yaw), 0.0, -_m.cos(yaw)])
    for j in range(1, n_fwd + 1):                 # now walking
        positions.append((fwd * 0.25 * j).tolist())
        rotations.append(q)
    return {"positions": positions, "rotations": rotations,
            "camera": {"hfov_deg": 90.0, "resolution": [448, 448],
                       "position_offset": [0.0, 1.5, 0.0]}}


def test_stationary_turn_steps_get_a_real_bearing():
    """During an in-place turn the next few positions are IDENTICAL. The
    control waypoint must skip them, or atan2(0, 0) labels the step 'straight'
    exactly where the expert is turning."""
    ts = targets_for_episode(_turn_then_walk())
    for t in ts[:6]:                              # the turning steps
        assert not t.is_stop
        assert (abs(t.dx) + abs(t.dy)) > 1e-6, "degenerate zero-displacement bearing"


def test_control_target_is_first_displaced_not_furthest():
    """Taking the FURTHEST waypoint inside the control horizon instead of the
    first genuinely-displaced one drops expert agreement 0.93 -> 0.75, because
    on a curving path it already aims past the turn."""
    ep = _turn_then_walk(n_turn=2, n_fwd=10)
    t0 = targets_for_episode(ep, control_lookahead=8)[0]
    # first real move is 0.25 m away, not the 8-steps-ahead 2.0 m one
    assert math.hypot(t0.dx, t0.dy) == pytest.approx(0.25, abs=0.02)


def test_point_and_control_horizons_are_independent():
    """Pointing answers 'where am I headed', control answers 'what now'. They
    must not be forced to share a horizon."""
    ep = _turn_then_walk(n_turn=0, n_fwd=24)
    near = targets_for_episode(ep, max_lookahead=24, control_lookahead=2)[0]
    far = targets_for_episode(ep, max_lookahead=24, control_lookahead=8)[0]
    # same pointing target regardless of the control horizon
    assert near.u == pytest.approx(far.u)
    assert near.visible == far.visible


def test_agent_that_never_moves_again_holds_heading():
    ep = {"positions": [[0.0, 0.0, 0.0]] * 4,
          "rotations": [[0.0, 0.0, 0.0, 1.0]] * 4,
          "camera": {"hfov_deg": 90.0, "resolution": [448, 448],
                     "position_offset": [0.0, 1.5, 0.0]}}
    ts = targets_for_episode(ep)
    for t in ts[:-1]:
        assert (t.dx, t.dy, t.dtheta) == (0.0, 0.0, 0.0)
    assert ts[-1].is_stop
