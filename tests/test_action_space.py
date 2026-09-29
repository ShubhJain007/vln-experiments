"""
Tests for Eq. 3 — A = {FWD, LEFT, RIGHT, STOP}

Required tests per AGENTS.md Sec. 3.1:
  - exactly 4 actions
  - the mapping to Habitat primitives matches the paper's real-robot params
    (FWD = 0.25 m, turns = 15 deg)

Run:  PYTHONPATH="" python -m pytest tests/test_action_space.py -v
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from model.action_space import (  # noqa: E402
    ACTIONS,
    FORWARD_STEP_METERS,
    HABITAT_ACTION_NAMES,
    NUM_ACTIONS,
    TURN_ANGLE_DEGREES,
    Action,
    action_from_index,
    action_from_name,
    habitat_agent_action_config,
)


# ---------------------------------------------------------------------------
# Eq. 3: exactly four actions
# ---------------------------------------------------------------------------
def test_exactly_four_actions():
    """AGENTS.md Sec. 3.1: exactly 4 actions."""
    assert NUM_ACTIONS == 4
    assert len(ACTIONS) == 4
    assert len(list(Action)) == 4


def test_action_set_matches_paper():
    """A = {FWD, LEFT, RIGHT, STOP}, exactly -- no extras, no renames."""
    assert {a.name for a in Action} == {"FWD", "LEFT", "RIGHT", "STOP"}


def test_action_indices_are_contiguous_from_zero():
    """Logit index i must map to ACTIONS[i]; gaps would corrupt the head."""
    assert [a.value for a in ACTIONS] == [0, 1, 2, 3]


def test_canonical_ordering_is_fixed():
    """The ordering defines what every logit means. Changing it invalidates
    every trained checkpoint, so pin it explicitly."""
    assert [a.name for a in ACTIONS] == ["FWD", "LEFT", "RIGHT", "STOP"]


# ---------------------------------------------------------------------------
# Motion primitives: the paper's real-robot parameters
# ---------------------------------------------------------------------------
def test_forward_step_is_quarter_metre():
    """AGENTS.md Sec. 3.1: FWD = 0.25 m."""
    assert FORWARD_STEP_METERS == 0.25


def test_turn_angle_is_fifteen_degrees():
    """AGENTS.md Sec. 3.1: turns = 15 degrees."""
    assert TURN_ANGLE_DEGREES == 15.0


def test_habitat_action_config_states_amounts_explicitly():
    """Amounts must be set explicitly, never inherited from habitat defaults,
    so a habitat version bump cannot silently change action semantics."""
    cfg = habitat_agent_action_config()
    assert cfg["move_forward"]["amount"] == 0.25
    assert cfg["turn_left"]["amount"] == 15.0
    assert cfg["turn_right"]["amount"] == 15.0


def test_habitat_config_covers_all_non_terminal_actions():
    """Every action that habitat actually executes needs a primitive spec.
    STOP is excluded: it is policy-side, habitat has no episode-ending action."""
    cfg = habitat_agent_action_config()
    expected = {a.habitat_name for a in ACTIONS if not a.is_terminal}
    assert set(cfg) == expected


def test_turn_angle_differs_from_habitat_default():
    """REGRESSION GUARD. habitat-sim 0.3.3 defaults to a 10-degree turn, but
    the paper specifies 15. Anyone who drops our explicit config and falls back
    to habitat's defaults gets 10 degrees with no error raised anywhere. This
    test exists so that silent deviation is caught here instead of showing up
    as unexplained metric drift much later."""
    HABITAT_DEFAULT_TURN_DEGREES = 10.0     # measured, habitat-sim 0.3.3
    assert TURN_ANGLE_DEGREES == 15.0
    assert TURN_ANGLE_DEGREES != HABITAT_DEFAULT_TURN_DEGREES
    assert habitat_agent_action_config()["turn_left"]["amount"] == 15.0


def test_habitat_config_is_not_shared_mutable_state():
    """Returning a fresh dict each call prevents one caller's mutation from
    silently changing another's action semantics."""
    a = habitat_agent_action_config()
    a["move_forward"]["amount"] = 99.0
    assert habitat_agent_action_config()["move_forward"]["amount"] == 0.25


# ---------------------------------------------------------------------------
# Habitat name mapping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("action,expected", [
    (Action.FWD, "move_forward"),
    (Action.LEFT, "turn_left"),
    (Action.RIGHT, "turn_right"),
    (Action.STOP, "stop"),
])
def test_habitat_name_mapping(action, expected):
    assert action.habitat_name == expected


def test_every_action_has_a_habitat_name():
    assert set(HABITAT_ACTION_NAMES) == {a.name for a in Action}


def test_habitat_names_are_unique():
    """Two actions mapping to the same primitive would make them indistinguishable."""
    names = [a.habitat_name for a in ACTIONS]
    assert len(set(names)) == len(names)


# ---------------------------------------------------------------------------
# STOP semantics (Eq. 2: terminates when a_t = STOP)
# ---------------------------------------------------------------------------
def test_only_stop_is_terminal():
    assert Action.STOP.is_terminal
    assert not Action.FWD.is_terminal
    assert not Action.LEFT.is_terminal
    assert not Action.RIGHT.is_terminal


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["FWD", "LEFT", "RIGHT", "STOP"])
def test_action_from_name_roundtrip(name):
    assert action_from_name(name).name == name


def test_action_from_name_is_case_insensitive():
    assert action_from_name("fwd") is Action.FWD


def test_action_from_name_rejects_unknown():
    with pytest.raises(ValueError):
        action_from_name("BACKWARD")


def test_action_from_index_roundtrip():
    for i, a in enumerate(ACTIONS):
        assert action_from_index(i) is a
        assert int(a) == i


@pytest.mark.parametrize("bad", [-1, 4, 99])
def test_action_from_index_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        action_from_index(bad)


# ---------------------------------------------------------------------------
# No backward action (a common way to accidentally deviate)
# ---------------------------------------------------------------------------
def test_no_backward_or_strafe_actions():
    """The paper's action space has no reverse or lateral motion. Adding one
    would be a STOP-and-ask deviation (AGENTS.md Sec. 0.4)."""
    names = {a.name for a in Action}
    for forbidden in ("BACK", "BACKWARD", "STRAFE_LEFT", "STRAFE_RIGHT", "UP", "DOWN"):
        assert forbidden not in names
