"""
Eq. 3 — Discrete action set.

    A = {FWD, LEFT, RIGHT, STOP}

"where each action corresponds to a small motion primitive in continuous
space" (Sec. 3.1). The episode terminates when a_t = STOP or t = T_max (Eq. 2).

Motion primitives (AGENTS.md Sec. 3.1, matching the paper's real-robot params):
    FWD           = 0.25 m forward
    LEFT / RIGHT  = 15 degrees turn
    STOP          = terminate episode

WARNING -- habitat's defaults do NOT match the paper. Measured on
habitat-sim 0.3.3:

    habitat default move_forward = 0.25 m   -> matches the paper
    habitat default turn_left    = 10 deg   -> paper says 15 deg
    habitat default turn_right   = 10 deg   -> paper says 15 deg

Relying on habitat's defaults would silently give 10-degree turns, changing
the action semantics with no error anywhere. Every amount is therefore set
EXPLICITLY from the constants below, never inherited.
"""

from enum import IntEnum
from typing import Dict, List

# --- Paper-specified motion primitives (AGENTS.md Sec. 3.1) -----------------
FORWARD_STEP_METERS = 0.25
TURN_ANGLE_DEGREES = 15.0

# Habitat action names for each primitive. STOP is handled by the policy, not
# by habitat -- there is no habitat primitive that ends an episode.
HABITAT_ACTION_NAMES: Dict[str, str] = {
    "FWD": "move_forward",
    "LEFT": "turn_left",
    "RIGHT": "turn_right",
    "STOP": "stop",
}


class Action(IntEnum):
    """Eq. 3:  A = {FWD, LEFT, RIGHT, STOP}

    Integer values are the class indices used by the action head (Eq. 7/13).
    The ordering is fixed: it defines the meaning of every logit vector, so
    changing it silently invalidates every trained checkpoint.
    """

    FWD = 0
    LEFT = 1
    RIGHT = 2
    STOP = 3

    @property
    def habitat_name(self) -> str:
        """habitat-sim action name for this primitive."""
        return HABITAT_ACTION_NAMES[self.name]

    @property
    def is_terminal(self) -> bool:
        """True for STOP, which ends the episode (Eq. 2)."""
        return self is Action.STOP


# Canonical ordering. Index i corresponds to logit i.
ACTIONS: List[Action] = [Action.FWD, Action.LEFT, Action.RIGHT, Action.STOP]
NUM_ACTIONS: int = len(ACTIONS)          # 4, per Eq. 3


def action_from_name(name: str) -> Action:
    """Look up an Action by its paper name ('FWD', 'LEFT', 'RIGHT', 'STOP')."""
    try:
        return Action[name.upper()]
    except KeyError:
        raise ValueError(
            f"unknown action {name!r}; expected one of {[a.name for a in ACTIONS]}"
        ) from None


def action_from_index(index: int) -> Action:
    """Map a head/logit index back to its Action."""
    if not 0 <= index < NUM_ACTIONS:
        raise ValueError(f"action index {index} out of range [0, {NUM_ACTIONS})")
    return ACTIONS[index]


def habitat_agent_action_config() -> dict:
    """Motion primitive parameters to install into a habitat AgentConfiguration.

    Returned as plain data so this module never imports habitat_sim -- habitat
    lives in a separate Python 3.9 env (`habitat_render`) and is not importable
    from the training env. The renderer-side code builds the real
    ActionSpec/ActuationSpec objects from these values.

    Amounts are stated explicitly rather than relying on habitat's defaults.
    """
    return {
        "move_forward": {"amount": FORWARD_STEP_METERS},   # metres
        "turn_left": {"amount": TURN_ANGLE_DEGREES},       # degrees
        "turn_right": {"amount": TURN_ANGLE_DEGREES},      # degrees
    }
