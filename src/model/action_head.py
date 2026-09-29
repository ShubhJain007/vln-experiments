"""
Eq. 7 — Action distribution.

    pi_theta(a_t | x, o_t, z_{t-1}) = Softmax(W_a h_t^act),   a_t in A

and the same projection in Eq. 13:

    p_theta(.) = Softmax(W_LM h_t^act)

NO NEW HEAD. Sec. 3.3 states actions are decoded by the backbone's native LM
output projection, so W_a IS W_LM -- the model's existing `lm_head`. This
resolves AGENTS.md Sec. 8 item 5 ("whether W_a and W_LM are the same matrix"):
they are, and the code uses one matrix for both.

The four actions are mapped to EXISTING single tokens in the vocabulary, so
nothing is added and no embedding resize happens:

    FWD -> "forward" (13435)    LEFT  -> "left"  (2359)
    RIGHT -> "right" (1291)     STOP  -> "stop"  (9495)

Verified single-token at build time (see resolve_action_token_ids). Reusing
real words rather than fresh special tokens keeps W_LM completely untouched and
starts from rows that already carry the right pretrained semantics.

Softmax is taken over the FOUR action logits, not the full 151669-token
vocabulary. Eq. 7 writes "a_t in A", i.e. a distribution over the action set,
and AGENTS.md's required test (uniform loss ~ ln(4) = 1.386, not
ln(151669) = 11.93) confirms that reading.
"""

# --- ROS guard: strip /opt/ros/* from sys.path before any other import. ------
import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]
# ----------------------------------------------------------------------------

from typing import Dict, List

import torch
import torch.nn.functional as F

from .action_space import ACTIONS, NUM_ACTIONS, Action

# Vocabulary words standing in for each action. Chosen because each is a single
# existing token and already carries the right meaning to the pretrained model.
ACTION_TOKEN_STRINGS: Dict[str, str] = {
    "FWD": "forward",
    "LEFT": "left",
    "RIGHT": "right",
    "STOP": "stop",
}


def resolve_action_token_ids(tokenizer) -> List[int]:
    """Token id for each action, in canonical ACTIONS order.

    Raises if any action word is not a single token -- a multi-token action
    could not be decoded by one application of W_LM, which would break Eq. 7's
    single-projection form and require the STOP-and-ask in AGENTS.md Sec. 3.1.
    """
    ids: List[int] = []
    for action in ACTIONS:
        word = ACTION_TOKEN_STRINGS[action.name]
        encoded = tokenizer(word, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            raise ValueError(
                f"action {action.name!r} maps to {word!r}, which tokenizes to "
                f"{len(encoded)} tokens {encoded}. Eq. 7 needs one token per "
                f"action so a single W_LM projection yields the distribution."
            )
        ids.append(encoded[0])

    if len(set(ids)) != NUM_ACTIONS:
        raise ValueError(f"action token ids are not distinct: {ids}")
    return ids


class ActionHead:
    """Eq. 7:  pi_theta(a_t | .) = Softmax(W_a h_t^act),  W_a == W_LM

    Holds no parameters of its own. It selects four rows of the backbone's
    existing output projection; there is nothing here to train beyond what
    LoRA already touches in the backbone.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.action_token_ids = resolve_action_token_ids(tokenizer)
        self._ids_tensor = torch.tensor(self.action_token_ids, device=model.device)

    @property
    def lm_head(self):
        """W_LM -- the backbone's native output projection (Eq. 13)."""
        return self.model.lm_head

    # -- Eq. 7 --------------------------------------------------------------
    def logits(self, h_act: torch.Tensor) -> torch.Tensor:
        """W_a h_t^act, restricted to the four action tokens.

        Args:
            h_act: (d,) or (B, d) -- hidden state(s) at the action position,
                   read from H_t (Eq. 6).

        Returns:
            (4,) or (B, 4) logits in canonical ACTIONS order.
        """
        full = self.lm_head(h_act)                       # (..., vocab)
        return full.index_select(-1, self._ids_tensor.to(full.device))

    def probs(self, h_act: torch.Tensor) -> torch.Tensor:
        """Eq. 7: Softmax over the four action logits. Sums to 1 over A."""
        return F.softmax(self.logits(h_act).float(), dim=-1)

    def predict(self, h_act: torch.Tensor) -> Action:
        """Greedy action for a single step (inference)."""
        if h_act.ndim != 1:
            raise ValueError(f"predict expects a single (d,) state, got {tuple(h_act.shape)}")
        return ACTIONS[int(self.logits(h_act).argmax().item())]
