"""
Tests for Eq. 7 (action distribution) and Eq. 13 (L_act).

Required tests per AGENTS.md Sec. 3.1:
  Eq. 7/13 — probabilities sum to 1; loss on a uniform-random model
             ~ ln(4) = 1.386
  L_act    — loss -> 0 when logits are one-hot on the correct action

Also asserts the structural claim that makes this faithful to Sec. 3.3: no new
head exists, W_a IS the backbone's native lm_head, and the vocabulary was not
resized.

Run:  PYTHONPATH="" python -m pytest tests/test_action_head.py -v
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import math
import pathlib

import numpy as np
import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
from losses.action_loss import action_loss  # noqa: E402
from model.action_head import (  # noqa: E402
    ACTION_TOKEN_STRINGS,
    ActionHead,
    resolve_action_token_ids,
)
from model.action_space import ACTIONS, NUM_ACTIONS, Action  # noqa: E402
from model.backbone import Backbone  # noqa: E402
from model.input_sequence import InputSequenceBuilder  # noqa: E402
from model.vision_encoder import VisionEncoder  # noqa: E402

EXPECTED_D = 2048
LN4 = math.log(4)


@pytest.fixture(scope="module")
def real_h_act(encoder):
    """A genuine h_t^act from a real Eq. 4 -> 5 -> 6 pass."""
    builder = InputSequenceBuilder(encoder.model, encoder.processor.tokenizer)
    bb = Backbone(encoder.model)
    frame = np.random.default_rng(0).integers(0, 255, (448, 448, 3), dtype=np.uint8)
    seq = builder.build("Walk forward and stop at the door.", encoder.encode(frame))
    return bb.forward(seq)[-1]


# ---------------------------------------------------------------------------
# Native LM projection — no new head (Sec. 3.3)
# ---------------------------------------------------------------------------
def test_action_head_uses_native_lm_head(head, encoder):
    """W_a IS W_LM -- resolves AGENTS.md Sec. 8 item 5."""
    assert head.lm_head is encoder.model.lm_head


def test_action_head_adds_no_parameters(head):
    """A genuinely new head would introduce trainable weights. This must not."""
    assert not hasattr(head, "weight")
    assert not isinstance(head, torch.nn.Module)


def test_vocabulary_was_not_resized(head, encoder):
    """Action tokens must be EXISTING vocabulary entries, not appended ones."""
    vocab_size = encoder.model.config.text_config.vocab_size
    for tid in head.action_token_ids:
        assert 0 <= tid < vocab_size


def test_four_distinct_single_token_actions(head):
    assert len(head.action_token_ids) == NUM_ACTIONS
    assert len(set(head.action_token_ids)) == NUM_ACTIONS


def test_action_token_ids_match_expected_words(encoder):
    tok = encoder.processor.tokenizer
    ids = resolve_action_token_ids(tok)
    for action, tid in zip(ACTIONS, ids):
        word = ACTION_TOKEN_STRINGS[action.name]
        assert tok(word, add_special_tokens=False)["input_ids"] == [tid]


def test_multi_token_action_word_is_rejected(encoder):
    """If a chosen word were multi-token, Eq. 7's single projection would not
    yield the distribution -- that must raise, not silently mis-decode."""
    import model.action_head as mod
    tok = encoder.processor.tokenizer
    original = mod.ACTION_TOKEN_STRINGS.copy()
    try:
        mod.ACTION_TOKEN_STRINGS["FWD"] = "move forward quickly"
        with pytest.raises(ValueError, match="tokenizes to"):
            resolve_action_token_ids(tok)
    finally:
        mod.ACTION_TOKEN_STRINGS.clear()
        mod.ACTION_TOKEN_STRINGS.update(original)


# ---------------------------------------------------------------------------
# Eq. 7: the distribution
# ---------------------------------------------------------------------------
def test_probabilities_sum_to_one(head, real_h_act):
    """AGENTS.md Sec. 3.1: probabilities sum to 1."""
    p = head.probs(real_h_act)
    assert p.shape == (NUM_ACTIONS,)
    assert abs(p.sum().item() - 1.0) < 1e-5


def test_probabilities_are_non_negative(head, real_h_act):
    assert (head.probs(real_h_act) >= 0).all()


def test_logits_shape_single_and_batched(head):
    single = head.logits(torch.zeros(EXPECTED_D, device=head.model.device,
                                     dtype=head.model.dtype))
    assert single.shape == (NUM_ACTIONS,)
    batched = head.logits(torch.zeros(7, EXPECTED_D, device=head.model.device,
                                      dtype=head.model.dtype))
    assert batched.shape == (7, NUM_ACTIONS)


def test_batched_probs_each_sum_to_one(head):
    h = torch.randn(5, EXPECTED_D, device=head.model.device, dtype=head.model.dtype)
    p = head.probs(h)
    assert torch.allclose(p.sum(-1), torch.ones(5, device=p.device), atol=1e-5)


def test_softmax_is_over_four_actions_not_full_vocab(head, real_h_act):
    """Restricted to A, per Eq. 7's 'a_t in A'. Over the full vocabulary the
    four probabilities would sum to far less than 1."""
    assert abs(head.probs(real_h_act).sum().item() - 1.0) < 1e-5


def test_logits_are_selected_rows_of_full_projection(head, real_h_act):
    """The action logits must literally be W_LM's rows for those tokens."""
    full = head.lm_head(real_h_act)
    for i, tid in enumerate(head.action_token_ids):
        assert head.logits(real_h_act)[i] == full[tid]


def test_predict_returns_an_action(head, real_h_act):
    assert head.predict(real_h_act) in set(ACTIONS)


def test_predict_matches_argmax(head, real_h_act):
    idx = int(head.logits(real_h_act).argmax().item())
    assert head.predict(real_h_act) is ACTIONS[idx]


def test_predict_rejects_batched_input(head):
    with pytest.raises(ValueError, match="single"):
        head.predict(torch.zeros(2, EXPECTED_D, device=head.model.device,
                                 dtype=head.model.dtype))


# ---------------------------------------------------------------------------
# Eq. 13: L_act
# ---------------------------------------------------------------------------
def test_uniform_model_loss_is_ln4():
    """AGENTS.md Sec. 3.1: loss on a uniform-random model ~ ln(4) = 1.386.

    That criterion is inherently PER-STEP, so it is checked with the diagnostic
    "mean" reduction. The training default is "sum" (Eq. 13 as written), under
    which the same uniform model gives T * ln(4) -- also asserted below.

    This is what proves the softmax is over 4 actions and not the full
    vocabulary: uniform over 151669 tokens would give ln(151669) = 11.93."""
    logits = torch.zeros(100, NUM_ACTIONS)          # uniform
    targets = torch.randint(0, NUM_ACTIONS, (100,))

    per_step = action_loss(logits, targets, reduction="mean").item()
    assert abs(per_step - LN4) < 1e-4, f"expected ln(4)={LN4:.4f}, got {per_step:.4f}"
    assert abs(LN4 - 1.386) < 1e-3

    # Eq. 13's sum over T=100 steps
    total = action_loss(logits, targets).item()      # default reduction="sum"
    assert abs(total - 100 * LN4) < 1e-2


def test_default_reduction_is_sum_per_eq13():
    """Eq. 13 is written as a sum over t. The default must implement that, not
    a per-step mean -- using mean would silently rescale lambda in Eq. 15 to
    lambda * T/(T-2), corrupting the paper's one stated hyperparameter."""
    logits = torch.zeros(7, NUM_ACTIONS)
    targets = torch.zeros(7, dtype=torch.long)
    assert action_loss(logits, targets).item() == pytest.approx(7 * LN4, abs=1e-4)


def test_batch_loss_sums_within_trajectory_and_means_across():
    """Eq. 15's E_{tau~D}: sum within each trajectory, mean across the batch.

    Trajectories of DIFFERENT lengths -- a longer one must contribute a larger
    total, which per-step normalisation would erase."""
    from losses.action_loss import batch_action_loss

    short = (torch.zeros(2, NUM_ACTIONS), torch.zeros(2, dtype=torch.long))
    long = (torch.zeros(10, NUM_ACTIONS), torch.zeros(10, dtype=torch.long))
    got = batch_action_loss([short[0], long[0]], [short[1], long[1]]).item()

    expected = ((2 * LN4) + (10 * LN4)) / 2
    assert got == pytest.approx(expected, abs=1e-4)


def test_batch_loss_weights_longer_trajectories_more():
    """A 10-step trajectory must contribute 5x a 2-step one, not equally."""
    from losses.action_loss import batch_action_loss

    two = batch_action_loss([torch.zeros(2, NUM_ACTIONS)],
                            [torch.zeros(2, dtype=torch.long)]).item()
    ten = batch_action_loss([torch.zeros(10, NUM_ACTIONS)],
                            [torch.zeros(10, dtype=torch.long)]).item()
    assert ten == pytest.approx(5 * two, abs=1e-4)


def test_batch_loss_rejects_mismatched_or_empty():
    from losses.action_loss import batch_action_loss

    with pytest.raises(ValueError, match="trajectories"):
        batch_action_loss([torch.zeros(2, NUM_ACTIONS)], [])
    with pytest.raises(ValueError, match="empty batch"):
        batch_action_loss([], [])


def test_loss_goes_to_zero_on_confident_correct_logits():
    """AGENTS.md Sec. 3.1: loss -> 0 when logits are one-hot on the correct
    action."""
    targets = torch.tensor([0, 1, 2, 3])
    logits = torch.full((4, NUM_ACTIONS), -1e4)
    logits[torch.arange(4), targets] = 1e4
    assert action_loss(logits, targets).item() < 1e-6


def test_loss_is_large_when_confidently_wrong():
    targets = torch.tensor([0])
    logits = torch.tensor([[-50.0, 50.0, 0.0, 0.0]])
    assert action_loss(logits, targets).item() > 50.0


def test_loss_reductions():
    logits = torch.zeros(10, NUM_ACTIONS)
    targets = torch.zeros(10, dtype=torch.long)
    per_step = action_loss(logits, targets, reduction="none")
    assert per_step.shape == (10,)
    assert abs(action_loss(logits, targets, reduction="mean").item() - LN4) < 1e-5
    # Eq. 13 is written as a sum: sum == mean * T
    assert abs(action_loss(logits, targets, reduction="sum").item() - 10 * LN4) < 1e-4


def test_loss_accepts_single_step():
    loss = action_loss(torch.zeros(NUM_ACTIONS), torch.tensor(2))
    assert abs(loss.item() - LN4) < 1e-5


def test_loss_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="targets"):
        action_loss(torch.zeros(5, NUM_ACTIONS), torch.zeros(3, dtype=torch.long))


def test_loss_is_differentiable_wrt_logits():
    logits = torch.zeros(4, NUM_ACTIONS, requires_grad=True)
    action_loss(logits, torch.tensor([0, 1, 2, 3])).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_loss_expects_logits_not_probabilities():
    """Passing probabilities would double-apply log-softmax and silently
    flatten the loss. Confirm the two paths differ, so the docstring warning
    corresponds to a real hazard."""
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([0])
    probs = torch.softmax(logits, dim=-1)
    assert action_loss(logits, target).item() != pytest.approx(
        action_loss(probs, target).item(), abs=1e-3
    )


# ---------------------------------------------------------------------------
# End-to-end: Eq. 4 -> 5 -> 6 -> 7 -> 13
# ---------------------------------------------------------------------------
def test_full_stage0_pipeline(encoder, head):
    """One untrained forward pass through the whole Stage 0 chain."""
    builder = InputSequenceBuilder(encoder.model, encoder.processor.tokenizer)
    bb = Backbone(encoder.model)
    frame = np.random.default_rng(1).integers(0, 255, (448, 448, 3), dtype=np.uint8)

    v_t = encoder.encode(frame)                      # Eq. 4
    seq = builder.build("Turn left at the sofa.", v_t)   # Eq. 5
    H_t = bb.forward(seq)                            # Eq. 6
    h_act = H_t[-1]
    probs = head.probs(h_act)                        # Eq. 7
    loss = action_loss(head.logits(h_act), torch.tensor(Action.FWD.value))  # Eq. 13

    assert v_t.shape == (196, EXPECTED_D)
    assert H_t.shape == (seq.seq_len, EXPECTED_D)
    assert abs(probs.sum().item() - 1.0) < 1e-5
    assert torch.isfinite(loss) and loss.item() > 0
