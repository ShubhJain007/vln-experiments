"""
Tests for Eq. 5 (Stage 0) — u_t = [ Tok(x) ; v_t ]

Required tests per AGENTS.md Sec. 3.1:
  - sequence length equals len(instr) + N_v
  - assert NO history frames are present

On validation strategy: this is a STRUCTURAL property, so the rigour comes from
invariants, not from realistic imagery. The decisive test is statelessness
(test_no_history_stepping_through_frames): a sequence built after processing
several frames must be BITWISE identical to one built fresh from the current
frame alone. Length checks alone would only catch history that grows the
sequence, not history silently mixed into it.

One integration test uses real walk.mov frames to make the "constant length
across a real rollout" claim concrete; it adds realism, not extra rigour.

Run:  PYTHONPATH="" python -m pytest tests/test_input_sequence.py -v
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import pathlib

import numpy as np
import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
from model.input_sequence import InputSequence, InputSequenceBuilder  # noqa: E402
from model.vision_encoder import VisionEncoder  # noqa: E402

EXPECTED_D = 2048
EXPECTED_N_V = 196
INSTRUCTION = "Exit the bedroom and turn left. Walk straight and stop near the rug."


@pytest.fixture(scope="module")
def frames():
    """Distinct synthetic frames standing in for consecutive observations."""
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (448, 448, 3), dtype=np.uint8) for _ in range(4)]


@pytest.fixture(scope="module")
def vtokens(encoder, frames):
    return [encoder.encode(f) for f in frames]


# ---------------------------------------------------------------------------
# Sequence length  =  len(Tok(x)) + N_v
# ---------------------------------------------------------------------------
def test_sequence_length_is_instr_plus_nv(builder, vtokens):
    """AGENTS.md Sec. 3.1: sequence length equals len(instr) + N_v.

    The approved layout adds exactly two marker tokens (<vision_start> and
    <vision_end>); the instruction and visual counts themselves are unchanged."""
    seq = builder.build(INSTRUCTION, vtokens[0])
    n_instr = len(builder.tokenize_instruction(INSTRUCTION))
    assert seq.n_instruction_tokens == n_instr
    assert seq.n_visual_tokens == EXPECTED_N_V
    assert seq.seq_len == n_instr + EXPECTED_N_V + 2


def test_literal_layout_has_no_markers(encoder, vtokens):
    """layout="eq5_literal" reproduces the paper's exact form: instruction
    first, image second, no marker tokens."""
    lit = InputSequenceBuilder(
        encoder.model, encoder.processor.tokenizer, layout="eq5_literal")
    seq = lit.build(INSTRUCTION, vtokens[0])
    n_instr = len(lit.tokenize_instruction(INSTRUCTION))
    assert seq.seq_len == n_instr + EXPECTED_N_V          # no +2
    assert seq.instr_slice.start == 0
    assert seq.visual_slice.start == seq.instr_slice.stop


def test_unknown_layout_rejected(encoder):
    with pytest.raises(ValueError, match="layout"):
        InputSequenceBuilder(
            encoder.model, encoder.processor.tokenizer, layout="sideways")


def test_expected_seq_len_helper_agrees_with_build(builder, vtokens):
    seq = builder.build(INSTRUCTION, vtokens[0])
    assert seq.seq_len == builder.expected_seq_len(INSTRUCTION, EXPECTED_N_V)


def test_embeds_shape_and_hidden_size(builder, vtokens):
    seq = builder.build(INSTRUCTION, vtokens[0])
    assert seq.inputs_embeds.shape == (1, seq.seq_len, EXPECTED_D)
    assert seq.hidden_size == EXPECTED_D


def test_attention_mask_matches_sequence(builder, vtokens):
    seq = builder.build(INSTRUCTION, vtokens[0])
    assert seq.attention_mask.shape == (1, seq.seq_len)
    assert seq.attention_mask.sum().item() == seq.seq_len   # no padding


def test_longer_instruction_grows_sequence_by_exact_token_delta(builder, vtokens):
    short, long = "Go left.", "Go left and then walk down the long hallway slowly."
    s1 = builder.build(short, vtokens[0])
    s2 = builder.build(long, vtokens[0])
    delta = len(builder.tokenize_instruction(long)) - len(
        builder.tokenize_instruction(short)
    )
    assert s2.seq_len - s1.seq_len == delta


# ---------------------------------------------------------------------------
# NO HISTORY — the paper's key efficiency property
# ---------------------------------------------------------------------------
def test_no_history_stepping_through_frames(builder, vtokens):
    """THE decisive test. Build sequences while stepping through a trajectory,
    then rebuild the last step in isolation. If ANY history were retained, the
    two would differ. Bitwise equality proves u_t depends only on (x, o_t)."""
    for i in range(len(vtokens)):
        builder.build(INSTRUCTION, vtokens[i])          # "step" through history

    after_history = builder.build(INSTRUCTION, vtokens[-1])
    fresh = InputSequenceBuilder(
        builder.model, builder.tokenizer
    ).build(INSTRUCTION, vtokens[-1])

    assert torch.equal(after_history.inputs_embeds, fresh.inputs_embeds), (
        "sequence depends on previously-seen frames -- history has leaked in, "
        "violating Eq. 5"
    )


def test_sequence_length_constant_across_timesteps(builder, vtokens):
    """Length must NOT grow with t. Growth is the signature of accumulated
    history and would destroy the paper's constant-cost claim."""
    lengths = [builder.build(INSTRUCTION, v).seq_len for v in vtokens]
    assert len(set(lengths)) == 1, f"sequence length grew across steps: {lengths}"


def test_sequence_depends_only_on_current_frame(builder, vtokens):
    """Different current frames must give different sequences, and the
    difference must be confined to the visual region."""
    a = builder.build(INSTRUCTION, vtokens[0])
    b = builder.build(INSTRUCTION, vtokens[1])
    assert not torch.equal(a.inputs_embeds, b.inputs_embeds)
    # instruction region identical...
    assert torch.equal(
        a.inputs_embeds[:, a.instr_slice], b.inputs_embeds[:, b.instr_slice]
    )
    # ...visual region differs
    assert not torch.equal(
        a.inputs_embeds[:, a.visual_slice], b.inputs_embeds[:, b.visual_slice]
    )


def test_build_rejects_stacked_history_frames(builder, vtokens):
    """Passing a (T, N_v, d) stack of frames must raise, not silently encode
    history into the sequence."""
    stacked = torch.stack(vtokens[:3])          # (3, N_v, d)
    with pytest.raises(ValueError, match="single frame"):
        builder.build(INSTRUCTION, stacked)


def test_builder_holds_no_per_episode_state(builder, vtokens):
    """Two interleaved 'episodes' must not contaminate each other."""
    a1 = builder.build("Episode A instruction.", vtokens[0])
    builder.build("Episode B instruction.", vtokens[1])
    a2 = builder.build("Episode A instruction.", vtokens[0])
    assert torch.equal(a1.inputs_embeds, a2.inputs_embeds)


# ---------------------------------------------------------------------------
# Region layout: [ Tok(x) ; v_t ] in that order
# ---------------------------------------------------------------------------
def test_approved_layout_is_image_first_with_markers(builder, vtokens):
    """APPROVED DEVIATION (human, 2026-08-31): the layout is
    [<vision_start> ; v_t ; <vision_end> ; Tok(x)], not Eq. 5's stated
    [Tok(x) ; v_t]. Measured 77.5% vs 21.2% zero-shot instruction following."""
    seq = builder.build(INSTRUCTION, vtokens[0])
    assert seq.layout == "image_first_markers"
    assert seq.visual_slice.start == 1                       # after <vision_start>
    assert seq.instr_slice.start == seq.visual_slice.stop + 1  # after <vision_end>
    assert seq.instr_slice.stop == seq.seq_len               # instruction is last


def test_marker_tokens_are_the_real_vision_markers(builder, vtokens):
    """The two markers must be the checkpoint's own vision_start/vision_end
    embeddings, not arbitrary vectors."""
    seq = builder.build(INSTRUCTION, vtokens[0])
    embed = builder.model.get_input_embeddings()
    dev = seq.inputs_embeds.device
    start = embed(torch.tensor([builder.vision_start_id], device=dev))[0]
    end = embed(torch.tensor([builder.vision_end_id], device=dev))[0]
    assert torch.equal(seq.inputs_embeds[0, 0], start.to(seq.inputs_embeds.dtype))
    assert torch.equal(
        seq.inputs_embeds[0, seq.visual_slice.stop], end.to(seq.inputs_embeds.dtype))


def test_action_index_is_last_position(builder, vtokens):
    """h_t^act (Eq. 7) is read from the final position, which under this layout
    is the last instruction token."""
    seq = builder.build(INSTRUCTION, vtokens[0])
    assert seq.action_index == seq.seq_len - 1
    assert seq.action_index == seq.instr_slice.stop - 1


def test_visual_region_contains_the_encoder_output(builder, vtokens):
    """The v_t region must be exactly Eq. 4's output, not a transform of it."""
    seq = builder.build(INSTRUCTION, vtokens[0])
    placed = seq.inputs_embeds[0, seq.visual_slice]
    assert torch.equal(placed, vtokens[0].to(placed.dtype))


def test_instruction_region_matches_embedded_tokens(builder, vtokens):
    seq = builder.build(INSTRUCTION, vtokens[0])
    ids = builder.tokenize_instruction(INSTRUCTION)
    expected = builder.model.get_input_embeddings()(ids)
    assert torch.equal(seq.inputs_embeds[0, seq.instr_slice], expected)


def test_no_pilot_slot_in_stage0(builder, vtokens):
    """Stage 0 has no Pilot Token. The sequence accounts for exactly the
    instruction, the visual tokens and the two markers -- nothing else. An
    unexplained extra position would mean a Pilot slot crept in early."""
    seq = builder.build(INSTRUCTION, vtokens[0])
    n_instr = len(builder.tokenize_instruction(INSTRUCTION))
    assert seq.seq_len == n_instr + EXPECTED_N_V + 2
    assert sum(n for _, n in seq.segments) == seq.seq_len


# ---------------------------------------------------------------------------
# Cached instruction ids
# ---------------------------------------------------------------------------
def test_precomputed_instruction_ids_give_identical_sequence(builder, vtokens):
    """Reusing tokenized ids across an episode must not change the result."""
    ids = builder.tokenize_instruction(INSTRUCTION)
    a = builder.build(INSTRUCTION, vtokens[0])
    b = builder.build(INSTRUCTION, vtokens[0], instruction_ids=ids)
    assert torch.equal(a.inputs_embeds, b.inputs_embeds)


def test_mismatched_hidden_size_raises(builder):
    with pytest.raises(ValueError, match="hidden size"):
        builder.build(INSTRUCTION, torch.zeros(EXPECTED_N_V, 999))


# ---------------------------------------------------------------------------
# Integration: real footage
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (_ROOT / "data" / "walk.mov").exists(), reason="walk.mov not present"
)
def test_constant_length_across_real_video_rollout(builder, encoder):
    """Integration check on real consecutive frames: stepping through an actual
    trajectory must produce a constant-length, history-free sequence."""
    sys.path.insert(0, str(_ROOT / "src"))
    from data.predictability_probe import collect_trajectories

    traj = collect_trajectories(str(_ROOT / "data" / "walk.mov"), fps=2.0,
                                max_frames=6)[0]
    seqs = [builder.build(INSTRUCTION, encoder.encode(f)) for f in traj]

    assert len({s.seq_len for s in seqs}) == 1, "length grew over a real rollout"
    # last step rebuilt in isolation must match the one built mid-rollout
    fresh = builder.build(INSTRUCTION, encoder.encode(traj[-1]))
    assert torch.equal(seqs[-1].inputs_embeds, fresh.inputs_embeds)
