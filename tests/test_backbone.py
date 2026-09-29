"""
Tests for Eq. 6 — H_t = F_theta(u_t) in R^{N x d}

Required tests per AGENTS.md Sec. 3.1:
  - one hidden vector per input position
  - causal mask verified: changing a later token must not alter an earlier
    position's output

Plus verification that the mRoPE position-id reconstruction is bitwise
identical to the model's own get_rope_index -- the measurement that justified
choosing mRoPE over sequential rests entirely on that reconstruction being
correct.

Run:  PYTHONPATH="" python -m pytest tests/test_backbone.py -v
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import dataclasses
import pathlib

import numpy as np
import pytest
import torch
from PIL import Image

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
from model.backbone import Backbone  # noqa: E402
from model.input_sequence import InputSequenceBuilder  # noqa: E402
from model.vision_encoder import VisionEncoder  # noqa: E402

EXPECTED_D = 2048
EXPECTED_N_V = 196
INSTRUCTION = "Exit the bedroom and turn left. Stop near the rug."


@pytest.fixture(scope="module")
def seq(builder, encoder):
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (448, 448, 3), dtype=np.uint8)
    return builder.build(INSTRUCTION, encoder.encode(frame))


# ---------------------------------------------------------------------------
# Shape: one hidden vector per input position
# ---------------------------------------------------------------------------
def test_one_hidden_vector_per_position(backbone, seq):
    """AGENTS.md Sec. 3.1: one hidden vector per input position."""
    H = backbone.forward(seq)
    assert H.shape == (seq.seq_len, EXPECTED_D)


def test_hidden_size_matches_d(backbone):
    assert backbone.hidden_size == EXPECTED_D


def test_hidden_at_matches_full_forward(backbone, seq):
    H = backbone.forward(seq)
    assert torch.equal(backbone.hidden_at(seq, -1), H[-1])
    assert torch.equal(backbone.hidden_at(seq, 0), H[0])


def test_output_is_finite(backbone, seq):
    H = backbone.forward(seq)
    assert torch.isfinite(H.float()).all()


# ---------------------------------------------------------------------------
# CAUSAL MASK — the load-bearing property
# ---------------------------------------------------------------------------
def test_causal_mask_later_token_does_not_change_earlier(backbone, seq):
    """AGENTS.md Sec. 3.1: changing a later token must not alter an earlier
    position's output.

    Perturb the FINAL position's embedding and confirm every earlier hidden
    state is unchanged. A failure here means information flows backwards and
    the paper's strict-causality claim does not hold."""
    H1 = backbone.forward(seq)

    perturbed = seq.inputs_embeds.clone()
    perturbed[0, -1] += 10.0
    seq2 = dataclasses.replace(seq, inputs_embeds=perturbed)
    H2 = backbone.forward(seq2)

    assert torch.equal(H1[:-1], H2[:-1]), "a later token altered earlier positions"
    assert not torch.equal(H1[-1], H2[-1]), "perturbation had no effect at all"


def test_causal_mask_holds_at_interior_position(backbone, seq):
    """Same property perturbing the middle, not just the end."""
    mid = seq.seq_len // 2
    H1 = backbone.forward(seq)

    perturbed = seq.inputs_embeds.clone()
    perturbed[0, mid] += 10.0
    seq2 = dataclasses.replace(seq, inputs_embeds=perturbed)
    H2 = backbone.forward(seq2)

    assert torch.equal(H1[:mid], H2[:mid]), "positions before the change moved"
    assert not torch.equal(H1[mid], H2[mid])


def test_earlier_token_does_change_later(backbone, seq):
    """Control for the causal tests: information MUST flow forwards, otherwise
    the two tests above would pass trivially on a broken model."""
    H1 = backbone.forward(seq)

    perturbed = seq.inputs_embeds.clone()
    perturbed[0, 0] += 10.0
    seq2 = dataclasses.replace(seq, inputs_embeds=perturbed)
    H2 = backbone.forward(seq2)
    assert not torch.equal(H1[-1], H2[-1]), "no forward information flow"


# ---------------------------------------------------------------------------
# mRoPE reconstruction correctness
# ---------------------------------------------------------------------------
def test_mrope_reconstruction_matches_model_ground_truth(encoder, backbone):
    """The choice of mRoPE over sequential rests on this reconstruction being
    right. Compare it bitwise against the model's own get_rope_index for an
    equivalent natively-processed input."""
    model, proc = encoder.model, encoder.processor
    img = Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (448, 448, 3), dtype=np.uint8)
    )
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": "go left"}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to(model.device)

    gt, _ = model.model.get_rope_index(
        input_ids=inp["input_ids"],
        mm_token_type_ids=inp["mm_token_type_ids"],
        image_grid_thw=inp["image_grid_thw"],
        attention_mask=inp["attention_mask"],
    )

    idx = (inp["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
    s, e = idx[0].item(), idx[-1].item() + 1
    start = gt[:, 0, s].min().item()

    mine = model.model.get_vision_position_ids(
        start, backbone.visual_grid_thw(EXPECTED_N_V), 1,
        backbone.spatial_merge_size, device=model.device,
    )
    assert torch.equal(mine.to(gt.device), gt[:, 0, s:e])


def test_image_advances_position_by_grid_side_not_n_v(backbone, seq):
    """The structural fact that makes sequential positions wrong: a 448x448
    image occupies 196 token slots but advances the position counter by only
    14 (= grid side / merge).

    Layout-independent: compares the position just before the visual span with
    the position just after it."""
    pos = backbone.build_position_ids(seq)
    v = seq.visual_slice
    before = pos[:, 0, v.start - 1].max().item()   # token preceding the image
    after = pos[:, 0, v.stop].max().item()         # token following the image
    assert after - before == 15, (
        f"image span advanced positions by {after - before - 1}, expected 14"
    )


def test_position_ids_shape_and_length(backbone, seq):
    pos = backbone.build_position_ids(seq)
    assert pos.shape == (3, 1, seq.seq_len)


def test_instruction_positions_are_sequential_across_channels(backbone, seq):
    """Text tokens must carry the same value in all three mRoPE channels, and
    run consecutively (offset by whatever precedes them in the layout)."""
    pos = backbone.build_position_ids(seq)
    instr = pos[:, 0, seq.instr_slice]
    assert torch.equal(instr[0], instr[1]) and torch.equal(instr[1], instr[2])
    expected = torch.arange(seq.n_instruction_tokens, device=instr.device) + instr[0][0]
    assert torch.equal(instr[0], expected)


def test_visual_positions_are_two_dimensional(backbone, seq):
    """Visual tokens must NOT be sequential: height and width channels differ,
    which is exactly what sequential positions would destroy."""
    pos = backbone.build_position_ids(seq)
    vis = pos[:, 0, seq.visual_slice]
    assert not torch.equal(vis[1], vis[2]), "h and w channels identical"
    assert vis[0].unique().numel() == 1, "temporal channel should be constant"
    assert vis[1].unique().numel() == 14   # 14 distinct rows
    assert vis[2].unique().numel() == 14   # 14 distinct columns


def test_sequential_mode_returns_none(encoder, seq):
    """Sequential mode defers to the model's default position handling."""
    bb = Backbone(encoder.model, position_mode="sequential")
    assert bb.build_position_ids(seq) is None


def test_mrope_and_sequential_differ_materially(encoder, seq):
    """Documents the measured finding: the two schemes are NOT interchangeable.
    If this ever stops being true, the mRoPE decision should be revisited."""
    bb_m = Backbone(encoder.model, position_mode="mrope")
    bb_s = Backbone(encoder.model, position_mode="sequential")
    Hm, Hs = bb_m.forward(seq).float(), bb_s.forward(seq).float()
    rel = ((Hm - Hs).norm() / Hs.norm()).item()
    assert rel > 0.1, f"expected a large difference, got relative L2 {rel:.4f}"


def test_invalid_position_mode_rejected(encoder):
    with pytest.raises(ValueError, match="position_mode"):
        Backbone(encoder.model, position_mode="rotary")


def test_visual_grid_thw_derivation(backbone):
    grid = backbone.visual_grid_thw(196)
    assert grid.tolist() == [1, 28, 28]      # 14 merged * merge_size 2


def test_visual_grid_rejects_non_square(backbone):
    with pytest.raises(ValueError, match="perfect square"):
        backbone.visual_grid_thw(200)


# ---------------------------------------------------------------------------
# Statelessness (Eq. 9: only the Pilot Token crosses steps)
# ---------------------------------------------------------------------------
def test_forward_is_deterministic(backbone, seq):
    assert torch.equal(backbone.forward(seq), backbone.forward(seq))


def test_no_state_carried_between_forwards(backbone, builder, encoder):
    """Running an intervening sequence must not change a later result -- no KV
    cache or hidden state may persist across steps."""
    rng = np.random.default_rng(1)
    f1 = rng.integers(0, 255, (448, 448, 3), dtype=np.uint8)
    f2 = rng.integers(0, 255, (448, 448, 3), dtype=np.uint8)
    s1 = builder.build(INSTRUCTION, encoder.encode(f1))
    s2 = builder.build(INSTRUCTION, encoder.encode(f2))

    a = backbone.forward(s1)
    backbone.forward(s2)
    b = backbone.forward(s1)
    assert torch.equal(a, b)
