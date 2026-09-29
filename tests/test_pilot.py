"""
Tests for Eq. 8 (G_psi), z_0, and the PilotCache (Sec. 3.3).

Required per AGENTS.md Sec. 4.1:
  - G_psi is a single linear layer; param count == d*d + d
  - z_0 is finite and non-zero
  - the Pilot slot occupies exactly one position; its index is recoverable
  - PilotCache never touches future-frame data

Plus the causality check that FORCED the Pilot to sit after the instruction:
h_pil must be able to see Tok(x), otherwise z_t cannot encode where the agent
is heading and Eq. 14's v_bar_{t+2} target is unreachable in principle.

Run:  PYTHONPATH="" python -m pytest tests/test_pilot.py -v
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import dataclasses
import pathlib

import numpy as np
import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
from model.input_sequence import InputSequenceBuilder  # noqa: E402
from model.pilot import (  # noqa: E402
    REAL_TOKEN_EMBED_NORM,
    PilotCache,
    PilotModule,
)

D = 2048
INSTRUCTION = "Exit the bedroom and turn left. Stop near the rug."


@pytest.fixture(scope="module")
def pilot_a():
    return PilotModule(D, mode="last", dtype=torch.bfloat16, device="cuda")


@pytest.fixture(scope="module")
def pilot_b():
    return PilotModule(D, mode="action_query", dtype=torch.bfloat16, device="cuda")


@pytest.fixture(scope="module")
def frame():
    return np.random.default_rng(0).integers(0, 255, (448, 448, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Eq. 8 — G_psi is a SINGLE linear layer
# ---------------------------------------------------------------------------
def test_g_psi_param_count_is_exactly_d_squared_plus_d(pilot_a):
    """AGENTS.md Sec. 4.1: parameter count == d*d + d. Anything else means
    G_psi stopped being 'a simple linear layer' (Sec. 3.2)."""
    assert pilot_a.g_psi_param_count == D * D + D == 4_196_352
    pilot_a.assert_shapes()


def test_g_psi_is_a_plain_linear(pilot_a):
    assert isinstance(pilot_a.G_psi, torch.nn.Linear)
    assert pilot_a.G_psi.in_features == D and pilot_a.G_psi.out_features == D


def test_assert_shapes_catches_a_deepened_g_psi(pilot_a):
    """If someone swaps in an MLP, this must fail rather than train quietly."""
    deep = PilotModule(D, mode="last", dtype=torch.bfloat16, device="cuda")
    deep.G_psi = torch.nn.Sequential(
        torch.nn.Linear(D, D), torch.nn.GELU(), torch.nn.Linear(D, D)
    ).to("cuda", torch.bfloat16)
    with pytest.raises(RuntimeError, match="single nn.Linear"):
        deep.assert_shapes()


def test_g_psi_output_shape(pilot_a):
    h = torch.randn(D, device="cuda", dtype=torch.bfloat16)
    assert pilot_a(h).shape == (D,)
    assert pilot_a(torch.randn(5, D, device="cuda", dtype=torch.bfloat16)).shape == (5, D)


# ---------------------------------------------------------------------------
# z_0
# ---------------------------------------------------------------------------
def test_z0_is_finite_and_nonzero(pilot_a):
    """AGENTS.md Sec. 4.1's explicit test."""
    assert torch.isfinite(pilot_a.z_0).all()
    assert pilot_a.z_0.abs().sum() > 0


def test_z0_is_trainable(pilot_a):
    """Sec. 3.2 says z_0 comes from a LEARNED embedding, so it must carry grad."""
    assert pilot_a.z_0.requires_grad


def test_z0_initialised_in_distribution(pilot_a):
    """Scaled to the real-token embedding norm (~1.44), not the untrained
    spare rows' ~0.36 -- the Pilot slot should enter the backbone at the scale
    it expects. Same reasoning as D10."""
    assert pilot_a.z_0.float().norm().item() == pytest.approx(
        REAL_TOKEN_EMBED_NORM, rel=0.02
    )


def test_z0_is_only_2048_params_not_an_embedding_matrix(pilot_a):
    """The approved deviation: an nn.Parameter, not a row of the 311M-param
    embedding table (which could not be trained without abandoning LoRA-only)."""
    assert pilot_a.z_0.numel() == D


def test_action_query_only_exists_in_mode_b(pilot_a, pilot_b):
    assert pilot_a.action_query is None
    assert pilot_b.action_query is not None and pilot_b.action_query.numel() == D
    assert pilot_b.action_query.requires_grad


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode must be"):
        PilotModule(D, mode="sideways")


# ---------------------------------------------------------------------------
# Slot placement — Designs A and B
# ---------------------------------------------------------------------------
def test_design_a_pilot_is_last_and_shares_state_with_action(encoder, pilot_a, frame):
    """A: [... ; Tok(x) ; PILOT(z)]. h_act and h_pil are the SAME position."""
    b = InputSequenceBuilder(encoder.model, encoder.processor.tokenizer, pilot=pilot_a)
    seq = b.build(INSTRUCTION, encoder.encode(frame), pilot_input=pilot_a.z_0)
    assert seq.has_pilot
    assert seq.pilot_index == seq.seq_len - 1 == seq.action_index
    assert seq.pilot_and_action_share_state


def test_design_b_pilot_precedes_a_distinct_action_query(encoder, pilot_b, frame):
    """B: [... ; Tok(x) ; PILOT(z) ; ACTION_QUERY]. Distinct positions."""
    b = InputSequenceBuilder(encoder.model, encoder.processor.tokenizer, pilot=pilot_b)
    seq = b.build(INSTRUCTION, encoder.encode(frame), pilot_input=pilot_b.z_0)
    assert seq.pilot_index == seq.action_index - 1
    assert not seq.pilot_and_action_share_state


def test_pilot_occupies_exactly_one_position(encoder, pilot_a, frame):
    """AGENTS.md Sec. 4.1: exactly one slot, index recoverable."""
    b0 = InputSequenceBuilder(encoder.model, encoder.processor.tokenizer)
    b1 = InputSequenceBuilder(encoder.model, encoder.processor.tokenizer, pilot=pilot_a)
    v = encoder.encode(frame)
    assert (b1.build(INSTRUCTION, v, pilot_input=pilot_a.z_0).seq_len
            - b0.build(INSTRUCTION, v).seq_len) == 1


def test_pilot_comes_after_the_instruction(encoder, pilot_a, frame):
    """Forced by causality, not style -- see test_h_pil_can_see_instruction."""
    b = InputSequenceBuilder(encoder.model, encoder.processor.tokenizer, pilot=pilot_a)
    seq = b.build(INSTRUCTION, encoder.encode(frame), pilot_input=pilot_a.z_0)
    assert seq.pilot_index >= seq.instr_slice.stop


def test_pilot_slot_holds_exactly_what_was_passed(encoder, pilot_a, frame):
    b = InputSequenceBuilder(encoder.model, encoder.processor.tokenizer, pilot=pilot_a)
    z = torch.randn(D, device="cuda", dtype=torch.bfloat16)
    seq = b.build(INSTRUCTION, encoder.encode(frame), pilot_input=z)
    assert torch.equal(seq.inputs_embeds[0, seq.pilot_index], z)


def test_missing_pilot_input_raises(encoder, pilot_a, frame):
    """Silently defaulting the slot would train on the wrong quantity."""
    b = InputSequenceBuilder(encoder.model, encoder.processor.tokenizer, pilot=pilot_a)
    with pytest.raises(ValueError, match="pilot_input"):
        b.build(INSTRUCTION, encoder.encode(frame))


def test_wrong_shape_pilot_input_raises(encoder, pilot_a, frame):
    b = InputSequenceBuilder(encoder.model, encoder.processor.tokenizer, pilot=pilot_a)
    with pytest.raises(ValueError, match=r"pilot_input must be"):
        b.build(INSTRUCTION, encoder.encode(frame),
                pilot_input=torch.randn(2, D, device="cuda", dtype=torch.bfloat16))


def test_expected_seq_len_accounts_for_pilot(encoder, pilot_a, pilot_b, frame):
    v = encoder.encode(frame)
    for pilot, extra in ((pilot_a, 1), (pilot_b, 2)):
        b = InputSequenceBuilder(encoder.model, encoder.processor.tokenizer, pilot=pilot)
        seq = b.build(INSTRUCTION, v, pilot_input=pilot.z_0)
        assert seq.seq_len == b.expected_seq_len(INSTRUCTION, 196)
        base = InputSequenceBuilder(
            encoder.model, encoder.processor.tokenizer
        ).expected_seq_len(INSTRUCTION, 196)
        assert seq.seq_len == base + extra


# ---------------------------------------------------------------------------
# THE CAUSALITY ARGUMENT that forced Pilot-after-instruction
# ---------------------------------------------------------------------------
def test_h_pil_can_see_the_instruction(encoder, backbone, pilot_a, frame):
    """z_t must encode WHERE THE AGENT IS HEADING to predict v_bar_{t+2}
    (Eq. 14), and that depends on the instruction. Under causal attention a
    Pilot placed before Tok(x) could not see it. Verify the chosen placement
    actually gives h_pil access: perturbing an instruction token MUST move
    h_pil."""
    b = InputSequenceBuilder(encoder.model, encoder.processor.tokenizer, pilot=pilot_a)
    seq = b.build(INSTRUCTION, encoder.encode(frame), pilot_input=pilot_a.z_0)

    h1 = backbone.forward(seq)[seq.pilot_index]
    perturbed = seq.inputs_embeds.clone()
    perturbed[0, seq.instr_slice.start] += 10.0
    h2 = backbone.forward(dataclasses.replace(seq, inputs_embeds=perturbed))[
        seq.pilot_index
    ]
    assert not torch.equal(h1, h2), (
        "h_pil is blind to the instruction -- z_t cannot encode intent"
    )


def test_h_pil_sees_the_visual_tokens_too(encoder, backbone, pilot_a, frame):
    b = InputSequenceBuilder(encoder.model, encoder.processor.tokenizer, pilot=pilot_a)
    seq = b.build(INSTRUCTION, encoder.encode(frame), pilot_input=pilot_a.z_0)
    h1 = backbone.forward(seq)[seq.pilot_index]
    perturbed = seq.inputs_embeds.clone()
    perturbed[0, seq.visual_slice.start] += 10.0
    h2 = backbone.forward(dataclasses.replace(seq, inputs_embeds=perturbed))[
        seq.pilot_index
    ]
    assert not torch.equal(h1, h2)


def test_pilot_slot_content_changes_the_action_state(encoder, backbone, pilot_a, frame):
    """The Pilot must actually influence the decision, else it is decoration."""
    b = InputSequenceBuilder(encoder.model, encoder.processor.tokenizer, pilot=pilot_a)
    v = encoder.encode(frame)
    g = torch.Generator(device="cpu").manual_seed(0)
    z1 = torch.randn(D, generator=g).to("cuda", torch.bfloat16)
    z2 = torch.randn(D, generator=g).to("cuda", torch.bfloat16)
    s1 = b.build(INSTRUCTION, v, pilot_input=z1)
    s2 = b.build(INSTRUCTION, v, pilot_input=z2)
    assert not torch.equal(
        backbone.forward(s1)[s1.action_index],
        backbone.forward(s2)[s2.action_index],
    )


# ---------------------------------------------------------------------------
# PilotCache (Sec. 3.3)
# ---------------------------------------------------------------------------
def test_cache_starts_at_z0(pilot_a):
    c = pilot_a.new_cache()
    assert torch.equal(c.read(), pilot_a.z_0.detach())
    assert c.is_fresh


def test_cache_read_write_cycle(pilot_a):
    c = pilot_a.new_cache()
    z = torch.randn(D, device="cuda", dtype=torch.bfloat16)
    c.write(z)
    assert torch.equal(c.read(), z)
    assert c.episode_steps == 1 and not c.is_fresh


def test_cache_reset_restores_z0_at_episode_boundary(pilot_a):
    """AGENTS.md Sec. 8 item 6 -- unresolved in the paper; we reset to z_0.
    Without this, one episode's latent leaks into the next."""
    c = pilot_a.new_cache()
    c.write(torch.randn(D, device="cuda", dtype=torch.bfloat16))
    c.reset()
    assert torch.equal(c.read(), pilot_a.z_0.detach())
    assert c.is_fresh


def test_cache_detaches_no_backprop_through_time(pilot_a):
    """Keeping the graph across a rollout would backprop through time -- not
    what the paper does, and it would OOM a 16 GB card."""
    c = pilot_a.new_cache()
    z = (pilot_a.G_psi(torch.randn(D, device="cuda", dtype=torch.bfloat16)))
    assert z.requires_grad
    c.write(z)
    assert c.read().requires_grad is False


def test_cache_rejects_wrong_shape(pilot_a):
    c = pilot_a.new_cache()
    with pytest.raises(ValueError, match="shape"):
        c.write(torch.randn(2, D, device="cuda", dtype=torch.bfloat16))


def test_cache_has_no_path_to_accept_future_frames(pilot_a):
    """AGENTS.md Sec. 4.1: future leakage at eval invalidates every number.
    The cache exposes only read/write/reset -- there is no v_bar entry point,
    so the eval path cannot express Eq. 12's teacher forcing at all."""
    api = {a for a in dir(pilot_a.new_cache()) if not a.startswith("_")}
    assert api == {"read", "write", "reset", "episode_steps", "is_fresh"}
    for banned in ("v_bar", "vbar", "future", "teacher", "set_future"):
        assert not any(banned in a.lower() for a in api)


# ---------------------------------------------------------------------------
# D18 bias init — untested until a live TypeError caught it at runtime
# ---------------------------------------------------------------------------
def test_init_bias_from_target_mean_sets_the_bias(pilot_a):
    """G_psi.bias must literally become the supplied mean v_bar.

    Regression guard: the first implementation called
    `.to(dtype, device)` positionally, but torch's signature is
    `.to(device, dtype)` -- it raised TypeError only when a training run
    actually reached it, because no test exercised this path.
    """
    p = PilotModule(D, mode="last", dtype=torch.bfloat16, device="cuda")
    mean_vbar = torch.full((D,), 0.25)
    p.init_bias_from_target_mean(mean_vbar)
    assert torch.allclose(
        p.G_psi.bias.float().cpu(), mean_vbar, atol=1e-2
    )


def test_init_bias_accepts_cpu_float32_input(pilot_a):
    """The caller reads mean v_bar off the fp32 cache as a CPU tensor, so the
    conversion to the module's device/dtype must happen inside."""
    p = PilotModule(D, mode="last", dtype=torch.bfloat16, device="cuda")
    cpu_f32 = torch.randn(D, dtype=torch.float32)     # cpu + fp32, like the cache
    p.init_bias_from_target_mean(cpu_f32)
    assert p.G_psi.bias.device.type == "cuda"
    assert p.G_psi.bias.dtype == torch.bfloat16


def test_init_bias_rejects_wrong_shape(pilot_a):
    p = PilotModule(D, mode="last", dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="mean_vbar must be"):
        p.init_bias_from_target_mean(torch.randn(D + 1))


def test_init_bias_actually_lowers_the_starting_loss():
    """D18's whole point: at default init z ~= 0 gives L_pil ~= ||v_bar||^2
    (~251, i.e. 25x L_act), which would drown the action objective in Eq. 15's
    opening steps. Seeding the bias must measurably reduce that."""
    sys.path.insert(0, str(_ROOT / "src"))
    from losses.pilot_loss import pilot_loss

    torch.manual_seed(0)
    # stand-in for cached v_bar: norm ~15.8, as measured
    targets = torch.randn(64, D)
    targets = targets / targets.norm(dim=-1, keepdim=True) * 15.8
    mean_vbar = targets.mean(dim=0)

    h = torch.zeros(64, D, device="cuda", dtype=torch.bfloat16)

    default = PilotModule(D, mode="last", dtype=torch.bfloat16, device="cuda")
    seeded = PilotModule(D, mode="last", dtype=torch.bfloat16, device="cuda")
    seeded.init_bias_from_target_mean(mean_vbar)

    with torch.no_grad():
        l_default = pilot_loss(default(h).float().cpu(), targets).item()
        l_seeded = pilot_loss(seeded(h).float().cpu(), targets).item()

    assert l_seeded < l_default, (
        f"seeded init ({l_seeded:.1f}) must beat default ({l_default:.1f})"
    )
