"""
Tests for Eq. 4 — v_t = E_phi(o_t) in R^{N_v x d}   (frozen vision encoder)

Required tests per AGENTS.md Sec. 3.1:
  - output shape is (N_v, d)
  - two calls on the same image are identical
  - requires_grad is False

Plus coverage of the RGBA trap (habitat returns 4 channels, Eq. 1 wants 3).

Run:  PYTHONPATH="" python -m pytest tests/test_vision_encoder.py -v
Note: loads the 2B backbone once (module-scoped fixture); needs ~5 GB VRAM.
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import pathlib

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from model.vision_encoder import (  # noqa: E402
    DEFAULT_IMAGE_SIZE,
    VisionEncoder,
    to_rgb_pil,
)

EXPECTED_D = 2048     # Cosmos-Reason2-2B hidden size, verified Session 1
EXPECTED_N_V = 196    # at 448x448, verified three ways in Session 1


# ---------------------------------------------------------------------------
# to_rgb_pil — pure, no model needed
# ---------------------------------------------------------------------------
def test_to_rgb_pil_drops_alpha_from_rgba():
    """Habitat's color_sensor returns (H, W, 4); Eq. 1 specifies C=3."""
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[..., 0] = 255           # red channel
    rgba[..., 3] = 7             # alpha that must not survive
    out = to_rgb_pil(rgba, (64, 64))
    assert out.mode == "RGB"
    arr = np.array(out)
    assert arr.shape == (64, 64, 3)
    assert arr[..., 0].max() == 255


def test_to_rgb_pil_preserves_rgb_values():
    rgb = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    out = np.array(to_rgb_pil(rgb, (32, 32)))
    assert np.array_equal(out, rgb), "no-op resize must not alter pixels"


def test_to_rgb_pil_resizes_to_target():
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    out = to_rgb_pil(arr, (448, 448))
    assert out.size == (448, 448)          # PIL reports (width, height)


def test_to_rgb_pil_accepts_pil_input():
    img = Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8))
    assert to_rgb_pil(img, (20, 20)).mode == "RGB"


def test_to_rgb_pil_converts_rgba_pil():
    img = Image.fromarray(np.zeros((20, 20, 4), dtype=np.uint8), mode="RGBA")
    assert to_rgb_pil(img, (20, 20)).mode == "RGB"


def test_to_rgb_pil_rejects_bad_shape():
    with pytest.raises(ValueError):
        to_rgb_pil(np.zeros((10, 10), dtype=np.uint8), (10, 10))
    with pytest.raises(ValueError):
        to_rgb_pil(np.zeros((10, 10, 2), dtype=np.uint8), (10, 10))


def test_to_rgb_pil_rejects_bad_type():
    with pytest.raises(TypeError):
        to_rgb_pil("not an image", (10, 10))


# ---------------------------------------------------------------------------
# VisionEncoder — needs the real checkpoint
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sample_image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (448, 448, 3), dtype=np.uint8)


def test_n_v_and_d_match_session1_values(encoder):
    assert encoder.d == EXPECTED_D
    assert encoder.n_v == EXPECTED_N_V


def test_encode_output_shape(encoder, sample_image):
    """AGENTS.md Sec. 3.1: output shape is (N_v, d)."""
    v_t = encoder.encode(sample_image)
    assert v_t.shape == (EXPECTED_N_V, EXPECTED_D)


def test_encode_shape_matches_declared_properties(encoder, sample_image):
    """The advertised n_v/d must agree with what the tower actually emits."""
    v_t = encoder.encode(sample_image)
    assert v_t.shape == (encoder.n_v, encoder.d)


def test_encode_is_deterministic(encoder, sample_image):
    """AGENTS.md Sec. 3.1: two calls on the same image are identical."""
    a = encoder.encode(sample_image)
    b = encoder.encode(sample_image)
    assert torch.equal(a, b), "frozen encoder in eval mode must be deterministic"


def test_encoder_is_frozen(encoder):
    """AGENTS.md Sec. 3.1: requires_grad is False.

    Deviation from Eq. 15 (which trains phi) is approved -- see the module
    docstring for the target-collapse rationale.
    """
    assert encoder.is_frozen
    assert all(not p.requires_grad for p in encoder.model.parameters())


def test_encode_output_requires_no_grad(encoder, sample_image):
    """No gradient may flow out of the encoder into the rest of the graph."""
    assert encoder.encode(sample_image).requires_grad is False


def test_encoder_in_eval_mode(encoder):
    """Dropout/norm must be in inference mode or determinism breaks."""
    assert not encoder.model.training


def test_rgba_input_matches_rgb_input(encoder, sample_image):
    """A habitat RGBA frame must encode identically to its RGB counterpart --
    the alpha channel must be dropped, not blended or misread."""
    rgba = np.concatenate(
        [sample_image, np.full((448, 448, 1), 255, dtype=np.uint8)], axis=2
    )
    assert rgba.shape == (448, 448, 4)
    assert torch.equal(encoder.encode(sample_image), encoder.encode(rgba))


def test_encode_batch_matches_individual_encodes(encoder):
    """Batching must be SEMANTICALLY equivalent to encoding one at a time.

    Not bitwise: in bf16 the same frame encoded at batch size 3 vs 1 differs by
    ~7% relative L2, because CUDA picks different kernels/reduction orders per
    batch shape and the error compounds through 24 ViT layers. In fp32 the same
    comparison agrees to ~1e-5 (see test_batch_invariance_is_a_bf16_artifact),
    which is what identifies this as precision, not a preprocessing bug.

    Consequence for Stage 1: v_bar caching (Sec. 3.2) must use a FIXED batch
    size, or cache in fp32 -- otherwise the Pilot target v_bar_{t+2} shifts
    depending on how frames happened to be batched.
    """
    rng = np.random.default_rng(1)
    imgs = [rng.integers(0, 255, (448, 448, 3), dtype=np.uint8) for _ in range(3)]

    batched = encoder.encode_batch(imgs)
    assert len(batched) == 3

    for i, img in enumerate(imgs):
        assert batched[i].shape == (EXPECTED_N_V, EXPECTED_D)
        b = batched[i].float().flatten()
        s = encoder.encode(img).float().flatten()
        cos = torch.nn.functional.cosine_similarity(
            b.unsqueeze(0), s.unsqueeze(0)
        ).item()
        assert cos > 0.99, f"frame {i}: cosine {cos:.4f} too low for bf16 noise"


@pytest.mark.skipif(
    torch.cuda.is_available()
    and torch.cuda.mem_get_info()[0] / 1024**3 < 10.0,
    reason="needs ~10 GB free VRAM for a second fp32 backbone; run standalone",
)
def test_batch_invariance_is_a_bf16_artifact():
    """Pins the root cause: in fp32, batch-vs-single agrees to ~1e-5.

    If this test ever fails, the batch/single discrepancy is NOT precision and
    something structural has broken (preprocessing, or pooler_output splitting).
    Loads its own fp32 encoder, so it is slow but self-contained.
    """
    rng = np.random.default_rng(1)
    imgs = [rng.integers(0, 255, (448, 448, 3), dtype=np.uint8) for _ in range(3)]

    enc32 = VisionEncoder(dtype=torch.float32)
    try:
        batched = enc32.encode_batch(imgs)
        single = enc32.encode(imgs[0])
        rel = ((batched[0] - single).norm() / single.norm()).item()
        assert rel < 1e-3, f"fp32 batch/single differ by {rel:.2e} -- not precision"
    finally:
        del enc32
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def test_encode_batch_empty_returns_empty(encoder):
    assert encoder.encode_batch([]) == []


def test_different_images_give_different_encodings(encoder):
    """Guards against a degenerate wrapper that ignores its input."""
    rng = np.random.default_rng(2)
    a = encoder.encode(rng.integers(0, 255, (448, 448, 3), dtype=np.uint8))
    b = encoder.encode(np.zeros((448, 448, 3), dtype=np.uint8))
    assert not torch.equal(a, b)


def test_accepts_non_square_input_by_resizing(encoder):
    """Habitat/phone frames are rarely 448x448; resizing must yield exactly N_v."""
    rng = np.random.default_rng(3)
    tall = rng.integers(0, 255, (1202, 1616, 3), dtype=np.uint8)
    assert encoder.encode(tall).shape == (EXPECTED_N_V, EXPECTED_D)


def test_default_image_size_is_448(encoder):
    assert tuple(DEFAULT_IMAGE_SIZE) == (448, 448)
    assert encoder.image_size == (448, 448)
