"""Shared pytest fixtures.

Every test module previously built its own VisionEncoder, so a full run held
one ~4.6 GB copy of the backbone PER MODULE and exhausted the 16 GB card. These
session-scoped fixtures load the weights once and share them, which both fixes
the OOM and makes the suite substantially faster.

Run tests as:  PYTHONPATH="" python -m pytest tests/ -q
(pytest's plugin autoloader imports ROS's `launch_testing` before any test code
runs, so the in-module sys.path guards cannot help it -- see README gotcha #8.)
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import pathlib

import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


@pytest.fixture(scope="session")
def encoder():
    """One frozen VisionEncoder (Eq. 4) shared by the whole session."""
    from model.vision_encoder import VisionEncoder
    return VisionEncoder()


@pytest.fixture(scope="session")
def builder(encoder):
    """Eq. 5 builder using the approved image-first-with-markers layout."""
    from model.input_sequence import InputSequenceBuilder
    return InputSequenceBuilder(encoder.model, encoder.processor.tokenizer)


@pytest.fixture(scope="session")
def backbone(encoder):
    """Eq. 6 backbone, mRoPE positions."""
    from model.backbone import Backbone
    return Backbone(encoder.model)


@pytest.fixture(scope="session")
def head(encoder):
    """Eq. 7 action head over the native W_LM."""
    from model.action_head import ActionHead
    return ActionHead(encoder.model, encoder.processor.tokenizer)


def free_vram_gb() -> float:
    """Free VRAM in GB, or inf when running on CPU."""
    if not torch.cuda.is_available():
        return float("inf")
    free, _ = torch.cuda.mem_get_info()
    return free / 1024 ** 3


requires_spare_vram = pytest.mark.skipif(
    torch.cuda.is_available() and free_vram_gb() < 10.0,
    reason="needs ~10 GB free VRAM to load a second fp32 copy of the backbone; "
           "run this test standalone",
)
