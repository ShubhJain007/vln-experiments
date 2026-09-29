"""
Eq. 4 — Visual encoding.

    v_t = E_phi(o_t)  in  R^{N_v x d}

where N_v is the number of visual tokens per frame and d matches the hidden
size of the LLM backbone.

DEVIATION FROM PAPER (approved by human, AGENTS.md Sec. 3.1):
Eq. 15 minimises over theta, phi, psi -- i.e. the paper TRAINS the vision
encoder E_phi. We FREEZE it for Stages 0-3, because from Stage 1 onward the
same encoder also produces the Pilot Token's regression target v_bar_{t+2}
(Eq. 11/14); a trainable target invites representation collapse. Revisit only
with an EMA target encoder. VRAM is a second, independent reason to freeze.

Backbone substitution (also approved, AGENTS.md Sec. 1.4): the paper uses
LLaVA-Video-7B; we use Cosmos-Reason2-2B (Qwen3-VL-2B-Instruct lineage), whose
vision tower is SigLIP-2 -- the same encoder family the paper uses, so the
Pilot Token's target space stays consistent with the paper's design.
"""

# --- ROS guard: strip /opt/ros/* from sys.path before any other import. ------
import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]
# ----------------------------------------------------------------------------

import pathlib
from typing import List, Optional, Sequence, Union

import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = str(_REPO_ROOT.parent / "models" / "cosmos-reason2-2b")

# NOT IN PAPER -- chosen by us. The paper never states an input resolution;
# N_v is backbone- and resolution-dependent (AGENTS.md Sec. 8, item 4).
# 448x448 -> N_v = (448/16)^2 / 2^2 = 196, verified three independent ways in
# Session 1 (formula, image_token count, encoder output rows). Recorded in
# docs/EQUATIONS.md. Changing this changes N_v and must be re-verified.
DEFAULT_IMAGE_SIZE = (448, 448)   # (H, W)


def to_rgb_pil(image, size: Sequence[int]):
    """Normalise any supported frame representation to an RGB PIL image.

    Accepts a PIL image, or a numpy array shaped (H, W, 3) / (H, W, 4).

    Habitat's `color_sensor` returns RGBA (H, W, 4) -- verified in Session 2 --
    while Eq. 1 defines o_t with C=3 for RGB. Dropping the alpha channel here
    means no caller can forget to do it.
    """
    from PIL import Image
    import numpy as np

    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(
                f"expected (H, W, 3) or (H, W, 4) array, got shape {arr.shape}"
            )
        arr = arr[..., :3]                      # drop alpha if present
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        pil = Image.fromarray(arr, mode="RGB")
    else:
        raise TypeError(f"unsupported image type: {type(image).__name__}")

    # PIL.resize takes (width, height); `size` is (H, W).
    target = (int(size[1]), int(size[0]))
    if pil.size != target:
        pil = pil.resize(target)
    return pil


class VisionEncoder:
    """Eq. 4:  v_t = E_phi(o_t)  in  R^{N_v x d}   (FROZEN)

    Wraps the Cosmos-Reason2-2B vision tower. The encoder is frozen and always
    runs under torch.no_grad(); `encode` returns one (N_v, d) tensor per frame.

    API note (verified against transformers 5.16.1 in Session 2): the vision
    tower returns a BaseModelOutputWithDeepstackFeatures, NOT a bare tensor.
    The merged visual tokens live in `.pooler_output`, a tuple holding one
    (N_v, d) tensor per image. Calling `model.visual(...)` directly hands back
    the wrapper object, which is the single easiest mistake to make here.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
        model=None,
        processor=None,
    ):
        """
        Args:
            model_path: local path to the Cosmos-Reason2-2B checkpoint.
            device:     "cuda" / "cpu"; auto-detected when None.
            dtype:      compute dtype for the tower (bf16 per AGENTS.md Sec. 3.3).
            image_size: (H, W) every frame is resized to. Determines N_v.
            model:      optional pre-loaded Qwen3VLForConditionalGeneration, so
                        Eq. 6's backbone can share one set of weights rather
                        than paying ~4.6 GB of VRAM twice.
            processor:  optional pre-loaded AutoProcessor, paired with `model`.
        """
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = tuple(image_size)
        self.model_path = model_path

        if model is None:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=dtype,
                device_map=self.device,
                # Session 1 noted the checkpoint stores embed_tokens and lm_head
                # separately with different values; declaring this silences a
                # warning and keeps the loaded weights as-is.
                tie_word_embeddings=False,
            )
        self.model = model
        self.processor = processor or AutoProcessor.from_pretrained(model_path)

        # Freeze: eval mode + requires_grad False on every parameter.
        self.model.eval()
        self.model.requires_grad_(False)

        cfg = self.model.config
        self.hidden_size = cfg.text_config.hidden_size            # d
        self.patch_size = cfg.vision_config.patch_size
        self.spatial_merge_size = cfg.vision_config.spatial_merge_size

    # -- properties ---------------------------------------------------------
    @property
    def d(self) -> int:
        """Backbone hidden size (2048 for Cosmos-Reason2-2B)."""
        return self.hidden_size

    @property
    def n_v(self) -> int:
        """Visual tokens per frame at the configured resolution.

        N_v = (H / patch) * (W / patch) / merge^2
        """
        h, w = self.image_size
        return (h // self.patch_size) * (w // self.patch_size) // (
            self.spatial_merge_size ** 2
        )

    @property
    def is_frozen(self) -> bool:
        """True when no parameter of the wrapped model requires grad."""
        return not any(p.requires_grad for p in self.model.parameters())

    # -- Eq. 4 --------------------------------------------------------------
    @torch.no_grad()
    def encode(self, image) -> torch.Tensor:
        """Eq. 4:  v_t = E_phi(o_t)  in  R^{N_v x d}

        Args:
            image: PIL image, or numpy (H, W, 3) / (H, W, 4) array.

        Returns:
            (N_v, d) tensor on self.device.
        """
        return self.encode_batch([image])[0]

    @torch.no_grad()
    def encode_batch(self, images: List) -> List[torch.Tensor]:
        """Eq. 4 applied to several frames.

        Returns a list of (N_v, d) tensors -- one per input frame. A list is
        returned rather than a stacked tensor because the underlying API emits
        one variable-length entry per image; at a fixed resolution every entry
        has the same N_v and callers may stack freely.

        WARNING -- batch size affects values in bf16. Measured in Session 3:
        the same frame encoded at batch size 3 vs 1 differs by ~7% relative L2
        (cosine still 0.998). Preprocessing is bitwise identical, and in fp32
        the same comparison agrees to ~1e-5, so this is CUDA choosing different
        kernels/reduction orders per batch shape, compounded over 24 ViT layers.

        Implication for Stage 1: when caching v_bar (Sec. 3.2) for the Pilot
        target v_bar_{t+2} (Eq. 11/14), use a FIXED batch size throughout, or
        cache in fp32. Otherwise the regression target silently depends on how
        frames were grouped, and cached-vs-live values will not agree.
        """
        if not images:
            return []

        pils = [to_rgb_pil(img, self.image_size) for img in images]

        # One <image> placeholder per frame; the text itself is irrelevant to
        # the vision tower, it only has to carry the image tokens through.
        inputs = self.processor(
            text=["<image>"] * len(pils), images=pils, return_tensors="pt"
        )
        inputs = {
            k: (v.to(self.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }

        out = self.model.model.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"]
        )
        return list(out.pooler_output)
