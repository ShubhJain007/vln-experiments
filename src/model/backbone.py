"""
Eq. 6 — LLM backbone forward pass.

    H_t = F_theta(u_t)  in  R^{N x d}

"The LLM backbone computes hidden states with causal attention" (Sec. 3.2).
N is the sequence length from Eq. 5, d the hidden size. From H_t the model
reads h_t^act (Eq. 7) and, from Stage 1, h_t^pil (Eq. 8).

Causality is the load-bearing property: position i's hidden state must depend
only on positions <= i. The paper's entire claim is that inference stays
strictly causal, so a leak here would invalidate every downstream number.

--------------------------------------------------------------------------
POSITION EMBEDDINGS -- mRoPE, and why it is the default
--------------------------------------------------------------------------
Cosmos-Reason2-2B (Qwen3-VL) uses mRoPE: image tokens get 3D (temporal,
height, width) position ids, NOT sequential ones. Crucially, an image advances
the position counter by only max(H,W)/merge, not by N_v:

    448x448 image -> 196 tokens, but positions run t=4, h=4..17, w=4..17
    position before image = 3,  position after image = 18   (+14, not +196)

transformers only takes that path when input_ids AND mm_token_type_ids AND
image_grid_thw are all present (verified in `compute_3d_position_ids`,
transformers 5.16.1). Eq. 5 must build sequences from inputs_embeds, because
PILOT(z_{t-1}) is a continuous latent with no token id -- so the automatic path
is unreachable and the model would silently fall back to sequential positions.

Measured cost of that fallback, same sequence, only position_ids differing
(5 real R2R-CE instructions x real walk.mov frames):

    hidden-state cosine        0.87 - 0.91
    hidden-state relative L2   0.42 - 0.50
    action-position cosine     0.39 - 0.94
    top-1 next token agreed    2 / 5

That is an order of magnitude larger than the ~7% bf16 batch-size noise, and
it changes the model's actual predictions most of the time. It is not a
rounding detail.

So this module reconstructs mRoPE ids explicitly and passes them in. The
reconstruction is verified BITWISE against the model's own get_rope_index
(tests/test_backbone.py::test_mrope_reconstruction_matches_model_ground_truth).

Why mRoPE rather than sequential, given the paper used LLaVA-Video-7B (which
has no mRoPE): the goal is to use each backbone the way it was pretrained.
LLaVA-Video at sequential positions is in-distribution FOR LLaVA-Video;
Cosmos-Reason2-2B at sequential positions is out-of-distribution FOR COSMOS.
Fidelity to the paper means preserving the relationship (backbone used
natively), not copying the mechanism. AGENTS.md Sec. 1.4 approved this
substitution on the grounds that the architecture and encoder family match --
running it in a non-native position scheme would undercut that equivalence.

`position_mode="sequential"` is kept so the alternative can be measured, but
it is not the default.
"""

# --- ROS guard: strip /opt/ros/* from sys.path before any other import. ------
import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]
# ----------------------------------------------------------------------------

import math
from typing import Optional

import torch


class Backbone:
    """Eq. 6:  H_t = F_theta(u_t)  in  R^{N x d}

    Holds no state between calls: each forward sees exactly one Eq. 5 sequence,
    matching the paper's step interface (Eq. 9), where the only thing carried
    across steps is the Pilot Token -- never a KV cache or a hidden state.
    """

    def __init__(self, model, position_mode: str = "mrope"):
        """
        Args:
            model: loaded Qwen3VLForConditionalGeneration. Pass the SAME
                   instance held by VisionEncoder -- loading twice costs
                   ~4.6 GB of VRAM for no benefit.
            position_mode: "mrope" (default, native to this checkpoint) or
                   "sequential" (the naive fallback; see module docstring).
        """
        if position_mode not in ("mrope", "sequential"):
            raise ValueError(
                f"position_mode must be 'mrope' or 'sequential', got {position_mode!r}"
            )
        self.model = model
        self.position_mode = position_mode
        self.spatial_merge_size = model.config.vision_config.spatial_merge_size

    @property
    def hidden_size(self) -> int:
        """d."""
        return self.model.config.text_config.hidden_size

    @property
    def num_layers(self) -> int:
        return self.model.config.text_config.num_hidden_layers

    # -- position ids -------------------------------------------------------
    def visual_grid_thw(self, n_v: int, n_frames: int = 1) -> torch.Tensor:
        """(T, H, W) patch grid that produced n_v merged visual tokens.

        n_v = (H/patch) * (W/patch) / merge^2, so for a square frame each
        post-merge side is sqrt(n_v). Square input is assumed -- Eq. 5 feeds
        frames resized to a square by the vision encoder. Non-square input
        would need the grid passed in explicitly.
        """
        per_frame = n_v // n_frames
        if per_frame * n_frames != n_v:
            raise ValueError(f"n_v={n_v} not divisible by n_frames={n_frames}")
        side = int(math.isqrt(per_frame))
        if side * side != per_frame:
            raise ValueError(
                f"per-frame n_v={per_frame} is not a perfect square; pass "
                f"grid_thw explicitly for non-square frames"
            )
        merged_side = side * self.spatial_merge_size
        # T is the TEMPORAL extent. This is the model's own video
        # representation: mRoPE indexes the frame in channel 0 and position
        # within the frame in channels 1-2. Verified: T=3 -> 588 tokens with 3
        # distinct temporal positions and 14x14 spatial positions per frame.
        # Flattening frames into one T=1 grid would make every frame share a
        # timestamp, so the model could not tell past from present.
        return torch.tensor([n_frames, merged_side, merged_side])

    def build_position_ids(
        self, sequence, grid_thw: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """mRoPE position ids for an Eq. 5 sequence -> (3, 1, N).

        Driven by `sequence.segments`, so it is correct for ANY layout (the
        approved image-first-with-markers form, the literal Eq. 5 ordering, and
        Stage 1's trailing Pilot slot) rather than assuming a fixed order.

        Text segments advance the counter by their token count and carry the
        same value in all three channels. The visual segment uses the model's
        own get_vision_position_ids and advances the counter by only
        max(H,W)/merge -- NOT by N_v.

        Returns None in "sequential" mode, letting the model use its default.
        """
        if self.position_mode == "sequential":
            return None

        device = sequence.inputs_embeds.device
        grid = (
            grid_thw
            if grid_thw is not None
            else self.visual_grid_thw(sequence.n_visual_tokens,
                                      getattr(sequence, "n_frames", 1) or 1)
        )

        parts, current_pos = [], 0
        for kind, length in sequence.segments:
            if length <= 0:
                continue
            if kind == "text":
                parts.append(
                    torch.arange(length, device=device).view(1, -1).expand(3, -1)
                    + current_pos
                )
                current_pos += length
            elif kind == "visual":
                parts.append(
                    self.model.model.get_vision_position_ids(
                        current_pos, grid, 1, self.spatial_merge_size, device=device
                    )
                )
                current_pos += int(max(grid[1], grid[2])) // self.spatial_merge_size
            else:
                raise ValueError(f"unknown segment kind {kind!r}")

        pos = torch.cat(parts, dim=1).reshape(3, -1)
        if pos.shape[1] != sequence.seq_len:
            raise RuntimeError(
                f"position id length {pos.shape[1]} != sequence length "
                f"{sequence.seq_len}; segments={sequence.segments}"
            )
        return pos.unsqueeze(1)          # (3, batch=1, N)

    # -- Eq. 6 --------------------------------------------------------------
    def forward(
        self,
        sequence,
        position_ids: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Eq. 6:  H_t = F_theta(u_t)

        Args:
            sequence:     InputSequence from Eq. 5.
            position_ids: explicit override; when None they are built according
                          to `position_mode`.
            grid_thw:     explicit (T, H, W) visual grid; derived when None.
            use_cache:    KV caching, default False. The paper's step interface
                          (Eq. 9) carries only the Pilot Token across steps; a
                          retained cache would silently reintroduce the history
                          Eq. 5 exists to exclude.

        Returns:
            H_t of shape (N, d) -- one hidden vector per input position, from
            the FINAL layer. The batch dim is squeezed; Eq. 6 is single-step.
        """
        if position_ids is None:
            position_ids = self.build_position_ids(sequence, grid_thw=grid_thw)

        out = self.model.model.language_model(
            inputs_embeds=sequence.inputs_embeds,
            attention_mask=sequence.attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
        return out.last_hidden_state.squeeze(0)          # (N, d)

    def hidden_at(self, sequence, index: int, **kwargs) -> torch.Tensor:
        """Hidden state at one position -- e.g. h_t^act for Eq. 7.

        Negative indices work as in Python, so -1 is the last position.
        """
        return self.forward(sequence, **kwargs)[index]
