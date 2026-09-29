"""
Eq. 5 — Multimodal input sequence.

Paper form (Stage 1+):
    u_t = [ Tok(x) ; v_t ; PILOT(z_{t-1}) ]

Stage 0 form (this module, no Pilot Token):
    u_t = [ Tok(x) ; v_t ]

where Tok(x) is the tokenized instruction and v_t = E_phi(o_t) in R^{N_v x d}
is the CURRENT frame's visual tokens (Eq. 4).

THE KEY PROPERTY: only the current frame appears. There are no history frames.
This is the paper's central efficiency claim -- the sequence stays ~300 tokens
however long the episode runs, instead of growing with t. Per AGENTS.md
Sec. 3.1 it is also "the most likely place to accidentally deviate", so the API
is built so history CANNOT be passed: `build()` takes exactly one frame's
visual tokens and holds no state between calls.

Sequences are built as INPUT EMBEDDINGS, not token ids. Eq. 5's PILOT(z_{t-1})
places a continuous latent at a token position, which has no token-id form;
going through embeddings now means Stage 1 adds the Pilot slot without
restructuring anything.

--------------------------------------------------------------------------
LAYOUT -- APPROVED DEVIATION FROM Eq. 5's STATED ORDER (human, 2026-08-31)
--------------------------------------------------------------------------
Eq. 5 writes [Tok(x) ; v_t]: instruction first, image second (Fig. 2 shows the
same order). We instead use the backbone's pretraining layout:

    [ <vision_start> ; v_t ; <vision_end> ; Tok(x) ]

Measured zero-shot instruction-following on 160 trials (40 unambiguous probes
x 4 real frames; chance = 25%), mRoPE correct in every arm:

    A  instr-first, no markers  (literal Eq. 5)   21.2%   <- at/below chance
    D  instr-first + markers                      41.2%
    E  image-first, no markers                    33.1%
    F  image-first + markers    (CHOSEN)          77.5%
    C  full native w/ chat template               68.8%

The two factors interact: markers alone +20, ordering alone +12, both +56.
Half the format is one the model has never seen. Note also that adding chat
scaffolding on top (C) HURTS relative to F, so the chat template is not used.

Rationale is the same principle that settled mRoPE in backbone.py: use the
backbone as it was PRETRAINED. With only 1000-2000 training episodes,
initialisation quality matters. Caveat: this is a zero-shot measurement on an
untrained model; LoRA may narrow the gap.

`layout="eq5_literal"` implements the paper's exact ordering and is kept so the
deviation can be re-measured after training. Whichever layout is used must be
IDENTICAL in training and evaluation.
"""

# --- ROS guard: strip /opt/ros/* from sys.path before any other import. ------
import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]
# ----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

LAYOUTS = ("image_first_markers", "eq5_literal")
DEFAULT_LAYOUT = "image_first_markers"


@dataclass(frozen=True)
class InputSequence:
    """One built multimodal sequence (Eq. 5), plus its structure.

    Attributes:
        inputs_embeds:  (1, L, d) -- the sequence u_t.
        attention_mask: (1, L)    -- all ones; no padding is produced here.
        instr_slice:    slice covering the Tok(x) region.
        visual_slice:   slice covering the v_t region (markers excluded).
        segments:       ordered [(kind, length)] with kind in {"text","visual"},
                        consumed by Backbone to build mRoPE position ids. Marker
                        tokens count as 1-length "text" segments.
        layout:         which layout produced this sequence.
    """

    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    instr_slice: slice
    visual_slice: slice
    segments: Tuple[Tuple[str, int], ...]
    layout: str = DEFAULT_LAYOUT
    pilot_index: Optional[int] = None
    n_frames: int = 1
    pilot_mode: str = "none"

    @property
    def seq_len(self) -> int:
        return self.inputs_embeds.shape[1]

    @property
    def n_instruction_tokens(self) -> int:
        return self.instr_slice.stop - self.instr_slice.start

    @property
    def n_visual_tokens(self) -> int:
        """N_v (marker tokens excluded)."""
        return self.visual_slice.stop - self.visual_slice.start

    @property
    def hidden_size(self) -> int:
        """d."""
        return self.inputs_embeds.shape[2]

    @property
    def action_index(self) -> int:
        """Position whose hidden state is h_t^act (Eq. 7).

        Always the LAST position: in an autoregressive model that is the state
        which predicts the next token, i.e. the action. Under pilot_mode="last"
        that position IS the Pilot slot (h_act == h_pil, Design A); under
        "action_query" it is the trailing query token, one past the Pilot
        (Design B). See DECISIONS.md.
        """
        return self.seq_len - 1

    @property
    def has_pilot(self) -> bool:
        return self.pilot_index is not None

    @property
    def pilot_and_action_share_state(self) -> bool:
        """True for Design A -- W_LM and G_psi read the SAME hidden vector."""
        return self.pilot_index is not None and self.pilot_index == self.action_index


class InputSequenceBuilder:
    """Eq. 5 (Stage 0). See the module docstring for the approved layout.

    Stateless by construction: each `build()` call sees one instruction and one
    frame, and nothing carries between calls, so no history frame can enter the
    sequence even by accident.
    """

    def __init__(self, model, tokenizer, layout: str = DEFAULT_LAYOUT,
                 use_chat_template: bool = False, pilot=None):
        """
        Args:
            model:     loaded Qwen3VLForConditionalGeneration (for its
                       input-embedding table, dtype and device).
            tokenizer: matching tokenizer, for Tok(x).
            layout:    "image_first_markers" (default, approved) or
                       "eq5_literal" (the paper's exact ordering).
            use_chat_template: adds role/system scaffolding. Default False --
                       measured to HURT (68.8% vs 77.5%) relative to the chosen
                       layout.
        """
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")

        self.model = model
        self.tokenizer = tokenizer
        self.layout = layout
        self.use_chat_template = use_chat_template
        self.pilot = pilot
        self.pilot_mode = getattr(pilot, "mode", "none") if pilot is not None else "none"
        self._embed = model.get_input_embeddings()

        cfg = model.config
        self.vision_start_id = cfg.vision_start_token_id
        self.vision_end_id = cfg.vision_end_token_id

    # -- Tok(x) -------------------------------------------------------------
    def tokenize_instruction(self, instruction: str) -> torch.Tensor:
        """Tok(x) -> (L_instr,) token ids on the model's device."""
        text = instruction
        if self.use_chat_template:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True,
            )
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return ids["input_ids"][0].to(self.model.device)

    def _marker(self, token_id: int, dtype) -> torch.Tensor:
        ids = torch.tensor([token_id], device=self.model.device)
        return self._embed(ids).to(dtype)

    # -- Eq. 5 --------------------------------------------------------------
    @torch.no_grad()
    def build(
        self,
        instruction: str,
        visual_tokens: torch.Tensor,
        instruction_ids: Optional[torch.Tensor] = None,
        pilot_input: Optional[torch.Tensor] = None,
        history_visual: Optional[List[torch.Tensor]] = None,
    ) -> InputSequence:
        """Build u_t for one step.

        Args:
            instruction:     the natural-language instruction x.
            visual_tokens:   (N_v, d) from Eq. 4 -- the CURRENT frame only.
            instruction_ids: optional pre-tokenized Tok(x), to avoid
                             re-tokenizing every step of an episode.

        Returns:
            InputSequence. Layout per `self.layout`.

        There is deliberately NO parameter for previous frames, previous
        sequences, or any other history; adding one would violate Eq. 5.
        """
        # -- frame history (D28) ------------------------------------------
        # Robostral Sec. 2.1: "The history enables the model to reason about
        # previously visited locations and track progress." Frames are ordered
        # OLDEST -> NEWEST so the current view sits adjacent to the instruction
        # and the action query, and mRoPE's temporal channel encodes recency.
        n_frames = 1
        if history_visual:
            for h in history_visual:
                if h.ndim != 2 or h.shape != visual_tokens.shape:
                    raise ValueError(
                        f"history frame {tuple(h.shape)} != current "
                        f"{tuple(visual_tokens.shape)}; every frame must have "
                        f"the same token count for the video grid to be valid"
                    )
            n_frames = len(history_visual) + 1
            visual_tokens = torch.cat(list(history_visual) + [visual_tokens], dim=0)

        if visual_tokens.ndim != 2:
            raise ValueError(
                f"visual_tokens must be (N_v, d) for a single frame; got shape "
                f"{tuple(visual_tokens.shape)}. Passing a batch or a stack of "
                f"history frames here would violate Eq. 5."
            )

        ids = (
            instruction_ids
            if instruction_ids is not None
            else self.tokenize_instruction(instruction)
        )
        instr_embeds = self._embed(ids)                    # (L_instr, d)

        d = instr_embeds.shape[-1]
        if visual_tokens.shape[-1] != d:
            raise ValueError(
                f"visual token dim {visual_tokens.shape[-1]} != model hidden size {d}"
            )

        dtype = instr_embeds.dtype
        v = visual_tokens.to(device=instr_embeds.device, dtype=dtype)
        n_instr, n_v = instr_embeds.shape[0], v.shape[0]

        parts: List[torch.Tensor]
        segments: List[Tuple[str, int]]

        if self.layout == "eq5_literal":
            # [ Tok(x) ; v_t ] -- the paper's exact ordering.
            parts = [instr_embeds, v]
            segments = [("text", n_instr), ("visual", n_v)]
            instr_slice = slice(0, n_instr)
            visual_slice = slice(n_instr, n_instr + n_v)
        else:
            # [ <vision_start> ; v_t ; <vision_end> ; Tok(x) ]
            parts = [
                self._marker(self.vision_start_id, dtype),
                v,
                self._marker(self.vision_end_id, dtype),
                instr_embeds,
            ]
            segments = [
                ("text", 1), ("visual", n_v), ("text", 1), ("text", n_instr)
            ]
            visual_slice = slice(1, 1 + n_v)
            instr_slice = slice(2 + n_v, 2 + n_v + n_instr)

        # -- Eq. 5's PILOT(.) slot ------------------------------------------
        # Always AFTER Tok(x): attention is causal, so a Pilot placed earlier
        # could not see the instruction, and z_t must encode where the agent is
        # heading in order to predict v_bar_{t+2}.
        pilot_index = None
        if self.pilot_mode != "none":
            if pilot_input is None:
                raise ValueError(
                    f"pilot_mode={self.pilot_mode!r} needs pilot_input: "
                    f"v_bar_{{t+1}} when training (Eq. 12) or z_{{t-1}} from "
                    f"PilotCache at inference (Eq. 5)"
                )
            if pilot_input.shape != (d,):
                raise ValueError(
                    f"pilot_input must be ({d},), got {tuple(pilot_input.shape)}"
                )
            pilot_index = sum(n for _, n in segments)
            parts.append(pilot_input.to(device=instr_embeds.device,
                                        dtype=dtype).unsqueeze(0))
            segments.append(("text", 1))

            if self.pilot_mode == "action_query":
                parts.append(self.pilot.action_query.to(
                    device=instr_embeds.device, dtype=dtype).unsqueeze(0))
                segments.append(("text", 1))

        seq = torch.cat(parts, dim=0).unsqueeze(0)          # (1, L, d)
        return InputSequence(
            inputs_embeds=seq,
            attention_mask=torch.ones(
                (1, seq.shape[1]), dtype=torch.long, device=seq.device
            ),
            instr_slice=instr_slice,
            visual_slice=visual_slice,
            segments=tuple(segments),
            layout=self.layout,
            pilot_index=pilot_index,
            n_frames=n_frames,
            pilot_mode=self.pilot_mode,
        )

    def expected_seq_len(self, instruction: str, n_v: int) -> int:
        """Sequence length. Independent of timestep, by design."""
        n_markers = 0 if self.layout == "eq5_literal" else 2
        n_pilot = {"none": 0, "last": 1, "action_query": 2}[self.pilot_mode]
        return (len(self.tokenize_instruction(instruction))
                + n_v + n_markers + n_pilot)


@dataclass(frozen=True)
class BatchedSequences:
    """Several Eq. 5 sequences padded into one batch.

    Attributes:
        inputs_embeds:  (B, L_max, d), right-padded with zeros.
        attention_mask: (B, L_max), 1 on real tokens.
        position_ids:   (3, B, L_max) mRoPE ids, or None for sequential.
        lengths:        (B,) true length of each sequence.
        action_index:   (B,) position of h_t^act in each row = length - 1.
        pilot_index:    (B,) position of h_t^pil, or None when pilot_mode is
                        "none". Under Design A this EQUALS action_index (one
                        hidden state, two heads); under Design B it is
                        action_index - 1. Eq. 8 reads z_t = G_psi(h_t^pil) from
                        here, so getting it wrong silently trains G_psi on the
                        action state instead of the Pilot state.
    """

    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: Optional[torch.Tensor]
    lengths: torch.Tensor
    action_index: torch.Tensor
    pilot_index: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        return self.inputs_embeds.shape[0]


def collate_sequences(sequences: List[InputSequence],
                      position_ids: Optional[List[torch.Tensor]] = None
                      ) -> BatchedSequences:
    """Pad a list of InputSequence into a batch.

    RIGHT padding, so `h_t^act` sits at index (length - 1) per row rather than
    at a shared -1. Left padding would put every action state at -1 but shifts
    each row's mRoPE ids, so right padding is the safer of the two here.

    Padded positions are masked out, so the embedding values there are never
    attended to; they are zeroed only to keep the tensor well defined.
    """
    if not sequences:
        raise ValueError("no sequences to collate")

    b = len(sequences)
    lengths = [s.seq_len for s in sequences]
    l_max = max(lengths)
    d = sequences[0].hidden_size
    ref = sequences[0].inputs_embeds

    embeds = torch.zeros((b, l_max, d), dtype=ref.dtype, device=ref.device)
    mask = torch.zeros((b, l_max), dtype=torch.long, device=ref.device)
    for i, s in enumerate(sequences):
        n = s.seq_len
        embeds[i, :n] = s.inputs_embeds[0]
        mask[i, :n] = 1

    # Eq. 8's h_t^pil. Sequences are RIGHT-padded, so each row's Pilot slot
    # keeps the absolute index it had before padding.
    pilot_idx = None
    if sequences[0].pilot_index is not None:
        if any(s.pilot_index is None for s in sequences):
            raise ValueError(
                "mixed batch: some sequences carry a Pilot slot and some do not"
            )
        pilot_idx = torch.tensor([s.pilot_index for s in sequences],
                                 device=ref.device)

    pos = None
    if position_ids is not None:
        if len(position_ids) != b:
            raise ValueError(
                f"{len(position_ids)} position id tensors for {b} sequences"
            )
        pos = torch.zeros((3, b, l_max), dtype=torch.long, device=ref.device)
        for i, p in enumerate(position_ids):
            n = lengths[i]
            pos[:, i, :n] = p[:, 0, :n]

    return BatchedSequences(
        inputs_embeds=embeds,
        attention_mask=mask,
        position_ids=pos,
        lengths=torch.tensor(lengths, device=ref.device),
        action_index=torch.tensor([n - 1 for n in lengths], device=ref.device),
        pilot_index=pilot_idx,
    )
