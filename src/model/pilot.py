"""
Eq. 8 — the Pilot module, plus the inference-time PilotCache (Sec. 3.3).

    z_t = G_psi(h_t^pil)  in  R^d

G_psi is "a simple linear layer, with only a few parameters that projects the
Pilot-position hidden state back to the latent space" (Sec. 3.2). Not an MLP,
not a transformer -- AGENTS.md Sec. 4.1 makes deepening it a STOP-and-ask.
Parameter count is asserted to be exactly d*d + d.

--------------------------------------------------------------------------
z_0 -- APPROVED DEVIATION IN MECHANISM (human, 2026-09-01)
--------------------------------------------------------------------------
Sec. 3.2 says the Pilot slot "is implemented by a special placeholder token
<|placeholder|> whose embedding provides z_0", and AGENTS.md Sec. 4.1 says to
add that token to the tokenizer. We use a standalone learned nn.Parameter
instead. Measured reasons:

  * `<|placeholder|>` is NOT in this tokenizer -- it splits into 5 tokens.
  * The embedding matrix is 311,164,928 params, 48x our whole LoRA adapter.
    z_0 must be LEARNED, and training one row of that matrix means either
    marking all 311M trainable (abandons LoRA-only, blows the VRAM budget) or
    hand-masking a single row's gradient, which fails silently when wrong.
  * The tokenizer never sees the Pilot slot at all: Eq. 5's PILOT(z) is a
    continuous latent with no token id, which is exactly why sequences are
    built from inputs_embeds (D6). A tokenizer entry would be an id nothing
    ever produces.

An nn.Parameter is 2048 trainable params and is functionally identical -- both
are "a learned d-dim vector occupying the Pilot slot". Only the literal
"has an embedding row" phrasing changes.

INITIALISATION: z_0 is scaled to the norm of REAL token embeddings (~1.44),
not to the untrained spare rows' ~0.36 and not to unit-Gaussian scale. The
Pilot slot is an out-of-distribution input to a pretrained backbone no matter
what; starting it at the scale the backbone actually expects is the same
reasoning that drove D10's layout choice.

--------------------------------------------------------------------------
EPISODE BOUNDARIES
--------------------------------------------------------------------------
AGENTS.md Sec. 8 item 6 lists this as unresolved in the paper: "presumably
reset to z_0 at each episode start; not stated explicitly". PilotCache.reset()
does exactly that, and `episode_steps` is exposed so a caller can assert the
cache was actually reset between episodes rather than leaking state across
them.
"""

# --- ROS guard: strip /opt/ros/* from sys.path before any other import. ------
import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]
# ----------------------------------------------------------------------------

from typing import Optional

import torch
import torch.nn as nn

# Mean L2 norm of real (non-spare) token embedding rows in Cosmos-Reason2-2B,
# measured Session 3. Spare/untrained rows sit at ~0.36 by comparison.
REAL_TOKEN_EMBED_NORM = 1.4439

PILOT_MODES = ("none", "last", "action_query")


class PilotModule(nn.Module):
    """Eq. 8's G_psi, the learned z_0, and (mode B) a learned action query.

    Modes -- the two readings of where h_t^act comes from, see DECISIONS.md:
      "none"          Stage 0: no Pilot slot at all.
      "last"          A: [... ; Tok(x) ; PILOT(z)].  h_act == h_pil == H[-1];
                      one hidden state, two heads on it.
      "action_query"  B: [... ; Tok(x) ; PILOT(z) ; ACTION_QUERY].
                      h_pil = H[-2], h_act = H[-1]; genuinely distinct states.

    Both put the Pilot AFTER the instruction, which is forced rather than
    chosen: attention is causal, so a Pilot placed before Tok(x) could not see
    the instruction, and z_t (which must predict v_bar_{t+2}, the view two
    steps ahead) cannot encode instruction-conditioned intent without it.
    """

    def __init__(self, hidden_size: int, mode: str = "last",
                 dtype: torch.dtype = torch.bfloat16,
                 device: Optional[torch.device] = None,
                 init_norm: float = REAL_TOKEN_EMBED_NORM,
                 generator: Optional[torch.Generator] = None):
        super().__init__()
        if mode not in PILOT_MODES:
            raise ValueError(f"mode must be one of {PILOT_MODES}, got {mode!r}")

        self.hidden_size = hidden_size
        self.mode = mode
        self.init_norm = init_norm

        # Eq. 8 -- a SINGLE linear layer. Deepening this is a STOP-and-ask.
        self.G_psi = nn.Linear(hidden_size, hidden_size, bias=True)

        self.z_0 = nn.Parameter(self._init_vector(hidden_size, generator))
        if mode == "action_query":
            self.action_query = nn.Parameter(
                self._init_vector(hidden_size, generator)
            )
        else:
            self.register_parameter("action_query", None)

        self.to(device=device, dtype=dtype)

    def _init_vector(self, d: int, generator) -> torch.Tensor:
        """A random direction scaled to the real-token embedding norm."""
        v = torch.randn(d, dtype=torch.float32, generator=generator)
        return v / v.norm() * self.init_norm

    # -- Eq. 8 --------------------------------------------------------------
    def forward(self, h_pil: torch.Tensor) -> torch.Tensor:
        """z_t = G_psi(h_t^pil). Accepts (d,) or (B, d)."""
        return self.G_psi(h_pil)

    # -- introspection ------------------------------------------------------
    @property
    def g_psi_param_count(self) -> int:
        return sum(p.numel() for p in self.G_psi.parameters())

    @property
    def expected_g_psi_param_count(self) -> int:
        """Eq. 8 as a single Linear(d, d): d*d weights + d biases."""
        return self.hidden_size * self.hidden_size + self.hidden_size

    def assert_shapes(self) -> None:
        """Fail loudly if G_psi has drifted from Eq. 8's single linear layer."""
        got, want = self.g_psi_param_count, self.expected_g_psi_param_count
        if got != want:
            raise RuntimeError(
                f"G_psi has {got:,} params, Eq. 8 requires exactly {want:,} "
                f"(a single nn.Linear({self.hidden_size}, {self.hidden_size}))"
            )
        if not torch.isfinite(self.z_0).all() or self.z_0.abs().sum() == 0:
            raise RuntimeError("z_0 must be finite and non-zero")

    @torch.no_grad()
    def init_bias_from_target_mean(self, mean_vbar: torch.Tensor) -> None:
        """Start G_psi predicting the mean v_bar instead of ~zero.

        NOT IN PAPER -- an initialisation choice, not an objective change. The
        standard trick of seeding an output layer's bias with the target mean.

        It matters here because of scale. Measured on cached v_bar (||v_bar||
        ~= 15.8, so ||v_bar||^2 ~= 250):

            predictor                     L_pil    lambda*L_pil / L_act
            z = 0  (default Linear init)  251.0            25x
            z = global mean v_bar          17.9           1.8x
            z = v_bar_{t+1} (copy)          4.8           0.5x

        With the default bias, Eq. 15's first steps are dominated ~25:1 by
        L_pil and can wreck the action policy before G_psi finds the output
        scale. Seeding the bias starts it at 1.8x, and a converged G_psi should
        reach ~0.5x -- which is where the paper's lambda=0.1 is well balanced.

        The weights are left at their default small init, so z starts near the
        target mean and is refined from there.
        """
        if mean_vbar.shape != (self.hidden_size,):
            raise ValueError(
                f"mean_vbar must be ({self.hidden_size},), got "
                f"{tuple(mean_vbar.shape)}"
            )
        self.G_psi.bias.copy_(mean_vbar.to(device=self.G_psi.bias.device,
                                           dtype=self.G_psi.bias.dtype))

    def new_cache(self) -> "PilotCache":
        return PilotCache(self.z_0)


class PilotCache:
    """Sec. 3.3, "Inference with stored pilot".

    Purely recurrent rollout: read z_{t-1}, forward, write z_t, repeat until
    STOP. No future frames, no separate Pilot computation.

    This class deliberately has NO path to accept a v_bar. Eq. 12 teacher-forces
    the slot with v_bar_{t+1} at TRAINING only; at inference the slot must carry
    the model's own z_{t-1}. AGENTS.md Sec. 4.1 calls that asymmetry "the single
    easiest thing to get wrong", and warns that future leakage at eval
    invalidates every number -- so the eval path simply cannot express it.
    """

    def __init__(self, z_0: torch.Tensor):
        self._z_0 = z_0.detach().clone()
        self._z: Optional[torch.Tensor] = None
        self.episode_steps = 0
        self.reset()

    def reset(self) -> None:
        """Start a new episode: z <- z_0 (AGENTS.md Sec. 8 item 6)."""
        self._z = self._z_0.clone()
        self.episode_steps = 0

    def read(self) -> torch.Tensor:
        """z_{t-1} -- the Pilot slot input for this step."""
        return self._z

    def write(self, z_t: torch.Tensor) -> None:
        """Store z_t for the next step.

        Detached on purpose: the cache spans steps, and keeping the graph alive
        across a rollout would backprop through time -- which the paper does not
        do and which would OOM a 16 GB card.
        """
        if z_t.shape != self._z_0.shape:
            raise ValueError(
                f"z_t has shape {tuple(z_t.shape)}, expected "
                f"{tuple(self._z_0.shape)}"
            )
        self._z = z_t.detach()
        self.episode_steps += 1

    @property
    def is_fresh(self) -> bool:
        """True when no step has been written since the last reset."""
        return self.episode_steps == 0
