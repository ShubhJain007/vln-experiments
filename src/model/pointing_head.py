"""
Pointing head (D27) — reads the waypoint off the action-query hidden state.

Seven outputs, three losses:

    u, v            image coords of the waypoint   (masked by `visible`)
    dir_x, dir_y    UNIT direction to the waypoint, agent frame
    stop_logit      binary, task complete
    visible_logit   binary, was a waypoint in frame

--------------------------------------------------------------------------
WHY STOP IS A SEPARATE BINARY OUTPUT
--------------------------------------------------------------------------
Under the previous 4-way softmax, STOP was one class at 2.5% of steps, and
argmax essentially never chose it: model_stop measured 0.0-4.7% across every
checkpoint, and the runs that did stop stopped in the wrong places. As its own
head it gets `pos_weight` during training and its own THRESHOLD at inference,
so precision/recall can be traded after the fact rather than being frozen into
an argmax. This is the single most important structural change of the pivot.

--------------------------------------------------------------------------
WHY IT READS MULTIPLE LAYERS (D29)
--------------------------------------------------------------------------
A per-layer probe over all 29 decoder layers, fit on train frames and scored
on held-out val_unseen, found the signals live at DIFFERENT depths -- and that
the final layer is the WORST place to read the stop decision:

    output        best layer            final layer (28)
    stop          15  (AUC 0.718)       AUC 0.441   <- below chance
    pointing u    26  (corr 0.357)      corr 0.317
    direction      0  (agree 0.646)     agree 0.407

The final layer of an LLM is shaped by next-token prediction, so "am I at the
described destination" is present mid-stack and washed out by the end. Reading
only H[-1] threw that away, which is why the deployed stop head managed just
4.1-5.1x separation.

Fusing layers recovers it (570-combination search, one block -> all outputs):

    layers          u corr    act     stop AUC
    28 (old)        0.317     0.407   0.442
    23,26           0.302     0.519   0.797
    23,26,28        ~0.30     ~0.55   ~0.74

DEFAULT 23,26,28 rather than the top-scoring 5,23,26,28: the top 12 combos
spanned only 0.815-0.784 with quite different layer sets, which is within noise
at n=1000. What was STABLE across them is layers 21-24 and 25-26 appearing
almost every time, so we take the stable structure over a noisy ranking.

--------------------------------------------------------------------------
WHY THE HEAD IS LINEAR
--------------------------------------------------------------------------
The action-query position already attends over the instruction AND all 196
visual tokens, so the fusion happens inside the backbone where it belongs. A
measured linear probe on the FROZEN vision grid alone reached only corr 0.37 /
R^2 -0.20 on u -- vision alone cannot resolve which way to go at a junction,
because that depends on the instruction. The head therefore only has to read
out a decision the LLM has already made; depth here would be papering over the
wrong layer.
"""

import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]

import torch
import torch.nn as nn

N_OUTPUTS = 6      # u, v, dir_x, dir_y, stop, visible
IDX_U, IDX_V, IDX_DIRX, IDX_DIRY, IDX_STOP, IDX_VISIBLE = range(6)


# D29: decoder layers to read, chosen by the probe documented above.
# Indices are into `hidden_states`, which has L+1 entries (embeddings at 0).
DEFAULT_FUSION_LAYERS = (23, 26, 28)


class PointingHead(nn.Module):
    """fused hidden states -> (u, v, dir_x, dir_y, stop_logit, visible_logit).

    `layers=None` reads a single hidden state, reproducing the original
    final-layer-only head exactly -- needed to load old checkpoints and to run
    the un-fused baseline.
    """

    def __init__(self, hidden_size: int, dtype=torch.float32, device=None,
                 layers=DEFAULT_FUSION_LAYERS):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = tuple(layers) if layers else None
        self.n_in = hidden_size * (len(self.layers) if self.layers else 1)
        # LayerNorm over the CONCATENATED block only. Activation scale differs
        # by roughly an order of magnitude between layers, so without
        # normalising, the largest-norm layer dominates the linear map
        # regardless of how much signal it carries.
        #
        # Identity in the single-layer case, which (a) is what the pre-fusion
        # head did, so checkpoints written before D29 still load, and (b) is
        # the right thing anyway -- there is nothing to rebalance against.
        self.norm = (nn.LayerNorm(self.n_in, dtype=dtype, device=device)
                     if self.layers else nn.Identity())
        self.proj = nn.Linear(self.n_in, N_OUTPUTS, dtype=dtype, device=device)
        with torch.no_grad():
            # Small init: start near the centre of the image rather than at an
            # arbitrary corner, so early gradients are informative.
            self.proj.weight.mul_(0.01)
            self.proj.bias.zero_()
            # u/v pass through a sigmoid, so bias 0 -> 0.5 (image centre).
            # Setting the bias to 0.5 would start at sigmoid(0.5) = 0.62.
            self.proj.bias[IDX_U] = 0.0
            self.proj.bias[IDX_V] = 0.0
            self.proj.bias[IDX_DIRY] = 1.0     # start pointing straight ahead
            # STOP fires at ~2.5% of steps; start pessimistic instead of at
            # p=0.5, which would flood the first epoch with false stops.
            self.proj.bias[IDX_STOP] = -3.0

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h is (B, n_in) -- already-fused features. See `fuse` below."""
        return self.proj(self.norm(h))

    @staticmethod
    def fuse(hidden_states, index, layers):
        """Concatenate the action-position state from each chosen layer.

        hidden_states: tuple of (B, N, d), length L+1 (embeddings first)
        index:         (B,) action position per row
        """
        rows = torch.arange(index.shape[0], device=index.device)
        return torch.cat([hidden_states[l][rows, index].float() for l in layers],
                         dim=-1)

    @property
    def param_count(self) -> int:
        n = self.n_in * N_OUTPUTS + N_OUTPUTS
        return n + (2 * self.n_in if self.layers else 0)


def split_outputs(out: torch.Tensor):
    """(B, 6) -> named pieces, with the direction normalised to a unit vector.

    WHY A UNIT DIRECTION rather than raw (dx, dy) in metres. The action is
    chosen by atan2(dx, dy), which is scale-free -- but regressing raw metres
    entangles scale with direction. Measured: typical steps are ~0.25 m, so the
    displacement loss sat at 0.009 (negligible gradient) while the bearing was
    ill-conditioned whenever both components were near zero. Action agreement
    stayed at chance (0.47 -> 0.32 -> 0.47 over 150 steps).

    Normalising makes the ONLY thing being learned the direction, which is the
    only thing the discrete controller consumes. It also removes the +/-pi
    wraparound that made a separate dtheta regression necessary.
    """
    d = out[:, [IDX_DIRX, IDX_DIRY]]
    d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return {
        "u": torch.sigmoid(out[:, IDX_U]),
        "v": torch.sigmoid(out[:, IDX_V]),
        "dir_x": d[:, 0],
        "dir_y": d[:, 1],
        "stop_logit": out[:, IDX_STOP],
        "visible_logit": out[:, IDX_VISIBLE],
    }
