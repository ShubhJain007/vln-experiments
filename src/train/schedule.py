"""
Learning-rate schedules for all training stages.

AGENTS.md Sec. 8 item 3 lists "learning rate, schedule, batch size, number of
epochs" as NOT stated in the paper, so every value here is ours and must be
reported with results (Sec. 7).

--------------------------------------------------------------------------
WHY THIS MODULE EXISTS -- the original schedule had no decay
--------------------------------------------------------------------------
Stages 0 and 0' originally used `min(1.0, (s+1)/warmup)`: a linear warmup and
then a CONSTANT LR for the remaining ~99% of the run. Two consequences:

  * The model takes full-size steps right up to the final checkpoint, so it
    never settles. The saved weights are a random point in a noisy basin
    rather than the basin's floor.
  * A loss plateau under constant LR is ambiguous -- it can mean "converged"
    or "LR-limited". Decaying resolves it, and usually buys another drop.

Stage 0's run showed exactly that signature: loss 0.66 / 0.69 / 0.65 across
steps 6000-8000 with accuracy pinned at ~0.775.

--------------------------------------------------------------------------
WHY THE PEAK LR STAYS AT 1e-4
--------------------------------------------------------------------------
1e-4 is the standard band for LoRA (typically 1e-4 to 3e-4). LoRA adapters are
initialised near zero and need a larger LR than full fine-tuning, where 1e-5
would be right. Lowering the PEAK would undertrain; the missing piece was
always the decay, not the magnitude.
"""

import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]

import math

import torch

SCHEDULES = ("cosine", "linear", "constant")

# NOT IN PAPER -- ours. A floor rather than 0 so the last steps still move.
DEFAULT_MIN_LR_FRAC = 0.05


def lr_lambda(step: int, warmup: int, total_steps: int,
              schedule: str = "cosine",
              min_lr_frac: float = DEFAULT_MIN_LR_FRAC) -> float:
    """Multiplier on the peak LR at `step`.

    Linear warmup for `warmup` steps, then decay to `min_lr_frac` of peak by
    `total_steps`. Exposed as a plain function so the shape can be unit-tested
    without building an optimizer.
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, got {schedule!r}")

    if step < warmup:
        return (step + 1) / max(1, warmup)

    if schedule == "constant":
        return 1.0

    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    if schedule == "cosine":
        decayed = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:                                       # linear
        decayed = 1.0 - progress
    return min_lr_frac + (1.0 - min_lr_frac) * decayed


def make_scheduler(optimizer, warmup: int, total_steps: int,
                   schedule: str = "cosine",
                   min_lr_frac: float = DEFAULT_MIN_LR_FRAC):
    """LambdaLR implementing warmup + decay.

    `total_steps` must be the run's FULL length -- the decay is defined
    relative to it, so passing a different number silently changes the shape.
    """
    if total_steps <= warmup:
        raise ValueError(
            f"total_steps ({total_steps}) must exceed warmup ({warmup}); "
            f"otherwise the run is pure warmup and never decays"
        )
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: lr_lambda(s, warmup, total_steps, schedule, min_lr_frac),
    )


def describe(warmup: int, total_steps: int, peak_lr: float,
             schedule: str = "cosine",
             min_lr_frac: float = DEFAULT_MIN_LR_FRAC) -> str:
    """One-line summary for the training log, so the schedule is recoverable
    from the log alone rather than only from the command line."""
    end = peak_lr * lr_lambda(total_steps - 1, warmup, total_steps,
                              schedule, min_lr_frac)
    return (f"LR {schedule}: warmup {warmup} -> peak {peak_lr:.2e} "
            f"-> {end:.2e} at step {total_steps} "
            f"(floor {min_lr_frac:.0%} of peak)")
