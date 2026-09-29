"""Capture the exact transformer kwargs Cosmos3OmniPipeline builds, for training.

The pipeline assembles its joint sequence (text + vision + action segments,
3D-mrope ids, per-modality sequence/loss indexes) inside a single ~700-line
__call__. Reimplementing that for a training loop is the obvious way to
introduce a silent bug -- the same class of bug as the +Y/+Z frame mismatch,
which would have trained the model to drive backwards without erroring.

So we don't reimplement it. We swap in a proxy transformer that records the
kwargs it is called with and aborts the denoising loop immediately. The result
is the library's own assembly, verbatim, ready to have the action tokens and
timesteps replaced with our own for a flow-matching step.
"""
import torch


class _Captured(Exception):
    """Control-flow signal: the pack is built, abandon the denoising loop."""


class _Proxy:
    """Stands in for pipe.transformer just long enough to record one call."""

    def __init__(self, real):
        self._real = real
        self.kwargs = None

    def __getattr__(self, name):          # config, dtype, device, parameters...
        return getattr(self._real, name)

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        raise _Captured


def capture_pack(pipe, *, prompt, image, chunk_size, fps,
                 domain_name="av", resolution_tier=480, view_point="ego_view"):
    """Return the transformer kwargs for one (prompt, image, chunk) sample.

    Runs the pipeline far enough to build the sequence, then bails out. Costs
    one prompt tokenization plus one VAE encode of the conditioning frame.
    """
    from diffusers import CosmosActionCondition

    real = pipe.transformer
    proxy = _Proxy(real)
    pipe.transformer = proxy
    try:
        pipe(
            prompt=prompt,
            action=CosmosActionCondition(
                mode="policy", chunk_size=chunk_size, domain_name=domain_name,
                resolution_tier=resolution_tier, view_point=view_point, image=image,
            ),
            fps=fps,
            num_inference_steps=1,
            guidance_scale=1.0,
            use_system_prompt=False,
        )
    except _Captured:
        pass
    finally:
        pipe.transformer = real
    if proxy.kwargs is None:
        raise RuntimeError("transformer was never called; pipeline layout changed")
    return proxy.kwargs
