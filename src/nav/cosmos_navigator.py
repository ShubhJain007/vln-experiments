"""Two-stage Cosmos3-Edge navigator for closed-loop R2R.

  stage 1  REASONER   last CTX frames of the robot's own view (as video) + the
                      R2R instruction -> one command in habitat's vocabulary.
                      This is the only place the instruction is read, and the
                      only place a STOP can originate: it is a reasoned
                      decision, not a magnitude threshold.
  stage 2  EXECUTION  either
             direct     command -> a fixed burst of primitives, then re-plan
             diffusion  command -> the action model's caption vocabulary ->
                        Cosmos3OmniPipeline policy chunk -> primitives

Both stages fit on the 16 GB card together (reasoner ~4.5 GB, action pipeline
~7.6 GB). All prompt text lives in nav.cosmos_prompts, with provenance.
"""
import collections

import numpy as np
import torch
from PIL import Image

from nav.cosmos_prompts import (SYSTEM, COMMANDS, reasoner_prompt, parse_command,
                                action_prompt, caption_prompt, parse_caption,
                                AV_ACTION_PREFIX, VIDEO_SPAN, video_kwargs)

CK = "checkpoints/Cosmos3-Edge"
CTX, FPS = VIDEO_SPAN, 15.0

# habitat primitive names, by command
DIRECT_BURST = {
    "move forward": ["move_forward", "move_forward"],       # 0.5 m
    "turn left":    ["turn_left", "turn_left"],             # 30 deg
    "turn right":   ["turn_right", "turn_right"],
    "stop":         [],
}


class CosmosNavigator:
    def __init__(self, policy="direct", chunk=24, steps=20, guidance=7.5,
                 translation_scale=7.47, max_new_tokens=768, think=True,
                 fwd_steps=2, turn_steps=2, device="cuda", adapter=None):
        from transformers import AutoProcessor, Cosmos3EdgeForConditionalGeneration

        self.policy, self.think = policy, think
        self.chunk, self.steps, self.guidance = chunk, steps, guidance
        self.scale, self.max_new_tokens = translation_scale, max_new_tokens
        self.burst = {"move forward": ["move_forward"] * fwd_steps,
                      "turn left": ["turn_left"] * turn_steps,
                      "turn right": ["turn_right"] * turn_steps, "stop": []}
        self.processor = AutoProcessor.from_pretrained(CK)
        self.reasoner = Cosmos3EdgeForConditionalGeneration.from_pretrained(
            CK, dtype=torch.bfloat16, device_map=device)
        if adapter:                      # SFT'd policy: trained with think=False
            from peft import PeftModel
            self.reasoner = PeftModel.from_pretrained(self.reasoner, adapter).merge_and_unload()
        self.pipe = None
        if policy.startswith("diffusion"):
            from diffusers import Cosmos3OmniPipeline
            # Reasoner (4.5 GB) + pipeline weights (7.6 GB) = 12.1 GB, leaving
            # nothing for the Wan VAE's activations on a 15.6 GB card -- it OOMs
            # inside the encoder. Offload keeps pipeline modules on CPU and pages
            # each onto the GPU as it runs, so only one is resident at a time.
            self.pipe = Cosmos3OmniPipeline.from_pretrained(CK, dtype=torch.bfloat16)
            self.pipe.enable_model_cpu_offload(device=device)
        self.buf = collections.deque(maxlen=CTX)
        self.last_answer = ""
        self.last_caption = ""

    # -- observation ----------------------------------------------------------
    def reset(self):
        self.buf.clear()

    def observe(self, frame):
        self.buf.append(np.asarray(frame)[:, :, :3].copy())

    def _clip(self):
        clip = list(self.buf)
        while len(clip) < CTX:                       # episode start: repeat frame 0
            clip.insert(0, clip[0])
        return np.stack(clip)

    # -- stage 1 --------------------------------------------------------------
    @torch.no_grad()
    def _generate(self, text):
        clip = self._clip()
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "video", "video": clip},
                                             {"type": "text", "text": text}]}]
        inputs = self.processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", enable_thinking=self.think, **video_kwargs(len(clip)),
        ).to(self.reasoner.device)
        kw = dict(do_sample=True, top_p=0.95, top_k=20, temperature=0.6) if self.think \
             else dict(do_sample=True, top_p=0.8, top_k=20, temperature=0.7)
        budget = self.max_new_tokens if self.think else 8
        out = self.reasoner.generate(**inputs, max_new_tokens=budget, **kw)
        self.last_answer = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return self.last_answer

    def decide(self, instruction):
        """Reasoner -> one of COMMANDS (None if unparseable)."""
        return parse_command(self._generate(
            reasoner_prompt(instruction, step=None, think=self.think)))

    def decide_caption(self, instruction):
        """Reasoner -> (caption in `<task> | <scene>` schema, is_stop).

        The caption carries the reasoner's scene understanding through to the
        action model instead of collapsing it to one of four words.
        """
        cap, is_stop, how = parse_caption(self._generate(
            caption_prompt(instruction, think=self.think)))
        self.last_how = how
        return cap, is_stop

    # -- stage 2 --------------------------------------------------------------
    @torch.no_grad()
    def plan(self, command, frame):
        """command -> ordered habitat primitives to execute before re-planning."""
        if command in (None, "stop"):
            return []
        if self.policy == "direct":
            return list(self.burst[command])
        if self.policy == "diffusion_grounded":
            # `command` IS the caption, already in their `<task> | <scene>` training
            # schema. No preamble: the AgiBot/Bridge captions carry none, and
            # "You are an autonomous vehicle..." + "The robot is in a bedroom"
            # is incoherent.
            prompt = command
            self.last_caption = command
        else:
            prompt = action_prompt(command)

        from diffusers import CosmosActionCondition
        from data.ego_pose_9d import rot_from_6d
        from eval.chunk_to_primitives import chunk_to_primitives

        img = Image.fromarray(np.asarray(frame)[:, :, :3]).convert("RGB").resize((480, 480))
        out = self.pipe(prompt=prompt,
                        action=CosmosActionCondition(
                            mode="policy", chunk_size=self.chunk, domain_name="av",
                            resolution_tier=480, view_point="ego_view", image=img),
                        fps=FPS, num_inference_steps=self.steps,
                        guidance_scale=self.guidance, use_system_prompt=False,
                        generator=torch.Generator(device="cuda").manual_seed(0))
        a = out.action
        if isinstance(a, (list, tuple)): a = a[0]
        if hasattr(a, "detach"): a = a.detach().float().cpu().numpy()
        a = np.asarray(a, np.float64).reshape(-1, 9)[:self.chunk]
        acts, _, _ = chunk_to_primitives(a, rot_from_6d, translation_scale=self.scale)
        return acts
