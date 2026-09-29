"""Every prompt we send to Cosmos3-Edge, in one place, with its provenance.

Two components, two prompt distributions -- mixing them was the bug in the
first reasoner probe:

  REASONER  Cosmos3EdgeForConditionalGeneration (transformers). Emits tokens.
            Prompts come from NVIDIA's Reasoner Prompt Guide. Reasoning is
            switched on by the chat template (`enable_thinking=True`, the
            default) which opens the assistant turn with "<think>"; the guide's
            format block is appended to the user text on top of that.

  ACTION    Cosmos3OmniPipeline policy mode (diffusers). Emits 9D chunks, never
            a token. Its prompt lands in the action-caption JSON `description`
            field, trained on short captions of the motion in a ~1.6 s clip.
            Their own example passes just "You are an autonomous vehicle
            planning system." for domain `av`.

Everything marked VERBATIM is copied from their docs. Everything marked OURS is
an adaptation for indoor navigation and is untested by definition -- there is no
navigation example in their guide, and no indoor-walker domain in the model.
"""

import re

# ------------------------------------------------------------ video window
# The Cosmos3-Edge video processor RESAMPLES by fps (do_sample_frames=True,
# default fps=2, min_frames=4): our old 16 frames @15 FPS spanned 1.07 s and
# were cut to the 4-frame floor. Tokens scale with the SAMPLED count, not the
# source count, so span is nearly free. We now hand it 120 source frames (8 s of
# the robot's own view) and sample at 1 FPS -> 8 frames x 196 tokens = 1568.
# Memory, not frame rate, is what turn-onset and "have I arrived" need.
VIDEO_SPAN = 120      # source frames kept (8 s at 15 FPS)
SRC_FPS    = 15.0     # what the frames actually are
SAMPLE_FPS = 1.0      # what the processor keeps: 8 frames over 8 s


def video_kwargs(n_frames):
    """apply_chat_template kwargs so train / probe / closed loop tokenise identically."""
    return dict(fps=SAMPLE_FPS,
                video_metadata=[{"fps": SRC_FPS, "frames_indices": list(range(n_frames)),
                                 "total_num_frames": n_frames, "duration": n_frames / SRC_FPS}])

# ------------------------------------------------------------------ reasoner
SYSTEM = "You are a helpful assistant."                                    # VERBATIM

THINK_FORMAT = (                                                           # VERBATIM
    "\nAnswer the question using the following format:\n\n"
    "<think>\nYour reasoning.\n</think>\n\n"
    "Write your final answer immediately after the </think> tag."
)

# OURS. Modelled sentence-for-sentence on their AV framing ("You are an
# autonomous vehicle planning system. The video depicts the observation from
# the vehicle's camera.") so the *shape* is in-distribution even though the
# embodiment is not. Without this the model describes the room in third person
# and speculates about "someone entering the scene" -- it does not know it IS
# the agent.
EMBODIMENT = (
    "You are a mobile robot navigating inside a building. The video depicts the "
    "observation from the robot's own forward-facing camera at eye level, ending "
    "at the present moment."
)

# OURS. The action vocabulary the reasoner must answer in -- habitat's
# primitives. Asking for a constrained answer is sanctioned by the guide ("ask
# directly for the desired answer format").
ANSWER_FORMAT = (
    "Choose the robot's next action. Answer with exactly one of: "
    "move forward, turn left, turn right, stop."
)

# VERBATIM structure (their "Assisted Task Next Action"), with our task text.
# Three parts: overall task, current step, question. The first probe dropped
# the middle line, so it never tested the template.
HIER_FULL = (
    "This is the overall task that the agent is trying to complete: \"{task}\"\n"
    "In the video, the agent is trying to follow the instruction (a single step "
    "out of many to complete the overall task): \"{step}\"\n"
    "What should be the next action of the agent?"
)

# OURS. Same template without the current-step line -- the version a closed
# loop can actually run, because R2R does not annotate progress. If HIER_FULL
# beats this by a lot offline, progress tracking is the next thing to build.
HIER_OVERALL = (
    "This is the overall task that the agent is trying to complete: \"{task}\"\n"
    "What should be the next action of the agent?"
)

COMMANDS = ("move forward", "turn left", "turn right", "stop")


def reasoner_prompt(task, step=None, think=True):
    """User-turn text for the navigation reasoner."""
    body = HIER_FULL.format(task=task.strip(), step=step.strip()) if step \
        else HIER_OVERALL.format(task=task.strip())
    text = f"{EMBODIMENT}\n{body}\n{ANSWER_FORMAT}"
    return text + THINK_FORMAT if think else text


def parse_command(answer):
    """Map the answer onto one of COMMANDS, or None.

    If </think> is present, only the text after it counts. If generation was
    truncated inside the thought (no </think>), fall back to the LAST command
    the reasoning mentions -- by then it has usually already decided, e.g.
    '...the action would be "move forward" to exit the bedroom.'
    """
    low = answer.lower().replace("move left", "turn left").replace("move right", "turn right")
    if "</think>" in low:
        tail = low.split("</think>")[-1]
    else:
        hits = [(low.rfind(c), c) for c in COMMANDS if c in low]
        return max(hits)[1] if hits else None
    # earliest-mentioned command wins; guards against "do not turn left, move forward"
    hits = [(tail.find(c), c) for c in COMMANDS if c in tail]
    if not hits:
        if "forward" in tail or "straight" in tail or "ahead" in tail:
            return "move forward"
        if "left" in tail:
            return "turn left"
        if "right" in tail:
            return "turn right"
        if "stop" in tail or "halt" in tail or "stay" in tail:
            return "stop"
        return None
    return min(hits)[1]


# ---------------------------------------------- grounded caption (AgiBot schema)
#
# NVIDIA's own action-policy training captions (tasks.parquet of their LeRobot
# examples):
#   AgiBotWorld  "Pickup items in the supermarket | The robot is positioned in
#                 front of the fruit stand in the supermarket environment."
#   Bridge       "Put the pot to the left of the purple item."
# So the action model is conditioned on a full natural-language task WITH scene
# context -- `<task> | <scene>` -- not on motion primitives. "The camera turns
# left." is out of distribution in the LOW-information direction; a raw R2R
# instruction in the other (multi-clause, whole-episode, no scene half).
#
# The reasoner is asked for two LABELLED lines and we join them into the schema
# ourselves: a bare "|" separator was followed 1 time in 10 (the model mostly
# emitted the action alone); labelled fields are followed far more reliably,
# and the schema is for the ACTION MODEL's input, not something the reasoner
# must emit literally.
CAPTION_PROMPT = (
    "This is the overall task that the agent is trying to complete: \"{task}\"\n"
    "Decide the robot's immediate next action and describe its current surroundings. "
    "The immediate action must be ONE short step the robot can complete in about one "
    "second -- such as turning toward a doorway or moving along a corridor -- not the "
    "whole task. If the overall task is already complete, the action is \"Stop\".\n"
    "Give your final answer as exactly two lines:\n"
    "Action: <the immediate action>\n"
    "Scene: <one sentence: where the robot is right now and what is around it>\n"
    "For example:\n"
    "Action: Pickup items in the supermarket\n"
    "Scene: The robot is positioned in front of the fruit stand in the supermarket environment."
)

_ACT_RE = re.compile(r"action\s*[:\-]\s*(.+)", re.I)
_SCN_RE = re.compile(r"scene\s*[:\-]\s*(.+)", re.I)
_THINK_DECISION = re.compile(
    r"(?:immediate|next)\s+action\s+(?:is|should be|would be)\s+(?:to\s+)?[\"']?([^.\n\"']{4,80})", re.I)


def caption_prompt(task, think=True):
    text = f"{EMBODIMENT}\n{CAPTION_PROMPT.format(task=task.strip())}"
    return text + THINK_FORMAT if think else text


def _clean(x):
    return x.strip().strip('"').strip("'").strip().rstrip(".") if x else ""


def parse_caption(answer):
    """-> (caption in `<task> | <scene>` schema, is_stop, how).

    how in {"schema", "action_only", "think_tail", "unparsed"} -- logged so the
    compliance rate is measured, not guessed.
    """
    closed = "</think>" in answer
    tail = answer.split("</think>")[-1] if closed else ""
    m = _ACT_RE.search(tail); act = _clean(m.group(1)) if m else ""
    m = _SCN_RE.search(tail); scn = _clean(m.group(1)) if m else ""
    how = "schema" if (act and scn) else ("action_only" if act else "")
    if not act and closed:
        lines = [l for l in tail.splitlines() if l.strip()]
        if lines and "|" in lines[0]:
            a, _, b = lines[0].partition("|"); act, scn, how = _clean(a), _clean(b), "schema"
        elif lines:
            act, how = _clean(lines[0])[:120], "action_only"
    if not act and not closed:
        # thought hit the token cap (usually a repetition loop); it has normally
        # already stated the decision -- take the LAST such statement
        hits = _THINK_DECISION.findall(answer)
        if hits:
            act, how = _clean(hits[-1]), "think_tail"
    if not act:
        return None, False, "unparsed"
    is_stop = act.lower().startswith(("stop", "remain", "stay", "halt"))
    if is_stop:
        act = "Stop"
    return (f"{act} | {scn}" if scn else act), is_stop, how


# -------------------------------------------------------------------- action
AV_ACTION_PREFIX = "You are an autonomous vehicle planning system."          # VERBATIM

# The captions that scored 95% turn-direction agreement at guidance 5-7.5 in the
# controllability sweep. These are the ACTION model's vocabulary; the reasoner's
# command is translated into them, never passed through raw.
ACTION_CAPTION = {
    "move forward": "The camera moves forward.",
    "turn left":    "The camera turns left.",
    "turn right":   "The camera turns right.",
    "stop":         "The camera remains stationary.",
}


def action_prompt(command, framed=True):
    cap = ACTION_CAPTION[command]
    return f"{AV_ACTION_PREFIX} {cap}" if framed else cap
