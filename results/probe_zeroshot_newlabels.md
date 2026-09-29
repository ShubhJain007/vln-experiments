# Reasoner navigation probe

160 decision points, 40 trajectories, val_unseen, video ctx 16 frames @ 15 FPS, horizon 16 frames

majority-class baseline: **0.544**

| variant | accuracy | parse | fwd | left | right | stop | s/call |
|---|---|---|---|---|---|---|---|
| hier_overall_nothink | **0.325** | 0.96 | 0.33 | 0.35 | 0.62 | 0.17 | 0.3 |
| hier_overall | **0.331** | 0.93 | 0.45 | 0.24 | 0.25 | 0.15 | 7.9 |

## Confusion (expert -> predicted)

**hier_overall_nothink**

```
move forward->UNPARSED           4
move forward->move forward       29
move forward->stop               10
move forward->turn left          13
move forward->turn right         31
stop->UNPARSED                   2
stop->move forward               10
stop->stop                       7
stop->turn left                  6
stop->turn right                 15
turn left->move forward          3
turn left->stop                  3
turn left->turn left             6
turn left->turn right            5
turn right->move forward         3
turn right->stop                 2
turn right->turn left            1
turn right->turn right           10
```

**hier_overall**

```
move forward->UNPARSED           5
move forward->move forward       39
move forward->stop               12
move forward->turn left          13
move forward->turn right         18
stop->UNPARSED                   2
stop->move forward               15
stop->stop                       6
stop->turn left                  5
stop->turn right                 12
turn left->UNPARSED              1
turn left->move forward          8
turn left->stop                  1
turn left->turn left             4
turn left->turn right            3
turn right->UNPARSED             3
turn right->move forward         6
turn right->stop                 1
turn right->turn left            2
turn right->turn right           4
```

## Sample transcripts

### hier_overall_nothink — 2azQ1b91cZZ/traj1039 t=99 expert=move forward pred=turn right

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Leave the bedroom and take a right in the hallway Go down the hall and enter the bathroom first on your left. Stop in the doorway to the bathroom."
What should be the next action of the agent?
Choose the robot's next action. Answer with exactly one of: move forward, turn left, turn right, stop.
```
**answer**
```
turn right
```

### hier_overall_nothink — 2azQ1b91cZZ/traj1039 t=108 expert=move forward pred=turn right

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Leave the bedroom and take a right in the hallway Go down the hall and enter the bathroom first on your left. Stop in the doorway to the bathroom."
What should be the next action of the agent?
Choose the robot's next action. Answer with exactly one of: move forward, turn left, turn right, stop.
```
**answer**
```
turn right
```

### hier_overall_nothink — 2azQ1b91cZZ/traj1039 t=195 expert=move forward pred=turn right

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Leave the bedroom and take a right in the hallway Go down the hall and enter the bathroom first on your left. Stop in the doorway to the bathroom."
What should be the next action of the agent?
Choose the robot's next action. Answer with exactly one of: move forward, turn left, turn right, stop.
```
**answer**
```
turn right
```

### hier_overall_nothink — 2azQ1b91cZZ/traj1039 t=225 expert=stop pred=turn right

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Leave the bedroom and take a right in the hallway Go down the hall and enter the bathroom first on your left. Stop in the doorway to the bathroom."
What should be the next action of the agent?
Choose the robot's next action. Answer with exactly one of: move forward, turn left, turn right, stop.
```
**answer**
```
turn right
```

### hier_overall_nothink — 8194nk5LbLH/traj1141 t=6 expert=turn left pred=turn right

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Go right around the counter and turn left. Continue until you are at the end of the second orange couch."
What should be the next action of the agent?
Choose the robot's next action. Answer with exactly one of: move forward, turn left, turn right, stop.
```
**answer**
```
turn right
```

### hier_overall_nothink — 8194nk5LbLH/traj1141 t=34 expert=move forward pred=turn right

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Go right around the counter and turn left. Continue until you are at the end of the second orange couch."
What should be the next action of the agent?
Choose the robot's next action. Answer with exactly one of: move forward, turn left, turn right, stop.
```
**answer**
```
turn right
```

### hier_overall — 2azQ1b91cZZ/traj1039 t=99 expert=move forward pred=turn right

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Leave the bedroom and take a right in the hallway Go down the hall and enter the bathroom first on your left. Stop in the doorway to the bathroom."
What should be the next action of the agent?
Choose the robot's next action. Answer with exactly one of: move forward, turn left, turn right, stop.
Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.
```
**answer**
```
Okay, let's break this down. The agent's current task is to leave the bedroom and take a right in the hallway, then go down the hall to enter the bathroom first on the left. They need to stop in the doorway of the bathroom.

Looking at the video, the robot is in the bedroom. The camera starts focused on the bed, then pans right towards the hallway. The hallway has an archway leading to the bathroom. The robot needs to exit the bedroom, so the first action would be moving towards the hallway. Since the camera is panning right, that suggests the robot is turning right to head down the hallway. The next action after panning would likely be moving forward down the hallway. But wait, the instruction says to take a right in the hallway. So maybe the robot needs to turn right upon entering the hallway. However, the current action impact isn't shown, so I need to assume the robot hasn't started moving yet. The next action would be moving towards the hallway. But the exact direction? Since the camera is panning right, maybe the robot is turning right. But the task says "take a right in the hallway," so after entering the hallway, the next step is to turn right. But the initial movement is exiting the bedroom. So first, the robot needs to move from the bedroom into the hallway. The initial action would be moving right (since the camera is panning right to follow the hallway). So the next action is probably "move right" to enter the hallway. Then, once in the hallway, take a right. But 
```

### hier_overall — 2azQ1b91cZZ/traj1039 t=108 expert=move forward pred=stop

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Leave the bedroom and take a right in the hallway Go down the hall and enter the bathroom first on your left. Stop in the doorway to the bathroom."
What should be the next action of the agent?
Choose the robot's next action. Answer with exactly one of: move forward, turn left, turn right, stop.
Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.
```
**answer**
```
Okay, let's break this down. The agent is supposed to leave the bedroom, take a right in the hallway, go down the hall, enter the bathroom first on the left, and stop in the bathroom doorway.

The current state is that the robot has just entered the bathroom. The last action shown is that the robot's camera stopped in the doorway. The task says to stop in the doorway, so maybe the next step is to actually enter the bathroom. But wait, the video shows the camera stopped in the doorway. So maybe the robot needs to move further into the bathroom now. But the instruction says "enter the bathroom first on your left" and then stop in the doorway. Hmm, maybe the robot is already at the doorway and needs to stop. But the next action after entering would be to move into the bathroom. Wait, the video shows "The camera stops in the doorway, indicating the robot has entered the bathroom." So the robot is now in the doorway. The next action should be to move forward into the bathroom. But according to the task, after entering the bathroom, the robot should stop in the doorway. Wait, maybe the robot has already entered and stopped. But the problem is to predict the next action after the current state. The current state is that the camera stopped in the doorway. So the next action would be to move forward into the bathroom. But the task says to enter the bathroom first on the left and stop in the doorway. So maybe the robot has already entered the bathroom and is now stopping. But the actio
```

### hier_overall — 2azQ1b91cZZ/traj1039 t=195 expert=move forward pred=turn right

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Leave the bedroom and take a right in the hallway Go down the hall and enter the bathroom first on your left. Stop in the doorway to the bathroom."
What should be the next action of the agent?
Choose the robot's next action. Answer with exactly one of: move forward, turn left, turn right, stop.
Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.
```
**answer**
```
Okay, let me try to figure out the next action here. The user provided  the video and the robot's current task. The task is to leave the bedroom, take a right in the hallway, go down the hall, enter the bathroom first on the left, and stop in the doorway.

Looking at the current state, the robot is in a hallway with archways and a light switch on the left. The camera is moving forward, which suggests the robot is already moving down the hallway. The next step after moving forward would be to turn right as per the task instruction. Since the instruction says to take a right in the hallway, the robot needs to make that turn. The bathroom is to the left after turning right, so after turning, the robot would then proceed down the hall to the bathroom. But since the current action is moving forward, the next logical step is to turn right to follow the hallway's direction. The robot hasn't reached the bathroom yet, so stopping isn't the next action yet. The correct next action should be turning right to navigate the hallway as instructed.
</think>

turn right
```

### hier_overall — 2azQ1b91cZZ/traj1039 t=225 expert=stop pred=turn left

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Leave the bedroom and take a right in the hallway Go down the hall and enter the bathroom first on your left. Stop in the doorway to the bathroom."
What should be the next action of the agent?
Choose the robot's next action. Answer with exactly one of: move forward, turn left, turn right, stop.
Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.
```
**answer**
```
Okay, let's break this down. The agent is supposed to leave the bedroom, take a right in the hallway, go down the hall, enter the bathroom first on the left, and stop in the bathroom doorway. The current environment is a hallway with a static camera view moving forward.

The robot's next action depends on where it is in the hallway. Since the task is to go down the hall and enter the bathroom first on the left, the robot probably needs to turn left into the bathroom. The previous action might have been moving forward along the hallway. Now, to enter the bathroom, turning left would be the logical next step. The user instructions specify entering the bathroom first on the left, so the robot should turn left to do that. After turning, it would then move forward into the bathroom. But since the next immediate action is turning left, that's the correct prediction.
</think>

turn left
```

### hier_overall — 8194nk5LbLH/traj1141 t=6 expert=turn left pred=turn left

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Go right around the counter and turn left. Continue until you are at the end of the second orange couch."
What should be the next action of the agent?
Choose the robot's next action. Answer with exactly one of: move forward, turn left, turn right, stop.
Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.
```
**answer**
```
Okay, let's try to figure out the next action the robot should take. The overall task is to "Go right around the counter and turn left. Continue until you are at the end of the second orange couch." 

First, the video shows a kitchen with a counter on the right side. The robot needs to go right around the counter. Since the counter is on the right, moving right around it would mean navigating along the counter's right side. But the counter has a sink and appliances, so maybe the robot needs to go around the right side of the counter to avoid obstacles.

Then, after going right around the counter, the next step is to turn left. So after completing the right movement, the robot should turn left. The task continues until reaching the end of the second orange couch. The video shows a dining table with chairs, but the video does not show couches. Wait, maybe the "second orange couch" refers to a seating area or a sofa in the dining area. The dining table is partially visible on the left, but the task mentions the second orange couch, which might be further along.

So the current position after going right around the counter would be near the kitchen counter. The next action should be to turn left as per the instructions. The robot needs to turn left to proceed towards the end of the second orange couch. Since the previous action was moving right around the counter, the next logical step is to turn left to change direction towards the couch. Therefore, the next action should be "tu
```

### hier_overall — 8194nk5LbLH/traj1141 t=34 expert=move forward pred=turn right

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Go right around the counter and turn left. Continue until you are at the end of the second orange couch."
What should be the next action of the agent?
Choose the robot's next action. Answer with exactly one of: move forward, turn left, turn right, stop.
Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.
```
**answer**
```
Okay, let's break this down. The agent is supposed to "Go right around the counter and turn left. Continue until you are at the end of the second orange couch." The current state shows the robot is moving forward, passing the first orange couch, and approaching the second one.

Since the task is to go right around the counter first, but the robot is moving forward, maybe they need to adjust direction. The next action after passing the first couch would be to turn right to go around the counter as instructed. Then, after turning left, they need to reach the end of the second couch. But right now, the robot is still approaching the second couch. Wait, the instruction says to go right around the counter first. So maybe the robot needs to turn right to navigate around the counter before turning left again. Since the robot is moving forward towards the second couch, but hasn't gone around the counter yet, the next action should be turning right to go around the counter. Then, after that, turn left to continue towards the end of the second couch. But the current step is predicting the next immediate action. The robot is moving forward, but according to the task, they need to go right around the counter first. So maybe the next action is turning right to start that maneuver. So the answer would be "turn right"
</think>

turn right
```
