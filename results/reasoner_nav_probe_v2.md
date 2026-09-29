# Reasoner navigation probe

96 decision points, 24 trajectories, val_unseen, video ctx 16 frames @ 15 FPS, horizon 16 frames

majority-class baseline: **0.573**

| variant | accuracy | parse | fwd | left | right | stop | s/call |
|---|---|---|---|---|---|---|---|
| hier_overall | **0.375** | 0.93 | 0.51 | 0.14 | 0.50 | 0.08 | 6.2 |
| hier_full | **0.385** | 0.91 | 0.51 | 0.57 | 0.30 | 0.08 | 6.5 |

## Confusion (expert -> predicted)

**hier_overall**

```
move forward->UNPARSED           3
move forward->move forward       28
move forward->stop               4
move forward->turn left          5
move forward->turn right         15
stop->UNPARSED                   1
stop->move forward               11
stop->stop                       2
stop->turn left                  3
stop->turn right                 7
turn left->UNPARSED              1
turn left->move forward          3
turn left->turn left             1
turn left->turn right            2
turn right->UNPARSED             2
turn right->move forward         3
turn right->turn right           5
```

**hier_full**

```
move forward->UNPARSED           3
move forward->move forward       28
move forward->stop               8
move forward->turn left          3
move forward->turn right         13
stop->UNPARSED                   3
stop->move forward               8
stop->stop                       2
stop->turn left                  5
stop->turn right                 6
turn left->UNPARSED              1
turn left->move forward          1
turn left->turn left             4
turn left->turn right            1
turn right->UNPARSED             2
turn right->move forward         5
turn right->turn right           3
```

## Sample transcripts

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
Okay, let's break this down. The agent is supposed to leave the bedroom, take a right in the hallway, go down the hall, enter the bathroom first on the left, and stop in the doorway. The current state is that the camera has moved from the bedroom into the hallway through an archway. Now, the next step would be to turn right in the hallway. Since the instruction says to take a right in the hallway, the agent needs to adjust their direction. So the next action should be turning right to follow the hallway's direction. After that, they'll need to go down the hall and enter the bathroom. But since the current position is in the hallway after moving through the archway, the immediate next action is turning right.
</think>

turn right
```

### hier_overall — 2azQ1b91cZZ/traj1039 t=108 expert=move forward pred=turn right

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
Okay, let's break this down. The agent's task is to leave the bedroom, take a right in the hallway, go down the hall to enter the bathroom first on the left, and stop in the doorway.

Looking at the video, the current state shows the camera is moving forward through the bedroom towards the doorway. The next action should be part of navigating to the bathroom. Since the instruction says to take a right in the hallway, the agent needs to turn right upon exiting the bedroom. The current movement is forward into the hallway, so the next logical step after moving forward would be to turn right to follow the hallway's direction. Then, after turning right, the agent would proceed down the hall towards the bathroom. So the immediate next action after moving forward would be turning right to adjust direction towards the hallway's right path.
</think>

turn right
```

### hier_overall — 2azQ1b91cZZ/traj1039 t=195 expert=move forward pred=move forward

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
Okay, let's break this down. The agent's task is to leave the bedroom, take a right in the hallway, go down the hall, enter the bathroom first on the left, and stop in the doorway.

From the video, the robot has just entered the hallway after exiting the bedroom. The next step is to take a right turn in the hallway. The camera movement shows the robot turning right into the hallway. Now, after entering the hallway, the next action should be moving forward along the hallway towards the bathroom. The instruction says to go down the hall and enter the bathroom first on the left. So after turning right into the hallway, the robot needs to proceed down the hall. The next action would logically be moving forward to navigate down the hallway towards the bathroom entrance. The video does not show any obstacles or stops needed yet, so "move forward" makes sense here.
</think>

move forward
```

### hier_overall — 2azQ1b91cZZ/traj1039 t=225 expert=stop pred=turn right

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
Okay, let's break this down. The agent is supposed to leave the bedroom, take a right in the hallway, go down the hall, enter the bathroom first on the left, and stop in the doorway. The current state shows the robot has entered the hallway, moving forward. The next step after entering the hallway would be turning right as per the task. The video shows the hallway is empty, so there's nothing blocking the turn. The robot needs to turn right to follow the instruction. So the next action should be turning right.
</think>

turn right
```

### hier_overall — 8194nk5LbLH/traj1141 t=6 expert=turn left pred=turn right

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
Okay, let's break this down. The video is a kitchen with a counter, sink, stove, cabinets, a table, chairs, and a window. The robot's task is to go right around the counter and turn left until it reaches the end of the second orange couch.

First, the robot starts in the kitchen. The initial position is facing the counter. The task involves moving around the counter. Since the instruction says "go right around the counter," the robot needs to navigate around it. Right around the counter would mean moving to the right side of the counter. But the robot is already facing the counter, so moving right might involve turning right to go around the counter's right side.

After going right around the counter, the next part is to turn left. So after navigating around the counter, the robot should turn left. Then, continue until reaching the end of the second orange couch. The second orange couch isn't shown in the video, so maybe it's in another area beyond the kitchen. The robot would need to move forward after turning left.

So the sequence would be: move right around the counter, then turn left. Since the robot's current action is starting, the first step is moving right. But wait, the robot is facing the counter. To go right around it, maybe it needs to turn right first? Or move right while facing the counter? Hmm. If the robot is facing the counter, moving right would take it towards the sink and stove. To go around the counter, perhaps it needs to move to the right side of the c
```

### hier_overall — 8194nk5LbLH/traj1141 t=34 expert=move forward pred=turn left

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
Okay, let me try to figure out the next action here. The video is a modern office or lounge with a conference table, couches, and a hallway. The robot's task is to go right around the counter and turn left, then continue until it's at the end of the second orange couch.

The current state says the robot is moving forward, passing the conference table and heading towards the hallway behind the couches. The hallway leads to other areas, and there's a door at the far end. The robot needs to navigate around the counter, which might be near the couches. Since the instruction mentions turning left after going right around the counter, the robot probably just passed the counter and now needs to turn left. But wait, the video shows the robot is moving forward towards the hallway. If the hallway is behind the couches, maybe the robot is approaching the end of the second orange couch. The next action after moving forward would be to turn left as per the task. But the robot might still need to reach the end of the couch first. However, the instruction says "continue until you are at the end of the second orange couch." So maybe the robot is still moving towards that end. Wait, the current action impact says the robot is moving forward, passing the conference table and heading towards the hallway. The next step would be to turn left once it's past the counter. But the problem is, the robot's current action is moving forward. The task requires going right around the counter and turning le
```

### hier_full — 2azQ1b91cZZ/traj1039 t=99 expert=move forward pred=turn right

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Leave the bedroom and take a right in the hallway Go down the hall and enter the bathroom first on your left. Stop in the doorway to the bathroom."
In the video, the agent is trying to follow the instruction (a single step out of many to complete the overall task): "Leave the bedroom and take a right in the hallway Go down the hall and enter the bathroom first on your left."
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
Okay, let me try to figure out the next action the agent should take. The overall task is to leave the bedroom, take a right in the hallway, go down the hall, enter the bathroom first on the left, and stop in the doorway. The current instruction is the first part of that task.

The video shows the camera starts in the bedroom, pans right towards the hallway, and the wooden door is on the right. The agent needs to leave the bedroom. So the first action would be to exit the bedroom. But the instruction says to take a right in the hallway. Wait, the hallway is beyond the bedroom. So maybe the agent is already in the hallway after leaving the bedroom. The next step is to go down the hall. Since the camera is panning right, maybe the agent needs to turn right to follow the hallway. But the instruction says "take a right in the hallway," so perhaps the agent is entering the hallway from the bedroom. The wooden door is on the right, so taking a right would mean turning right into the hallway. But the video shows the camera is moving forward from the bedroom, panning right towards the hallway. So maybe the agent is moving forward and turning right. But the next action after leaving the bedroom would be to take a right in the hallway. So the next action should be "turn right" to enter the hallway. Then, after that, go down the hall. But the user is asking for the next action. Since the agent has just left the bedroom, the next step is to take a right in the hallway. So the action woul
```

### hier_full — 2azQ1b91cZZ/traj1039 t=108 expert=move forward pred=move forward

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Leave the bedroom and take a right in the hallway Go down the hall and enter the bathroom first on your left. Stop in the doorway to the bathroom."
In the video, the agent is trying to follow the instruction (a single step out of many to complete the overall task): "Leave the bedroom and take a right in the hallway Go down the hall and enter the bathroom first on your left."
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
Okay, let me try to figure out the next action the agent should take. The overall task is to leave the bedroom, take a right in the hallway, go down the hall, enter the bathroom first on the left, and stop in the doorway. The current instruction is a bit broken down but the main steps are leaving the bedroom, navigating the hallway, and entering the bathroom.

Looking at the video, the agent starts in the bedroom. The camera pans right as they move towards the hallway. The wooden door is slightly ajar, and the agent moves past it into the hallway. The hallway has a tiled floor, and the agent is supposed to take a right turn. The video shows the camera moves forward and pans right, indicating the agent is turning right into the hallway. Then, as they move further in, the camera continues panning right, suggesting they're heading towards the bathroom, which is on the left side when entering it.

Wait, the instruction says to enter the bathroom first on the left. So after taking a right in the hallway, the agent needs to go down the hall and then turn left into the bathroom. The current action shown is moving into the hallway after leaving the bedroom. The next step would be to actually turn right in the hallway to follow the path towards the bathroom. But the video shows the camera pans right as they move forward into the hallway. Maybe they've already taken the right turn and are now moving down the hall. The next action would be to go down the hall towards the bathroom. Once 
```

### hier_full — 2azQ1b91cZZ/traj1039 t=195 expert=move forward pred=stop

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Leave the bedroom and take a right in the hallway Go down the hall and enter the bathroom first on your left. Stop in the doorway to the bathroom."
In the video, the agent is trying to follow the instruction (a single step out of many to complete the overall task): "Stop in the doorway to the bathroom."
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
Okay, let's break this down. The agent is trying to follow the instruction "Stop in the doorway to the bathroom." The current environment is a hallway with the robot moving forward. The camera is panning right as the robot approaches the bathroom doorway.

So, the robot is moving down the hallway towards the bathroom. The instruction is to stop in the doorway. If the robot is still moving towards the bathroom, the next action would be to continue moving forward until it reaches the doorway. Once there, it needs to stop. But since the instruction is to stop in the doorway, the robot might need to adjust its position to be properly stopped there. However, the current action is moving forward. If the robot hasn't reached the doorway yet, the next step would still be moving forward. If it's already there, then stop. But based on the video, the camera is approaching the bathroom, so maybe the robot is still in motion. Therefore, the next action should be to stop once it's in position. Wait, but the instruction says "stop in the doorway," which implies that after moving forward, the robot needs to stop. Since the current action is moving forward, the next logical step after reaching the doorway would be to stop. So the answer should be "stop."
</think>

stop
```

### hier_full — 2azQ1b91cZZ/traj1039 t=225 expert=stop pred=stop

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Leave the bedroom and take a right in the hallway Go down the hall and enter the bathroom first on your left. Stop in the doorway to the bathroom."
In the video, the agent is trying to follow the instruction (a single step out of many to complete the overall task): "Stop in the doorway to the bathroom."
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
Okay, let's break this down. The agent's current instruction is to "Stop in the doorway to the bathroom." The video is a hallway with the robot moving forward. The camera is moving towards the bathroom door at the end of the hallway.

First, the robot is in the hallway, moving forward. The next step after moving forward would be to reach the bathroom doorway. Since the instruction is to stop in the doorway, the robot needs to position itself there. But if the robot is still moving forward, maybe it hasn't reached the door yet. So the next action should be to continue moving forward until it's in position to stop. Alternatively, if it's already at the doorway, stopping would be next. But the problem says if unsuccessful, predict the next action. The current action is moving forward. If the agent hasn't stopped yet, the next action should be to stop once in the doorway. But maybe the robot hasn't arrived yet. Wait, the instruction is to stop in the doorway. So the robot needs to move forward to reach the doorway and then stop. Since the camera is moving forward, the next action after moving forward would be to stop. But the user's instruction is the current step, which is "Stop in the doorway to the bathroom." If the robot is still moving towards the doorway, the next action would be to continue moving forward. But maybe it's already there. Hmm. The video shows the camera moves forward, approaching the bathroom door. So maybe the robot is still moving towards it. If the agent h
```

### hier_full — 8194nk5LbLH/traj1141 t=6 expert=turn left pred=turn left

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Go right around the counter and turn left. Continue until you are at the end of the second orange couch."
In the video, the agent is trying to follow the instruction (a single step out of many to complete the overall task): "Go right around the counter and turn left."
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
Okay, let's break this down. The agent is supposed to go right around the counter and then turn left. The current instruction is just "Go right around the counter and turn left." The video has a kitchen with a countertop, cabinets, a sink, and a dining area with an orange couch. The robot is moving around, probably navigating through the space.

First, the robot needs to go right around the counter. Since the counter is on the right side of the kitchen, moving right around it would mean navigating along the counter's edge. After that, the next step is to turn left. But where is the second orange couch? The video shows an orange chair at the dining table, but the task refers to a "second orange couch." Maybe there's another couch further along the counter or in another part of the kitchen.

Assuming the robot has already navigated around the counter, the next action would be to turn left to head towards the second orange couch. The previous action was moving right around the counter, so now turning left makes sense to follow the path towards the couch. The agent might need to adjust their direction to reach the specified location. Since the instruction is a step in a larger task, the next action should logically be the turn left as per the given instruction. If the robot successfully went right around the counter, the next step is turning left. If not, maybe they need to adjust their path, but based on the info, the next action is likely "turn left."
</think>

turn left
```

### hier_full — 8194nk5LbLH/traj1141 t=34 expert=move forward pred=turn right

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Go right around the counter and turn left. Continue until you are at the end of the second orange couch."
In the video, the agent is trying to follow the instruction (a single step out of many to complete the overall task): "Go right around the counter and turn left."
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
Okay, let's break this down. The agent is supposed to "Go right around the counter and turn left." The video has a counter with a blue and white checkered tablecloth. The robot is moving forward through the space.

First, the instruction says to go right around the counter. So the robot needs to navigate around the counter. If the robot is moving forward, maybe it's approaching the counter. To go right around it, the robot would need to turn right to avoid the counter on the left side. But the current action is just moving forward. So the next step after moving forward would be to turn right to go around the counter. Then, after that, turn left as per the instruction. Since the current step is "Go right around the counter and turn left," and the agent is moving forward, the next action should be turning right to maneuver around the counter. Then, after turning right, they might need to turn left to continue. But the immediate next action after moving forward would be turning right. So the predicted action is "turn right."
</think>

turn right
```
