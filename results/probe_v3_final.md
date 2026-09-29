# Reasoner navigation probe

160 decision points, 40 trajectories, val_unseen, video ctx 120 frames @ 15 FPS, horizon 16 frames

majority-class baseline: **0.544**

| variant | accuracy | parse | fwd | left | right | stop | s/call |
|---|---|---|---|---|---|---|---|
| hier_overall_nothink | **0.631** | 1.00 | 0.86 | 0.71 | 0.62 | 0.10 | 0.2 |

## Confusion (expert -> predicted)

**hier_overall_nothink**

```
move forward->move forward       75
move forward->turn left          6
move forward->turn right         6
stop->move forward               31
stop->stop                       4
stop->turn left                  2
stop->turn right                 3
turn left->move forward          3
turn left->stop                  1
turn left->turn left             12
turn left->turn right            1
turn right->move forward         6
turn right->turn right           10
```

## Sample transcripts

### hier_overall_nothink — 2azQ1b91cZZ/traj1039 t=99 expert=move forward pred=move forward

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Leave the bedroom and take a right in the hallway Go down the hall and enter the bathroom first on your left. Stop in the doorway to the bathroom."
What should be the next action of the agent?
Choose the robot's next action. Answer with exactly one of: move forward, turn left, turn right, stop.
```
**answer**
```
move forward
```

### hier_overall_nothink — 2azQ1b91cZZ/traj1039 t=108 expert=move forward pred=move forward

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Leave the bedroom and take a right in the hallway Go down the hall and enter the bathroom first on your left. Stop in the doorway to the bathroom."
What should be the next action of the agent?
Choose the robot's next action. Answer with exactly one of: move forward, turn left, turn right, stop.
```
**answer**
```
move forward
```

### hier_overall_nothink — 2azQ1b91cZZ/traj1039 t=195 expert=move forward pred=move forward

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Leave the bedroom and take a right in the hallway Go down the hall and enter the bathroom first on your left. Stop in the doorway to the bathroom."
What should be the next action of the agent?
Choose the robot's next action. Answer with exactly one of: move forward, turn left, turn right, stop.
```
**answer**
```
move forward
```

### hier_overall_nothink — 2azQ1b91cZZ/traj1039 t=225 expert=stop pred=move forward

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Leave the bedroom and take a right in the hallway Go down the hall and enter the bathroom first on your left. Stop in the doorway to the bathroom."
What should be the next action of the agent?
Choose the robot's next action. Answer with exactly one of: move forward, turn left, turn right, stop.
```
**answer**
```
move forward
```

### hier_overall_nothink — 8194nk5LbLH/traj1141 t=6 expert=turn left pred=turn left

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Go right around the counter and turn left. Continue until you are at the end of the second orange couch."
What should be the next action of the agent?
Choose the robot's next action. Answer with exactly one of: move forward, turn left, turn right, stop.
```
**answer**
```
turn left
```

### hier_overall_nothink — 8194nk5LbLH/traj1141 t=34 expert=move forward pred=move forward

**prompt**
```
You are a mobile robot navigating inside a building. The video depicts the observation from the robot's own forward-facing camera at eye level, ending at the present moment.
This is the overall task that the agent is trying to complete: "Go right around the counter and turn left. Continue until you are at the end of the second orange couch."
What should be the next action of the agent?
Choose the robot's next action. Answer with exactly one of: move forward, turn left, turn right, stop.
```
**answer**
```
move forward
```
