# Test Environments for Reinforcement Learning with Tensor State and Action Spaces

## Introduction
This repo contains the test environments for IJCAI Submission:
Reinforcement Learning with Tensor State and Action Spaces

## Brief Introduction to Environment

### _passing_ Domain
*passing* is a grid world environment with two teams. One is composed
of agents while the other is controlled by the environment, which is
the opponent team. In
*passing*, we only have the control over passing action of the agent with the ball and its goal is to pass the ball to a
teammate as soon as possible. Other agents and the opponents are given random actions in
the environment.

For notice, in the **Result** column, **terminate** means the episode ends while
**proceed** means the episode continues.


The following table describes the details of the transition and reward
functions:

| Condition                                          | Result                       | Reward |
|----------------------------------------------------|:----------------------------:|:------:|
| pass outside the same column or the same row       | invalid pass, terminate      | -10    |
| pass to valid empty location or opponent locations | invalid pass, terminate      | -10    |
| pass to the teammate if an opponent is in between  | unsuccessful pass, terminate | -10    |
| hold the ball and haven't passed                   | proceed                      | -1     |
| pass to a teammate validly without interception    | successful pass, terminate   | 10     |

### *take-the-treasure* Domain
*take-the-treasure* is a grid world environment with two teams. One is
composed of agents while the opponent team is controlled by the
environment.

Here we have two tables to describe the details. The first table
describes the conditions, outcomes and rewards for the overall team
reward and task. The second table describes the outcomes and rewards
for a single agent in the team.


| Condition                                                         | Result                   | Reward |
|-------------------------------------------------------------------|:------------------------:|:------:|
| intercepted by opponent                                           | fail, terminate          | -1     |
| an opponent is right next to the agent with the ball (touch)      | fail, terminate          | -1     |
| pass to a teammate in the same column or row without interception | successful pass, proceed | 0      |
| agent with the ball moves without being touched                   | proceed                  | 0      |

| Condition                                          | Result                              | Reward |
|----------------------------------------------------|:-----------------------------------:|:------:|
| pass outside the same column or the same row       | invalid pass, terminate             | -1     |
| pass to valid empty location or opponent locations | invalid pass, terminate             | -1     |
| pass to the teammate if an opponent is in between  | unsuccessful pass, terminate        | -1     |
| move with actions out side of 4-connected grid     | invalid move, stay still, proceed   | -1     |
| pass action when the agent doesn't have the ball   | invalid action, stay still, proceed | -1     |
| pass to a teammate validly without interception    | proceed                             | 0      |

### *break-out* Domain
*break-out* is a clone of the Atari Breakout
game, which
allows for different screen heights and widths. Changing the height
affects the number of rows of bricks, and spacing between the paddle
of bricks. Changing the width affects the number of bricks in each
row.


## How to use this environment
### Install the environment
These environments are actually openai-gym style envs, and it does
depend on several gym features such as spaces objects from the gym. To
use this, first of all in a python virtual environment (Should be at
least Python 3.6) do:
``` shell
pip install -e .
```

### How to import the environment?
Here it is a little bit different from openai gym's
initialization. Here is how you import an environment
``` python
from tensor_state_action_env import grid_soccer_passing

env = grid_soccer_passing.GridSoccerPassing()
```

And the environment will be initialized with default parameters.

If you want to initialize, say, the GridSoccerPassing environment in a
field of 20x20, here is what you can do:
``` python
from tensor_state_action_env import grid_soccer_passing

env = grid_soccer_passing.GridSoccerPassing(length=20, height=20)
```

I'll add documentation later to the specific parameters for each
environments, and you can initialize tasks with different environment
parameters according to your need.


