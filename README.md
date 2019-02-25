# Test Environments for Zero-shot Transfer Learning

## Introduction
This repo contains the environments for zero-shot transfer learning
presented in AAMAS Extended Abstract [Zero Shot Transfer Learning for
Robot
Soccer](https://dl.acm.org/ft_gateway.cfm?id=3238075&ftid=1986331&dwn=1&CFID=105147742&CFTOKEN=a948d5965970ca19-D8519914-DEED-3FFD-E94AE98E2B143805).

## How to use this
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


