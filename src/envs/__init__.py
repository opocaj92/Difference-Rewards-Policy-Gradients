from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from .gridworld import MultiRover, PredatorPrey
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["multi_rover"] = partial(env_fn, env=MultiRover)
REGISTRY["predator_prey"] = partial(env_fn, env=PredatorPrey)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))