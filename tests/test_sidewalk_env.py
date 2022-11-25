import os
import unittest
import gym
import numpy as np

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(WORKING_DIR)

import sys

sys.path.append(WORKING_DIR+'\sidewalk_env')
# print(sys.path)
from SideWalkEnv import*

class TestModels(unittest.TestCase):
    def test_sidewalkenv(self):
        # Test the sidewalk environment
        env = side_walk_env(p_obstacle=0.1,p_litter=0.1)
        env1 = side_walk_env_with_obstacle(p_obstacle = 0.1)
        env2 = side_walk_env_with_litter(p_litter = 0.1)
        assert env.nx == 50, "wrong road length"
        assert env.ny == 15, "wrong road width"
        assert env1.nx == 50, "wrong road length"
        assert env1.ny == 15, "wrong road width"
        assert env2.nx == 50, "wrong road length"
        assert env2.ny == 15, "wrong road width"
        assert env.action_space.n == 4, "wrong number of actions"
        assert env1.action_space.n == 4, "wrong number of actions"
        assert env1.observation_space.n == 16, "wrong observation space"
        assert env2.action_space.n == 4, "wrong number of actions"
        assert env2.observation_space.n == 16, "wrong observation space"

if __name__ == "__main__":
    unittest.main()