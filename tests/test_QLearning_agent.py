import os
import unittest
import gym
import numpy as np

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(WORKING_DIR)

import sys

sys.path.append(WORKING_DIR+'\QLearning_Agent')
# print(sys.path)
from QQL_learner_trainer import*

class TestModels(unittest.TestCase):
    def test_QLearning(self):
        # Test QLearning using Classical Learner with the FrozenLake-v1 environment
        envtest = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
        QL = ClassicalLearner(envtest, env_type = 'global')
        hyperp = {'alpha': 0.1,
          'gamma': 0.99,
          'eps': 0.0,
          'max_epochs': 500,
          'max_steps': 15}
        QL.set_hyperparams(hyperp)
        steps_in_all_epochs,target_reached_in_all_epochs,_ = QL.train()
        # 7 steps to reach the goal
        assert steps_in_all_epochs[-1] == 7, "QLearning failed"
        assert target_reached_in_all_epochs[-1] == 1, "QLearning failed"

if __name__ == "__main__":
    unittest.main()