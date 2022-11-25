import os
import unittest
import gym
import numpy as np

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(WORKING_DIR)

import sys

sys.path.insert(1, os.path.join(WORKING_DIR, "QLearning_Agent"))
from QLearning_Agent.QQL_learner_trainer import*

class TestModels(unittest.TestCase):
    def test_QLearning(self):
        # Test QLearning
        envtest = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
        QL = ClassicalLearner(envtest, env_type = 'global')
        QL.train()

if __name__ == "__main__":
    unittest.main()