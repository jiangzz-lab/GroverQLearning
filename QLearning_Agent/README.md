# `qrllearner`
This folder contains the [source code](./QQL_learner_trainer.py) of the package `qrllearner` and some test notebooks.
## Classes
- `RL_Qlearning_trainer`: superclass that provides the general api for Q-learning; takes env and env_type (='global' or 'local') as the arguments of its constructor
- `GroverQlearner`: inherits the superclass and implements the Grover's algorithm
- `ClassicalLearner`: inherits the superclass and implements a classical search algorithm

## Major methods
- `set_hyperparams`: sets the hyperparameters used for training the Q-learning agent which takes the following as arguments
~~~python
hyperp = {'k': float,
          'alpha': float,
          'gamma': float,
          'eps': float,
          'max_epochs': int,
          'max_steps': int}
~~~
- `train`: train the agent and returns the steps, target_reached, and trajectories during the training. Those returned values are only meaningful if the env is the Frozenlake. For the sidewalk env, just ignore the output.

## Major variables
- `Q_values` (np.array): the Q-function in the reinforcement Q-learning.
- `state_dimension` (int): the dimension of the state space
- `action_dimension` (int): the dimension of the action space