# `sidewalkdemo`
This folder contains the [source code](./SideWalkEnv.py) of the package `sidewalkdemo` and some test notebooks.
## Classes
- `side_walk_env`: superclass that provides the general api for the sidewalk env; takes the following keywords as arguments of its constructor
    - `nx` (int): length of the road in the x direction
    - `ny` (int): width of the road in the y direction
    - `upper_border` (int): upper border of the sidewalk
    - `lower_border` (int): lower border of the sidewalk
    - `p_obstacle` (float): probability of an obstacle appearing on the road
    - `p_litter` (float): probability of litter appearing on the road
- `side_walk_env_with_obstacle`: inherits the superclass with `p_litter=0`
- `side_walk_env_with_litter`: inherits the superclass with `p_obstacle=0`

## Major methods
- `generate_roadmap`: generates a random roadmap given the arguments in the constructor
- `reset`: resets the initial state of the agent; returns the current `state`, `current_position`, `reward`, `done` (boolean), and `info` (for debugging)
- `position_to_state`: converts the position of the agent on the road to its current state and returns that state
- `plot_roadmap`: plots the current roadmap
- `trajectory`: takes `Q_values` as its argument and returns the trajectory of the agent on a roadmap given that Q_value
- `plot_roadmap_with_trajectory`: takes `tasks` (str, "picking up litters" or "avoiding obstacles") and `trajectory` as its arguments and returns the roadmap with the trajectory of the agent
- `step` (subclass method): takes `action` as its argument and updates the agent by taking that action; returns the next `state`, `current_position`, `reward`, `done` (boolean), and `info` (for debugging)

## Major variables
- `roadmap` (np.array): roadmap of the road containing the obstacles (1), litter (2), or none (0)