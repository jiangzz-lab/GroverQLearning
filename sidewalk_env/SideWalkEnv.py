from matplotlib import pyplot as plt
import numpy as np
import operator
from functools import reduce

class Observation_space:
    '''
    Returns the state space
    '''
    def __init__(self, state_space):
        self.state_space = state_space
        self.n = state_space.size

class Action_space:
    '''
    Returns the action space
    '''
    def __init__(self, action_space):
        self.action_space = action_space
        self.n = action_space.size

class side_walk_env:
    '''
    This class defines a sidewalk environment containing two main tasks for the agent to learn
    1. Avoid obstacles
    2. Collect litter

    Arguments:
    nx: length of the road in the x direction
    ny: width of the road in the y direction
    upper_border: upper border of the sidewalk
    lower_border: lower border of the sidewalk
    (We assume that the sidewalk is centered at the y axis and we have 0<lower_border<upper_border<ny-1)
    p_obstacle: probability of an obstacle appearing in the road
    p_litter: probability of litter appearing in the road

    Parameters:
    state: current local state of the agent
    action_space: action space of the agent containing four different actions: 0: left, 1: right, 2: up, 3: down
    reward: reward of the agent after taking an action
    done: boolean variable indicating whether the agent has reached the end of the road
    info: for debugging
    p_obstacle: probability of an obstacle appearing in the road
    p_litter: probability of litter appearing in the road
    position_obstacles: position of the obstacles in the road
    position_litter: position of the litter in the road
    nx: length of the road in the x direction
    ny: width of the road in the y direction
    roadmap: roadmap of the road containing the obstacles and litter
    upper_border: upper border of the sidewalk
    lower_border: lower border of the sidewalk
    current_position: current position of the agent in the road
    '''
    def __init__(self,nx,ny,upper_border,lower_border,p_obstacle,p_litter):
        self.state = 0
        self.action_space = Action_space(np.array([0, 1, 2, 3]))
        self.reward = 0
        self.done = False
        self.info = {} # for debugging
        self.p_obstacle = p_obstacle
        self.p_litter = p_litter
        self.position_obstacles = None
        self.position_litter = None
        self.nx = nx
        self.ny = ny
        self.roadmap = self.generate_roadmap()
        self.upper_border = upper_border
        self.lower_border = lower_border
        self.current_position = self._init_position()

    def _init_position(self):
        '''
        Returns a random initial position of the agent
        '''
        y = np.random.randint(self.lower_border,self.upper_border)
        return np.array([0,y])

    def generate_roadmap(self):
        '''
        Generates a roadmap of the road containing the obstacles and litter, where 0 represents 
        an empty space, 1 represents an obstacle, and 2 represents litter
        '''
        roadmap = np.random.choice([0,1,2], (self.nx-1, self.ny),p=[1-self.p_obstacle-self.p_litter,self.p_obstacle,self.p_litter])
        # the first row of the roadmap is always empty
        roadmap = np.insert(roadmap, 0, 0, axis=0)
        return roadmap

    def update_env(self):
        '''
        Updates the environment after certain number of epochs
        '''
        self.roadmap = self.generate_roadmap()

    def reset(self):
        '''
        Resets the environment
        '''
        pass

    def position_to_state(self,current_position,position_objects):
        '''
        Returns the state of the agent given the current position and the position of the objects (obstacles or litter)
        '''
        # position_obstacles = [[objects_position[0][i], objects_position[1][i]] for i in range(len(objects_position[0]))]
        right = ([current_position[0]+1,current_position[1]] in position_objects)+0
        up = ([current_position[0],current_position[1]+1] in position_objects)+0
        left = ([current_position[0]-1,current_position[1]] in position_objects)+0
        down = ([current_position[0],current_position[1]-1] in position_objects)+0
        state = right + up*2 + left*4 + down*8
        return state

    def render(self):
        '''
        Renders the environment
        '''
        print('state: ', self.state)

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def plot_roadmap(self):
        '''
        Plots the roadmap of the road
        '''
        x = [[i] for i in range(self.nx)]
        road = [[i for i in range(self.ny)] for j in range(self.nx)]
        road_up = [[self.upper_border] for i in range(self.nx)]
        road_low = [[self.lower_border] for i in range(self.nx)]
        obstacle = np.array([np.where(self.roadmap == 1)[0], np.where(self.roadmap == 1)[1]])
        litter = np.array([np.where(self.roadmap == 2)[0], np.where(self.roadmap == 2)[1]])
        plt.figure(figsize=(15,4))
        plt.plot(x, road, linestyle='--', color = "0.7", zorder=10)
        plt.fill_between(reduce(operator.add, x), reduce(operator.add, road_low), reduce(operator.add, road_up), color = '#539caf', alpha = 0.2, zorder=20)
        plt.plot(np.array(x), (np.array(road_up)+np.array(road_low))/2, linestyle='--', color = "y", linewidth = 3, zorder=30)
        plt.scatter(obstacle[0], obstacle[1], marker='x', color = 'k', zorder=40)
        plt.scatter(litter[0], litter[1], marker='s', color='r', zorder=40)
        plt.scatter([0 for i in range(self.lower_border,self.upper_border+1,2)],[i for i in range(self.lower_border,self.upper_border+1,2)], marker='>',zorder=50)
        plt.title('Road Map')

    def trajectory(self):
        '''
        Returns the trajectory of the agent
        '''
        pass

    def plot_roadmap_with_trajectory(self,task,trajectory):
        '''
        Plots the roadmap of the road with the trajectory of the agent
        '''
        x = [[i] for i in range(self.nx)]
        road = [[i for i in range(self.ny)] for j in range(self.nx)]
        road_up = [[self.upper_border] for i in range(self.nx)]
        road_low = [[self.lower_border] for i in range(self.nx)]
        obstacle = np.array([np.where(self.roadmap == 1)[0], np.where(self.roadmap == 1)[1]])
        litter = np.array([np.where(self.roadmap == 2)[0], np.where(self.roadmap == 2)[1]])
        plt.figure(figsize=(15,4))
        plt.plot(x, road, linestyle='--', color = "0.7", zorder=10)
        plt.fill_between(reduce(operator.add, x), reduce(operator.add, road_low), reduce(operator.add, road_up), color = '#539caf', alpha = 0.2, zorder=20)
        plt.plot(np.array(x), (np.array(road_up)+np.array(road_low))/2, linestyle='--', color = "y", linewidth = 3, zorder=30)
        plt.scatter(obstacle[0], obstacle[1], marker='x', color = 'k', zorder=40)
        plt.scatter(litter[0], litter[1], marker='s', color='r', zorder=40)
        plt.scatter([0 for i in range(self.lower_border,self.upper_border+1,2)],[i for i in range(self.lower_border,self.upper_border+1,2)], marker='>',zorder=50)
        plt.plot(trajectory[0],trajectory[1], linestyle='-', zorder=40)
        plt.title('Road Map: ' + task)

class side_walk_env_with_obstacle(side_walk_env):
    '''
    Side walk environment with obstacles: arguments are the same as the side_walk_env class but with p_litter = 0

    Parameters:
    position_obstacles: list of the position of the obstacles
    observation_space: observation space of the agent: 16 possible states (four neighbors having an obstacle or not)
    '''
    def __init__(self,nx,ny,upper_border,lower_border,p_obstacle,p_litter=0):
        super().__init__(nx,ny,upper_border,lower_border,p_obstacle,p_litter)
        self.position_obstacles = [[np.where(self.roadmap == 1)[0][i], np.where(self.roadmap == 1)[1][i]] for i in range(len(np.where(self.roadmap == 1)[0]))]
        self.observation_space = Observation_space(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]))

    def step(self, action):
        '''
        Performs an action and returns the next state, the next position, the reward, a boolean indicating if the episode is finished, and the information
        '''
        current_position = self.current_position
        panelity = -15
        if current_position[0] == self.nx-1:
            self.done = True
            self.reward = 10
            return self.state, self.reward, self.current_position, self.done, self.info
        if action == 1: # move backward
            next_position = [current_position[0] - 1, current_position[1]]
            next_state = self.position_to_state(next_position,self.position_obstacles)
            if next_position in self.position_obstacles:
                self.reward = panelity
            else:
                self.reward = -6
        elif action == 0: # move forward
            next_position = [current_position[0] + 1, current_position[1]]
            next_state = self.position_to_state(next_position,self.position_obstacles)
            if next_position in self.position_obstacles:
                self.reward = panelity
            else:
                self.reward = 3
        elif action == 2: # move upward
            next_position = [current_position[0], current_position[1] + 1]
            next_state = self.position_to_state(next_position,self.position_obstacles)
            if next_position in self.position_obstacles:
                self.reward = panelity
            else:
                self.reward = 0
        elif action == 3: # move downward
            next_position = [current_position[0], current_position[1] - 1]
            next_state = self.position_to_state(next_position,self.position_obstacles)
            if next_position in self.position_obstacles:
                self.reward = panelity
            else:
                self.reward = 0
        else:
            raise ValueError('Invalid action')

        # self.reward = self.Reward[self.state, action]
        self.state = next_state
        self.current_position = next_position

        return self.state, self.reward, self.current_position, self.done, self.info

    def reset(self):
        '''
        Resets the environment
        '''
        self.current_position = self._init_position()
        self.state = self.position_to_state(self.current_position,self.position_obstacles)
        self.reward = 0
        self.done = False
        self.info = {} 
        return self.state, self.current_position, self.reward, self.done, self.info
    
    def trajectory(self, Q_values):
        '''
        Returns the trajectory of the agent given the Q_values
        '''
        self.update_env()
        self.reset()
        self.position_obstacles = [[np.where(self.roadmap == 1)[0][i], np.where(self.roadmap == 1)[1][i]] for i in range(len(np.where(self.roadmap == 1)[0]))]
        trajectory_x = []
        trajectory_y = []
        trajectory_x.append(self.current_position[0])
        trajectory_y.append(self.current_position[1])
        n_step = 0
        while self.done == False and self.current_position[1] <= self.ny-1 and self.current_position[1] >= 0 and n_step < 4*self.nx:
            self.state = self.position_to_state(self.current_position, self.position_obstacles)
            # print(self.state)
            action = np.argmax(Q_values[self.state,:])
            self.step(action)
            trajectory_x.append(self.current_position[0])
            trajectory_y.append(self.current_position[1])
            n_step += 1
        print('Done')
        return [trajectory_x, trajectory_y]

    def render(self):
        '''
        Renders the environment
        '''
        print('state: ', self.state)
        # print('roadmap: ', self.roadmap)

class side_walk_env_with_litter(side_walk_env):
    '''
    Side walk environment with litter: arguments are the same as the side_walk_env class but with p_obstacle = 0

    Parameters:
    position_litter: list of the position of the litter
    observation_space: observation space of the agent: 16 possible states (four neighbors having a litter or not)
    '''
    def __init__(self,nx,ny,upper_border,lower_border,p_litter,p_obstacle=0):
        super().__init__(nx,ny,upper_border,lower_border,p_obstacle,p_litter)
        self.position_litter = [[np.where(self.roadmap == 2)[0][i], np.where(self.roadmap == 2)[1][i]] for i in range(len(np.where(self.roadmap == 2)[0]))]
        self.observation_space = Observation_space(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]))

    def step(self, action):
        '''
        Performs an action and returns the next state, the next position, the reward, a boolean indicating if the episode is finished, and the information
        '''
        current_position = self.current_position
        reward = 15
        if current_position[0] == self.nx-1:
            self.done = True
            self.reward = 20
            return self.state, self.reward, self.current_position, self.done, self.info
        if action == 1: # move backward
            next_position = [current_position[0] - 1, current_position[1]]
            next_state = self.position_to_state(next_position,self.position_litter)
            if next_position in self.position_litter:
                self.position_litter.remove(next_position)
                self.reward = -5
            else:
                self.reward = -5
        elif action == 0: # move forward
            next_position = [current_position[0] + 1, current_position[1]]
            next_state = self.position_to_state(next_position,self.position_litter)
            if next_position in self.position_litter:
                self.position_litter.remove(next_position)
                self.reward = reward
            else:
                self.reward = 5
        elif action == 2: # move upward
            next_position = [current_position[0], current_position[1] + 1]
            next_state = self.position_to_state(next_position,self.position_litter)
            if next_position in self.position_litter:
                self.position_litter.remove(next_position)
                self.reward = reward
            else:
                self.reward = 0
        elif action == 3: # move downward
            next_position = [current_position[0], current_position[1] - 1]
            next_state = self.position_to_state(next_position,self.position_litter)
            if next_position in self.position_litter:
                self.position_litter.remove(next_position)
                self.reward = reward
            else:
                self.reward = 0
        else:
            raise ValueError('Invalid action')

        self.state = next_state
        self.current_position = next_position
        return self.state, self.reward, self.current_position, self.done, self.info

    def reset(self):
        '''
        Resets the environment
        '''
        self.current_position = self._init_position()
        self.position_litter = [[np.where(self.roadmap == 2)[0][i], np.where(self.roadmap == 2)[1][i]] for i in range(len(np.where(self.roadmap == 2)[0]))]
        self.state = self.position_to_state(self.current_position,self.position_litter)
        self.reward = 0
        self.done = False
        self.info = {} 
        return self.state, self.current_position, self.reward, self.done, self.info

    def trajectory(self, Q_values):
        '''
        Returns the trajectory of the agent given the Q_values
        '''
        self.update_env()
        self.reset()
        self.position_litter = [[np.where(self.roadmap == 2)[0][i], np.where(self.roadmap == 2)[1][i]] for i in range(len(np.where(self.roadmap == 2)[0]))]
        trajectory_x = []
        trajectory_y = []
        trajectory_x.append(self.current_position[0])
        trajectory_y.append(self.current_position[1])
        n_step = 0
        while self.done == False and self.current_position[1] <= self.ny-1 and self.current_position[1] >= 0 and n_step < 4*self.nx:
            self.state = self.position_to_state(self.current_position, self.position_litter)
            # print(self.state)
            action = np.argmax(Q_values[self.state,:])
            self.step(action)
            trajectory_x.append(self.current_position[0])
            trajectory_y.append(self.current_position[1])
            n_step += 1
        print('Done')
        return [trajectory_x, trajectory_y]

    def render(self):
        print('state: ', self.state)
        # print('roadmap: ', self.roadmap)

class side_walk_env_stay_on_road(side_walk_env):
    def __init__(self,nx,ny,upper_border,lower_border,p_litter=0,p_obstacle=0):
        "The rows inbetween the upper and lower border (defined as half integers) are area where the agent can move. We assume that the agent can move in the rows"
        "above and below the upper and lower border but with a penalty of -10."
        super().__init__(nx,ny,upper_border,lower_border,p_obstacle,p_litter)
        self.observation_space = Observation_space(np.array([0, 1, 2, 3, 4, 5, 6]))

    def position_to_state(self,current_position):
        if current_position[1]-1 > self.upper_border: # above upper border
            return 0
        elif current_position[1] > self.upper_border and current_position[1]-1 < self.upper_border: # above the upper border but near the border
            return 1
        elif current_position[1] < self.upper_border and current_position[1]+1 > self.upper_border: # on the road but near the upper border
            return 2
        elif current_position[1]+1 < self.upper_border and current_position[1]-1 > self.lower_border: # on the road
            return 3
        elif current_position[1] > self.lower_border and current_position[1]-1 < self.lower_border: # on the road but near the lower border
            return 4
        elif current_position[1] < self.lower_border and current_position[1]+1 > self.lower_border: # below the lower border but near the border
            return 5
        else: # below the lower border
            return 6
        

    def step(self, action):
        '''
        Performs a step in the environment
        '''
        current_position = self.current_position
        if current_position[0] == self.nx-1:
            self.done = True
            self.reward = 10
            return self.state, self.reward, self.done, self.info
        current_state = self.state
        if action == 0: # move backward
            next_position = [current_position[0] - 1, current_position[1]]
            self.state = self.position_to_state(next_position)
            self.current_position = next_position
            if current_state == 0 or current_state == 1:
                self.reward = -20
            elif current_state == 2 or current_state == 3 or current_state == 4:
                self.reward = -2
            else:
                self.reward = -20
        elif action == 1: # move forward
            next_position = [current_position[0] + 1, current_position[1]]
            self.state = self.position_to_state(next_position)
            self.current_position = next_position
            if current_state == 0 or current_state == 1:
                self.reward = -20
            elif current_state == 2 or current_state == 3 or current_state == 4:
                self.reward = 2
            else:
                self.reward = -20
        elif action == 2: # move upward
            next_position = [current_position[0], current_position[1] + 1]
            self.state = self.position_to_state(next_position)
            self.current_position = next_position
            if current_state == 0 or current_state == 1:
                self.reward = -20
            elif current_state == 2:
                self.reward = -10
            elif current_state == 3 or current_state == 4:
                self.reward = 0
            else:
                self.reward = 10
        elif action == 3: # move downward
            next_position = [current_position[0], current_position[1] - 1]
            self.state = self.position_to_state(next_position)
            self.current_position = next_position
            if current_state == 0 or current_state == 1:
                self.reward = 10
            elif current_state == 2 or current_state == 3:
                self.reward = 0
            elif current_state == 4:
                self.reward = -10
            else:
                self.reward = -20
        else:
            raise ValueError('Invalid action')

        return self.state, self.reward, self.done, self.info

    def reset(self):
        '''
        Resets the environment
        '''
        self.current_position = self._init_position()
        self.state = self.position_to_state(self.current_position)
        self.reward = 0
        self.done = False
        self.info = {} 
        return self.state, self.current_position, self.reward, self.done, self.info