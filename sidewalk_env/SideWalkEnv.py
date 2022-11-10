from matplotlib import pyplot as plt
import numpy as np
import operator
from functools import reduce

class side_walk_env:
    def __init__(self,nx,ny,upper_border,lower_border,p_obstacle,p_litter):
        "The action space is discrete with 4 actions: 0: left, 1: right, 2: up, 3: down"
        self.state = 0
        self.action_space = np.array([0, 1, 2, 3])
        self.reward = 0
        self.done = False
        self.info = {} # for debugging
        self.p_obstacle = p_obstacle
        self.p_litter = p_litter
        self.nx = nx
        self.ny = ny
        self.roadmap = self.generate_roadmap()
        self.upper_border = upper_border
        self.lower_border = lower_border
        self.current_position = np.array([0,0])

    def generate_roadmap(self):
        roadmap = np.random.choice([0,1,2], (self.nx, self.ny),p=[1-self.p_obstacle-self.p_litter,self.p_obstacle,self.p_litter])
        return roadmap

    def reset(self):
        self.state = 0
        self.reward = 0
        self.done = False
        self.info = {} 
        return self.state

    def position_to_state(self,current_position,ob_bo):
        right = current_position + np.array([0,1])
        left = current_position + np.array([0,-1])
        down = current_position + np.array([-1,0])
        up = current_position + np.array([1,0])
        state = 2**0 * (self.roadmap[up[0], up[1]]==ob_bo) + 2**1 * (self.roadmap[down[0], down[1]]==ob_bo) + 2**2 * (self.roadmap[left[0], left[1]]==ob_bo) + 2**3 * (self.roadmap[right[0], right[1]]==ob_bo)
        return state

    def render(self):
        print('state: ', self.state)

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def plot_roadmap(self):
        x = [[i+1] for i in range(self.nx)]
        road = [[i+1 for i in range(self.ny)] for j in range(self.nx)]
        road_up = [[self.upper_border] for i in range(self.nx)]
        road_low = [[self.lower_border] for i in range(self.nx)]
        obstacle = np.array([np.where(self.roadmap == 1)[0]+1, np.where(self.roadmap == 1)[1]+1])
        litter = np.array([np.where(self.roadmap == 2)[0]+1, np.where(self.roadmap == 2)[1]+1])
        plt.figure(figsize=(15,4))
        plt.plot(x, road, linestyle='--', color = "0.7", zorder=10)
        plt.fill_between(reduce(operator.add, x), reduce(operator.add, road_low), reduce(operator.add, road_up), color = '#539caf', alpha = 0.2, zorder=20)
        plt.plot(np.array(x), (np.array(road_up)+np.array(road_low))/2, linestyle='--', color = "y", linewidth = 3, zorder=30)
        plt.scatter(obstacle[0], obstacle[1], marker='x', color = 'k', zorder=40)
        plt.scatter(litter[0], litter[1], marker='s', color='r', zorder=40)
        plt.scatter([1,1,1,1,1,1],[3,5,7,9,11,13], marker='>',zorder=50)
        plt.title('Road Map')

class side_walk_env_with_obstacle(side_walk_env):
    def __init__(self,nx,ny,upper_border,lower_border,p_obstacle,p_litter=0):
        super().__init__(nx,ny,upper_border,lower_border,p_obstacle,p_litter)
        self.observation_space = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    def step(self, action):
        current_position = self.current_position
        if current_position[0] == self.nx-1:
            self.done = True
            self.reward = 100
            return self.state, self.reward, self.done, self.info
        if action == 0: # move left
            next_position = [current_position[0], current_position[1] - 1]
            self.state = self.position_to_state(next_position,1)
            self.current_position = next_position
            if self.roadmap[next_position[0], next_position[1]] == 1:
                self.reward = -50
            elif self.roadmap[next_position[0], next_position[1]] == 0:
                self.reward = -2
        elif action == 1: # move right
            next_position = [current_position[0], current_position[1] + 1]
            self.state = self.position_to_state(next_position,1)
            self.current_position = next_position
            if self.roadmap[next_position[0], next_position[1]] == 1:
                self.reward = -50
            elif self.roadmap[next_position[0], next_position[1]] == 0:
                self.reward = 2
        elif action == 2: # move up
            next_position = [current_position[0] + 1, current_position[1]]
            self.state = self.position_to_state(next_position,1)
            self.current_position = next_position
            if self.roadmap[next_position[0], next_position[1]] == 1:
                self.reward = -50
            elif self.roadmap[next_position[0], next_position[1]] == 0:
                self.reward = 0
        elif action == 3: # move down
            next_position = [current_position[0] - 1, current_position[1]]
            self.state = self.position_to_state(next_position,1)
            self.current_position = next_position
            if self.roadmap[next_position[0], next_position[1]] == 1:
                self.reward = -100
            elif self.roadmap[next_position[0], next_position[1]] == 0:
                self.reward = 0
        else:
            raise ValueError('Invalid action')

        return self.state, self.reward, self.done, self.info

    def render(self):
        print('state: ', self.state)
        # print('roadmap: ', self.roadmap)

class side_walk_env_with_litter(side_walk_env):
    def __init__(self,nx,ny,upper_border,lower_border,p_litter,p_obstacle=0):
        super().__init__(nx,ny,upper_border,lower_border,p_obstacle,p_litter)
        self.observation_space = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    def step(self, action):
        current_position = self.current_position
        if current_position[0] == self.nx-1:
            self.done = True
            self.reward = 100
            return self.state, self.reward, self.done, self.info
        if action == 0: # move left
            next_position = [current_position[0], current_position[1] - 1]
            self.state = self.position_to_state(next_position,2)
            self.current_position = next_position
            if self.roadmap[next_position[0], next_position[1]] == 2:
                self.reward = 10
            elif self.roadmap[next_position[0], next_position[1]] == 0:
                self.reward = -2
        elif action == 1: # move right
            next_position = [current_position[0], current_position[1] + 1]
            self.state = self.position_to_state(next_position,2)
            self.current_position = next_position
            if self.roadmap[next_position[0], next_position[1]] == 2:
                self.reward = 10
            elif self.roadmap[next_position[0], next_position[1]] == 0:
                self.reward = 2
        elif action == 2: # move up
            next_position = [current_position[0] + 1, current_position[1]]
            self.state = self.position_to_state(next_position,2)
            self.current_position = next_position
            if self.roadmap[next_position[0], next_position[1]] == 2:
                self.reward = 10
            elif self.roadmap[next_position[0], next_position[1]] == 0:
                self.reward = 0
        elif action == 3: # move down
            next_position = [current_position[0] - 1, current_position[1]]
            self.state = self.position_to_state(next_position,2)
            self.current_position = next_position
            if self.roadmap[next_position[0], next_position[1]] == 2:
                self.reward = 10
            elif self.roadmap[next_position[0], next_position[1]] == 0:
                self.reward = 0
        else:
            raise ValueError('Invalid action')

        return self.state, self.reward, self.done, self.info

    def render(self):
        print('state: ', self.state)
        # print('roadmap: ', self.roadmap)

class side_walk_env_stay_on_road(side_walk_env):
    def __init__(self,nx,ny,upper_border,lower_border,p_litter=0,p_obstacle=0):
        "The rows inbetween the upper and lower border (defined as half integers) are area where the agent can move. We assume that the agent can move in the rows"
        "above and below the upper and lower border but with a penalty of -10."
        super().__init__(nx,ny,upper_border,lower_border,p_obstacle,p_litter)
        self.observation_space = np.array([0, 1, 2, 3, 4, 5, 6])

    def position_to_state(self,current_position):
        if current_position[0]-1 > self.upper_border: # above upper border
            return 0
        elif current_position[0] > self.upper_border and current_position[0]-1 < self.upper_border: # above the upper border but near the border
            return 1
        elif current_position[0] < self.upper_border and current_position[0]+1 > self.upper_border: # on the road but near the upper border
            return 2
        elif current_position[0]+1 < self.upper_border and current_position[0]-1 > self.lower_border: # on the road
            return 3
        elif current_position[0] > self.lower_border and current_position[0]-1 < self.lower_border: # on the road but near the lower border
            return 4
        elif current_position[0] < self.lower_border and current_position[0]+1 > self.lower_border: # below the lower border but near the border
            return 5
        else: # below the lower border
            return 6
        

    def step(self, action):
        current_position = self.current_position
        if current_position[0] == self.nx-1:
            self.done = True
            self.reward = 100
            return self.state, self.reward, self.done, self.info
        current_state = self.state
        if action == 0: # move left
            next_position = [current_position[0], current_position[1] - 1]
            self.state = self.position_to_state(next_position)
            self.current_position = next_position
            if current_state == 0 or current_state == 1:
                self.reward = -20
            elif current_state == 2 or current_state == 3 or current_state == 4:
                self.reward = -2
            else:
                self.reward = -20
        elif action == 1: # move right
            next_position = [current_position[0], current_position[1] + 1]
            self.state = self.position_to_state(next_position)
            self.current_position = next_position
            if current_state == 0 or current_state == 1:
                self.reward = -20
            elif current_state == 2 or current_state == 3 or current_state == 4:
                self.reward = 2
            else:
                self.reward = -20
        elif action == 2: # move up
            next_position = [current_position[0] + 1, current_position[1]]
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
        elif action == 3: # move down
            next_position = [current_position[0] - 1, current_position[1]]
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