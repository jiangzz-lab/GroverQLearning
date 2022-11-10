from qiskit import *
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram

import numpy as np
from math import ceil

class RL_Qlearning_trainer():
    ''' 
    Implement a Reinforcement Q-Learning.

    Assumption:
    The dimensions of the state space and action space are both finite

    Parameters:
    env: the environment to solve; default is OpenAI gym "FrozenLake"
    state (int): current state
    action (int): current action 
    state_dimension (int): dimension of the state space
    Q_values (2D np array): Q values of all (state, action) combinations; shape = (state_dimension, action_dimention)
    hyperparameters (dict): hyperparameters of learning; 
                            {
                                'alpha': learning rate, 'gamma': discount, 
                                'eps': tolerance of the Q values,
                                'max_epochs': max number of epochs for training,
                                'max_steps': max number of steps in every epoch
                            }
    '''
    def __init__(self, env) -> None:
        self.env = env
        self.state = env.reset()[0]
        self.action = 0
        self.state_dimension = env.observation_space.n
        self.action_dimension = env.action_space.n
        self.Q_values = np.zeros((self.state_dimension, self.action_dimension), dtype=float) 
        self.state_values = np.zeros(self.state_dimension, dtype=float)
        self.hyperparameters = {
            'alpha': 0.05,
            'gamma': 0.99,
            'eps': 0.01,
            'max_epochs': 1000,
            'max_steps': 100
        }
            
    # hyperparameter setter
    def set_hyperparams(self, hyperdict):
        """
        Set learner's hyperparameters
        :param hyperdict: a dict with same keys as self's
        :return:
        """
        self.hyperparameters = hyperdict


    def _take_action(self):
        ''' 
        Take an action under the current state by measuring the corresponding action wavefunction
        '''
        raise NotImplementedError

    def _update_Q_values(self, reward, new_state):
        ''' 
        Update the Q_value after one state transition
        '''
        alpha = self.hyperparameters['alpha']
        gamma = self.hyperparameters['gamma']
        # self.Q_values[self.state, self.action] = self.Q_values[self.state, self.action] + alpha * (reward + gamma * np.max(self.Q_values[new_state]) - self.Q_values[self.state, self.action])
        self.Q_values[self.state, self.action] = self.Q_values[self.state, self.action] + alpha * (reward + gamma * self.state_values[new_state] - self.Q_values[self.state, self.action])

    def _update_learner(self):
        '''
        Update the learner after one state transition if necessary
        '''
        raise NotImplementedError
        
    # def _reach_terminal_state(self) -> bool:
    #     '''
    #     Check if the current state is a terminal state
    #     '''
    #     raise NotImplementedError

    def train(self):
        ''' 
        Train the GroverQlearner agent by running multiple epochs with the given max epoch.
        Record the step used in each epoch, whether the target is reached, and the trajectory
        '''
        # eps = self.hyperparameters['eps'] # the Q_value table tolerance
        max_epochs = self.hyperparameters['max_epochs']
        print(max_epochs)
        optimal_steps = self.hyperparameters['max_steps'] # initial the steps to be the max steps
        target_reached = False
        trajectory = []
        steps_in_all_epochs = []
        target_reached_in_all_epochs = []
        trajectories_in_all_epochs = [] # stores 

        for epoch in range(max_epochs):
            if epoch % 10 == 0:
                print("Processing epoch {} ...".format(epoch)) # monitor the training process
            self.state = self.env.reset()[0] # reset env
            target_reached = False # init target reached flag
            trajectory = [self.state] # list to record the trajectory of the current epoch
            
            for step in range(optimal_steps): # take steps
                print('Taking step {0}/{1}'.format(step, optimal_steps), end='\r')
                self.action = self._take_action() # take an action
                new_state, reward, _, done, _ = self.env.step(self.action) # step to the new state
                if new_state == self.state:
                    reward -= 10 # take less reward if stay in the same state
                    done = True # no need to step more if the state cannot be changed
                if new_state == self.state_dimension - 1:
                    reward += 99 # take very large reward when the target is reached
                    optimal_steps = step + 1 # update the optimal steps when the target is reached
                    target_reached = True
                elif not done: 
                    reward -= 1  # when the state is changed but the target is not reached, take a moderate reward
                self._update_Q_values(reward, new_state)
                self._update_learner(reward, new_state)
                trajectory.append(new_state) # append the new state to the trajectory
                if done:
                    break
                self.state = new_state # update the state if it is changed

            steps_in_all_epochs.append(optimal_steps)
            target_reached_in_all_epochs.append(target_reached)
            trajectories_in_all_epochs.append(trajectory)

        return steps_in_all_epochs, target_reached_in_all_epochs, trajectories_in_all_epochs

class GroverQlearner(RL_Qlearning_trainer):
    ''' 
    Implement a quanutm reinforcement learning agent based on Grover amplitute enhancement and QLearing algorithm.

    Assumption:
    The dimensions of the state space and action space are both finite

    Parameters:
    env: the environment passed to the super class
    action_qregister_size (int): number of qubits on the quantum register for storing the action wavefunction 
    max_grover_length (int): maximum of the length of the grover iteration
    grover_lengths (2D np array): lengths of grover iterations of all (state, action) combinaitons; shape = (state_dimension, action_dimention)
    grover_operators (1D np array): grover_operators for all actions; grover_operators[a] records the grover operator constructed from action eigenfucntion a 
    action_circuits (1D np array): action quantum circuits for all states; action_circuits[s] records the quanutm circuit encoding the up-to-date action wavefunction of state s
    backend: machine to execute the quanutm circuit jobs; could be either a simulator or a true quantum computer
    '''
    def __init__(self, env):
        super().__init__(env)
        self.action_qregister_size = ceil(np.log2(self.action_dimension))
        self.max_grover_length = int(round(np.pi / (4 * np.arcsin(1. / np.sqrt(2 ** self.action_qregister_size))) - 0.5))
        self.grover_lengths = np.zeros((self.state_dimension, self.action_dimension), dtype=int)
        self.max_grover_length_reached = np.zeros((self.state_dimension, self.action_dimension), dtype=bool)
        self.grover_operators = self._init_grover_operators()
        self.action_circuits = self._init_action_circuits()
        self.backend = Aer.get_backend('qasm_simulator')
        self.hyperparameters['k'] = 0.1 # prefactor of max grover length


    def _init_action_circuits(self):
        '''
        Initialize the quanutm circuits encoding the action wavefunction of every state. Every initial action wavefunction is a equally weighted superposition of all action eignenfucntions. 
        '''
        # action_circuits = np.zeros(self.state_dimension)
        action_circuits = np.empty(self.state_dimension, dtype=object)
        for s in range(self.state_dimension):
            action_circuits[s] = QuantumCircuit(self.action_qregister_size, name='action_s{}'.format(s)) # construct the quanutm circuit
        for circuit in action_circuits:
            circuit.h(list(range(self.action_qregister_size))) # apply H gate to every qubit register to create the equally weighted superposition
        return action_circuits

    def _init_grover_operators(self):
        ''' 
        Initialize the grover operators of every action. U_grover := U_a0 * Ua where a0 is the equally superposition of all action eigenfunctions and a is an action eigenfunction. In fact,
        U_grover is not updated during the training process within the scope of this project.
        '''
        # grove_operators = np.zeros(self.action_dimension)
        grove_operators = np.empty(self.action_dimension, dtype=object)
        target_states = np.zeros(self.action_dimension)
        for i in range(self.action_dimension):
            state_binary = format(i, '0{}b'.format(self.action_qregister_size)) # generate the statevector binary string for encoding every action using the quantum register
            grove_operators[i] = GroverOperator(oracle=Statevector.from_label(state_binary)).to_instruction()
        return grove_operators

    def _get_grover_length(self, reward, new_state):
        ''' 
        Calculate length of the Grover iteration after taking an action
        '''
        k = self.hyperparameters['k']
        # return int(k * (reward + np.max(self.Q_values[new_state]))) # here we use max(Q_value[new_state]), it is also possible to use the expectation of Q_value[new_state] based on Born's rule 
        return int(k * (reward + self.state_values[new_state]))
        
    def _run_grover_iterations(self):
        '''
        Run grover iterations at one state
        '''
        length = min(self.grover_lengths[self.state, self.action], self.max_grover_length) # number of grover operators(iterations) to append in this steps
        circuit = self.action_circuits[self.state] # the up-to-date quanutm circuit encodeing the action of current state
        grover_operator = self.grover_operators[self.action] # the grover operator of current action
        max_grover_length_reached = self.max_grover_length_reached[self.state]
        if not(max_grover_length_reached.any()):
            for _ in range(length):
                circuit.append(grover_operator, list(range(self.action_qregister_size)))
        if length >= self.max_grover_length and (not(max_grover_length_reached.any())): 
            self.max_grover_length_reached[self.state, self.action] = True  # update the self.max_grover_length_reached when the max grove length is reached
        self.action_circuits[self.state] = circuit # update the circuit

    def _take_action(self):
        '''
        Take an action at the current state
        '''
        
        circuit = self.action_circuits[self.state] # the quanutm circuit encoding the up-to-date action wavefunction of the current state
        circuit_to_measure = circuit.copy() # make a copy of the circuit for measurement so that the original circuit is not broken by the measurement
        circuit_to_measure.measure_all() # take the action by measuring the current action wavefunciton
        job = execute(circuit_to_measure, backend=self.backend, shots=1) # execute the circuit using the backend
        result = job.result()
        counts = result.get_counts()
        action = int((list(counts.keys()))[0], 2) # take the action with highest probablity
        self.state_values[self.state] = self.Q_values[self.state, action] # update the state value
        return action

    def _update_learner(self, reward, new_state):
        '''
        Update the learner after one state transition if necessary
        '''
        self.grover_lengths[self.state, self.action] = self._get_grover_length(reward, new_state) # get grover length
        self._run_grover_iterations() # run grover iterations to tune the amplitude of the action eigenfunctions

class ClassicalLearner(RL_Qlearning_trainer):
    '''
    Classical Q-learning algorithm
    '''
    def __init__(self, env):
        super().__init__(env)

    def _take_action(self):
        '''
        Take an action at the current state
        '''
        if np.random.random() < self.hyperparameters['eps']:
            action = np.random.randint(self.action_dimension)
        else:
            action = np.argmax(self.Q_values[self.state])
            # self.state_values[self.state] = self.Q_values[self.state, action] # update the state value
        return action
    
    def _update_Q_values(self, reward, new_state):
        ''' 
        Update the Q_value after one state transition
        '''
        alpha = self.hyperparameters['alpha']
        gamma = self.hyperparameters['gamma']
        self.Q_values[self.state, self.action] = self.Q_values[self.state, self.action] + alpha * (reward + gamma * np.max(self.Q_values[new_state]) - self.Q_values[self.state, self.action])

    def _update_learner(self,*args):
        pass