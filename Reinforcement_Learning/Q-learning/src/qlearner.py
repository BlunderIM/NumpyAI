import random as rand
import numpy as np

class Qlearner():
    def __init__(self, num_states, num_actions, alpha, gamma, rar, radr, memory_replay):
        """
        Constructor

        Inputs:
            num_states (int): number of possible states
            num_actions (int): number of possible actions
            alpha (float): learning rate
            gamma (float): future reward discount rate
            rar (float): random action rate
            radr (float): random action decay rate
            memory_replay (int): Number of iterations to replay experiences
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.q_table = np.zeros((num_states, num_actions))
        self.memory_replay = memory_replay
        self.experiences = []

    def query_no_update(self, s):
        """
        Query to decide which action to take without updating the q-table

        Args:
            s (int): value of the current state
        Returns:
            action (int): value of the suggested action
        """
        self.s = s
        if self.rar > np.random.uniform(0, 1):
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.q_table[s])

        return action

    def update_q_table(self, s, a, sprime, r):
        """
        Updates the Q table

        Args:
            s (int): Value of current state
            a (int): Value of action
            sprime (int): Value of new state
            r (int): Value of reward
        Returns:
            None
        """
        # Q update
        self.q_table[s, a] = (1 - self.alpha) * (self.q_table[s, a]) + \
                             self.alpha * (r + (self.gamma * self.q_table[sprime, np.argmax(self.q_table[sprime])]))

    def query(self, sprime, r):
        """
             Update the Q-table and return an action

             Args:
                 sprime (int): Value of the new state
                 r (int): reward
             """
        # Update the experiences memory
        self.experiences.append((self.s, self.a, sprime, r))

        # Update the Q table
        self.update_q_table(self.s, self.a, sprime, r)
        # Exploration vs Exploitation
        if self.rar > np.random.uniform(0, 1):
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.q_table[sprime, :])

        # Update the state, action and random action rate
        self.s = sprime
        self.a = action
        self.rar = self.rar * self.radr

        # Replaying the memory
        for i in range(self.memory_replay):
            random_index = np.random.choice(len(self.experiences), 1)[0]
            # Pulling a random tuple containing the state, action, sprime and the reward
            exp = self.experiences[random_index]
            temp_s, temp_a, temp_sprime, temp_r = exp[0], exp[1], exp[2], exp[3]
            self.update_q_table(temp_s, temp_a, temp_sprime, temp_r)

        return self.a, self.rar








