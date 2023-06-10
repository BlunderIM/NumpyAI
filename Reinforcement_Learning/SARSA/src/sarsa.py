import random as rand
import numpy as np

class SarsaLearner():
    def __init__(self, num_states, num_actions, alpha, gamma, rar, radr):
        """
        Constructor

        Inputs:
            num_states (int): number of possible states
            num_actions (int): number of possible actions
            alpha (float): learning rate
            gamma (float): future reward discount rate
            rar (float): random action rate
            radr (float): random action decay rate
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
        # Getting the new action at sprime
        anew = self.query_no_update(sprime)
        # Q update
        self.q_table[s, a] = (1 - self.alpha) * (self.q_table[s, a]) + \
                             self.alpha * (r + (self.gamma * self.q_table[sprime, anew]))

    def query(self, sprime, r):
        """
             Update the Q-table and return an action

             Args:
                 sprime (int): Value of the new state
                 r (int): reward
             """

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

        return self.a, self.rar








