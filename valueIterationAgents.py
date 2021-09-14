# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        # import pdb;pdb.set_trace()
        "*** YOUR CODE HERE ***"

        # iterate through the convergence
        for _iter in range(self.iterations):

            # init a temp val counter to store the intermediate results of q_vals
            val_counter = util.Counter()
            # iterate through each states to find the optimal policy
            for state in self.mdp.getStates():
                # init a max_val as some very high number
                max_val = float("-inf")
                # iterate through each actions to find the possible set of actions for the given state
                for action in self.mdp.getPossibleActions(state):
                    # call compute qval functions to find the value for that state
                    q_val = self.computeQValueFromValues(state, action)

                    # if q_val is greater than max val then assign it as it's val
                    if q_val > max_val:
                        max_val = q_val

                    # update state val in val counter
                    val_counter[state] = max_val

            self.values = val_counter  # update the values to new val counter

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        _val = 0  # init a _val with zero

        # find all the next states and it's probablity
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(
                state, action):
            # based on next states, rewards and discount find the value for the given state and action
            _val += prob * (self.mdp.getReward(state, action, next_state) +
                            self.discount * self.values[next_state])

        # reutrn the val
        return _val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # set best_action as none and max val as -inf
        best_action = None
        max_val = float('-inf')
        # iterate through each action and finds it max val
        for action in self.mdp.getPossibleActions(state):
            # find the q_val for the next action
            q_val = self.computeQValueFromValues(state, action)

            # if q_val is greater than max_val assign it as max_val and best_action as current action
            if q_val > max_val:
                max_val = q_val
                best_action = action

        # return best action
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
