# qlearningAgents.py
# ------------------
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

from game import *
from learningAgents import ReinforcementAgent

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.q_vals = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # return q_vals for state action pair
        return self.q_vals[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # find all the legal allowed actions for the this state
        _allowed_actions = self.getLegalActions(state)

        # if no actions then return 0
        if not _allowed_actions:
            return 0.0

        # create an instance of q_val and store all it's val
        _q_vals = util.Counter()

        # iterte throught all the list actions and store it's q_val
        for action in _allowed_actions:
            _q_vals[action] = self.getQValue(state, action)

        # return the best q_val from set of all the q_val for that actions
        return _q_vals[_q_vals.argMax()]

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # init best_action and max_val
        best_action = None
        max_val = float('-inf')

        # iterate through each legal action and store it's q_val
        for action in self.getLegalActions(state):
            # compute q_vals
            _q_val = self.getQValue(state, action)

            # if q_val is gte than max_val then, assign that action as best action
            # and max_val as current q_Val
            if _q_val >= max_val:
                max_val = _q_val
                best_action = action

        return best_action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # check if you should explore or exploit
        explore = util.flipCoin(self.epsilon)

        if explore:
            # if in explore mode then select any random action from legal set of action
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)

        # return action
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # compute old q vals and reward
        _old_q_vals = self.getQValue(state, action)
        _reward = self.alpha * reward

        # update q vals w.r.t. old q vals and reward
        self.q_vals[(state, action)] = (1 - self.alpha) * _old_q_vals + _reward

        # check if next state exists for the given state and action
        # if next state exists then add it's val too to q_val
        if nextState:
            self.q_vals[(
                state, action
            )] += self.alpha * self.discount * self.getValue(nextState)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self,
                 epsilon=0.05,
                 gamma=0.8,
                 alpha=0.2,
                 numTraining=0,
                 **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action
