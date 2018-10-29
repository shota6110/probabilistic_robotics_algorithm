#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     bayesian_filter
# CreatedDate:  2018-10-29 21:51:29
#

import numpy as np
from configurations import State, Action, Observation


class BayesianFilter(object):
    """ The calculation of BayesianFilter

    Attributes:
        _actions (:obj: Action): Action obj which holds action name list
        _observes (:obj: Observation): Observation obj which holds the matrix of probability of sensor data
        _state (:obj: State): State obj of initial state list and matrix of conditional probabilities
        _prob_list (:np.array: `float`): numpy array of conditional probabilities which are belongs to State class
    """

    def __init__(self, state, action, observation):
        """ Constructor

        Args:
            state (State obj) : state object
            action_list (Action obj) : action object
            observe_list (Observation obj) : observation object
        """
        self._state = state
        self._actions = action
        self._observes = observation
        self._prob_matrix = state.prob_matrix

    def _get_ita(self, prob_list):
        """ Returns the reciprocal number of input list sum

        Args:
            prob_list (list or np.array) : probability list

        Returns:
            float: reciprocal num of input list summation
        """
        return np.reciprocal(np.sum(prob_list))

    def calculate_belief(self, action, t=0):
        """ Get belief of interested action

        Args:
            action (string) : Action name in action list

        Returns:
            list (float): The belief list of states
        """
        # create column vector of before beliefs
        if t == 0:
            belief_before = np.reshape(self._state.states, (self._state.size, 1))
        else:
            belief_before = np.reshape(self.calculate_belief(action, t-1), (self._state.size, 1))
        # get action's index
        action_index = self._actions.get_index(action)
        # calculate belief_bar
        belief_bar = self._prob_matrix[:, action_index, :].dot(belief_before)
        # calculate belief
        beliefs = belief_bar * self._observes.list[:, 0].reshape((self._observes.shape[0], 1))  # element-wise
        # normalized by ita
        normalized_beliefs = self._get_ita(beliefs) * beliefs

        return np.round(normalized_beliefs.flatten(), 3)


if __name__ == '__main__':
    state_prob = np.array([0.5, 0.5])
    prob_matrix = np.array([[[1.0, 0.0], [1.0, 0.0]],
                            [[0.8, 0.2], [0.0, 1.0]]], dtype=np.float32)

    state = State(state_prob, prob_matrix)
    actions = Action(['push', 'do_nothing'])
    observations = Observation([[0.6, 0.4], [0.2, 0.8]])

    b = BayesianFilter(state, actions, observations)
    print(b.calculate_belief('push', 9))
