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

    def calculate_belief(self, action_sequence, t=1):
        """ Get belief of interested action

        Args:
            action_sequence (list): the sequence of actions like [action0_1, action1_2, action2_3, ....]
            t (int `Optional`): the loop number

        Returns:
            list (float): The belief list of states
        """

        assert len(action_sequence) == t, "Action num should be equal to loop num !!!"

        # create column vector of before beliefs
        if t == 1:
            belief_before = np.reshape(self._state.states, (self._state.size, 1))
        else:
            belief_before = np.reshape(self.calculate_belief(action_sequence[1:], t-1), (self._state.size, 1))

        # get action's index
        action_index = self._actions.get_index(action_sequence[0])
        # calculate belief_bar
        belief_bar = self._prob_matrix[:, action_index, :].dot(belief_before)
        # calculate belief
        beliefs = belief_bar * self._observes.list[:, 0].reshape((self._observes.shape[0], 1))  # element-wise
        # normalized by ita
        normalized_beliefs = self._get_ita(beliefs) * beliefs

        return np.round(normalized_beliefs.flatten(), 3)
