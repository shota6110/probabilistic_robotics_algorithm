#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     configurations
# CreatedDate:  2018-10-30 01:15:12
#


import numpy as np


class State(object):
    """ About state

    Attributes:
        _action_list (:list: string): the name list of actions
        _prob_matrix (:list or np.array: float): the matrix of conditional probability.
            the shape is:
                1st dimention => before state: Ex) open, close
                2nd dimention => action: Ex) push, nothing
                3rd dimention => state: Ex) open, close
    """

    def __init__(self, x_list, prob_matrix):
        """ Constructor

        Args:
            x_list (:list: float): the state probability matrix
            prob_matrix (:list or np.array:): the conditional probabilitiy matrix

        """
        self._x_list = x_list
        self._prob_matrix = prob_matrix

    @property
    def size(self):
        """ return the size of x_list """
        return len(self._x_list)

    @property
    def states(self):
        """ return the states probability vector """
        return self._x_list

    @property
    def prob_matrix(self):
        """ return the prob matrix """
        return self._prob_matrix


class Action(object):
    """ About action

    Attributes:
        _action_list (:list: string): the name list of actions
    """

    def __init__(self, action_list):
        """ Constructor

        Args:
            action_list (:list: string): the name of action list
        """
        self._action_list = action_list

    @property
    def actions(self):
        """ return the action name list """
        return self._action_list

    def get_index(self, action):
        return self._action_list.index(action)


class Observation(object):
    """ About observation

    Attributes:
        _observe_list (:list or np.array: float): observation probability matrix
    """

    def __init__(self, observe_list):
        """ Constructor

        Args:
            observe_list (:list: float): the observation probability matrix
        """
        self._observe_list = np.array(observe_list)

    @property
    def list(self):
        """ return the list of observation probability """
        return self._observe_list

    @property
    def shape(self):
        """ return the shape of observe list """
        return self._observe_list.shape

