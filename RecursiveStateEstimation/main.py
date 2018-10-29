#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     main
# CreatedDate:  2018-10-30 01:53:18
#

import numpy as np
from configurations import State, Action, Observation
from bayesian_filter import BayesianFilter


if __name__ == '__main__':
    state_prob = np.array([0.5, 0.5])
    prob_matrix = np.array([[[1.0, 0.0], [1.0, 0.0]],
                            [[0.8, 0.2], [0.0, 1.0]]], dtype=np.float32)

    state = State(state_prob, prob_matrix)
    actions = Action(['push', 'do_nothing'])
    observations = Observation([[0.6, 0.4], [0.2, 0.8]])

    b = BayesianFilter(state, actions, observations)
    action_list = ['push', 'do_nothing', 'push']
    print(b.calculate_belief(action_list, 3))
