"""
BSD 3-Clause License

Copyright (c) 2019, HJ Reachability Group
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author(s): David Fridovich-Keil ( dfk@eecs.berkeley.edu )
"""
################################################################################
#
# Script to run a 3 player collision avoidance example intended to model
# a T-intersection.
#
################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt

from unicycle_4d import Unicycle4D
from bicycle_4d import Bicycle4D
from car_5d import Car5D
from product_multiplayer_dynamical_system import \
    ProductMultiPlayerDynamicalSystem

from point import Point
from polyline import Polyline

from ilq_solver_potential import ILQSolver_Potential
from ilq_solver_openloop import ILQSolver_Openloop
from ilq_solver import ILQSolver
from proximity_cost import ProximityCost
from product_state_proximity_cost import ProductStateProximityCost
from semiquadratic_cost import SemiquadraticCost
from quadratic_cost import QuadraticCost
from semiquadratic_polyline_cost import SemiquadraticPolylineCost
from quadratic_polyline_cost import QuadraticPolylineCost
from player_cost import PlayerCost
from box_constraint import BoxConstraint

from visualizer import Visualizer
from logger import Logger

TIME_HORIZON = 5.0    # s
TIME_RESOLUTION = 0.1 # s
HORIZON_STEPS = int(TIME_HORIZON / TIME_RESOLUTION)
Planning_Steps = 5
LOG_DIRECTORY = "./logs/two_player_unicycle/"

# Create dynamics.
# car1 = Car5D(4.0)
# car2 = Car5D(4.0)
unicycle1 = Unicycle4D()
unicycle2 = Unicycle4D()
unicycle3 = Unicycle4D()
dynamics = ProductMultiPlayerDynamicalSystem(
    [unicycle1, unicycle2, unicycle3], T=TIME_RESOLUTION)

unicycle1_theta0 = np.arctan(1/3) # moving right
unicycle1_v0 = 2.0   # initial speed
unicycle1_x0 = np.array([
    [0.0],
    [0.0],
    [unicycle1_theta0],
    [unicycle1_v0]
])

unicycle2_theta0 = -0.5*np.pi # moving up
unicycle2_v0 = 1.5   # initial speed
unicycle2_x0 = np.array([
    [10.0],
    [10.0],
    [unicycle2_theta0],
    [unicycle2_v0]
])

unicycle3_theta0 = np.pi - np.arctan(1/3) # moving left
unicycle3_v0 = 3      # initial speed
unicycle3_x0 = np.array([
    [20.0],
    [0.0],
    [unicycle3_theta0],
    [unicycle3_v0]
])

stacked_x0 = np.concatenate([unicycle1_x0, unicycle2_x0, unicycle3_x0], axis=0)

states_array = np.zeros([12,HORIZON_STEPS+1])

states_array[:,0] = stacked_x0[:,0]

R = 5

for t in range(HORIZON_STEPS):

    print("Running step", t)

    unicycle1_Ps = [np.zeros((unicycle1._u_dim, dynamics._x_dim))] * Planning_Steps
    unicycle2_Ps = [np.zeros((unicycle2._u_dim, dynamics._x_dim))] * Planning_Steps
    unicycle3_Ps = [np.zeros((unicycle3._u_dim, dynamics._x_dim))] * Planning_Steps

    unicycle1_alphas = [np.zeros((unicycle1._u_dim, 1))] * Planning_Steps
    unicycle2_alphas = [np.zeros((unicycle2._u_dim, 1))] * Planning_Steps
    unicycle3_alphas = [np.zeros((unicycle3._u_dim, 1))] * Planning_Steps

    unicycle1_position_indices_in_product_state = (0, 1)
    unicycle1_goal = Point(15.0, 5.0)
    unicycle1_goal_cost = ProximityCost(
        unicycle1_position_indices_in_product_state, unicycle1_goal, np.inf, "unicycle1_goal")

    unicycle2_position_indices_in_product_state = (4, 5)
    unicycle2_goal = Point(10.0, 0.0)
    unicycle2_goal_cost = ProximityCost(
        unicycle2_position_indices_in_product_state, unicycle2_goal, np.inf, "unicycle2_goal")

    unicycle3_position_indices_in_product_state = (8, 9)
    unicycle3_goal = Point(5.0, 5.0)
    unicycle3_goal_cost = ProximityCost(
        unicycle3_position_indices_in_product_state, unicycle3_goal, np.inf, "unicycle3_goal")

    unicycle1_v_index_in_product_state = 3
    unicycle1_maxv = 4.0 # m/s
    unicycle1_minv_cost = SemiquadraticCost(
        unicycle1_v_index_in_product_state, 0.0, False, "unicycle1_minv")
    unicycle1_maxv_cost = SemiquadraticCost(
        unicycle1_v_index_in_product_state, unicycle1_maxv, True, "unicycle1_maxv")

    unicycle2_v_index_in_product_state = 7
    unicycle2_maxv = 4.0 # m/s
    unicycle2_minv_cost = SemiquadraticCost(
        unicycle2_v_index_in_product_state, 0.0, False, "unicycle2_minv")
    unicycle2_maxv_cost = SemiquadraticCost(
        unicycle2_v_index_in_product_state, unicycle2_maxv, True, "unicycle2_maxv")

    unicycle3_v_index_in_product_state = 11
    unicycle3_maxv = 4.0 # m/s
    unicycle3_minv_cost = SemiquadraticCost(
        unicycle3_v_index_in_product_state, 0.0, False, "unicycle3_minv")
    unicycle3_maxv_cost = SemiquadraticCost(
        unicycle3_v_index_in_product_state, unicycle3_maxv, True, "unicycle3_maxv")

    unicycle1_steering_cost = QuadraticCost(0, 0.0, "unicycle1_steering")
    unicycle1_a_cost = QuadraticCost(1, 0.0, "unicycle1_a")

    unicycle2_steering_cost = QuadraticCost(0, 0.0, "unicycle2_steering")
    unicycle2_a_cost = QuadraticCost(1, 0.0, "unicycle2_a")

    unicycle3_steering_cost = QuadraticCost(0, 0.0, "unicycle3_steering")
    unicycle3_a_cost = QuadraticCost(1, 0.0, "unicycle3_a")

    UNICYCLE_PROXIMITY_THRESHOLD = 2.0

    coef = 1

    unicycle1_proximity_cost = ProductStateProximityCost(
        [unicycle1_position_indices_in_product_state,
         unicycle2_position_indices_in_product_state,
         unicycle3_position_indices_in_product_state],
        UNICYCLE_PROXIMITY_THRESHOLD, coef,
        "unicycle1_proximity")

    unicycle2_proximity_cost = ProductStateProximityCost(
        [unicycle1_position_indices_in_product_state,
         unicycle2_position_indices_in_product_state,
         unicycle3_position_indices_in_product_state],
        UNICYCLE_PROXIMITY_THRESHOLD, coef,
        "unicycle2_proximity")

    unicycle3_proximity_cost = ProductStateProximityCost(
        [unicycle1_position_indices_in_product_state,
         unicycle2_position_indices_in_product_state,
         unicycle3_position_indices_in_product_state],
        UNICYCLE_PROXIMITY_THRESHOLD, coef,
        "unicycle3_proximity")

    unicycle1_cost = PlayerCost()
    unicycle1_cost.add_cost(unicycle1_goal_cost, "x", -100.0)
    unicycle1_cost.add_cost(unicycle1_proximity_cost, "x", 200.0)
    unicycle1_cost.add_cost(unicycle1_maxv_cost, "x", 100.0)
    unicycle1_cost.add_cost(unicycle1_minv_cost, "x", 100.0)

    unicycle1_player_id = 0

    unicycle1_cost.add_cost(unicycle1_steering_cost, unicycle1_player_id, 50.0)
    unicycle1_cost.add_cost(unicycle1_a_cost, unicycle1_player_id, 1.0)

    unicycle2_cost = PlayerCost()
    unicycle2_cost.add_cost(unicycle2_goal_cost, "x", -100.0)
    unicycle2_cost.add_cost(unicycle2_proximity_cost, "x", 200.0)
    unicycle2_cost.add_cost(unicycle2_maxv_cost, "x", 100.0)
    unicycle2_cost.add_cost(unicycle2_minv_cost, "x", 100.0)

    unicycle2_player_id = 1

    unicycle2_cost.add_cost(unicycle2_steering_cost, unicycle2_player_id, 50.0)
    unicycle2_cost.add_cost(unicycle2_a_cost, unicycle2_player_id, 1.0)

    unicycle3_cost = PlayerCost()
    unicycle3_cost.add_cost(unicycle3_goal_cost, "x", -100.0)
    unicycle3_cost.add_cost(unicycle3_proximity_cost, "x", 200.0)
    unicycle3_cost.add_cost(unicycle3_maxv_cost, "x", 100.0)
    unicycle3_cost.add_cost(unicycle3_minv_cost, "x", 100.0)

    unicycle3_player_id = 2

    unicycle3_cost.add_cost(unicycle3_steering_cost, unicycle3_player_id, 50.0)
    unicycle3_cost.add_cost(unicycle3_a_cost, unicycle3_player_id, 1.0)

    visualizer = Visualizer(
        [unicycle1_position_indices_in_product_state,
         unicycle2_position_indices_in_product_state,
         unicycle3_position_indices_in_product_state],
        [unicycle1_goal_cost,
         unicycle2_goal_cost,
         unicycle3_goal_cost],
        [".-r", ".-g",".-b"],
        1,
        False,
        plot_lims=[-2.5, 22.5,-2.5, 12.5])

    if not os.path.exists(LOG_DIRECTORY):
        os.makedirs(LOG_DIRECTORY)

    logger = Logger(os.path.join(LOG_DIRECTORY, 'trial.pkl'))

    x = states_array[:,t]

    d12 = np.sqrt(((x[0:2]-x[4:6])**2).sum())
    d13 = np.sqrt(((x[0:2]-x[8:10])**2).sum())
    d23 = np.sqrt(((x[8:10]-x[4:6])**2).sum())

    #if min([d12,d13,d23]) <= 5:
    # Set up ILQSolver.
    solver = ILQSolver(dynamics,
                       [unicycle1_cost, unicycle2_cost, unicycle3_cost],
                       stacked_x0,
                       [unicycle1_Ps, unicycle2_Ps, unicycle3_Ps],
                       [unicycle1_alphas, unicycle2_alphas, unicycle3_alphas],
                       0.1,
                       None,
                       None,
                       visualizer,
                       None)

    xs, us, costs = solver.run()

    stacked_x0 = xs[1]

    noise = np.random.multivariate_normal(np.zeros(12), 0.01*np.eye(12), size = 1)

    stacked_x0[:,0] += noise[0,:]

    states_array[:,t+1] = stacked_x0[:,0]

xs = states_array

np.savetxt("./data/three_unicycle_mpc/xs_three_unicycle_mpc_feedback_2.csv",
           xs,
           delimiter =", ",
           fmt ='% s')
