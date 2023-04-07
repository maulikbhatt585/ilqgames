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
from three_unicycle_4d import Three_Unicycle4D
from bicycle_4d import Bicycle4D
from car_5d import Car5D
from product_multiplayer_dynamical_system import \
    ProductMultiPlayerDynamicalSystem

from point import Point
from polyline import Polyline

from ilq_solver_potential import ILQSolver_Potential
from ilq_solver import ILQSolver
from proximity_cost import ProximityCost
from product_state_proximity_cost import ProductStateProximityCost
from product_state_proximity_cost import ProductStateProximityCost_Potential
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
LOG_DIRECTORY = "./logs/three_player_unicycle/"

# Create dynamics.

three_unicycle = Three_Unicycle4D()

dynamics = ProductMultiPlayerDynamicalSystem(
    [three_unicycle], T=TIME_RESOLUTION)

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

three_unicycle_Ps = [np.zeros((three_unicycle._u_dim, three_unicycle._x_dim))] * HORIZON_STEPS

three_unicycle_alphas = [np.zeros((three_unicycle._u_dim, 1))] * HORIZON_STEPS


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

unicycle2_steering_cost = QuadraticCost(2, 0.0, "unicycle2_steering")
unicycle2_a_cost = QuadraticCost(3, 0.0, "unicycle2_a")

unicycle3_steering_cost = QuadraticCost(4, 0.0, "unicycle3_steering")
unicycle3_a_cost = QuadraticCost(5, 0.0, "unicycle3_a")

UNICYCLE_PROXIMITY_THRESHOLD = 2.0

coef = 1

unicycle_proximity_cost = ProductStateProximityCost(
    [unicycle1_position_indices_in_product_state,
     unicycle2_position_indices_in_product_state,
     unicycle3_position_indices_in_product_state],
    UNICYCLE_PROXIMITY_THRESHOLD, coef,
    "unicycle_proximity")

unicycle_cost = PlayerCost()

unicycle_cost.add_cost(unicycle1_goal_cost, "x", -100.0)
unicycle_cost.add_cost(unicycle_proximity_cost, "x", 200.0)
unicycle_cost.add_cost(unicycle1_maxv_cost, "x", 100.0)
unicycle_cost.add_cost(unicycle1_minv_cost, "x", 100.0)

unicycle1_player_id = 0

unicycle_cost.add_cost(unicycle1_steering_cost, unicycle1_player_id, 50.0)
unicycle_cost.add_cost(unicycle1_a_cost, unicycle1_player_id, 1.0)

unicycle_cost.add_cost(unicycle2_goal_cost, "x", -100.0)
unicycle_cost.add_cost(unicycle2_maxv_cost, "x", 100.0)
unicycle_cost.add_cost(unicycle2_minv_cost, "x", 100.0)

unicycle2_player_id = 0

unicycle_cost.add_cost(unicycle2_steering_cost, unicycle2_player_id, 50.0)
unicycle_cost.add_cost(unicycle2_a_cost, unicycle2_player_id, 1.0)

unicycle_cost.add_cost(unicycle3_goal_cost, "x", -100.0)
unicycle_cost.add_cost(unicycle3_maxv_cost, "x", 100.0)
unicycle_cost.add_cost(unicycle3_minv_cost, "x", 100.0)

unicycle3_player_id = 0

unicycle_cost.add_cost(unicycle3_steering_cost, unicycle3_player_id, 50.0)
unicycle_cost.add_cost(unicycle3_a_cost, unicycle3_player_id, 1.0)

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

logger = Logger(os.path.join(LOG_DIRECTORY, 'three_unicycle_potential.pkl'))

# Set up ILQSolver.
solver = ILQSolver_Potential(dynamics,
                   [unicycle_cost],
                   stacked_x0,
                   [three_unicycle_Ps],
                   [three_unicycle_alphas],
                   0.1,
                   None,
                   logger,
                   visualizer,
                   None)

save_path = './data/three_unicycle_potential/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

xs, us, costs = solver.run()

x = np.zeros([12, HORIZON_STEPS])
u = np.zeros([6, HORIZON_STEPS])

for i in range(len(xs)):
    x[:,i] = xs[i].ravel()
    u[:,i] = us[0][i].ravel()

np.savetxt("./data/three_unicycle_potential/xs_three_unicycle_potential.csv",
           x,
           delimiter =", ",
           fmt ='% s')

np.savetxt("./data/three_unicycle_potential/us_three_unicycle_potential.csv",
           u,
           delimiter =", ",
           fmt ='% s')
