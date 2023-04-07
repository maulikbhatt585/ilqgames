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
# Proximity cost for state spaces that are Cartesian products of individual
# systems' state spaces. Penalizes
#      ``` sum_{i \ne j} min(distance(i, j) - max_distance, 0)^2 ```
# for all players i, j.
#
################################################################################

import torch

from cost import Cost

class ProductStateProximityCost(Cost):
    def __init__(self, position_indices, max_distance, coef, name=""):
        """
        Initialize with dimension to add cost to and threshold BELOW which
        to impose quadratic cost.

        :param position_indices: list of index tuples corresponding to (x, y)
        :type position_indices: [(uint, uint)]
        :param max_distance: maximum value of distance to penalize
        :type max_distance: float
        """
        self._position_indices = position_indices
        self._max_distance = max_distance
        self._num_players = len(position_indices)
        self.c = coef
        super(ProductStateProximityCost, self).__init__(name)

    def __call__(self, x, k=0):
        """
        Evaluate this cost function on the given state.
        NOTE: `x` should be a PyTorch tensor with `requires_grad` set `True`.
        NOTE: `x` should be a column vector.

        :param x: concatenated state vector of all systems
        :type x: torch.Tensor
        :return: scalar value of cost
        :rtype: torch.Tensor
        """
        total_cost = torch.zeros(1, 1, requires_grad=True).double()

        for ii in range(self._num_players):
            xi_idx, yi_idx = self._position_indices[ii]

            for jj in range(self._num_players):
                if ii == jj:
                    continue

                # Compute relative distance.
                xj_idx, yj_idx = self._position_indices[jj]
                dx = x[xi_idx, 0] - x[xj_idx, 0]
                dy = x[yi_idx, 0] - x[yj_idx, 0]
                relative_distance = torch.sqrt(dx*dx + dy*dy)

                total_cost += self.c*min(
                    relative_distance - self._max_distance, 0.0)**2

        return total_cost

class ProductStateProximityCost_Potential(Cost):
    def __init__(self, position_indices, max_distance, name=""):
        """
        Initialize with dimension to add cost to and threshold BELOW which
        to impose quadratic cost.

        :param position_indices: list of index tuples corresponding to (x, y)
        :type position_indices: [(uint, uint)]
        :param max_distance: maximum value of distance to penalize
        :type max_distance: float
        """
        self._position_indices = position_indices
        self._max_distance = max_distance
        self._num_players = len(position_indices)
        super(ProductStateProximityCost_Potential, self).__init__(name)

    def __call__(self, x, k=0):
        """
        Evaluate this cost function on the given state.
        NOTE: `x` should be a PyTorch tensor with `requires_grad` set `True`.
        NOTE: `x` should be a column vector.

        :param x: concatenated state vector of all systems
        :type x: torch.Tensor
        :return: scalar value of cost
        :rtype: torch.Tensor
        """
        total_cost = torch.zeros(1, 1, requires_grad=True).double()

        for ii in range(self._num_players):
            xi_idx, yi_idx = self._position_indices[ii]

            for jj in range(self._num_players):
                if ii >= jj:
                    continue

                # Compute relative distance.
                xj_idx, yj_idx = self._position_indices[jj]
                dx = x[xi_idx, 0] - x[xj_idx, 0]
                dy = x[yi_idx, 0] - x[yj_idx, 0]
                relative_distance = torch.sqrt(dx*dx + dy*dy)

                total_cost += min(
                    relative_distance - self._max_distance, 0.0)**2

        return total_cost

class ProximityCost_1(Cost):
    def __init__(self, player_indices, other_indices, max_distance, coef, name=""):
        """
        Initialize with dimension to add cost to and threshold BELOW which
        to impose quadratic cost.

        :param position_indices: list of index tuples corresponding to (x, y)
        :type position_indices: [(uint, uint)]
        :param max_distance: maximum value of distance to penalize
        :type max_distance: float
        """
        self.player_indices = player_indices
        self._position_indices = other_indices
        self._max_distance = max_distance
        self._num_players = len(other_indices)
        self.c = coef
        super(ProximityCost_1, self).__init__(name)

    def __call__(self, x, k=0):
        """
        Evaluate this cost function on the given state.
        NOTE: `x` should be a PyTorch tensor with `requires_grad` set `True`.
        NOTE: `x` should be a column vector.

        :param x: concatenated state vector of all systems
        :type x: torch.Tensor
        :return: scalar value of cost
        :rtype: torch.Tensor
        """
        total_cost = torch.zeros(1, 1, requires_grad=True).double()

        xi_idx, yi_idx = self.player_indices

        for jj in range(self._num_players):
            # Compute relative distance.
            xj_idx, yj_idx = self._position_indices[jj]
            dx = x[xi_idx, 0] - x[xj_idx, 0]
            dy = x[yi_idx, 0] - x[yj_idx, 0]
            relative_distance = torch.sqrt(dx*dx + dy*dy)

            total_cost += self.c*min(
                relative_distance - self._max_distance, 0.0)**2
        return total_cost

class ProximityCost_Potential_1(Cost):
    def __init__(self, position_indices, max_distance, coefs, name=""):
        """
        Initialize with dimension to add cost to and threshold BELOW which
        to impose quadratic cost.

        :param position_indices: list of index tuples corresponding to (x, y)
        :type position_indices: [(uint, uint)]
        :param max_distance: maximum value of distance to penalize
        :type max_distance: float
        """
        self._position_indices = position_indices
        self._max_distance = max_distance
        self._num_players = len(position_indices)
        self.cs = coefs
        super(ProximityCost_Potential_1, self).__init__(name)

    def __call__(self, x, k=0):
        """
        Evaluate this cost function on the given state.
        NOTE: `x` should be a PyTorch tensor with `requires_grad` set `True`.
        NOTE: `x` should be a column vector.

        :param x: concatenated state vector of all systems
        :type x: torch.Tensor
        :return: scalar value of cost
        :rtype: torch.Tensor
        """
        total_cost = torch.zeros(1, 1, requires_grad=True).double()

        c = 1
        for coef in self.cs:
            c = c*coef

        for ii in range(self._num_players-1):
            xi_idx, yi_idx = self._position_indices[ii]

            for jj in range(ii+1,self._num_players):
                # Compute relative distance.
                xj_idx, yj_idx = self._position_indices[jj]
                dx = x[xi_idx, 0] - x[xj_idx, 0]
                dy = x[yi_idx, 0] - x[yj_idx, 0]
                relative_distance = torch.sqrt(dx*dx + dy*dy)

                total_cost += min(
                    relative_distance - self._max_distance, 0.0)**2
        return c*total_cost


class ProximityCost_Potential_2(Cost):
    def __init__(self, position_indices, max_distance, coefs, name=""):
        """
        Initialize with dimension to add cost to and threshold BELOW which
        to impose quadratic cost.

        :param position_indices: list of index tuples corresponding to (x, y)
        :type position_indices: [(uint, uint)]
        :param max_distance: maximum value of distance to penalize
        :type max_distance: float
        """
        self._position_indices = position_indices
        self._max_distance = max_distance
        self._num_players = len(position_indices)
        self.cs = coefs
        super(ProximityCost_Potential_1, self).__init__(name)

    def __call__(self, x, k=0):
        """
        Evaluate this cost function on the given state.
        NOTE: `x` should be a PyTorch tensor with `requires_grad` set `True`.
        NOTE: `x` should be a column vector.

        :param x: concatenated state vector of all systems
        :type x: torch.Tensor
        :return: scalar value of cost
        :rtype: torch.Tensor
        """
        total_cost = torch.zeros(1, 1, requires_grad=True).double()

        for ii in range(self._num_players-1):
            xi_idx, yi_idx = self._position_indices[ii]

            for jj in range(ii+1,self._num_players):
                # Compute relative distance.
                xj_idx, yj_idx = self._position_indices[jj]
                dx = x[xi_idx, 0] - x[xj_idx, 0]
                dy = x[yi_idx, 0] - x[yj_idx, 0]
                relative_distance = torch.sqrt(dx*dx + dy*dy)

                total_cost += self.cs[ii]*self.cs[jj]*min(
                    relative_distance - self._max_distance, 0.0)**2
        return total_cost
