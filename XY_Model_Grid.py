"""
This file defines XY_Model Grid Class
Name: Shangkun LI, Yukai CHEN
"""

import numpy as np
import math
from math import cos, sin


class Grid(object):
    """
    Grid is a 2-D, periodic-boundaried and square canvas consisting of spins.
    We use a angle to represent the spin of each site.
    Hence, the spin of a site is (cos(phi),sin(phi))
    """

    def __init__(self, size, Jfactor):
        self.size = size
        self.Jfactor = Jfactor
        self.phi = np.zeros([size, size])

    def randomize(self):
        self.phi = np.random.uniform(0, 2 * math.pi, [self.size, self.size])

    # We define the following functions to find the neighbours that satisfy the periodic boundary.
    def left(self, x, y):
        if x < 0.5:
            return [self.size - 1, y]
        else:
            return [x - 1, y]

    def right(self, x, y):
        if x > self.size - 1.5:
            return [0, y]
        else:
            return [x + 1, y]

    def up(self, x, y):
        if y < 0.5:
            return [x, self.size - 1]
        else:
            return [x, y - 1]

    def down(self, x, y):
        if y > self.size - 1.5:
            return [x, 0]
        else:
            return [x, y + 1]

    ############################################################

    # Define some basic results
    # single energy of a site
    def single_E(self, x, y):
        [left_x, left_y] = self.left(x, y)
        [right_x, right_y] = self.right(x, y)
        [up_x, up_y] = self.up(x, y)
        [down_x, down_y] = self.down(x, y)
        single_energy = -self.Jfactor * (
            cos(self.phi[left_x, left_y] - self.phi[x, y])
            + cos(self.phi[right_x, right_y] - self.phi[x, y])
            + cos(self.phi[up_x, up_y] - self.phi[x, y])
            + cos(self.phi[down_x, down_y] - self.phi[x, y])
        )
        return single_energy / 2

    # total energy of the grid
    def E_total(self):
        total_energy = 0
        for x in range(0, self.size):
            for y in range(0, self.size):
                total_energy += self.single_E(x, y)
        return total_energy

    # average energy per site
    def E_per_site(self):
        return self.total_E() / (self.size * self.size)

    # the change of energy in wolff algorithm
    def delta_E(self, x1, y1, x2, y2, phi):
        phi1 = self.phi[x1, y1]
        phi2 = self.phi[x2, y2]
        return -self.Jfactor * (-cos(phi1 - phi2) + cos(phi1 + phi2 - 2 * phi))

    def M_total(self):
        total_mag_x = 0
        total_mag_y = 0
        for x in range(0, self.size):
            for y in range(0, self.size):
                total_mag_x += cos(self.phi[x, y])
                total_mag_y += sin(self.phi[x, y])
        return total_mag_x + total_mag_y * 1j

    # average magnetization per site
    def avr_single_M(self):
        return self.total_M() / (self.size * self.size)

    ###########################
    # Cluster Filp (Wolff Algorithm)
    def ClusterFlip(self, temperature):
        # Randomly pick a seed spin
        x = np.random.randint(0, self.size)
        y = np.random.randint(0, self.size)

        # stack is the list of spins in cluster
        random_phi = np.random.uniform(0, math.pi)
        stack = [[x, y]]
        lable = np.ones([self.size, self.size], int)
        lable[x, y] = 0

        while len(stack) > 0:
            # While stack is not empty, pop and flip a spin
            [current_x, current_y] = stack.pop()
            origin = self.phi[current_x, current_y]
            self.phi[current_x, current_y] = (
                2 * random_phi - origin
            )  # flip the seed spin first

            # Append neighbor spins
            neighbour = [
                self.left(current_x, current_y),
                self.right(current_x, current_y),
                self.up(current_x, current_y),
                self.down(current_x, current_y),
            ]

            for x_n, y_n in neighbour:
                dE = self.delta_E(x_n, y_n, current_x, current_y, random_phi)
                if (
                    dE < 0
                    and lable[x_n, y_n]
                    and (np.random.rand() < 1 - math.exp(dE / temperature))
                ):
                    stack.append([x_n, y_n])
                    lable[x_n, y_n] = 0
        # Return cluster size
        return self.size * self.size - sum(sum(lable))

    # Single flip (Metropolis Algorithm)
    def SingleFlip(self, temperature):
        # Randomly pick a spin to flip

        x = np.random.randint(0, self.size)
        y = np.random.randint(0, self.size)

        origin = self.phi[x, y]
        origin_E = self.E_total()
        self.phi[x, y] = np.random.uniform(0, 2 * math.pi)

        # Metropolis acceptance rate
        delta_E = -origin_E + self.E_total()

        if (delta_E > 0) and (np.random.rand() > np.exp(-delta_E / temperature)):
            self.phi[x, y] = origin

        # Return cluster size
        return 1
