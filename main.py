"""
This file is the main function of this project
Name: Shangkun LI, Yukai CHEN
"""

import XY_Model_Grid
import numpy as np
import matplotlib.pyplot as plt
import math

# Fundamental parameetrs
size = [10]
overall_steps = 90000
interval = 10000
Jfactor = 1
length = 15
T = np.linspace(0.2, 2.5, length)

# Figure of e-T Settings
fig1, ax1 = plt.subplots()

# Figure of Cv-T Settings
fig2, ax2 = plt.subplots()

# Figure of m-T Settings
fig3, ax3 = plt.subplots()

# Figure of Cai-T Settings
fig4, ax4 = plt.subplots()

# Figure of Spin Settings
fig5, ax5 = plt.subplots()

# Simulation
for i in range(len(size)):
    e = []  # Energy per site
    c_v = []  # Heat per site
    m = []  # Magnetization per site
    Cai = []  # Spin Susceptibility
    print("Simulation ", "n = ", size[i], "begins.")
    # Generate grid
    g = XY_Model_Grid.Grid(size[i], Jfactor)

    for j in range(length):
        E_sum = 0  # the sum of #steps times of g.E_total()
        E_2_sum = 0

        E_total = 0  # the average of the E_total
        E_2_total = 0

        M_2_sum = 0  # the sum value of #steps times of g.M_total()**2
        M_sum_x = 0
        M_sum_y = 0

        M_total = 0  # the average of the M_total
        M_2_total = 0

        g.randomize()

        for step in range(overall_steps):
            # Using the Wolff Algorithm
            ClusterSize = g.ClusterFlip(T[j])
            if step % interval == 0:
                print("+ ", interval, "steps")
                print("total ", step, "steps")

            # Using the Metropolis Algorithm
            # ClusterSize = g.SingleFlip(T[j])
        steps = 10000

        for step in range(steps):
            # Using the Wolff Algorithm
            ClusterSize = g.ClusterFlip(T[j])

            # Using the Metropolis Algorithm
            # ClusterSize = g.SingleFlip(T[j])
            E_sum += abs(g.E_total())
            E_2_sum += g.E_total() ** 2
            M = g.M_total()
            M_sum_x += np.abs(np.real(M))
            M_sum_y += np.abs(np.imag(M))
            M_2_sum += np.abs(M) ** 2
            # 2023 12 9 凌晨 2:21 目前m 和cv是正确的

        E_total = E_sum / steps
        E_2_total = E_2_sum / steps

        M_total = math.sqrt(M_sum_x**2 + M_sum_y**2) / steps
        M_2_total = M_2_sum / steps

        e.append(-g.Jfactor * E_total / (g.size * g.size))
        c_v.append(abs(E_total**2 - E_2_total) / (g.size * g.size * T[j] * T[j]))
        Cai.append((M_2_total - M_total**2) / (T[j] * g.size * g.size))
        m.append(M_total / (g.size * g.size))
        X, Y = np.meshgrid(np.arange(0, g.size), np.arange(0, g.size))
        # U = np.zeros([g.size, g.size])
        # V = np.zeros([g.size, g.size])
        U = np.cos(g.phi)
        V = np.sin(g.phi)
        # ax5.quiver(X, Y, U, V)
        # ax5.set_xlabel("L_x")
        # ax5.set_ylabel("L_y")
        # ax5.set_title("T={}, #spins= 14*14".format(T[j]),loc='center')
        # plt.show()

    ax1.plot(T, e, "o-", label="n={}".format(size[i]))
    ax1.set_xlabel("T")
    ax1.set_ylabel("Energy per Site e")
    ax1.grid(color="#C0C0C0", linestyle="--")
    ax1.legend()

    ax2.plot(T, c_v, "o-", label="n={}".format(size[i]))
    ax2.set_xlabel("T")
    ax2.set_ylabel("Heat per Site c_v")
    ax2.grid(color="#C0C0C0", linestyle="--")
    ax2.legend()

    ax3.plot(T, m, "o-", label="n={}".format(size[i]))
    ax3.set_xlabel("T")
    ax3.set_ylabel("Magnetization per Site m")
    ax3.grid(color="#C0C0C0", linestyle="--")
    ax3.legend()

    # ax4.plot(T, Cai, "o-", label="n={}".format(size[i]))
    # ax4.set_xlabel("T")
    # ax4.set_ylabel("Spin Susceptibility Chi")
    # ax4.legend()
    # ax4.grid(color="#C0C0C0", linestyle="--")
    print("Simulation ", "n = ", size[i], "ends.")

plt.show()
