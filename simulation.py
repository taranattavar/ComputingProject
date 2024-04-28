"""
Module to simulate collisions between H-atoms in a closed container (with radius = 10*a0) and obtain thermodynamic quantities along with position and velocity distributions.
"""

import numpy as np
import sys
import tkinter
import matplotlib.pyplot as plt
import random as rn
import ball
from tqdm import tqdm, trange
from ball import Ball
from scipy.optimize import curve_fit

np.random.seed(0)
a0 = 5.29177e-11  # in m
m_H = 1.67e-27  # in kg
k_B = 1.38064852e-23  # in m^2 kg s^{-2} K^{-1}


def maxwell_distribution(v, sigma):
    f_v = v * np.exp(- v**2 / (2 * sigma**2)) / (sigma**2)
    return f_v


def overlaps(p1, pos_array):
    for p in pos_array:
        r = ball.mag(np.asarray(p) - p1)
        if r <= 2 * a0:
            return True
    return False


def get_max_energy(T):
    E_k = k_B * T  # 0.5*m*v^2 = n*k_B*T/2 (n = degrees of freedom)
    return E_k


def get_random_pos():
    mag = rn.uniform(0, 9 * a0)
    ang = rn.uniform(-np.pi, np.pi)
    x = mag * np.cos(ang)
    y = mag * np.sin(ang)
    # print(x, y)
    return x, y


def get_random_velocities(m, E):
    vel = []
    for i in range(len(E)):
        v_mag = np.sqrt(3 * E[i] / m)
        ang = rn.uniform(-np.pi, np.pi)
        vx = v_mag * np.cos(ang)
        vy = v_mag * np.sin(ang)
        vel.append([vx, vy])
    return vel


def get_next_collision(balls):
    min_time = 1e9
    min_balls = [balls[0], balls[1]]
    # print("__________ \n")
    # print("Getting next collision..." + "\n" * 2)
    for i in range(len(balls) - 1):
        ball_1 = balls[i]
        for j in range(i + 1, len(balls)):
            ball_2 = balls[j]
            time = ball_1.time_to_coll(ball_2)
            if time < min_time:
                if not np.isreal(time):
                    pass
                else:
                    min_balls = [ball_1, ball_2]
                    min_time = time
    # print("{} -> {}".format(min_balls[0], min_balls[1]))
    # print("{}={}".format("min time", min_time))
    return min_time, min_balls


def extrapolate(x, m, c):
    y = m * x + c
    return y


class Simulation:
    """
    Graphically simulates collisions between 2(or more) balls and/or ball(s) and the container.
    Plots a histogram of the relative position between the balls
    """

    def __init__(self, num, T=300, p=0, t=0, con=Ball(m=sys.maxsize, rad=10 * a0, con=True)):
        self._container = con
        self._balls = []
        self.num = num
        self.T_intial = T
        self._dp = p
        self._time = t  # in K
        positions = [get_random_pos()]
        max_tries = 10
        for _ in range(num - 1):
            tries = 1
            while tries < max_tries:
                p1 = get_random_pos()
                if overlaps(p1, positions):
                    tries += 1
                else:
                    positions.append(p1)
                    break
            if tries == max_tries:
                raise ValueError("Maximum tries reached to find positions. Try lower number of balls.")

        E_max = get_max_energy(self.T_intial)
        self._v_max = np.sqrt(np.sqrt(2) * E_max / m_H)
        energy = []
        for i in range(num - 1):
            E = rn.uniform(0, E_max)
            energy.append(E)
            E_max -= E
        energy.append(E_max)

        velocities = get_random_velocities(m_H, energy)
        # print(self._v_max)

        for i in range(num):
            r = positions[i]
            v = velocities[i]
            b = Ball(m=m_H, rad=a0, pos=r, vel=v)
            self._balls.append(b)

        p = []
        E = []
        for i in range(num):
            b = self._balls[i]
            mom = b._mass * ball.mag(b._velocity)
            energy = 0.5 * b._mass * ball.vsquare(b._velocity)
            p.append(mom)
            E.append(energy)
        self._p_intial = sum(p)
        self._E_intial = sum(E)

    def next_coll(self):
        balls = self._balls + [self._container]
        min_time, min_balls = get_next_collision(balls)
        # print("Moving system..." + "\n" * 2)
        for i in range(self.num):
            balls[i].move(min_time)
        # print("Colliding..." + "\n" * 2)

        self._time += min_time
        self._dp += min_balls[0].collide(min_balls[1])

    def momentum_conservation(self):
        mom = []
        balls = self._balls + [self._container]
        for i in range(len(balls)):
            b = balls[i]
            p = (b._mass * ball.mag(b._velocity))
            mom.append(p)
        return self._p_intial - sum(mom)

    def energy_conservation(self):
        total_E = []
        balls = self._balls + [self._container]
        for i in range(len(balls)):
            b = balls[i]
            E = 0.5 * b._mass * ball.vsquare(b._velocity)
            total_E.append(E)
        return self._E_intial - sum(total_E)

    def get_relative_position(self):
        r = []
        for i in range(self.num - 1):
            b1 = self._balls[i]
            for j in range(i + 1, self.num):
                b2 = self._balls[j]
                pos = b1._position - b2._position
                r.append(ball.mag(pos))
        return r

    def get_velocity(self):
        v = []
        for i in range(self.num):
            b = self._balls[i]
            vel = ball.mag(b._velocity)
            v.append(vel)
        return v

    def get_position_from_origin(self):
        r = []
        for i in range(self.num):
            b = self._balls[i]
            pos = b._position
            r.append(ball.mag(pos))
        return r

    def get_pressure(self):
        # Force per unit length (2D)
        c = self._container
        return self._dp / (2 * self._time * (c._radius - a0))

    def run(self, num_frames, animate=False):
        pos = []
        r = []
        v = []
        p = []
        E = []
        P = []
        f1, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=[20, 10])
        plt.subplots_adjust(wspace=0.35, hspace=0.35)
        ax1.set_xlim([-12.5 * a0, 12.5 * a0])
        ax1.set_ylim([-12.5 * a0, 12.5 * a0])
        ax1.add_artist(self._container._patch)
        ax1.set_xlabel(r"$x$ $[m]$")
        ax1.set_ylabel(r"$y$ $[m]$")
        ax7 = ax2.twinx()
        for i in range(self.num):
            ax1.add_patch(self._balls[i]._patch)
        if animate:
            with tqdm(total=num_frames) as pbar:
                for frame in range(num_frames):
                    self.next_coll()
                    p.append(self.momentum_conservation())
                    E.append(self.energy_conservation())
                    P.append(self.get_pressure())
                    pos.extend(self.get_relative_position())
                    r.extend(self.get_position_from_origin())
                    v.extend(self.get_velocity())
                    ax2.cla()
                    ax3.cla()
                    ax4.cla()
                    ax5.cla()
                    ax6.cla()
                    ax2.plot(p, color='darkorange')
                    ax7.plot(E, color='m')
                    ax3.plot(P, color='crimson')
                    ax4.hist(pos, density=1, bins=16, color='lightgreen')
                    ax5.hist(r, density=1, bins=9, color='skyblue')
                    ax6.hist(v, density=1, bins=20, color='lightcoral')
                    ax2.grid()
                    ax3.grid()
                    ax2.set_xlabel('Event')
                    ax3.set_xlabel('Event')
                    ax2.set_ylabel(r'Momentum $[ms^{-1}]$')
                    ax7.set_ylabel(r'Energy $[J]$')
                    ax3.set_ylabel(r'Pressure $[Nm^{-1}]$')
                    ax2.legend([r'$\Delta \sum_{i} p_i$'], loc='upper right')
                    ax7.legend([r'$\Delta E$'], loc='upper left')
                    ax3.legend([r'$P$'])
                    ax4.set_xticks([0, 2 * a0, 9 * a0, 18 * a0])
                    ax4.set_xticklabels(['0', r'$2R_b$', r'$R_c - R_b$', r'$2(R_c - R_b)$'])
                    ax4.set_xlim([0, 20 * a0])
                    ax4.set_xlabel(r"Distance $[m]$")
                    ax5.set_xlabel(r"Distance $[m]$")
                    ax5.set_xlim([0, 10 * a0])
                    ax5.set_xticks([0, 9 * a0])
                    ax5.set_xticklabels(['0', r'$(R_c - R_b)$'])
                    ax4.legend(['Relative position'], loc='upper right')
                    ax5.legend(['Position from origin'], loc='upper left')
                    ax6.set_xlabel(r"Velocity [$ms^{-1}$]")
                    ax6.set_xticks([0, 0.5 * self._v_max, self._v_max])
                    ax6.set_xlim([0, 1.1 * self._v_max])
                    ax6.set_xticklabels(['0', r'$0.5v_{max}$', r'$v_{max}$'])
                    ax6.legend(['Velocity distribution'], loc='upper right')
                    pbar.update(1)
                    if animate:
                        plt.pause(0.25)
        if not animate:
            for frame in range(num_frames):
                self.next_coll()
                p.append(self.momentum_conservation())
                E.append(self.energy_conservation())
                P.append(self.get_pressure())
                pos.extend(self.get_relative_position())
                r.extend(self.get_position_from_origin())
                v.extend(self.get_velocity())
            P_mean = np.mean(P[99:])
            # np.savetxt('vel_300K.txt', v)
            print(P_mean)
            v_hist, edges = np.histogram(v, bins=20, density=True)
            # print(v_hist)
            centers = 0.5 * (edges[1:] + edges[:-1])
            sigma = np.sqrt(k_B * self.T_intial / m_H)
            popt, pcov = curve_fit(maxwell_distribution, centers, v_hist, p0=sigma)
            x = np.linspace(0, centers[-1], 100000)
            ax4.plot(p, color='darkorange')
            ax5.plot(E, color='m')
            ax6.plot(P, color='crimson')
            ax2.hist(pos, density=1, bins=16, color='lightgreen')
            ax3.hist(r, density=1, bins=9, color='skyblue')
            ax4.grid()
            ax5.grid()
            ax6.grid()
            ax4.set_xlabel('Event')
            ax5.set_xlabel('Event')
            ax6.set_xlabel('Event')
            ax4.set_ylabel(r'Momentum $[ms^{-1}]$')
            ax5.set_ylabel(r'Energy $[J]$')
            ax6.set_ylabel(r'Pressure $[Nm^{-1}]$')
            ax4.legend([r'$\Delta \sum_{i} p_i$'], loc='upper right')
            ax5.legend([r'$\Delta E$'], loc='upper left')
            ax6.legend([r'$P$'])
            ax2.set_xticks([0, 2 * a0, 9 * a0, 18 * a0])
            ax2.set_xticklabels(['0', r'$2R_b$', r'$R_c - R_b$', r'$2(R_c - R_b)$'])
            ax2.set_xlim([0, 20 * a0])
            ax2.set_xlabel(r"Distance $[m]$")
            ax3.set_xlabel(r"Distance $[m]$")
            ax3.set_xlim([0, 10 * a0])
            ax3.set_xticks([0, 9 * a0])
            ax3.set_xticklabels(['0', r'$(R_c - R_b)$'])
            ax2.legend(['Relative position'], loc='upper right')
            ax3.legend(['Position from origin'], loc='upper left')
            plt.show()
            plt.hist(v, density=1, bins=20, color='lightcoral')
            plt.plot(x, maxwell_distribution(x, *popt), '--', color='0.27')
            plt.xlabel(r"Velocity [$ms^{-1}$]")
            plt.xticks([0, 0.5 * self._v_max, self._v_max], ['0', r'$0.5v_{max}$', r'$v_{max}$'])
            plt.xlim([0, self._v_max])
            plt.legend(['Maxwell-Boltzmann curve', 'Velocity distribution'], loc='upper right')
            plt.show()
            # ax2.set_xlim([0, 5000])
            # ax7.set_xlim([0, 5000])
        if animate:
            plt.show()
