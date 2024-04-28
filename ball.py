"""
Module that creates a ball/container object.
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

def vdot(a, b):
    return np.dot(a, b)


def vsquare(a):
    return np.dot(a, a)


def mag(a):
    return np.linalg.norm(a)


class Ball:
    """
    Sets the mass, radius, position and velocity of a ball.
    Can also define a container (should use mass of container >> mass of ball).
    Creates a patch for container/ball on set of axes.
    """

    def __init__(self, m=0.0, rad=0.0, pos=[0.0, 0.0], vel=[0.0, 0.0], con=False):
        self._mass = m
        self._radius = rad
        self._position = np.array(pos)
        self._velocity = np.array(vel)
        self._container = con
        self._patch = plt.Circle([0, 0], 0, fc='None', ec='None')
        if self._container:
            self._patch = plt.Circle(self._position, self._radius, fc='None', ec='b')
        else:
            self._patch = plt.Circle(self._position, self._radius, fc='Red', ec='None')

    def __repr__(self):
        a = "Ball"
        if self._container:
            a = "Con"
        return "{}(m={},r={},pos{},vel{})".format(a, self._mass, self._radius, self._position, self._velocity)

    def __str__(self):
        return self.__repr__()

    def position(self):
        return self._position

    def velocity(self):
        return self._velocity

    def move(self, dt):
        self._position = self._position + self._velocity * dt
        self._patch.center = self._position

    def time_to_coll(self, other):
        delr = self._position - other._position
        delv = self._velocity - other._velocity
        # print("{} -> {} | {}".format(self, other, delv))
        # print("r={}".format(delr))
        # print("v={}".format(delv))
        # print("modr={}".format(vsquare(delr)))
        # print("modv={}".format(vsquare(delv)))
        # print("vdotr={}".format(vdot(delv, delr)))
        if self._container or other._container:
            R = self._radius - other._radius
        else:
            R = self._radius + other._radius
        # print("R={}".format(R))

        vr = vdot(delv, delr)
        v_square = vsquare(delv)
        r_square = vsquare(delr)
        D = (vr**2 - (v_square * (r_square - R**2)))
        dt = 1e9
        sols = []
        if D >= 0:
            sols = [(-vr - np.sqrt(D)) / v_square, (-vr + np.sqrt(D)) / v_square]

        if D < 0:
            dt = 1e9
        elif self._container or other._container:
            dt = max(sols)
        elif vr >= 0:
            dt = 1e9
        else:
            dt = min(sols)
        return 0.9999 * dt

    def collide(self, other):
        delr = self._position - other._position
        delv = self._velocity - other._velocity
        v1_prime = self._velocity - ((2 * other._mass) / (self._mass + other._mass)) * ((vdot(delv, delr)) / (vsquare(delr))) * (delr)
        v2_prime = other._velocity + ((2 * self._mass) / (self._mass + other._mass)) * ((vdot(delv, delr)) / (vsquare(delr))) * (delr)
        # return print("{} = {}, {} = {}".format("v1 after collision", v1_prime, "v2 after collision", v2_prime))
        # print(other)
        dp = 0
        if other._container:
            dp = self._mass * (mag(v1_prime - self._velocity))
        # print(dp)
        self._velocity, other._velocity = v1_prime, v2_prime
        return dp
