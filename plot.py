"""
Loads the data for pressure vs temperature
"""
import scipy as sp
import matplotlib.pyplot as plt
from simulation import *


A = np.pi * (10 * a0)**2
N = 20


def linear(x, m, c):
    y = m * x + c
    return y


P, T = np.loadtxt('docs/PvsT.csv', skiprows=1, delimiter=',', unpack=True)

lfit, lcov = np.polyfit(T, P, deg=1, cov=True)
print(lfit)

a = - lfit[1] * A**2 / N**2
# print(A, k_B, N, lfit[0])
# exit()
b = A / N - k_B / lfit[0]
m = N * k_B / (A - b * N)
# print(m)
# exit()
print("a = {}, b = {}".format(a, b))
err = np.sqrt(np.diag(lcov))
print(err)
del_a = err[1] / lfit[1] * a
del_b = err[0] / lfit[0] * b
print(del_a)
print(del_b)

# exit()
x = np.linnpace(0, 500, 1000000)

plt.scatter(T, P, marker='x', color='r')
plt.plot(x, lfit[0] * x + lfit[1], 'k--')
plt.xlim([0, 500])
plt.ylim([0, 0.075])
plt.ylabel(r'$P$ $[Nm^{-1}]$')
plt.xlabel(r'$T$ $[K]$')
plt.legend([r'Linear fit', r'Scatter plot of $P$ vs $T$'], loc='upper left')
plt.grid()
plt.show()
