import scipy as sp
from scipy.optimize import curve_fit
from simulation import *

v = np.loadtxt('docs/vel_300K.txt')


def get_max_velocity():
    E_max = get_max_energy(300)
    v_max = np.sqrt(np.sqrt(2) * E_max / m_H)
    return v_max


v_max = get_max_velocity()

print(v_max)

v_hist, edges = np.histogram(v, bins=20, density=True)
centers = 0.5 * (edges[1:] + edges[:-1])
sigma = np.sqrt(k_B * 300 / m_H)
print(sigma)
popt, pcov = curve_fit(maxwell_distribution, centers, v_hist, p0=sigma)
x = np.linspace(0, centers[-1], 100000)

plt.hist(v, density=1, bins=100, color='lightcoral')
plt.plot(x, maxwell_distribution(x, *popt), '--', color='0.27')
plt.xlabel(r"Velocity [$ms^{-1}$]")
plt.xticks([0, np.sqrt(1 / 12) * sigma, np.sqrt(1 / 9) * sigma, np.sqrt(1 / 6) * sigma], ['0', r'$v_{p}$', r'$\bar v$', r'$v_{rms}$'])
plt.xlim([0, v_max])
plt.legend(['Maxwell-Boltzmann curve', 'Velocity distribution'], loc='upper right')
plt.show()
