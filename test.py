import numpy as np
import matplotlib.pyplot as plt


from rate_functions import RateFunctionWrapper as RF


def tau(energy, alpha, beta, gamma):
    return alpha * np.power(energy, - beta) + gamma

def scale(energy, mu, xi, nu, zeta):
    return - mu * np.power((xi - energies), nu) + zeta

def band_counts(energies, E_0, nu, xi):
    return np.where(y == 0, 0, x/y)

energies = np.geomspace(25, 600, 20)
tau_array = tau(energies, 10, 0.1, 1)
scale_array = scale(energies, 3e-6, 300, 2, 1)
print(energies)
print(tau_array)
print(scale_array)
plt.plot(energies, scale_array)
plt.show()


dt = np.ones(999) * 0.1
col = ['r', 'orange', 'g', 'b']
for i in range(4):
    y = RF.one_FRED_rate(dt, 0, 0, scale_array[i], 5, tau_array[i], 0.6)
    plt.plot(np.arange(1000) * 0.1, y, color = col[i])

plt.show()
