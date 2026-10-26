# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:33:25 2026

@author: Hp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

L = 1.0
x = np.linspace(0, L, 500)

def shoot(E):
    def schrodinger(x, y):
        return [y[1], -2*E*y[0]]

    sol = solve_ivp(schrodinger, [0, L], [0, 1], t_eval=x)
    return sol.y[0]

def f(E):
    return shoot(E)[-1]
E = [brentq(f, a, b) for a, b in
     [(4, 6), (18, 21), (43, 46)]]

for n, En in enumerate(E, 1):
    print(f"n={n}, Numerical={En:.4f}, "
          f"Exact={n**2*np.pi**2/2:.4f}")

    psi = shoot(En)
    psi /= np.sqrt(np.trapezoid(psi**2, x))
    plt.plot(x, psi, label=f"Numerical n={n}")
    plt.plot(x, np.sqrt(2)*np.sin(n*np.pi*x), "--")

plt.xlabel("Position x")
plt.ylabel("Wavefunction ψ(x)")
plt.legend()
plt.grid()
plt.show()