# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:20:38 2026

@author: Hp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

V0, a = 10, 1
p = a*np.sqrt(2*V0)
x = np.linspace(-4, 4, 1000)

states = []

for parity in ["even", "odd"]:
    for j in range(5):
        lo = j*np.pi if parity == "even" else j*np.pi+np.pi/2
        hi = lo + np.pi/2
        lo, hi = max(lo, 1e-6), min(hi, p-1e-6)
        if lo >= hi:
            continue

        f = lambda z: (z*np.tan(z) if parity == "even"
                       else -z/np.tan(z)) - np.sqrt(p**2-z**2)
        grid = np.linspace(lo, hi, 200)

        for u, v in zip(grid[:-1], grid[1:]):
            if f(u)*f(v) < 0:
                z = brentq(f, u, v)
                E = -V0 + z**2/(2*a**2)
                states.append((E, z, parity))

for E, z, parity in sorted(states):
    k, kap = z/a, np.sqrt(-2*E)
    psi = np.where(abs(x)<a,
        np.cos(k*x) if parity=="even" else np.sin(k*x),
        (np.cos(z) if parity=="even" else
         np.sign(x)*np.sin(z))*np.exp(-kap*(abs(x)-a)))
    psi /= np.sqrt(np.trapezoid(psi**2, x))
    print(f"{parity}: E = {E:.4f}")
    plt.plot(x, psi, label=f"E={E:.2f}")
    plt.axvline(-a, ls="--", color="gray")
plt.axvline(a, ls="--", color="gray")
plt.xlabel("x")
plt.ylabel("Normalized wavefunction")
plt.legend()
plt.grid()
plt.show()