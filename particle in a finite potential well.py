# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:44:30 2026

@author: Hp
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

V0, a, X = 10, 1, 6
x = np.linspace(0, X, 500)

def shoot(E):
    def f(x, y):
        V = -V0 if x < a else 0
        return [y[1], 2*(V-E)*y[0]]
    return solve_ivp(f, [0, X], [1, 0],
                     t_eval=x).y

def R(E):
    y = shoot(E)
    return y[1, -1] + np.sqrt(-2*E)*y[0, -1]
es = np.linspace(-9.99, -0.01, 1000)
br = next((u, v) for u, v in zip(es[:-1], es[1:])
          if R(u)*R(v) < 0)

E = brentq(R, *br)
psi = shoot(E)[0]
psi /= np.sqrt(np.trapezoid(psi**2, x))

print("Ground-state energy =", E)
plt.plot(x, psi)
plt.axvline(a, ls="--", color="red")
plt.xlabel("x")
plt.ylabel("Wavefunction")
plt.grid()
plt.show()