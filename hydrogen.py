# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:43:52 2026

@author: Hp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# Radial coordinate
r = np.linspace(0.01, 30, 1000)
dr = r[1] - r[0]

# Hydrogen potential: V = -e^2/r
# In atomic units: e = 1
V = -1/r

# Hamiltonian for l = 0
H_diag = 1/dr**2 + V
H_off = -0.5/dr**2 * np.ones(len(r)-1)

# Find first two eigenvalues and eigenfunctions
E, psi = eigh_tridiagonal(
    H_diag, H_off,
    select='i', select_range=(0, 1))
                              
# Energy in eV
E_eV = E * 27.2114

print("Ground state energy =", E_eV[0], "eV")
print("First excited energy =", E_eV[1], "eV")

# Plot wave functions
plt.plot(r, psi[:,0], label="1s")
plt.plot(r, psi[:,1], label="2s")

plt.axhline(0, color='black')
plt.xlabel("r / a₀")
plt.ylabel("Wave function")
plt.title("Hydrogen Atom: s-wave")
plt.legend()
plt.grid()
plt.show()