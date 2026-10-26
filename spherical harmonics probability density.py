# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:18:47 2026

@author: Hp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y

# Quantum numbers
states = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]

theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 150)
TH, PH = np.meshgrid(theta, phi)

fig = plt.figure(figsize=(12, 8))

for i, (l, m) in enumerate(states):
    Y = sph_harm_y(l, m, TH, PH)
    P = np.abs(Y)**2
    # Radius proportional to probability density
    r = P / P.max()

    X = r * np.sin(TH) * np.cos(PH)
    Yc = r * np.sin(TH) * np.sin(PH)
    Z = r * np.cos(TH)

    ax = fig.add_subplot(2, 3, i+1, projection='3d')
    ax.plot_surface(X, Yc, Z, cmap='viridis', linewidth=0)
    ax.set_title(f"l={l}, m={m}")
    ax.set_axis_off()

plt.tight_layout()
plt.show()