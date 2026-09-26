import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# Constants
hbar = 1
m = 1
L = 1
N = 200

# Grid
dx = L / (N + 1)
x = np.linspace(dx, L-dx, N)

# Hamiltonian matrix
main = np.ones(N) * (hbar**2 / (m * dx**2))
off = np.ones(N-1) * (-hbar**2 / (2*m*dx**2))

# Eigenvalues and eigenvectors
E, psi = eigh_tridiagonal(main, off)

# First 3 energy levels
print("Numerical energies:")
for n in range(3):
    print(f"E{n+1} = {E[n]:.5f}")

# Plot wavefunctions
for n in range(3):
    psi[:, n] /= np.sqrt(np.trapezoid(psi[:, n]**2, x))
    plt.plot(x, psi[:, n], label=f"n={n+1}")

plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.title("Particle in a Box - Finite Difference Method")
plt.legend()
plt.grid()
plt.show()
