"""Laplacian eigenvalues (Dirichlet/Neumann).

Computes eigenvalues of -u'' = lambda*u with various boundary conditions,
comparing with exact values n^2*pi^2 / L^2.

Credit: Chebfun example ode-eig/DirLap.m.
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Laplacian eigenvalues")
    print("=" * 60)

    # --- Dirichlet on [0, pi] ----------------------------------------
    # -u'' = lambda*u, u(0) = u(pi) = 0
    # Eigenvalues: lambda_n = n^2, n = 1, 2, 3, ...
    pi = float(jnp.pi)
    N = Chebop(lambda x, u: -u.diff(2), domain=(0.0, pi))
    N.lbc = 0.0
    N.rbc = 0.0
    k = 6
    lam = N.eigs(k=k)
    lam_real = np.sort(np.real(np.array(lam)))
    exact = np.array([n**2 for n in range(1, k+1)], dtype=float)
    print(f"\nDirichlet on [0, pi]: eigenvalues (exact: n^2)")
    print(f"  {'n':>4}  {'computed':>16}  {'exact':>8}  {'error':>10}")
    for i in range(k):
        err = abs(lam_real[i] - exact[i])
        print(f"  {i+1:>4}  {lam_real[i]:>16.10f}  {exact[i]:>8.1f}  {err:.2e}")
    max_err = np.max(np.abs(lam_real - exact))
    print(f"  Max error: {max_err:.2e}")
    assert max_err < 1e-6, f"Eigenvalue error too large: {max_err}"

    # --- Neumann on [0, pi] ------------------------------------------
    # -u'' = lambda*u, u'(0) = u'(pi) = 0
    # Eigenvalues: lambda_n = n^2, n = 0, 1, 2, ...
    N2 = Chebop(lambda x, u: -u.diff(2), domain=(0.0, pi))
    # Neumann BC: derivative = 0 at endpoints
    N2.lbc = lambda u: u.diff()
    N2.rbc = lambda u: u.diff()
    lam2 = N2.eigs(k=k)
    lam2_real = np.sort(np.real(np.array(lam2)))
    exact2 = np.array([n**2 for n in range(0, k)], dtype=float)
    print(f"\nNeumann on [0, pi]: eigenvalues (exact: n^2, n=0,1,...)")
    for i in range(k):
        err2 = abs(lam2_real[i] - exact2[i])
        print(f"  {i:>4}  {lam2_real[i]:>16.10f}  {exact2[i]:>8.1f}  {err2:.2e}")
    max_err2 = np.max(np.abs(lam2_real - exact2))
    assert max_err2 < 1e-6, f"Neumann eigenvalue error: {max_err2}"

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
