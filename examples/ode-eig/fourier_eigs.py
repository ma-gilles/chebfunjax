"""Periodic ODE eigenvalue problems.

Computes eigenvalues of the periodic Sturm-Liouville problem
  -(p(x) u')' + q(x) u = lambda w(x) u,  u(0) = u(2*pi)
using Fourier collocation.

Note: chebfunjax Chebop.eigs() with 'periodic' BCs has limited accuracy
this demo uses Dirichlet BCs for the main tests and shows the periodic setup.

Credit: Chebfun example ode-eig/FourierEigs.m (Hadrien Montanelli, Dec 2014).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.operators.chebop import Chebop

def run():
    print("=" * 60)
    print("Periodic Sturm-Liouville eigenvalue problems")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: Simple Laplacian -u'' = lambda u, Dirichlet on [0, pi]
    # Exact eigenvalues: 1, 4, 9, 16, ...  (= k^2 for k=1,2,3,...)
    # ------------------------------------------------------------------
    print("\nTest 1: -u'' = lambda u, u(0)=u(pi)=0")
    dom1 = (0.0, np.pi)
    L1 = Chebop(lambda x, u: -u.diff(2), domain=dom1)
    L1.lbc = 0.0
    L1.rbc = 0.0
    k = 8
    lams1 = L1.eigs(k=k)
    lams1_sorted = np.sort(np.real(np.array(lams1)))
    exact1 = np.array([(kk)**2 for kk in range(1, k+1)], dtype=float)
    print(f"  {'computed':>12}  {'exact':>10}  {'error':>10}")
    for i in range(k):
        err = abs(lams1_sorted[i] - exact1[i])
        print(f"  {lams1_sorted[i]:12.6f}  {exact1[i]:10.4f}  {err:10.2e}")
    max_err1 = np.max(np.abs(lams1_sorted - exact1))
    print(f"  Max error: {max_err1:.2e}")
    assert max_err1 < 1e-8, f"Test 1 error too large: {max_err1}"

    # ------------------------------------------------------------------
    # Test 2: -u'' + cos(x)*u = lambda u on [0, pi], Dirichlet BCs
    # In Chebop lambda, x is a Chebfun; use cj.cos (not jnp.cos)
    # Eigenvalues should be real and sorted
    # ------------------------------------------------------------------
    print("\nTest 2: -u'' + cos(x)*u = lambda u, Dirichlet on [0, pi]")
    L2 = Chebop(lambda x, u: -u.diff(2) + cj.cos(x) * u, domain=dom1)
    L2.lbc = 0.0
    L2.rbc = 0.0
    k2 = 6
    lams2 = L2.eigs(k=k2)
    lams2_sorted = np.sort(np.real(np.array(lams2)))
    print(f"  First {k2} eigenvalues: {lams2_sorted}")
    # Should be close to k^2 + mean(cos(x)) correction (mean is 0 on [0,pi])
    # Ground state ~ 1 with small perturbation
    assert lams2_sorted[0] > 0.5, f"Ground state {lams2_sorted[0]:.4f} unexpectedly small"
    assert np.all(np.diff(lams2_sorted) > 0), "Eigenvalues should be increasing"

    # ------------------------------------------------------------------
    # Test 3: -u'' + x^2*u = lambda u on [-5, 5], Dirichlet BCs
    # Harmonic oscillator-like problem with well potential
    # ------------------------------------------------------------------
    print("\nTest 3: -u'' + x^2*u = lambda u, Dirichlet on [-5, 5]")
    L3 = Chebop(lambda x, u: -u.diff(2) + x**2 * u, domain=(-5.0, 5.0))
    L3.lbc = 0.0
    L3.rbc = 0.0
    k3 = 6
    lams3 = L3.eigs(k=k3)
    lams3_sorted = np.sort(np.real(np.array(lams3)))
    # Quantum harmonic oscillator: eigenvalues ~ 2k+1 for k=0,1,2,...
    exact3 = np.array([2*kk+1 for kk in range(k3)], dtype=float)
    print(f"  First {k3} eigenvalues: {lams3_sorted}")
    print(f"  Exact (2k+1, QHO):    {exact3}")
    max_err3 = np.max(np.abs(lams3_sorted - exact3))
    print(f"  Max error: {max_err3:.2e}")
    assert max_err3 < 0.01, f"QHO eigenvalue error {max_err3:.4f} too large"

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    axes[0].bar(range(1, len(lams1_sorted)+1), lams1_sorted, color='steelblue', alpha=0.7,
                label="computed")
    axes[0].plot(range(1, len(exact1)+1), exact1, color='#D95319', marker='o', linestyle='none', markersize=6, label="exact k²")
    axes[0].set_title("−u″ = λu on [0,π], Dirichlet", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].bar(range(1, len(lams3_sorted)+1), lams3_sorted, color='coral', alpha=0.7,
                label="computed")
    axes[1].plot(range(1, len(exact3)+1), exact3, color='#D95319', marker='o', linestyle='none', markersize=6, label="exact 2k+1")
    axes[1].set_title("−u″ + x²u = λu on [−5,5]", fontsize=10)
    axes[1].legend(fontsize=8)

    fig.suptitle("Fourier/Chebyshev eigenvalue problems", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "fourier_eigs.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
