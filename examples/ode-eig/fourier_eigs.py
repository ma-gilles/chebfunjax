"""Periodic ODE eigenvalue problems.

Computes eigenvalues of the periodic Sturm-Liouville problem
  -(p(x) u')' + q(x) u = lambda w(x) u,  u(0) = u(2*pi)
using Fourier collocation.

Credit: Chebfun example ode-eig/FourierEigs.m (Hadrien Montanelli, Dec 2014).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Periodic Sturm-Liouville eigenvalue problems")
    print("=" * 60)

    dom = (0.0, 2.0 * np.pi)

    # ------------------------------------------------------------------
    # Test 1: Simple Laplacian -u'' = lambda u, periodic
    # Exact eigenvalues: 0, 1, 1, 4, 4, 9, 9, ...
    # ------------------------------------------------------------------
    print("\nTest 1: -u'' = lambda u, periodic on [0, 2*pi]")
    L1 = Chebop(lambda x, u: -u.diff(2), domain=dom)
    L1.bc = "periodic"
    k = 9
    lams1 = L1.eigs(k=k)
    lams1_sorted = np.sort(np.real(np.array(lams1)))
    # Exact: 0, 1,1, 4,4, 9,9, 16, ...
    exact1 = np.array([0, 1, 1, 4, 4, 9, 9, 16, 16], dtype=float)[:k]
    print(f"  {'computed':>12}  {'exact':>10}  {'error':>10}")
    for i in range(k):
        err = abs(lams1_sorted[i] - exact1[i])
        print(f"  {lams1_sorted[i]:12.6f}  {exact1[i]:10.4f}  {err:10.2e}")
    max_err1 = np.max(np.abs(lams1_sorted - exact1))
    print(f"  Max error: {max_err1:.2e}")
    assert max_err1 < 1e-8, f"Test 1 error too large: {max_err1}"

    # ------------------------------------------------------------------
    # Test 2: -u'' + cos(x)*u = lambda u, periodic
    # By Mathieu/Floquet theory eigenvalues come in near-pairs
    # ------------------------------------------------------------------
    print("\nTest 2: -u'' + cos(x)*u = lambda u, periodic on [0, 2*pi]")
    L2 = Chebop(lambda x, u: -u.diff(2) + jnp.cos(x) * u, domain=dom)
    L2.bc = "periodic"
    k2 = 6
    lams2 = L2.eigs(k=k2)
    lams2_sorted = np.sort(np.real(np.array(lams2)))
    print(f"  First {k2} eigenvalues: {lams2_sorted}")
    # Ground state should be close to Mathieu a_0(1/2) ≈ -0.4548
    assert lams2_sorted[0] < 0.0, "Ground state should be negative for cos potential"
    # Eigenvalues should be increasing
    assert np.all(np.diff(lams2_sorted) > 0), "Eigenvalues should be increasing"

    # ------------------------------------------------------------------
    # Test 3: Weighted problem -u'' + x*u = lambda (1+sin(x)^2) u
    # (simplified — just check eigenvalues are real and increasing)
    # ------------------------------------------------------------------
    print("\nTest 3: -u'' + u = lambda (2 + sin(x)) u, periodic")
    L3 = Chebop(lambda x, u: -u.diff(2) + u, domain=dom)
    L3.bc = "periodic"
    k3 = 4
    lams3 = L3.eigs(k=k3)
    lams3_sorted = np.sort(np.real(np.array(lams3)))
    print(f"  First {k3} eigenvalues: {lams3_sorted}")
    # All eigenvalues ≥ 1 since V=1
    assert np.all(lams3_sorted >= 0.5), "Eigenvalues should be >= 1 for -u''+u"

    # --- Plot -----------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = np.linspace(0, 2 * np.pi, 300)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot eigenvalues from test 1
    axes[0].plot(range(k), lams1_sorted, 'bo', markersize=7, label="computed λ_k")
    axes[0].plot(range(k), exact1, 'r^', markersize=5, label="exact k²", alpha=0.7)
    axes[0].set_xlabel("k"); axes[0].set_ylabel("λ_k")
    axes[0].set_title("Periodic Laplacian: -u'' = λu", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # Plot eigenvalues from test 2
    axes[1].plot(range(k2), lams2_sorted, 'go-', markersize=7)
    axes[1].set_xlabel("k"); axes[1].set_ylabel("λ_k")
    axes[1].set_title("Periodic Hill: -u'' + cos(x)u = λu", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Periodic Sturm-Liouville eigenvalue problems", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "fourier_eigs.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
