"""Frequencies of a circular drum.

The axisymmetric vibrations of a circular drum satisfy
  u''(r) + u'(r)/r = -omega^2 u(r),  u'(0)=0, u(1)=0.
The frequencies are zeros of J_0(omega), the Bessel function.

Credit: Chebfun example ode-eig/Drum.m (Toby Driscoll, Nov 2010).
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
from chebfunjax.operators.chebop import Chebop
from scipy.special import j0, jn_zeros


def run():
    print("=" * 60)
    print("Drum frequencies: u'' + u'/r = -omega^2 u")
    print("=" * 60)

    # Avoid r=0 singularity by working on [eps, 1]
    eps = 1e-3
    dom = (eps, 1.0)

    # Eigenvalue problem: -u'' - u'/r = omega^2 u
    # i.e., -(r u')'/r = omega^2 u (Bessel-type)
    L = Chebop(lambda r, u: -(u.diff(2) + u.diff() / r), domain=dom)
    L.lbc = 0.0   # u'(eps) ≈ 0 (Neumann at r=0)
    L.rbc = 0.0   # u(1) = 0

    k = 6
    lams = L.eigs(k=k)
    lams_sorted = np.sort(np.real(np.array(lams)))
    omegas = np.sqrt(np.maximum(lams_sorted, 0))
    print(f"\nFirst {k} drum frequencies:")

    # Exact: zeros of J_0
    j0_zeros = jn_zeros(0, k)
    print(f"\n  {'omega_k (computed)':>22}  {'J_0 zero (exact)':>20}  {'error':>10}")
    for i in range(k):
        err = abs(omegas[i] - j0_zeros[i])
        print(f"  {omegas[i]:22.8f}  {j0_zeros[i]:20.8f}  {err:10.2e}")
    max_err = np.max(np.abs(omegas[:k] - j0_zeros[:k]))
    print(f"\n  Max error: {max_err:.2e}")
    assert max_err < 0.01, f"Frequency error too large: {max_err}"

    # Plot J_0 and the computed zeros
    r_plot = np.linspace(0, 1, 300)
    J0_vals = j0(j0_zeros[0] * r_plot)  # mode 1

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(range(1, k+1), omegas, color='steelblue', alpha=0.7, label="computed ω_k")
    axes[0].plot(range(1, k+1), j0_zeros, 'ro', markersize=6, label="exact J₀ zeros")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("ω_k")
    axes[0].set_title("Drum frequencies (J₀ zeros)", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3, axis='y')

    for i, (omega, col) in enumerate(zip(j0_zeros[:4],
                                          ['b', 'r', 'g', 'm'])):
        axes[1].plot(r_plot, j0(omega * r_plot), color=col, linewidth=1.4,
                     label=f"mode {i+1}: ω={omega:.3f}")
    axes[1].axhline(0, color='k', linewidth=0.5)
    axes[1].set_xlabel("r"); axes[1].set_ylabel("u(r)")
    axes[1].set_title("Radial drum modes J₀(ω_k r)", fontsize=10)
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Circular drum: eigenfrequencies = zeros of J₀", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "drum.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
