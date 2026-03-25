"""Continuous analogue of the Wilkinson matrix.

Considers the Sturm-Liouville eigenvalue problem
  -u'' + |x| u = lambda u,  u(±N) = 0
which is a continuous version of Wilkinson's tridiagonal matrix
with near-equal extreme eigenvalues.

Credit: Chebfun example ode-eig/ContinuousWilkinson.m (Nick Trefethen, Mar 2017).
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
    print("Continuous Wilkinson: -u'' + |x| u = λu, u(±N) = 0")
    print("=" * 60)

    N = 8.0
    dom = (-N, N)

    print(f"\nDomain: [{-N}, {N}]")
    L = Chebop(lambda x, u: -u.diff(2) + jnp.abs(x) * u, domain=dom)
    L.lbc = 0.0
    L.rbc = 0.0

    k = 12
    lams = L.eigs(k=k)
    lams_sorted = np.sort(np.real(np.array(lams)))
    print(f"\nFirst {k} eigenvalues:")
    for i, lam in enumerate(lams_sorted):
        print(f"  λ_{i+1} = {lam:.8f}")

    # Check eigenvalues are positive (potential is positive)
    assert np.all(lams_sorted > 0), "All eigenvalues should be positive"

    # Check they are increasing
    assert np.all(np.diff(lams_sorted) > 0), "Eigenvalues should be increasing"

    # The middle eigenvalues should be close to each other (Wilkinson effect)
    # For the Wilkinson analogue, eigenvalues around N come in near-pairs
    # For N=8, look at eigenvalues near index 8
    if len(lams_sorted) >= 10:
        gaps = np.diff(lams_sorted)
        min_gap = np.min(gaps)
        max_gap = np.max(gaps)
        print(f"\n  Min eigenvalue gap: {min_gap:.6f}")
        print(f"  Max eigenvalue gap: {max_gap:.6f}")
        print(f"  Gap ratio max/min: {max_gap/min_gap:.2f}")

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(range(1, len(lams_sorted)+1), lams_sorted, color='steelblue', alpha=0.7)
    axes[0].set_xlabel("k"); axes[0].set_ylabel("λ_k")
    axes[0].set_title(f"Eigenvalues of −u″ + |x|u on [−{N:.0f},{N:.0f}]", fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Plot eigenvalue gaps
    if len(lams_sorted) >= 2:
        gaps = np.diff(lams_sorted)
        axes[1].bar(range(1, len(gaps)+1), gaps, color='coral', alpha=0.7)
        axes[1].set_xlabel("k"); axes[1].set_ylabel("λ_{k+1} − λ_k")
        axes[1].set_title("Eigenvalue gaps (Wilkinson effect)", fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')

    fig.suptitle("Continuous Wilkinson eigenvalue problem", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "continuous_wilkinson.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
