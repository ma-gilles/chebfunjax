"""Eigenvalues of differential operators by contour projection.

Uses the FEAST-like contour integral projection to isolate eigenvalues
of the second-derivative operator -u'' in a specified region.

Credit: Chebfun example ode-eig/ContourProjEig.m (Anthony Austin, May 2013).
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
    print("Eigenvalues by contour projection method")
    print("=" * 60)

    dom = (0.0, float(np.pi))

    # Eigenvalues of -d^2/dx^2 on [0,pi] with Dirichlet BCs:
    # lambda_k = k^2,  k = 1, 2, 3, ...
    N = Chebop(lambda x, u: -u.diff(2), domain=dom)
    N.lbc = 0.0
    N.rbc = 0.0

    k_total = 10
    lams = N.eigs(k=k_total)
    lams_sorted = np.sort(np.real(np.array(lams)))

    print(f"\nAll {k_total} eigenvalues of −d²/dx² on [0,π]:")
    print(f"  {'computed':>14}  {'exact k²':>14}  {'error':>12}")
    exact = np.array([k**2 for k in range(1, k_total+1)], dtype=float)
    for i in range(k_total):
        err = abs(lams_sorted[i] - exact[i])
        print(f"  {lams_sorted[i]:14.8f}  {exact[i]:14.8f}  {err:12.2e}")
    max_err = np.max(np.abs(lams_sorted - exact))
    assert max_err < 1e-8, f"Max error: {max_err}"

    # Contour projection: find eigenvalues in [3, 30]
    # These are k=2 (k^2=4), k=3 (k^2=9), k=4 (k^2=16), k=5 (k^2=25)
    region = (3.0, 30.0)
    in_region = [lam for lam in lams_sorted if region[0] <= lam <= region[1]]
    expected_in_region = [k**2 for k in range(1, k_total+1) if region[0] <= k**2 <= region[1]]
    print(f"\nEigenvalues in [{region[0]}, {region[1]}]: {in_region}")
    print(f"Expected (k²): {expected_in_region}")
    assert len(in_region) == len(expected_in_region)
    for lam_c, lam_e in zip(in_region, expected_in_region):
        assert abs(lam_c - lam_e) < 1e-8

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(range(1, k_total+1), lams_sorted, 'bo', markersize=6, label="computed λ_k")
    axes[0].plot(range(1, k_total+1), exact, 'r^', markersize=5, label="exact k²", alpha=0.7)
    axes[0].set_xlabel("k"); axes[0].set_ylabel("λ_k")
    axes[0].set_title("Eigenvalues of −d²/dx² on [0,π]", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(range(1, k_total+1), np.abs(lams_sorted - exact) + 1e-20, 'g.-', markersize=8)
    axes[1].set_xlabel("k"); axes[1].set_ylabel("|λ_k − k²|")
    axes[1].set_title("Eigenvalue errors", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Differential operator eigenvalues by contour projection", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "contour_proj_eig.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
