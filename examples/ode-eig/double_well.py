"""Double-well Schrodinger eigenstates.

The Schrodinger equation with a smooth double-well potential:
    -c * u''(x) + V(x)*u(x) = lambda * u(x), u(-1) = u(1) = 0

where V(x) = 5*(x^2 - 0.5)^2 has minima at x = ±1/sqrt(2) and a
barrier at x=0, creating a quantum double well.

Credit: Inspired by Chebfun example ode-eig/DoubleWell.m.
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

from chebfunjax.plotting import plot
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Double-well Schrodinger equation eigenstates")
    print("=" * 60)

    c = 0.007
    dom = (-1.0, 1.0)
    n_eigs = 6

    # Use a smooth double-well potential: V(x) = 5*(x^2 - 0.5)^2
    # This has minima at x = ±1/sqrt(2) ≈ ±0.707 and a barrier at x=0
    N = Chebop(
        lambda x, u: -c * u.diff(2) + 5.0 * (x**2 - 0.5)**2 * u,
        domain=dom
    )
    N.lbc = 0.0
    N.rbc = 0.0

    lam = N.eigs(k=n_eigs)
    lam_real = np.sort(np.real(np.array(lam)))

    print(f"\nDouble-well potential V(x)=5*(x^2-0.5)^2, c={c}:")
    print(f"  First {n_eigs} eigenvalues:")
    for i, l in enumerate(lam_real):
        print(f"    lam_{i+1} = {l:.8f}")

    # Basic sanity checks:
    # All eigenvalues should be positive (V >= 0 and c > 0)
    assert np.all(lam_real > 0), "Eigenvalues should be positive"
    # They should be increasing
    assert np.all(np.diff(lam_real) > 0), "Eigenvalues should be increasing"
    # First eigenvalue is the ground state energy
    print(f"\n  Ground state energy: lam_1 = {lam_real[0]:.6f}")

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    import matplotlib.pyplot as _plt
    import numpy as _np
    fig, ax = _plt.subplots(figsize=(6, 3.5))
    _lam_real = _np.sort(_np.real(_np.array(lam[:n_eigs])))
    ax.bar(_np.arange(n_eigs), _lam_real, color="#4169E1", alpha=0.8)
    ax.set_xlabel("eigenvalue index", fontsize=10)
    ax.set_ylabel("λ", fontsize=10)
    ax.set_title("Double-well Schrödinger eigenvalues", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "double_well.png"),
                dpi=150, bbox_inches="tight")
    _plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
