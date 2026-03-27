"""Quantum harmonic oscillator eigenvalues.

Computes eigenvalues of -h^2*u'' + x^2*u = lambda*u, which are
lambda_n = h*(2n+1) for n = 0, 1, 2, ...

Credit: Chebfun example ode-eig/QHO.m (Nick Trefethen).
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
    print("Quantum harmonic oscillator eigenvalues")
    print("=" * 60)

    # -h^2*u'' + x^2*u = lambda*u on [-L, L], u(±L) = 0
    h = 1.0   # "hbar" parameter
    L = 6.0   # truncation domain (should be large enough)
    dom = (-L, L)
    n_eigs = 6

    x_dom = cj.chebfun(lambda t: t, domain=dom)
    V = x_dom**2  # harmonic potential

    N = Chebop(
        lambda x, u: -h**2 * u.diff(2) + x**2 * u,
        domain=dom
    )
    N.lbc = 0.0
    N.rbc = 0.0

    lam = N.eigs(k=n_eigs)
    lam_real = np.sort(np.real(np.array(lam)))
    # Exact: lambda_n = h*(2n+1) for n=0,1,2,...
    exact = np.array([h * (2*n + 1) for n in range(n_eigs)], dtype=float)

    print(f"\nQHO eigenvalues on [{-L}, {L}] (h={h}):")
    print(f"  {'n':>4}  {'computed':>16}  {'exact h(2n+1)':>16}  {'error':>10}")
    for i in range(n_eigs):
        err = abs(lam_real[i] - exact[i])
        print(f"  {i:>4}  {lam_real[i]:>16.10f}  {exact[i]:>16.10f}  {err:.2e}")

    max_err = np.max(np.abs(lam_real - exact))
    print(f"\n  Max error: {max_err:.2e}")
    assert max_err < 1e-5, f"QHO eigenvalue error too large: {max_err}"

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    import matplotlib.pyplot as plt
    import numpy as _np
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(_np.arange(n_eigs), _np.array(lam[:n_eigs]), color="#4169E1",
           alpha=0.8, label="computed")
    ax.plot(_np.arange(n_eigs), exact, "ro", markersize=6, label="exact")
    ax.set_xlabel("eigenvalue index", fontsize=10)
    ax.set_ylabel("λ", fontsize=10)
    ax.set_title("Quantum harmonic oscillator eigenvalues", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "harmonic_oscillator.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
