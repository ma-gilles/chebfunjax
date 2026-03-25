"""Sturm-Liouville eigenvalue problems.

Demonstrates several Sturm-Liouville problems with known
eigenvalues, including the Laplacian and weighted operators.

Credit: Chebfun example ode-eig/SturmLiouville.m.
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
from chebfunjax.plotting import plot
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Sturm-Liouville eigenvalue problems")
    print("=" * 60)

    pi = float(jnp.pi)

    # --- 1. Laplacian on [0, pi], Dirichlet --------------------------
    # -u'' = lambda*u, u(0) = u(pi) = 0
    # Eigenvalues: n^2, n = 1, 2, 3, ...
    N1 = Chebop(lambda x, u: -u.diff(2), domain=(0.0, pi))
    N1.lbc = 0.0
    N1.rbc = 0.0
    lam1 = N1.eigs(k=5)
    lam1_real = np.sort(np.real(np.array(lam1)))
    exact1 = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    print(f"\n1. -u'' = lam*u on [0,pi], Dirichlet:")
    print(f"  Eigenvalues: {lam1_real}")
    print(f"  Exact:       {exact1}")
    max_err1 = np.max(np.abs(lam1_real - exact1))
    assert max_err1 < 1e-6, f"Error: {max_err1}"

    # --- 2. Laplacian on [0, 1], Dirichlet ---------------------------
    # -u'' = lambda*u, u(0) = u(1) = 0
    # Eigenvalues: (n*pi)^2, n = 1, 2, 3, ...
    N2 = Chebop(lambda x, u: -u.diff(2), domain=(0.0, 1.0))
    N2.lbc = 0.0
    N2.rbc = 0.0
    lam2 = N2.eigs(k=4)
    lam2_real = np.sort(np.real(np.array(lam2)))
    exact2 = np.array([(n * pi)**2 for n in range(1, 5)])
    print(f"\n2. -u'' = lam*u on [0,1], Dirichlet:")
    for i in range(4):
        err = abs(lam2_real[i] - exact2[i])
        print(f"  n={i+1}: computed={lam2_real[i]:.6f}, exact=(n*pi)^2={exact2[i]:.6f}, err={err:.2e}")
    max_err2 = np.max(np.abs(lam2_real - exact2))
    assert max_err2 < 1e-5, f"Error: {max_err2}"

    # --- 3. QHO on small domain (repeat from harmonic_oscillator) ----
    # -u'' + x^2*u = lambda*u, eigenvalues = 2n+1
    N3 = Chebop(lambda x, u: -u.diff(2) + x**2 * u, domain=(-5.0, 5.0))
    N3.lbc = 0.0
    N3.rbc = 0.0
    lam3 = N3.eigs(k=4)
    lam3_real = np.sort(np.real(np.array(lam3)))
    exact3 = np.array([1.0, 3.0, 5.0, 7.0])
    print(f"\n3. QHO: -u'' + x^2*u = lam*u on [-5,5]:")
    for i in range(4):
        err = abs(lam3_real[i] - exact3[i])
        print(f"  n={i}: computed={lam3_real[i]:.6f}, exact={exact3[i]:.1f}, err={err:.2e}")
    max_err3 = np.max(np.abs(lam3_real - exact3))
    assert max_err3 < 1e-5, f"QHO error: {max_err3}"

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    import matplotlib.pyplot as _plt
    import numpy as _np
    fig, ax = _plt.subplots(figsize=(6, 3.5))
    _lam1_arr = _np.sort(_np.real(_np.array(lam1[:5])))
    ax.plot(_np.arange(1, 6), _lam1_arr, "o-", color="#4169E1",
            linewidth=1.5, markersize=6, label="−u″ on [0,π]")
    ax.set_xlabel("n", fontsize=10)
    ax.set_ylabel("λ", fontsize=10)
    ax.set_title("Sturm-Liouville eigenvalues", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "sturm_liouville.png"),
                dpi=150, bbox_inches="tight")
    _plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
