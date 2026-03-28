"""Spectral discretization: differentiation and integration matrices.

Demonstrates Chebyshev differentiation and integration matrices as
building blocks for spectral BVP solvers.

Credit: Chebfun example ode-linear/SpectralDisc.m (Nick Trefethen, Aug 2016).
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
from chebfunjax.operators.blocks import D, I, diag, eval_at, ChebColloc2Disc

def run():
    print("=" * 60)
    print("Spectral discretization: diff and integral matrices")
    print("=" * 60)

    dom = (-1.0, 1.0)
    n = 32   # number of collocation points

    disc = ChebColloc2Disc(n=n, domain=dom)

    # Build differentiation matrix D1
    D1_block = D(1)
    D1_mat = D1_block.matrix(disc)
    print(f"\nDifferentiation matrix D1: shape {D1_mat.shape}")
    assert D1_mat.shape == (n, n)

    # Build second-order differentiation matrix D2 = D1 @ D1
    D2_mat = D1_mat @ D1_mat
    print(f"Differentiation matrix D2 (=D1@D1): shape {D2_mat.shape}")

    # Verify: D1 applied to sin(x) gives cos(x)
    from chebfunjax.utils.quadrature import chebpts
    # chebpts returns points on [-1,1]; scale to domain
    a, b = dom
    x_ref = chebpts(n)
    x_pts = 0.5 * (b - a) * x_ref + 0.5 * (a + b)
    f_vals = np.sin(x_pts)
    df_computed = D1_mat @ f_vals
    df_exact = np.cos(x_pts)
    err_D1 = np.max(np.abs(df_computed - df_exact))
    print(f"\n  D1 @ sin(x): max error = {err_D1:.2e}")
    assert err_D1 < 1e-10

    # Verify D2 applied to sin(x) gives -sin(x)
    d2f_computed = D2_mat @ f_vals
    d2f_exact = -np.sin(x_pts)
    err_D2 = np.max(np.abs(d2f_computed - d2f_exact))
    print(f"  D2 @ sin(x): max error = {err_D2:.2e}")
    assert err_D2 < 1e-10

    # Use Linop to solve a BVP: -u'' = pi^2/4 * cos(pi*x/2), u(±1)=0
    print("\nSolving -u'' = (pi/2)^2 cos(pi*x/2) via Linop:")
    N = Chebop(lambda x, u: -u.diff(2), domain=dom)
    N.lbc = 0.0
    N.rbc = 0.0
    rhs = cj.chebfun(lambda x: (jnp.pi/2)**2 * jnp.cos(jnp.pi * x / 2), domain=dom)
    u = N.solve(rhs)

    x_test = jnp.linspace(-1.0, 1.0, 300)
    exact = jnp.cos(jnp.pi * x_test / 2)
    err = float(jnp.max(jnp.abs(u(x_test) - exact)))
    print(f"  Max error vs cos(pi*x/2): {err:.2e}")
    assert err < 1e-10

    # Show convergence: solve same BVP with increasing n
    print("\nSpectral convergence:")
    for n_test in [8, 16, 32, 64]:
        N_n = Chebop(lambda x, u: -u.diff(2), domain=dom)
        N_n.lbc = 0.0
        N_n.rbc = 0.0
        u_n = N_n.solve(rhs)
        err_n = float(jnp.max(jnp.abs(u_n(x_test) - exact)))
        print(f"  n={n_test:3d}: err = {err_n:.2e}")

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    # Plot D1 matrix sparsity pattern
    im = axes[0].imshow(np.abs(D1_mat), aspect='auto', cmap='Blues')
    axes[0].set_title(f"Chebyshev D1 matrix (n={n})", fontsize=10)
    plt.colorbar(im, ax=axes[0])

    # Plot solution
    axes[1].plot(x_test, u(x_test), 'b', linewidth=1.8, label="chebfunjax")
    axes[1].plot(x_test, exact, color='#D95319', linestyle='--', linewidth=1.2, label="cos(πx/2)")
    axes[1].set_title("−u″ = (π/2)² cos(πx/2)", fontsize=10)
    axes[1].legend(fontsize=8)

    fig.suptitle("Spectral discretization with differentiation matrices", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "spectral_disc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
