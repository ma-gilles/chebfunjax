"""Allen-Cahn equation time-stepping.

Solves the Allen-Cahn equation using an implicit-explicit (IMEX) scheme,
following pde/AllenCahn2.m from Chebfun.

Allen-Cahn: u_t = ε²u_xx + u - u³,  u(-1) = -1, u(1) = 1

The linear diffusion term ε²u_xx is treated implicitly (for stability),
and the nonlinear term u - u³ is treated explicitly.

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

from chebfunjax.utils.quadrature import chebpts
from chebfunjax.domain import Domain
from scipy.linalg import solve as scipy_solve


def run():
    print("=" * 60)
    print("Allen-Cahn equation (IMEX time-stepping)")
    print("=" * 60)

    # Parameters
    eps = 0.1   # interface width
    T = 0.5     # final time
    dt = 5e-3   # time step

    print(f"\nAllen-Cahn: ε={eps}, T={T}, dt={dt}")

    # Chebyshev collocation
    n = 64
    xs = np.array(chebpts(n))  # Chebyshev points in [-1,1]

    # Initial condition: tanh profile shifted slightly
    u_init = np.tanh((xs - 0.2) / eps)

    # Chebyshev differentiation matrix (2nd order)
    from chebfunjax.utils.diffmat import diffmat
    D = np.array(diffmat(n, 1))
    D2 = D @ D

    # IMEX scheme: (I - dt*ε²*D2) * u^{n+1} = u^n + dt*(u^n - (u^n)^3)
    # with Dirichlet BCs: u[0] = 1, u[-1] = -1
    # Chebfun ordering: xs[0]=1 (right), xs[-1]=-1 (left)

    # Build implicit operator
    L = np.eye(n) - dt * eps**2 * D2

    # Modify for boundary conditions (rows 0 and n-1)
    L[0, :] = 0.0; L[0, 0] = 1.0
    L[-1, :] = 0.0; L[-1, -1] = 1.0

    # LU factorize once for efficiency
    from scipy.linalg import lu_factor, lu_solve
    L_lu = lu_factor(L)

    u_curr = u_init.copy()
    n_steps = int(T / dt)
    history = [u_curr.copy()]
    t_vals = [0.0]
    save_every = max(1, n_steps // 8)

    for k in range(n_steps):
        # Explicit nonlinear RHS
        nonlin = u_curr - u_curr**3
        rhs = u_curr + dt * nonlin
        # Enforce BCs in RHS
        rhs[0] = 1.0    # u(1) = 1
        rhs[-1] = -1.0  # u(-1) = -1
        # Solve implicit system
        u_new = lu_solve(L_lu, rhs)
        u_curr = u_new

        if (k + 1) % save_every == 0:
            history.append(u_curr.copy())
            t_vals.append((k + 1) * dt)

    print(f"Time-stepped Allen-Cahn to T={t_vals[-1]:.3f}")
    print(f"Final u range: [{u_curr.min():.4f}, {u_curr.max():.4f}]")
    assert np.isfinite(u_curr).all(), "Solution contains NaN/Inf"
    assert u_curr.min() > -1.2 and u_curr.max() < 1.2

    # Final solution via chebfun interpolation
    u_final = cj.chebfun.from_values(jnp.array(u_curr), domain=Domain([-1.0, 1.0]))

    # Check boundary values
    # Note: Chebfun ordering has xs[0]=+1 (right), xs[-1]=-1 (left)
    val_right = float(u_final(jnp.array(1.0)))
    val_left = float(u_final(jnp.array(-1.0)))
    print(f"u(-1) = {val_left:.4f}  (expected: -1 in Chebfun ordering)")
    print(f"u(+1) = {val_right:.4f}  (expected: +1 in Chebfun ordering)")
    # Chebfun ordering: u_curr[0] corresponds to x=+1, u_curr[-1] to x=-1
    assert abs(u_curr[0] - 1.0) < 0.01, f"BC at x=1 failed: {u_curr[0]}"
    assert abs(u_curr[-1] - (-1.0)) < 0.01, f"BC at x=-1 failed: {u_curr[-1]}"

    # The solution should approach the stable equilibrium tanh(x/eps)
    u_eq = np.tanh(xs / eps)
    print(f"Distance from equilibrium tanh: {np.max(np.abs(u_curr - u_eq)):.4f}")

    # Plot
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    colors = plt.cm.viridis(np.linspace(0, 1, len(history)))
    for i, (u_h, t_h) in enumerate(zip(history, t_vals)):
        label = f't={t_h:.2f}' if i in [0, len(history)-1] else ''
        axes[0].plot(xs, u_h, color=colors[i], alpha=0.8, linewidth=1.5,
                     label=label)
    axes[0].axhline(1, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[0].axhline(-1, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[0].set_title(f"Allen-Cahn: ε={eps}", fontsize=12)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("u")
    axes[0].set_xlim(-1, 1); axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(vmin=0, vmax=t_vals[-1]))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[0], label='t')

    # Space-time plot
    H = np.array(history)
    im = axes[1].imshow(H.T, aspect='auto', origin='lower',
                         extent=[0, t_vals[-1], -1, 1],
                         cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title("Space-time diagram", fontsize=12)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("x")
    fig.colorbar(im, ax=axes[1])

    fig.suptitle("Allen-Cahn equation", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "allen_cahn.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
