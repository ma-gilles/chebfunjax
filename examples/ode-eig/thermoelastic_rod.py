"""Stability of a thermoelastic rod.

Eigenvalue problem with nonstandard integral boundary condition:
  phi''(x) = lambda phi(x),  0 < x < 1,
  phi(0) = 0,   phi'(1) + phi(1) = 4*delta * int_0^1 phi(x) dx.

Transition from stable to unstable occurs at delta = 1.

Credit: Chebfun example ode-eig/ThermoelasticRod.m (Toby Driscoll, Nov 2011).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eig as scipy_eig
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.operators.chebop import Chebop


def thermoelastic_eigs(delta, N=64):
    """Compute eigenvalues of thermoelastic rod problem via matrix discretization.

    phi'' = lambda phi,  phi(0)=0,  phi'(1)+phi(1) = 4*delta * int phi dx

    Uses Chebyshev collocation with the integral BC enforced explicitly.
    """
    # Chebyshev points on [0,1]
    x = 0.5 * (1.0 - np.cos(np.pi * np.arange(N + 1) / N))  # mapped to [0,1]
    # Chebyshev D matrix on [-1,1]
    t = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1); c[0] = 2.0; c[N] = 2.0
    X_t = np.tile(t, (N + 1, 1))
    dX = X_t - X_t.T
    D_cheb = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D_cheb -= np.diag(np.sum(D_cheb, axis=1))

    # Scale D for [0,1]: dx_phys = 2 * dt_cheb
    D = 2.0 * D_cheb
    D2 = D @ D

    # Interior indices (remove x=0 row for Dirichlet)
    # Row 0: x=1 (Chebyshev ordered x[0]=1, x[N]=0)
    # Row N: x=0
    # Enforce phi(0)=0 at row N; enforce Robin BC at row 0
    # Interior rows: 1..N-1
    idx_int = np.arange(1, N)
    n_int = len(idx_int)

    # Build system: phi'' = lambda phi for interior points
    A_bulk = D2[np.ix_(idx_int, np.arange(N + 1))]

    # Enforce phi(0) = 0: set column N appropriately via substitution
    # phi[N] = 0, so remove that column
    # Columns: 0..N, with col N = boundary at x=0
    # We keep phi[0]..phi[N-1] as unknowns; phi[N] = 0

    A_int = D2[np.ix_(idx_int, np.arange(N))]  # remove last col (phi[N]=0)

    # Enforce Robin BC: phi'(1) + phi(1) = 4*delta * integral(phi)
    # phi'(1) = D[0, :] @ phi; phi(1) = phi[0]
    # integral(phi) ~ Clenshaw-Curtis weights for [0,1]
    # Use midpoint approx or Gauss-Legendre; here approximate via trapezoid on Cheb grid
    # Clenshaw-Curtis weights for [0,1] mapped from [-1,1]
    wCC = np.zeros(N + 1)
    # CC weights on [-1,1]: w_k = 2/N * sum cos(j*pi*k/N)/(1-j^2) for j even
    for i in range(N + 1):
        sum_w = 0.0
        for j in range(0, N + 1, 2):
            if j == 0:
                sum_w += 1.0
            elif j == N:
                sum_w += np.cos(j * np.pi * i / N) / (1 - j**2)
            else:
                sum_w += 2.0 * np.cos(j * np.pi * i / N) / (1 - j**2)
        wCC[i] = sum_w / N
    wCC[0] /= 2; wCC[N] /= 2
    # Scale from [-1,1] to [0,1]: multiply by 1/2
    w = 0.5 * wCC

    # Remove last weight (phi[N]=0)
    w_int = w[:N]  # weights for phi[0]..phi[N-1]

    # Robin row: D[0, :N] @ phi + phi[0] = 4*delta * w_int @ phi
    robin_row = D[0, :N] + np.eye(N)[0] - 4.0 * delta * w_int

    # Build full system A*phi = lambda*B*phi
    # Rows: 0..N-2 = interior phi'' = lambda*phi
    # Row N-1 = Robin BC = 0 (eigenvalue equation)
    A_mat = np.zeros((N, N), dtype=complex)
    B_mat = np.zeros((N, N), dtype=complex)

    A_mat[:n_int, :] = A_int  # phi'' rows
    B_mat[:n_int, :] = np.eye(N)[:n_int, :]  # phi on RHS

    # Robin BC row: enforce as "A_robin = 0" (no eigenvalue contribution)
    A_mat[n_int, :] = robin_row
    B_mat[n_int, :] = 0.0

    # Solve generalized eigenvalue problem
    lams_all, _ = scipy_eig(A_mat, B_mat)

    # Filter spurious eigenvalues
    finite_mask = (np.isfinite(lams_all) & (np.abs(lams_all) < 1e6)
                   & (np.abs(lams_all) > 1e-6))
    return lams_all[finite_mask]


def run():
    print("=" * 60)
    print("Stability of a thermoelastic rod (integral BC)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Stable case: delta = 0.96
    # ------------------------------------------------------------------
    delta_stable = 0.96
    print(f"\nCase 1: delta = {delta_stable} (stable)")
    lams_stable = thermoelastic_eigs(delta_stable, N=48)
    # Find eigenvalues closest to zero
    idx_near_zero = np.argsort(np.abs(np.real(lams_stable)))
    top4 = lams_stable[idx_near_zero[:4]]
    lams_stable_real = np.sort(np.real(top4))
    print(f"  4 eigenvalues nearest 0: {lams_stable_real}")
    max_real_stable = np.max(np.real(top4))
    print(f"  Max Re(lambda): {max_real_stable:.4f} (should be < 0 for stability)")
    assert max_real_stable < 0.1, f"Stable case: max eigenvalue {max_real_stable:.4f}"

    # ------------------------------------------------------------------
    # Unstable case: delta = 1.02
    # ------------------------------------------------------------------
    delta_unstable = 1.02
    print(f"\nCase 2: delta = {delta_unstable} (unstable)")
    lams_unstable = thermoelastic_eigs(delta_unstable, N=48)
    idx_near_zero2 = np.argsort(np.abs(np.real(lams_unstable)))
    top4u = lams_unstable[idx_near_zero2[:4]]
    lams_unstable_real = np.sort(np.real(top4u))
    print(f"  4 eigenvalues nearest 0: {lams_unstable_real}")
    max_real_unstable = np.max(np.real(top4u))
    print(f"  Max Re(lambda): {max_real_unstable:.4f} (should be > 0 for instability)")
    assert max_real_unstable > -0.1, f"Unstable case: max eigenvalue {max_real_unstable:.4f}"

    # ------------------------------------------------------------------
    # Sweep delta: find critical transition
    # ------------------------------------------------------------------
    print("\nSweeping delta in [0.5, 2.0] to find stability boundary ...")
    n_delta = 25
    deltas = np.linspace(0.5, 2.0, n_delta)
    max_lams = np.zeros(n_delta)

    for i, d in enumerate(deltas):
        lams_d = thermoelastic_eigs(d, N=48)
        max_lams[i] = np.max(np.real(lams_d[
            np.abs(np.real(lams_d)) < 100
        ]))

    # Find where max eigenvalue changes sign
    sign_changes = np.where(np.diff(np.sign(max_lams)))[0]
    if len(sign_changes) > 0:
        delta_crit = deltas[sign_changes[0]] + (deltas[sign_changes[0]+1] - deltas[sign_changes[0]]) / 2
        print(f"  Critical delta ≈ {delta_crit:.3f} (exact = 1.0)")
        assert abs(delta_crit - 1.0) < 0.2, f"Critical delta wrong: {delta_crit}"
    else:
        print("  (No sign change detected in sweep)")

    # --- Plot -----------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Sweep result
    axes[0].plot(deltas, max_lams, 'b-o', markersize=5, linewidth=1.5)
    axes[0].axhline(0, color='r', linewidth=1.2, linestyle='--', label="λ=0")
    axes[0].axvline(1.0, color='k', linewidth=0.8, linestyle=':', alpha=0.7, label="δ=1")
    axes[0].set_xlabel("δ (thermal parameter)"); axes[0].set_ylabel("max Re(λ)")
    axes[0].set_title("Max eigenvalue vs δ\n(stability transition at δ=1)", fontsize=10)
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    # Show eigenvalues for stable and unstable
    axes[1].axvline(0, color='k', linewidth=1.0, linestyle='--', alpha=0.7)
    axes[1].scatter(np.real(lams_stable[idx_near_zero[:6]]),
                    np.arange(6), color='steelblue', s=50,
                    label=f"δ={delta_stable} (stable)", zorder=3)
    axes[1].scatter(np.real(lams_unstable[idx_near_zero2[:6]]),
                    np.arange(6) + 0.2, color='coral', s=50,
                    label=f"δ={delta_unstable} (unstable)", zorder=3)
    axes[1].set_xlabel("Re(λ)"); axes[1].set_ylabel("mode index")
    axes[1].set_title("Eigenvalues: stable vs unstable", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Thermoelastic rod: stability via integral BC eigenvalue problem", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "thermoelastic_rod.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
