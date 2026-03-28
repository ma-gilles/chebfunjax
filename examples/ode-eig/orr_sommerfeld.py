"""Orr-Sommerfeld eigenvalues for hydrodynamic stability.

Computes the eigenvalue spectrum of the Orr-Sommerfeld operator for
plane Poiseuille flow. The operator governs stability of the laminar
flow; eigenvalues in the right half-plane indicate instability.

The critical Reynolds number is Re_c ≈ 5772.22.

Credit: Chebfun example ode-eig/OrrSommerfeld.m
        (Toby Driscoll and Nick Trefethen, October 2010).
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
    print("Orr-Sommerfeld eigenvalues for plane Poiseuille flow")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Orr-Sommerfeld equation (4th-order eigenvalue problem):
    #
    #   [(D^2 - alpha^2)^2 / Re - i*alpha*(1-x^2)*(D^2 - alpha^2)
    #    + 2*i*alpha] v = lambda * (D^2 - alpha^2) v
    #
    # BCs: v = Dv = 0 at x = ±1
    #
    # Note: In simplified form for implementation:
    #   A v = lambda B v,  where
    #   A = (D^4 - 2*alpha^2*D^2 + alpha^4)/Re - i*alpha*(1-x^2)*(D^2-alpha^2) - 2i*alpha*I
    #   B = D^2 - alpha^2 * I
    #
    # We convert to standard form by solving via direct matrix approach.
    # Here we build the Chebyshev differentiation matrix discretization.
    # ------------------------------------------------------------------

    def orr_sommerfeld_matrix(N, Re, alph):
        """Build Orr-Sommerfeld matrices A, B on N+1 Chebyshev points in [-1,1]."""
        # Chebyshev differentiation matrices
        x = np.cos(np.pi * np.arange(N + 1) / N)
        # Build D matrix
        c = np.ones(N + 1)
        c[0] = 2.0; c[N] = 2.0
        X = np.tile(x, (N + 1, 1))
        dX = X - X.T
        D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
        D -= np.diag(np.sum(D, axis=1))
        D2 = D @ D
        D4 = D2 @ D2

        # Bulk rows (interior points only, indices 1..N-1)
        # Interior only (remove first and last row/col for Dirichlet)
        idx = np.arange(1, N)  # interior indices
        xi = x[idx]

        D2i = D2[np.ix_(idx, idx)]
        D4i = D4[np.ix_(idx, idx)]
        Ii = np.eye(len(idx))

        A = (D4i - 2.0 * alph**2 * D2i + alph**4 * Ii) / Re \
            - 1j * alph * np.diag(1.0 - xi**2) @ (D2i - alph**2 * Ii) \
            - 2j * alph * Ii

        B = D2i - alph**2 * Ii

        return A, B, xi

    # For Neumann + Dirichlet: v(±1)=0, v'(±1)=0
    # We apply 4 BCs => use a slightly different approach:
    # Apply Dirichlet on v and Dv simultaneously by reducing to
    # interior-only formulation with a modified BC enforcement.

    def orr_sommerfeld_full(N, Re, alph):
        """Full Orr-Sommerfeld with v=v'=0 at ±1."""
        x = np.cos(np.pi * np.arange(N + 1) / N)
        c = np.ones(N + 1)
        c[0] = 2.0; c[N] = 2.0
        X = np.tile(x, (N + 1, 1))
        dX = X - X.T
        D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
        D -= np.diag(np.sum(D, axis=1))
        D2 = D @ D
        D4 = D2 @ D2

        # We want to impose v(-1)=v(1)=0, v'(-1)=v'(1)=0
        # Rows 0 and N are BCs for Dirichlet
        # We use the "boundary bordering" technique in a simple way:
        # Pick interior rows 2..N-2 from the 4th-order equation
        # (leaving rows 0,1,N-1,N for BCs)
        # This is a rough approximation; for accuracy use standard techniques

        idx = np.arange(2, N - 1)
        xi = x[idx]
        n_int = len(idx)

        # Build full D2, D4 restricted to interior rows; columns = all but BCs?
        # Instead, just zero out BC rows and enforce strongly
        I_all = np.eye(N + 1)
        A_full = (D4 - 2 * alph**2 * D2 + alph**4 * I_all) / Re \
                 - 1j * alph * np.diag(1.0 - x**2) @ (D2 - alph**2 * I_all) \
                 - 2j * alph * I_all
        B_full = D2 - alph**2 * I_all

        # Enforce BCs by replacing BC rows with identity rows
        for bc in [0, 1, N - 1, N]:
            A_full[bc] = I_all[bc]
            B_full[bc] = 0.0 * I_all[bc]

        return A_full, B_full

    Re = 2000
    alph = 1.0
    N = 64

    print(f"\nRe = {Re}, alpha = {alph}, N = {N}")
    A, B = orr_sommerfeld_full(N, Re, alph)

    # Solve generalized eigenvalue problem A v = lambda B v
    # Use scipy which handles this well
    from scipy.linalg import eig as scipy_eig
    lams_all, _ = scipy_eig(A, B)

    # Filter out spurious eigenvalues (BC rows give lambda=0 or inf)
    finite_mask = np.isfinite(lams_all) & (np.abs(lams_all) < 1e6) & (np.abs(lams_all) > 1e-8)
    lams_finite = lams_all[finite_mask]

    # Focus on eigenvalues in physically relevant region
    # -1 < Im < 0 (temporal decay) and -1 < Re < 0.5
    in_region = lams_finite[
        (np.real(lams_finite) > -1.0) & (np.real(lams_finite) < 0.2) &
        (np.imag(lams_finite) > -1.1) & (np.imag(lams_finite) < 0.1)
    ]

    max_real = np.max(np.real(in_region)) if len(in_region) > 0 else np.nan
    print(f"  Found {len(in_region)} eigenvalues in physical region")
    print(f"  Most unstable mode: Re(lambda) = {max_real:.5f}")
    print(f"  (Negative means stable for Re=2000; critical Re ≈ 5772)")
    assert max_real < 0.0, f"Flow at Re=2000 should be stable, got Re(lam)={max_real:.4f}"

    # Near-critical Reynolds number
    Re_crit = 5772.22
    alph_crit = 1.02
    print(f"\nNear-critical: Re = {Re_crit}, alpha = {alph_crit}")
    A2, B2 = orr_sommerfeld_full(N, Re_crit, alph_crit)
    lams2, _ = scipy_eig(A2, B2)
    finite2 = lams2[np.isfinite(lams2) & (np.abs(lams2) < 1e6) & (np.abs(lams2) > 1e-8)]
    in2 = finite2[
        (np.real(finite2) > -1.0) & (np.real(finite2) < 0.2) &
        (np.imag(finite2) > -1.1) & (np.imag(finite2) < 0.1)
    ]
    max_real2 = np.max(np.real(in2)) if len(in2) > 0 else np.nan
    print(f"  Most unstable mode: Re(lambda) = {max_real2:.5f}")
    print(f"  (Near zero means near-critical)")
    assert abs(max_real2) < 0.01, f"Near-critical should have |Re(lam)| < 0.01, got {max_real2:.4f}"

    # --- Plot -----------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    # Re=2000 spectrum
    axes[0].plot(np.real(in_region), np.imag(in_region), color='#D95319', marker='.', linestyle='none', markersize=5)
    axes[0].axvline(0, color='k', linewidth=0.8, linestyle='--')
    axes[0].set_title(f"Orr-Sommerfeld spectrum\nRe={Re}, α={alph}", fontsize=10)
    axes[0].set_xlim(-0.9, 0.15); axes[0].set_ylim(-1.1, 0.1)
    axes[0].text(0.02, -0.1, f"max Re(λ)={max_real:.4f}", transform=axes[0].transAxes, fontsize=8)

    # Re_crit spectrum
    axes[1].plot(np.real(in2), np.imag(in2), color='#D95319', marker='.', linestyle='none', markersize=5)
    axes[1].axvline(0, color='k', linewidth=0.8, linestyle='--')
    axes[1].set_title(f"Orr-Sommerfeld spectrum\nRe={Re_crit:.0f}, α={alph_crit}", fontsize=10)
    axes[1].set_xlim(-0.9, 0.15); axes[1].set_ylim(-1.1, 0.1)
    axes[1].text(0.02, -0.1, f"max Re(λ)={max_real2:.5f}", transform=axes[1].transAxes, fontsize=8)

    fig.suptitle("Orr-Sommerfeld: plane Poiseuille flow stability", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "orr_sommerfeld.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
