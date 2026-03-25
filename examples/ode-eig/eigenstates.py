"""Eigenstates of the Schrodinger equation.

Computes quantum eigenstates for several 1D potentials:
harmonic oscillator, infinite square well, and anharmonic oscillator.

Credit: Chebfun example ode-eig/Eigenstates.m (Nick Trefethen, Jan 2012).
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


def run():
    print("=" * 60)
    print("Schrodinger eigenstates: -u'' + V(x) u = E u")
    print("=" * 60)

    dom = (-6.0, 6.0)
    n_eigs = 6

    # Harmonic oscillator: V(x) = x^2/2
    # Exact eigenvalues: E_n = n + 1/2 for n=0,1,2,...
    print("\nHarmonic oscillator: V(x) = x²/2")
    L_harm = Chebop(lambda x, u: -u.diff(2) + x**2 / 2.0 * u, domain=dom)
    L_harm.lbc = 0.0
    L_harm.rbc = 0.0
    lams_harm = L_harm.eigs(k=n_eigs)
    lams_sorted = np.sort(np.real(np.array(lams_harm)))
    exact_harm = np.array([n + 0.5 for n in range(n_eigs)])
    print(f"  {'E_n computed':>14}  {'E_n exact':>14}  {'error':>10}")
    for i in range(n_eigs):
        err = abs(lams_sorted[i] - exact_harm[i])
        print(f"  {lams_sorted[i]:14.8f}  {exact_harm[i]:14.8f}  {err:10.2e}")
    max_err_harm = np.max(np.abs(lams_sorted[:n_eigs] - exact_harm[:n_eigs]))
    print(f"  Max error: {max_err_harm:.2e}")
    assert max_err_harm < 1e-8, f"Harmonic eigenvalue error: {max_err_harm}"

    # Double well: V(x) = x^4 - 2x^2
    print("\nDouble well: V(x) = x^4 - 2x^2")
    L_dw = Chebop(lambda x, u: -u.diff(2) + (x**4 - 2*x**2) * u, domain=dom)
    L_dw.lbc = 0.0
    L_dw.rbc = 0.0
    lams_dw = L_dw.eigs(k=n_eigs)
    lams_dw_sorted = np.sort(np.real(np.array(lams_dw)))
    print(f"  First 6 eigenvalues: {lams_dw_sorted}")
    assert lams_dw_sorted[0] > -2.0, "Ground state should be above bottom of well"
    # Ground state (n=0) and first excited (n=1) should be close (tunneling)
    gap_01 = lams_dw_sorted[1] - lams_dw_sorted[0]
    print(f"  Tunneling gap (E1-E0): {gap_01:.6f}")

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = jnp.linspace(-6.0, 6.0, 400)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Harmonic oscillator modes
    V_harm = x_plot**2 / 2.0
    axes[0].plot(x_plot, V_harm, 'k', linewidth=1.5, label="V(x)")
    colors = plt.cm.jet(np.linspace(0, 1, n_eigs))
    # Recompute eigenvectors
    L_harm2 = Chebop(lambda x, u: -u.diff(2) + x**2 / 2.0 * u, domain=dom)
    L_harm2.lbc = 0.0; L_harm2.rbc = 0.0
    for i in range(min(4, n_eigs)):
        E = lams_sorted[i]
        axes[0].axhline(E, color=colors[i], linewidth=0.8, linestyle='--', alpha=0.6)
    axes[0].set_ylim(-0.5, 6)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("E")
    axes[0].set_title("Harmonic oscillator energy levels", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # Double well
    V_dw = x_plot**4 - 2 * x_plot**2
    axes[1].plot(x_plot, V_dw, 'k', linewidth=1.5, label="V(x) = x⁴−2x²")
    for i in range(min(6, n_eigs)):
        axes[1].axhline(lams_dw_sorted[i], color=colors[i], linewidth=0.8,
                        linestyle='--', alpha=0.7, label=f"E_{i}={lams_dw_sorted[i]:.2f}")
    axes[1].set_ylim(-2, 8)
    axes[1].set_xlabel("x"); axes[1].set_ylabel("E")
    axes[1].set_title("Double-well energy levels", fontsize=10)
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Schrodinger eigenstates: −u″ + V(x)u = Eu", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "eigenstates.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
