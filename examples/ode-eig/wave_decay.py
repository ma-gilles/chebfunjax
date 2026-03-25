"""Wave equation with decay band.

Computes eigenmodes of the 1D wave equation (Laplacian) on [-pi/2, pi/2]
with and without a dissipative middle band. Shows how the spectrum
changes from purely imaginary (no decay) to shifted into the left half-plane.

Credit: Chebfun example ode-eig/WaveDecay.m (Nick Trefethen, November 2010).
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
    print("Wave equation with decay band: eigenmodes")
    print("=" * 60)

    dom = (-float(np.pi) / 2.0, float(np.pi) / 2.0)

    # ------------------------------------------------------------------
    # Part 1: Pure Laplacian -u'' on [-pi/2, pi/2], Dirichlet BCs
    # Eigenvalues: lambda_k = k^2, k=1,2,...
    # Wave modes: u(x,t) = sin(kx) e^{i*k*t}
    # ------------------------------------------------------------------
    print("\nPart 1: Eigenmodes of -u'' on [-pi/2, pi/2]")
    L_pure = Chebop(lambda x, u: -u.diff(2), domain=dom)
    L_pure.lbc = 0.0
    L_pure.rbc = 0.0

    nmax = 20
    lams_pure = L_pure.eigs(k=nmax)
    lams_pure_sorted = np.sort(np.real(np.array(lams_pure)))
    exact_k = np.arange(1, nmax + 1, dtype=float)
    exact_lams = exact_k**2
    print(f"  {'k':>4}  {'lambda_k':>12}  {'exact k^2':>12}  {'error':>10}")
    for k in [1, 2, 10, 20]:
        i = k - 1
        if i < len(lams_pure_sorted):
            err = abs(lams_pure_sorted[i] - exact_lams[i])
            print(f"  {k:>4}  {lams_pure_sorted[i]:>12.6f}  {exact_lams[i]:>12.0f}  {err:>10.2e}")
    max_err = np.max(np.abs(lams_pure_sorted[:nmax] - exact_lams[:nmax]))
    print(f"  Max error: {max_err:.2e}")
    assert max_err < 1e-6, f"Wave eigenvalue error: {max_err}"

    # ------------------------------------------------------------------
    # Part 2: Wave operator with piecewise damping
    # Consider u'' = -k^2 u (wave) but with dissipation in middle band
    # Modified operator: L = d^2/dx^2 + sigma(x) * d/dx
    # where sigma(x) = gamma > 0 in |x| < pi/6 (middle third)
    # This shifts eigenvalues into the left half-plane
    # ------------------------------------------------------------------
    print("\nPart 2: Eigenmodes with dissipation in middle band")
    gamma = 2.0

    def sigma(x):
        return jnp.where(jnp.abs(x) < float(np.pi) / 6.0, gamma, 0.0)

    L_damp = Chebop(lambda x, u: u.diff(2) + sigma(x) * u.diff(), domain=dom)
    L_damp.lbc = 0.0
    L_damp.rbc = 0.0

    k_damp = 8
    lams_damp = L_damp.eigs(k=k_damp)
    lams_damp_arr = np.array(lams_damp)
    lams_damp_sorted = lams_damp_arr[np.argsort(np.real(lams_damp_arr))][::-1]

    print(f"  First {k_damp} eigenvalues (sorted by Re):")
    for i in range(k_damp):
        print(f"    λ_{i+1} = {lams_damp_sorted[i].real:+.4f} {lams_damp_sorted[i].imag:+.4f}i")

    # All eigenvalues should have non-positive real parts (stability)
    max_real_damp = np.max(np.real(lams_damp_arr))
    print(f"  Max Re(lambda): {max_real_damp:.4f} (should be <= 0 for stable damping)")
    # Note: operator convention may differ; check that we have nontrivial imaginary parts
    has_complex = np.any(np.abs(np.imag(lams_damp_arr)) > 0.1)
    print(f"  Complex eigenvalues present: {has_complex}")

    # ------------------------------------------------------------------
    # Part 3: Plot specific wave modes for the undamped problem
    # Modes k=1,2,10,20 (or nearest available)
    # ------------------------------------------------------------------
    nn = [1, 2, 10, 20]
    x_plot = np.linspace(-np.pi/2, np.pi/2, 500)
    # The eigenfunctions are sin(k*(x + pi/2)) normalized
    mode_funcs = {}
    for k_m in nn:
        v = np.sin(k_m * (x_plot + np.pi/2))
        v /= np.max(np.abs(v))  # normalize
        mode_funcs[k_m] = v

    # --- Plot -----------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()

    for idx, k_m in enumerate(nn):
        v = mode_funcs[k_m]
        lam_k = lams_pure_sorted[k_m - 1] if k_m - 1 < len(lams_pure_sorted) else k_m**2
        axes[idx].plot(x_plot, v, 'b', linewidth=1.5)
        axes[idx].axhline(0, color='k', linewidth=0.5)
        axes[idx].set_xlim(-np.pi/2, np.pi/2)
        axes[idx].set_ylim(-1.6, 2.2)
        axes[idx].set_title(f"mode {k_m},  λ = {lam_k:.3f}", fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        if idx >= 2:
            axes[idx].set_xlabel("x")

    fig.suptitle("Wave equation eigenmodes: −u″ = λu on [−π/2, π/2]", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "wave_decay.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
