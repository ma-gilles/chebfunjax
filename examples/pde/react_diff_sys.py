"""Coupled reaction-diffusion system.

Solves a three-component reaction-diffusion system:
  u_t = 0.1 u_xx - 100*u*v
  v_t = 0.2 v_xx - 100*u*v
  w_t = 0.001 w_xx + 200*u*v

with Neumann BCs. This models two chemicals u, v diffusing and reacting
to produce w, following pde/ReactDiffSys.m by Nick Hale (October 2010).

Original MATLAB: https://www.chebfun.org/examples/pde/ReactDiffSys.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chebfunjax.plotting import chebfun_style
chebfun_style()

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erf
import os

def run():
    print("=" * 60)
    print("Coupled reaction-diffusion system")
    print("=" * 60)

    # Spatial domain [-1, 1]
    N = 100
    x = np.linspace(-1, 1, N)
    dx = x[1] - x[0]

    # Initial conditions
    u0 = 1 - erf(10 * (x + 0.7))
    v0 = 1 + erf(10 * (x - 0.7))
    w0 = np.zeros_like(x)

    print("\nInitial conditions:")
    print(f"  u0: mainly on left (max={u0.max():.3f})")
    print(f"  v0: mainly on right (max={v0.max():.3f})")
    print(f"  w0: zero")

    # Build second-derivative matrix with Neumann BCs
    def build_d2(n, dx):
        D2 = (np.diag(-2 * np.ones(n)) +
              np.diag(np.ones(n - 1), 1) +
              np.diag(np.ones(n - 1), -1)) / dx**2
        # Neumann: u'=0 at endpoints (ghost point approach)
        D2[0, 1] = 2 / dx**2
        D2[-1, -2] = 2 / dx**2
        return D2

    D2 = build_d2(N, dx)

    # RHS function — stiff due to fast reaction
    def rhs(t, uvw):
        u, v, w = uvw[:N], uvw[N:2*N], uvw[2*N:]
        # Clip to non-negative to avoid negative concentrations blowing up
        u = np.maximum(u, 0.0)
        v = np.maximum(v, 0.0)
        reaction = 100.0 * u * v
        du = 0.1 * D2 @ u - reaction
        dv = 0.2 * D2 @ v - reaction
        dw = 0.001 * D2 @ w + 2.0 * reaction
        return np.concatenate([du, dv, dw])

    # Use BDF (stiff) solver
    T = 2.0
    t_snap = [0.0, 0.5, 1.0, 1.5, 2.0]
    uvw0 = np.concatenate([u0, v0, w0])

    print(f"\nIntegrating to T={T} (BDF solver, stiff)...")
    sol = solve_ivp(rhs, [0.0, T], uvw0, method='BDF',
                    t_eval=t_snap, rtol=1e-5, atol=1e-7,
                    jac_sparsity=None)

    snapshots = {t: sol.y[:, k] for k, t in enumerate(t_snap)}
    uvw = sol.y[:, -1]

    # Conservation check
    total_init = np.sum(u0 + v0) * dx
    total_final = np.sum(uvw[:N] + uvw[N:2*N]) * dx
    print(f"\n  Total (u+v) at t=0: {total_init:.4f}")
    print(f"  Total (u+v) at t=T: {total_final:.4f}")
    w_max = np.max(uvw[2*N:])
    print(f"  w_max at T={T}: {w_max:.4f}")

    # w should have grown (product formed by reaction)
    assert w_max > 0.01, f"w should have grown from reactions: {w_max}"
    print("  PASS: product w formed by reaction")

    # Total u+v should decrease (consumed by reaction); total of w should grow
    u_final = uvw[:N]
    w_final = uvw[2*N:]
    total_u_init = np.sum(u0) * dx
    total_u_final = np.sum(u_final) * dx
    total_w_final = np.sum(w_final) * dx
    assert total_u_final < total_u_init, \
        f"u should decrease (consumed by reaction): {total_u_final:.4f} >= {total_u_init:.4f}"
    print("  PASS: reactant u consumed during reaction")
    assert total_w_final > 0.05, \
        f"w should accumulate as product: {total_w_final:.4f}"
    print("  PASS: product w accumulated")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(x, u0, color='#0072BD', linestyle='-', label='u₀', linewidth=2)
    axes[0].plot(x, v0, color='#D95319', linestyle='-', label='v₀', linewidth=2)
    axes[0].set_title("Initial conditions", fontsize=11)
    axes[0].legend()

    u_f = uvw[:N]; v_f = uvw[N:2*N]; w_f = uvw[2*N:]
    axes[1].plot(x, u_f, color='#0072BD', linestyle='-', label='u', linewidth=2)
    axes[1].plot(x, v_f, color='#D95319', linestyle='-', label='v', linewidth=2)
    axes[1].plot(x, w_f, color='#77AC30', linestyle='-', label='w', linewidth=2)
    axes[1].set_title(f"Solution at t = {T}", fontsize=11)
    axes[1].legend()

    fig.suptitle("Reaction-diffusion system: u, v react to form w", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "react_diff_sys.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True

if __name__ == "__main__":
    run()
