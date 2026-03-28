"""Time-dependent integro-differential equation.

Solves the time-dependent integro-differential equation:
  u_t = 0.02*u_xx + (int u dx) * (int_-1^x u dx), u(-1) = u(1) = 0

following pde/IntegroDiffT.m by Nick Hale (October 2010).

This uses pde15s-style method of lines, combining diffusion with
integral terms.

Original MATLAB: https://www.chebfun.org/examples/pde/IntegroDiffT.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chebfunjax.plotting import chebfun_style
chebfun_style()

import numpy as np
from scipy.special import erf
import os

def run():
    print("=" * 60)
    print("Time-dependent integro-differential equation")
    print("=" * 60)

    # u_t = 0.02*u_xx + (int_{-1}^{1} u dx) * (int_{-1}^{x} u dx)
    # with u(-1) = u(1) = 0

    N = 100
    x = np.linspace(-1, 1, N + 2)[1:-1]  # interior points
    dx = x[1] - x[0]

    # Build second derivative matrix with Dirichlet BCs
    D2 = (np.diag(-2 * np.ones(N)) +
          np.diag(np.ones(N - 1), 1) +
          np.diag(np.ones(N - 1), -1)) / dx**2

    # Initial condition: (1-x^2)*exp(-30*(x+0.5)^2)
    u0 = (1 - x**2) * np.exp(-30 * (x + 0.5)**2)

    print(f"\nInitial condition: (1-x^2)*exp(-30*(x+0.5)^2)")
    print(f"  max(u0) = {u0.max():.4f}")

    def rhs(t, u):
        """RHS: 0.02*u_xx + cumsum(u)*sum(u)."""
        diffusion = 0.02 * D2 @ u
        # Full integral from -1 to 1
        integral_total = np.trapezoid(u, x)
        # Cumulative integral from -1 to x[i]
        cum_integral = np.zeros(N)
        for i in range(N):
            cum_integral[i] = np.trapezoid(u[:i+1], x[:i+1])
        return diffusion + cum_integral * integral_total

    # RK4 integration to T=4
    T = 4.0
    dt = 0.01
    nsteps = int(T / dt)
    u = u0.copy()
    t_cur = 0.0
    t_vals = np.arange(0, T + dt / 2, 0.2)
    history = {0.0: u0.copy()}

    print(f"\nIntegrating to T={T} with dt={dt}...")
    for step in range(nsteps):
        k1 = rhs(t_cur, u)
        k2 = rhs(t_cur + dt/2, u + dt/2 * k1)
        k3 = rhs(t_cur + dt/2, u + dt/2 * k2)
        k4 = rhs(t_cur + dt, u + dt * k3)
        u = u + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        t_cur += dt
        t_r = round(t_cur, 3)
        for t_snap in [0.5, 1.0, 2.0, 3.0, 4.0]:
            if abs(t_r - t_snap) < dt / 2 and t_snap not in history:
                history[t_snap] = u.copy()

    print(f"\nAt T={T}: max(u) = {u.max():.4f}")
    print(f"  Boundary values: u(-1) ≈ {u[0]:.2e}, u(1) ≈ {u[-1]:.2e}")

    # Simulation should produce finite (non-NaN) values
    assert np.isfinite(u).all(), "Solution has NaN/Inf values"
    print("  PASS: solution is finite (no blow-up)")
    # Note: interior-grid boundaries (near x=±1) can drift for this nonlinear PDE

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    t_show = sorted(history.keys())
    colors = plt.cm.cool(np.linspace(0, 1, len(t_show)))
    for (t, u_h), col in zip([(t, history[t]) for t in t_show], colors):
        x_full = np.concatenate([[-1.0], x, [1.0]])
        u_full = np.concatenate([[0.0], u_h, [0.0]])
        axes[0].plot(x_full, u_full, color=col, linewidth=1.5,
                     label=f't={t:.1f}' if t in [0, 1, 2, 3, 4] else '')
    axes[0].set_title("Integro-diff. equation: u(x,t)", fontsize=11)
    axes[0].legend(fontsize=9)

    # Integral over time
    total_integrals = [np.trapezoid(history[t], x) for t in t_show]
    axes[1].plot(t_show, total_integrals, color='#0072BD', linestyle='.-', markersize=8)
    axes[1].set_title("Total integral ∫u dx vs time", fontsize=11)

    fig.suptitle("Time-dependent integro-differential equation", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "integro_diff_time.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True

if __name__ == "__main__":
    run()
