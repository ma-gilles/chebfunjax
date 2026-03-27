"""Time-dependent integro-differential equation.

Solves the PDE u_t = 0.02*u'' + (∫u dξ)(∫_a^x u dξ) with Dirichlet BCs,
demonstrating how nonlocal integral operators can appear in PDEs.
Translated from temp/IntegroDiffT.m (original: pde/IntegroDiffT.m).

Original: https://www.chebfun.org/examples/pde/IntegroDiffT.html
Author: Nick Hale, October 2010
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import solve_ivp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()



def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/temp')
    os.makedirs(outdir, exist_ok=True)

    # Grid
    n = 60
    x = np.linspace(-1, 1, n)
    dx = x[1] - x[0]

    # Initial condition: pulse
    u0 = (1 - x**2) * np.exp(-30 * (x + 0.5)**2)

    # PDE: u_t = 0.02*u'' + cumsum(u)*sum(u)
    # where sum(u) = int_{-1}^{1} u dx
    #       cumsum(u)(x) = int_{-1}^{x} u dt

    def pde_rhs(t, u_flat):
        u = u_flat.copy()
        # Enforce BCs
        u[0] = 0; u[-1] = 0

        # Second derivative (finite difference)
        d2u = np.zeros_like(u)
        d2u[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2

        # sum(u) = integral over [-1,1]
        total_u = np.trapezoid(u, x)

        # cumsum(u)(x) = integral from -1 to x
        cumsum_u = np.zeros_like(u)
        for i in range(1, n):
            cumsum_u[i] = np.trapezoid(u[:i+1], x[:i+1])

        rhs = 0.02 * d2u + cumsum_u * total_u
        rhs[0] = 0; rhs[-1] = 0
        return rhs

    t_eval = np.linspace(0, 4, 41)
    print("Solving integro-differential PDE...")
    sol = solve_ivp(pde_rhs, [0, 4], u0, t_eval=t_eval,
                    method='RK45', rtol=1e-4, atol=1e-6)
    print(f"  Status: {sol.message}")

    fig = plt.figure(figsize=(15, 5))

    # --- Panel 1: 3D waterfall / surface ---
    ax1 = fig.add_subplot(131, projection='3d')
    T_grid, X_grid = np.meshgrid(t_eval, x)
    ax1.plot_surface(X_grid, T_grid, sol.y, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x'); ax1.set_ylabel('t'); ax1.set_zlabel('u')
    ax1.set_title('u_t = 0.02·u\'\' + (∫u)(∫u)\nSurface plot', fontsize=9)

    # --- Panel 2: Snapshots ---
    ax2 = fig.add_subplot(132)
    snap_times = [0, 0.5, 1.0, 2.0, 4.0]
    colors2 = plt.cm.Blues(np.linspace(0.3, 1.0, len(snap_times)))
    for t_s, col in zip(snap_times, colors2):
        idx = np.argmin(np.abs(t_eval - t_s))
        ax2.plot(x, sol.y[:, idx], '-', color=col, linewidth=2,
                 label=f't={t_eval[idx]:.1f}')
    ax2.set_title('Snapshots at various times\nPulse diffusion + nonlocal', fontsize=10)
    ax2.set_xlabel('x'); ax2.set_ylabel('u(t,x)')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    # --- Panel 3: Total mass vs time ---
    ax3 = fig.add_subplot(133)
    masses = [np.trapezoid(sol.y[:, i], x) for i in range(len(t_eval))]
    ax3.plot(t_eval, masses, 'b-', linewidth=2.5)
    ax3.set_title('Total mass ∫u dx vs time', fontsize=10)
    ax3.set_xlabel('t'); ax3.set_ylabel('∫u dx')
    ax3.grid(True, alpha=0.3)
    print(f"  Initial mass: {masses[0]:.4f}")
    print(f"  Final mass: {masses[-1]:.4f}")

    fig.suptitle('Time-Dependent Integro-Differential Equation', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'integro_diff_t.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("integro_diff_t: done")
    return True


if __name__ == "__main__":
    run()
