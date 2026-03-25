"""Compacting colloids in a centrifuge.

Solves the Auzerais-Jackson-Russel PDE describing sedimentation of particles
in a centrifuge:
    u_t + [(1-u)^6.55 * (u - (1.85/Pe) * phi_m * u' / (phi_m-u)^2)]' = 0
Translated from temp/CompactingColloids.m (original: pde/CompactingColloids.m).

Original: https://www.chebfun.org/examples/pde/CompactingColloids.html
Authors: Julia Schollick and Rob Style, September 2014
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj


def make_rhs(pe, phi_m, n_pts=80):
    """Build PDE right-hand side using finite differences."""
    x = np.linspace(0, 1, n_pts)
    dx = x[1] - x[0]

    def rhs(t, u):
        # Flux: F = (1-u)^6.55 * (u - (1.85/pe)*phi_m*u' / (phi_m-u)^2)
        eps = 1e-6
        u_c = np.clip(u, eps, phi_m - eps)

        # Gradient of u (interior, Neumann at boundaries)
        du = np.zeros_like(u_c)
        du[1:-1] = (u_c[2:] - u_c[:-2]) / (2*dx)
        du[0] = (u_c[1] - u_c[0]) / dx
        du[-1] = (u_c[-1] - u_c[-2]) / dx

        diff_term = 1.85 / pe * phi_m * du / (phi_m - u_c)**2
        F = (1 - u_c)**6.55 * (u_c - diff_term)

        # dF/dx (interior, no-flux BCs: F=0 at boundaries)
        dFdx = np.zeros_like(u_c)
        dFdx[1:-1] = (F[2:] - F[:-2]) / (2*dx)
        dFdx[0] = 0; dFdx[-1] = 0

        return -dFdx

    return x, rhs


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/temp')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pe = 200
    phi_m = 0.64
    u0_val = 0.3
    t_end = 10.0

    x, rhs_fn = make_rhs(pe, phi_m, n_pts=60)
    u0 = np.full(len(x), u0_val)

    t_eval = np.linspace(0, t_end, 50)

    print(f"Solving compacting colloids PDE (Pe={pe}, phi_m={phi_m})")
    sol = solve_ivp(rhs_fn, [0, t_end], u0, t_eval=t_eval,
                    method='Radau', rtol=1e-4, atol=1e-6)
    print(f"  Status: {sol.message}")

    # --- Panel 1: Space-time plot ---
    T, X = np.meshgrid(t_eval, x)
    im = axes[0].pcolormesh(T, X, sol.y, cmap='YlOrRd',
                              vmin=0, vmax=phi_m, shading='auto')
    axes[0].set_title(f'Concentration u(t,x)\nPe={pe}, φ_m={phi_m:.2f}', fontsize=10)
    axes[0].set_xlabel('Time t'); axes[0].set_ylabel('Position x')
    plt.colorbar(im, ax=axes[0], label='u (concentration)')

    # --- Panel 2: Snapshots at several times ---
    snap_times = [0, 1, 2, 5, 10]
    colors2 = plt.cm.Blues(np.linspace(0.3, 1.0, len(snap_times)))
    for t_snap, col in zip(snap_times, colors2):
        idx = np.argmin(np.abs(t_eval - t_snap))
        axes[1].plot(x, sol.y[:, idx], '-', color=col, linewidth=2,
                     label=f't={t_eval[idx]:.1f}')
    axes[1].axhline(phi_m, color='r', linestyle='--', linewidth=1.5,
                     label=f'φ_m={phi_m}')
    axes[1].set_title('Concentration profiles\nat various times', fontsize=10)
    axes[1].set_xlabel('x'); axes[1].set_ylabel('u(t,x)')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    # --- Panel 3: Different Pe values (steady-state profiles) ---
    pe_vals = [10, 50, 200, 500]
    colors3 = plt.cm.viridis(np.linspace(0, 1, len(pe_vals)))

    for pe_v, col in zip(pe_vals, colors3):
        x_v, rhs_v = make_rhs(pe_v, phi_m, n_pts=60)
        u0_v = np.full(len(x_v), u0_val)
        sol_v = solve_ivp(rhs_v, [0, 30], u0_v, method='Radau',
                          rtol=1e-4, atol=1e-6)
        axes[2].plot(x_v, sol_v.y[:, -1], '-', color=col,
                     linewidth=2, label=f'Pe={pe_v}')

    axes[2].axhline(phi_m, color='r', linestyle='--', linewidth=1.5,
                     label=f'φ_m={phi_m}')
    axes[2].set_title('Steady-state profiles\nfor various Pe', fontsize=10)
    axes[2].set_xlabel('x'); axes[2].set_ylabel('u(∞,x)')
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    fig.suptitle('Compacting Colloids in a Centrifuge', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'compacting_colloids.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("compacting_colloids: done")
    return True


if __name__ == "__main__":
    run()
