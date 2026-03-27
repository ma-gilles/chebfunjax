"""Phase portraits with vector field quiver plots.

Illustrates phase portraits for several nonlinear ODE systems,
plotting trajectories and direction fields.

Credit: Chebfun example ode-nonlin/ChebopQuiver.m (Asgeir Birkisson, Nov 2015).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.operators.chebop import Chebop

def phase_portrait(ax, f, x_range, y_range, title, trajectories=None):
    """Draw quiver plot and optional trajectories for system x'=f1(x,y), y'=f2(x,y)."""
    xs = np.linspace(*x_range, 15)
    ys = np.linspace(*y_range, 15)
    X, Y = np.meshgrid(xs, ys)
    DX, DY = f(X, Y)
    N = np.sqrt(DX**2 + DY**2 + 1e-15)
    ax.quiver(X, Y, DX/N, DY/N, alpha=0.5, scale=20, color='gray')

    if trajectories:
        for x0, col in trajectories:
            def rhs(t, y): return list(f(y[0], y[1]))
            for T in [3.0, -3.0]:
                sol = solve_ivp(rhs, [0, T], x0, max_step=0.05,
                                dense_output=True, rtol=1e-8)
                ax.plot(sol.y[0], sol.y[1], color=col, linewidth=1.2, alpha=0.7)
            ax.plot(*x0, 'o', color=col, markersize=4)

    ax.set_xlim(*x_range); ax.set_ylim(*y_range)
    ax.set_title(title, fontsize=9)

def run():
    print("=" * 60)
    print("Phase portraits with quiver field plots")
    print("=" * 60)

    # System 1: van der Pol  x' = y, y' = mu*(1-x^2)*y - x
    mu = 0.5
    def vdp(x, y): return y, mu * (1 - x**2) * y - x

    # System 2: Lotka-Volterra  x' = ax - bxy, y' = -cy + dxy
    a, b, c, d = 1.0, 0.5, 1.0, 0.5
    def lotka_volterra(x, y): return a*x - b*x*y, -c*y + d*x*y

    # System 3: nonlinear pendulum  theta' = omega, omega' = -sin(theta)
    def pendulum(x, y): return y, -np.sin(x)

    # System 4: saddle node  x' = x^2 - 1, y' = -y
    def saddle_node(x, y): return x**2 - 1, -y

    print("\nGenerating phase portraits for 4 systems...")

    ics = [
        ([0.5, 0.5], 'b'), ([1.5, 0.0], 'r'),
        ([0.0, 1.0], 'g'), ([-1.5, 0.5], 'm'),
    ]

    fig, axes = plt.subplots(2, 2)
    phase_portrait(axes[0,0], vdp, (-3, 3), (-3, 3), f"van der Pol (μ={mu})", ics)
    phase_portrait(axes[0,1], lotka_volterra, (0, 4), (0, 4), "Lotka-Volterra",
                   [([0.5, 1.0], 'b'), ([1.5, 0.5], 'r'), ([2.0, 1.5], 'g')])
    phase_portrait(axes[1,0], pendulum, (-4, 4), (-4, 4), "Nonlinear pendulum",
                   [([0.5, 0.0], 'b'), ([2.0, 0.0], 'r'), ([0.0, 2.0], 'g')])
    phase_portrait(axes[1,1], saddle_node, (-2, 2), (-2, 2), "Saddle node",
                   [([0.0, 0.5], 'b'), ([0.5, -1.0], 'r'), ([-0.5, 1.0], 'g')])

    fig.suptitle("Phase portraits with direction fields", fontsize=11)
    fig.tight_layout()
    _here = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(os.path.join(_here, "chebop_quiver.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Phase portraits generated successfully.")

    # Quick assertion on van der Pol limit cycle
    def rhs_vdp(t, y): return [y[1], mu*(1 - y[0]**2)*y[1] - y[0]]
    sol = solve_ivp(rhs_vdp, [0, 20], [1.0, 0.0], rtol=1e-8, atol=1e-10,
                    t_eval=np.linspace(15, 20, 500))
    amp = np.max(np.abs(sol.y[0]))
    print(f"\nvan der Pol limit cycle amplitude ≈ {amp:.4f}")
    assert 1.5 < amp < 2.5  # limit cycle amplitude ≈ 2 for small mu

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
