"""Square limit cycle (heteroclinic cycle).

Integrates the Johnson-Tucker ODE system:
  x' = (delta*x + y)(x^2 - 1)
  y' = (delta*y - x)(y^2 - 1)
whose solutions approach a square limit cycle with four saddle points.

Credit: Chebfun example ode-nonlin/SquareCycle.m (Nick Trefethen, May 2019).
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



def run():
    print("=" * 60)
    print("Square limit cycle (heteroclinic cycle)")
    print("=" * 60)

    delta = -0.01  # negative delta gives limit cycle approaching (±1, ±1) saddles

    def rhs(t, state):
        x, y = state
        dx = (delta*x + y) * (x**2 - 1)
        dy = (delta*y - x) * (y**2 - 1)
        return [dx, dy]

    # Start near interior (limit cycle attracts for delta < 0)
    ics = [
        [0.5, 0.1],
        [0.3, 0.5],
        [-0.5, 0.3],
    ]
    T = 200.0

    print(f"\nIntegrating with delta={delta}...")
    solutions = []
    for ic in ics:
        sol = solve_ivp(rhs, [0, T], ic, max_step=0.05,
                        rtol=1e-9, atol=1e-11,
                        t_eval=np.linspace(0, T, 20000))
        solutions.append(sol)
        x_final = sol.y[0, -2000:]
        y_final = sol.y[1, -2000:]
        print(f"  ic={ic}: late-time x in [{np.min(x_final):.3f}, {np.max(x_final):.3f}]")
        # Should be cycling around the square corners (±1, ±1)
        assert np.max(np.abs(x_final)) > 0.8

    # The saddle points are at (±1, 0), (0, ±1)... actually the limit cycle
    # connects the saddles at (1,0), (0,-1), (-1,0), (0,1)? No - at the corners.
    # The system has fixed points where dx=dy=0:
    # (x^2=1 or y=-delta*x) and (y^2=1 or x=delta*y)
    # Corner fixed points (saddles): (±1, ±1)
    print("\nVerifying fixed points at corners (±1, ±1):")
    for sx, sy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
        dx = (delta*sx + sy) * (sx**2 - 1)
        dy = (delta*sy - sx) * (sy**2 - 1)
        print(f"  ({sx:+d},{sy:+d}): f=(dx,dy)=({dx:.2e},{dy:.2e})")
        assert abs(dx) < 1e-10 and abs(dy) < 1e-10

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    colors = ['b', 'r', 'g']
    fig, axes = plt.subplots(1, 2)

    for sol, ic, c in zip(solutions, ics, colors):
        # Plot last part to show the limit cycle
        n_show = 15000
        x_show = sol.y[0, -n_show:]
        y_show = sol.y[1, -n_show:]
        axes[0].plot(x_show, y_show, color=c, linewidth=0.8, alpha=0.7,
                     label=f"ic={ic}")

    # Mark the corner saddles
    for sx, sy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
        axes[0].plot(sx, sy, 'k*', markersize=8)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].set_title(f"Square limit cycle (δ={delta})", fontsize=10)
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.2)

    # Time series
    sol0 = solutions[0]
    n_ts = 5000
    t_ts = sol0.t[-n_ts:]
    axes[1].plot(t_ts, sol0.y[0, -n_ts:], 'b', linewidth=0.8, label="x(t)")
    axes[1].plot(t_ts, sol0.y[1, -n_ts:], 'r', linewidth=0.8, label="y(t)")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("state")
    axes[1].set_title("Time series along limit cycle", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Johnson-Tucker square limit cycle (δ={delta}, limit cycle at corners ±1)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "square_cycle.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
