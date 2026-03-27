"""Four bugs on a rectangle.

Four bugs start at corners of a 2×1 rectangle, each chasing the next.
We find when the first collision occurs by solving the ODE system.

Credit: Chebfun example ode-nonlin/FourBugs.m (Hrothgar, Nov 2013).
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
    print("Four bugs on a rectangle")
    print("=" * 60)

    # Initial positions: corners of 2x1 rectangle
    # Bug 0: (0,0), Bug 1: (2,0), Bug 2: (2,1), Bug 3: (0,1)
    # Each bug chases the next (cyclically)
    x0 = np.array([0.0, 2.0, 2.0, 0.0])
    y0 = np.array([0.0, 0.0, 1.0, 1.0])

    def rhs(t, state):
        x = state[0:4]
        y = state[4:8]
        # Each bug chases the next: bug i chases bug (i+1)%4
        dx = np.zeros(4)
        dy = np.zeros(4)
        for i in range(4):
            j = (i + 1) % 4
            d = np.sqrt((x[j]-x[i])**2 + (y[j]-y[i])**2)
            if d < 1e-10:
                continue
            dx[i] = (x[j] - x[i]) / d
            dy[i] = (y[j] - y[i]) / d
        return list(dx) + list(dy)

    def min_dist(state):
        x = state[0:4]
        y = state[4:8]
        dists = []
        for i in range(4):
            j = (i + 1) % 4
            dists.append(np.sqrt((x[j]-x[i])**2 + (y[j]-y[i])**2))
        return min(dists)

    # Event: collision (min distance < threshold)
    def event_collision(t, state):
        return min_dist(state) - 0.01
    event_collision.terminal = True
    event_collision.direction = -1

    state0 = list(x0) + list(y0)
    sol = solve_ivp(rhs, [0, 5.0], state0, events=event_collision,
                    rtol=1e-9, atol=1e-11, max_step=0.01,
                    t_eval=np.linspace(0, 5.0, 1000))

    t_all = sol.t
    x_all = sol.y[0:4, :]
    y_all = sol.y[4:8, :]

    t_coll = sol.t_events[0][0] if len(sol.t_events[0]) > 0 else t_all[-1]
    print(f"\nCollision time: t ≈ {t_coll:.6f}")
    print(f"  (For a square with side 1: t = 0.5;  for 2x1 rectangle: larger)")
    assert 0.5 < t_coll < 4.0  # collision should happen between 0.5 and 4 seconds

    # Final positions (at collision)
    x_final = x_all[:, -1]
    y_final = y_all[:, -1]
    center = np.array([np.mean(x_final), np.mean(y_final)])
    print(f"  Collision point ≈ {center}")

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    colors = ['b', 'r', 'g', 'm']
    for i in range(4):
        axes[0].plot(x_all[i], y_all[i], color=colors[i], linewidth=1.4,
                     label=f"Bug {i}")
        axes[0].plot(x_all[i, 0], y_all[i, 0], 'o', color=colors[i], markersize=6)
        axes[0].plot(x_all[i, -1], y_all[i, -1], 's', color=colors[i], markersize=6)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].set_title(f"Bug trajectories (t_coll ≈ {t_coll:.3f})", fontsize=10)
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

    # Distance between successive bugs vs time
    for i in range(4):
        j = (i + 1) % 4
        dist = np.sqrt((x_all[j] - x_all[i])**2 + (y_all[j] - y_all[i])**2)
        axes[1].plot(t_all, dist, color=colors[i], linewidth=1.4, label=f"|{i}→{j}|")
    axes[1].axvline(t_coll, color='k', linestyle='--', linewidth=0.8, label=f"t_coll")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("distance")
    axes[1].set_title("Distance between successive bugs", fontsize=10)
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Four bugs on a 2×1 rectangle", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "four_bugs.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
