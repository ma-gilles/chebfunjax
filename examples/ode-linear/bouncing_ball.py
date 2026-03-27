"""Bouncing ball trajectory.

A ball is launched at angle theta with initial speed v0.  Under gravity
it follows a parabolic arc; on each bounce the vertical velocity is
reduced by factor k.  The resulting piecewise-quadratic trajectory is
assembled analytically and plotted.

Credit: Chebfun example ode-linear/BouncingBall.m (Filomena Di Tommaso, Feb 2013).
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



def run():
    print("=" * 60)
    print("Bouncing ball trajectory")
    print("=" * 60)

    g = 9.81
    v0 = 30.0
    theta = np.pi / 4
    k = 0.9        # coefficient of restitution
    n_bounces = 8

    v0x = v0 * np.cos(theta)
    v0y_init = v0 * np.sin(theta)

    # Simulate bounces
    times = []
    x_segs = []
    y_segs = []

    t_start = 0.0
    x_pos = 0.0
    v0y = v0y_init

    for i in range(n_bounces):
        # Time of flight for this arc: 0 = v0y*t - 0.5*g*t^2  => t = 2*v0y/g
        if v0y <= 0:
            break
        t_flight = 2.0 * v0y / g
        t_end = t_start + t_flight

        t_arr = np.linspace(t_start, t_end, 200)
        dt = t_arr - t_start
        x_arr = x_pos + v0x * dt
        y_arr = v0y * dt - 0.5 * g * dt**2
        y_arr = np.maximum(y_arr, 0.0)

        times.append(t_arr)
        x_segs.append(x_arr)
        y_segs.append(y_arr)

        x_pos += v0x * t_flight
        t_start = t_end
        v0y *= k
        print(f"  Bounce {i+1}: t_flight={t_flight:.3f}s, x={x_pos:.2f}m, v0y_next={v0y:.3f}")

    assert len(times) > 0
    total_x = max(x_segs[-1])
    print(f"\nTotal horizontal distance: {total_x:.2f} m")
    assert total_x > 50.0  # should travel a reasonable distance

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(times)))

    # x-y trajectory
    for i, (xs, ys) in enumerate(zip(x_segs, y_segs)):
        axes[0].plot(xs, ys, color=colors[i], linewidth=1.6)
    axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
    axes[0].set_title("Bouncing ball: x-y trajectory")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, alpha=0.3)

    # y vs t
    for i, (ts, ys) in enumerate(zip(times, y_segs)):
        axes[1].plot(ts, ys, color=colors[i], linewidth=1.6)
    axes[1].set_xlabel("t (s)"); axes[1].set_ylabel("y (m)")
    axes[1].set_title("Bouncing ball: height vs time")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(_here, "bouncing_ball.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
