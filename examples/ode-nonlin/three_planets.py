"""Pythagorean three-planet problem.

Three planets at rest at positions forming a 3-4-5 right triangle,
attracting each other with 1/r^2 gravity. Demonstrates chaotic dynamics.

Credit: Chebfun example ode-nonlin/ThreePlanets.m (Hashemi & Trefethen, Dec 2014).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj


def run():
    print("=" * 60)
    print("Pythagorean three-planet problem")
    print("=" * 60)

    # Masses: m1=3, m2=4, m3=5 at (1,3), (-2,-1), (1,-1)
    # (Pythagorean configuration: sides 3,4,5)
    m1, m2, m3 = 3.0, 4.0, 5.0
    G = 1.0

    # Initial positions
    x10, y10 =  1.0,  3.0
    x20, y20 = -2.0, -1.0
    x30, y30 =  1.0, -1.0

    def rhs(t, state):
        x1, y1, x2, y2, x3, y3 = state[:6]
        vx1, vy1, vx2, vy2, vx3, vy3 = state[6:]

        def force(xi, yi, xj, yj, mj):
            dx, dy = xj-xi, yj-yi
            r3 = (dx**2 + dy**2)**1.5
            return G*mj*dx/r3, G*mj*dy/r3

        f12x, f12y = force(x1,y1,x2,y2,m2)
        f13x, f13y = force(x1,y1,x3,y3,m3)
        f21x, f21y = force(x2,y2,x1,y1,m1)
        f23x, f23y = force(x2,y2,x3,y3,m3)
        f31x, f31y = force(x3,y3,x1,y1,m1)
        f32x, f32y = force(x3,y3,x2,y2,m2)

        return [vx1, vy1, vx2, vy2, vx3, vy3,
                (f12x+f13x)/m1, (f12y+f13y)/m1,
                (f21x+f23x)/m2, (f21y+f23y)/m2,
                (f31x+f32x)/m3, (f31y+f32y)/m3]

    ic = [x10, y10, x20, y20, x30, y30,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    T = 100.0
    print(f"\nIntegrating Pythagorean problem to T={T}...")
    sol = solve_ivp(rhs, [0, T], ic, t_eval=np.linspace(0, T, 20000),
                    rtol=1e-10, atol=1e-12)

    x1, y1 = sol.y[0], sol.y[1]
    x2, y2 = sol.y[2], sol.y[3]
    x3, y3 = sol.y[4], sol.y[5]

    # Center of mass should be conserved
    xcm0 = (m1*x10 + m2*x20 + m3*x30) / (m1+m2+m3)
    ycm0 = (m1*y10 + m2*y20 + m3*y30) / (m1+m2+m3)
    xcm = (m1*sol.y[0] + m2*sol.y[2] + m3*sol.y[4]) / (m1+m2+m3)
    ycm = (m1*sol.y[1] + m2*sol.y[3] + m3*sol.y[5]) / (m1+m2+m3)
    print(f"  Initial CM: ({xcm0:.4f}, {ycm0:.4f})")
    print(f"  CM drift: max {np.max(np.abs(xcm - xcm0)):.2e}")
    assert np.max(np.abs(xcm - xcm0)) < 0.01

    # The system should show complex behavior
    r12 = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    min_r12 = np.min(r12)
    print(f"  Min separation (1-2): {min_r12:.4f}")
    assert min_r12 > 0.01  # no collision detected

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    colors = ['b', 'r', 'g']
    labels = [f'm₁={m1:.0f}', f'm₂={m2:.0f}', f'm₃={m3:.0f}']
    for (xi, yi), c, lab in zip([(x1,y1),(x2,y2),(x3,y3)], colors, labels):
        axes[0].plot(xi, yi, color=c, linewidth=0.5, alpha=0.6, label=lab)
        axes[0].plot(xi[0], yi[0], 'o', color=c, markersize=6)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].set_title("Pythagorean three-body trajectories", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.2)

    t_short = sol.t[:2000]
    for (xi, yi), c, lab in zip([(x1,y1),(x2,y2),(x3,y3)], colors, labels):
        axes[1].plot(t_short, xi[:2000], color=c, linewidth=1.0, label=lab)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("x(t)")
    axes[1].set_title("x-coordinates vs time", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Pythagorean three-planet problem", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "three_planets.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
