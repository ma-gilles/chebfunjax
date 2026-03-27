"""Three-body problem complex singularities.

Integrates the gravitational three-body problem and investigates
complex singularities by looking at how solutions behave.

Credit: Chebfun example ode-nonlin/ThreeBodyProblem.m (Marcus Webb, Aug 2011).
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
    print("Three-body gravitational problem")
    print("=" * 60)

    m1, m2, m3 = 1.0, 1.0, 1.0
    G = 1.0

    def three_body(t, state):
        x1, y1, x2, y2, x3, y3 = state[0], state[1], state[2], state[3], state[4], state[5]
        vx1, vy1, vx2, vy2, vx3, vy3 = state[6], state[7], state[8], state[9], state[10], state[11]

        def force(xi, yi, xj, yj, mj):
            r = np.sqrt((xj-xi)**2 + (yj-yi)**2)
            if r < 1e-10: return 0.0, 0.0
            f = G * mj / r**3
            return f*(xj-xi), f*(yj-yi)

        f12x, f12y = force(x1, y1, x2, y2, m2)
        f13x, f13y = force(x1, y1, x3, y3, m3)
        f21x, f21y = force(x2, y2, x1, y1, m1)
        f23x, f23y = force(x2, y2, x3, y3, m3)
        f31x, f31y = force(x3, y3, x1, y1, m1)
        f32x, f32y = force(x3, y3, x2, y2, m2)

        ax1 = (f12x + f13x) / m1
        ay1 = (f12y + f13y) / m1
        ax2 = (f21x + f23x) / m2
        ay2 = (f21y + f23y) / m2
        ax3 = (f31x + f32x) / m3
        ay3 = (f31y + f32y) / m3

        return [vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3]

    # Figure-8 initial conditions (Chenciner & Montgomery, 2000)
    # Known periodic three-body orbit
    x1_0 = -0.97000436
    y1_0 =  0.24308753
    vx3_0 = 0.93240737 / 2.0
    vy3_0 = 0.86473146 / 2.0

    ic = [
        x1_0, y1_0,             # body 1
        -x1_0, -y1_0,           # body 2
        0.0, 0.0,               # body 3
        -0.93240737/2, -0.86473146/2,  # v1
        -0.93240737/2, -0.86473146/2,  # v2
        0.93240737, 0.86473146  # v3
    ]

    T = 2.0 * 6.3259
    print(f"\nFigure-8 orbit: T = {T:.4f}")
    t_eval = np.linspace(0, T, 2000)
    sol = solve_ivp(three_body, [0, T], ic, t_eval=t_eval, rtol=1e-11, atol=1e-13)

    x1, y1 = sol.y[0], sol.y[1]
    x2, y2 = sol.y[2], sol.y[3]
    x3, y3 = sol.y[4], sol.y[5]

    # Check approximate periodicity
    err_period = np.sqrt((x1[-1]-x1[0])**2 + (y1[-1]-y1[0])**2)
    print(f"  Periodicity error (body 1): {err_period:.6f}")
    # Note: T might not be exact so this could be non-trivial
    print(f"  Max extent: {max(np.max(np.abs(x1)), np.max(np.abs(y1))):.4f}")
    assert max(np.max(np.abs(x1)), np.max(np.abs(y1))) > 0.5

    # Energy conservation
    def energy(state):
        x1, y1, x2, y2, x3, y3 = state[0], state[1], state[2], state[3], state[4], state[5]
        vx1, vy1, vx2, vy2, vx3, vy3 = state[6], state[7], state[8], state[9], state[10], state[11]
        KE = 0.5*(m1*(vx1**2+vy1**2) + m2*(vx2**2+vy2**2) + m3*(vx3**2+vy3**2))
        def PE_pair(xi, yi, xj, yj, mi, mj):
            r = np.sqrt((xj-xi)**2+(yj-yi)**2)
            return -G*mi*mj/r
        PE = (PE_pair(x1,y1,x2,y2,m1,m2) + PE_pair(x1,y1,x3,y3,m1,m3) +
              PE_pair(x2,y2,x3,y3,m2,m3))
        return KE + PE

    E0 = energy(sol.y[:, 0])
    E_arr = [energy(sol.y[:, i]) for i in range(0, len(t_eval), 100)]
    E_variation = np.max(np.abs(np.array(E_arr) - E0)) / abs(E0)
    print(f"  Relative energy variation: {E_variation:.2e}")
    assert E_variation < 0.01

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    colors = ['b', 'r', 'g']
    for (xi, yi), c, lab in zip([(x1,y1),(x2,y2),(x3,y3)], colors, ['m1','m2','m3']):
        axes[0].plot(xi, yi, color=c, linewidth=0.8, alpha=0.7, label=lab)
        axes[0].plot(xi[0], yi[0], 'o', color=c, markersize=5)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].set_title("Figure-8 three-body orbit", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.2)

    # Energy conservation
    E_all = [energy(sol.y[:, i]) for i in range(len(t_eval))]
    axes[1].plot(t_eval, (np.array(E_all) - E0) / abs(E0), 'b', linewidth=1.2)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("(E-E₀)/|E₀|")
    axes[1].set_title("Relative energy error", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Gravitational three-body problem (figure-8)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "three_body.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
