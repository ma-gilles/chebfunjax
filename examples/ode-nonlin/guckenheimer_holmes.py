"""Guckenheimer-Holmes nonlinear ODE system.

Solves the three-variable system with b<1<c:
  u' = u(1 - u^2 - b v^2 - c w^2)
  v' = v(1 - v^2 - b w^2 - c u^2)
  w' = w(1 - w^2 - b u^2 - c v^2)

Credit: Chebfun example ode-nonlin/GuckenheimerHolmes.m (Nick Trefethen, Feb 2015).
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
    print("Guckenheimer-Holmes system")
    print("=" * 60)

    b = 0.55
    c = 1.5
    assert b < 1.0 < c, "Need b < 1 < c for interesting dynamics"

    def rhs(t, state):
        u, v, w = state
        du = u * (1 - u**2 - b*v**2 - c*w**2)
        dv = v * (1 - v**2 - b*w**2 - c*u**2)
        dw = w * (1 - w**2 - b*u**2 - c*v**2)
        return [du, dv, dw]

    # Start near a saddle point (1, 0, epsilon)
    eps = 0.1
    ic1 = [1.0, eps, eps]
    ic2 = [eps, 1.0, eps]

    T = 50.0
    t_eval = np.linspace(0, T, 5000)

    print(f"\nParameters: b={b}, c={c}")
    print(f"Initial conditions: {ic1}")

    sol1 = solve_ivp(rhs, [0, T], ic1, t_eval=t_eval, rtol=1e-10, atol=1e-12)
    sol2 = solve_ivp(rhs, [0, T], ic2, t_eval=t_eval, rtol=1e-10, atol=1e-12)

    u1, v1, w1 = sol1.y
    u2, v2, w2 = sol2.y

    # The solution should cycle: dominance passes from u to v to w and back
    print(f"\n  At t=0: max component = {np.argmax(np.abs(ic1))} (u)")
    print(f"  At t=T: max component = {np.argmax(np.abs(sol1.y[:, -1]))} ({['u','v','w'][np.argmax(np.abs(sol1.y[:, -1]))]})")

    # Check that trajectory stays bounded
    max_norm = np.max(np.sqrt(u1**2 + v1**2 + w1**2))
    print(f"  Max |state|: {max_norm:.4f}")
    assert max_norm < 5.0

    # Solution oscillates among saddle points (heteroclinic cycle)
    # Check that u, v, w each approach 1 at some point
    assert np.max(u1) > 0.9
    assert np.max(v1) > 0.9
    assert np.max(w1) > 0.9

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(sol1.t, u1, 'b', linewidth=1.2, label="u(t)", alpha=0.8)
    axes[0].plot(sol1.t, v1, 'r', linewidth=1.2, label="v(t)", alpha=0.8)
    axes[0].plot(sol1.t, w1, 'g', linewidth=1.2, label="w(t)", alpha=0.8)
    axes[0].set_xlabel("t"); axes[0].set_ylabel("component")
    axes[0].set_title(f"Guckenheimer-Holmes (b={b}, c={c})", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # 3D trajectory
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    ax3d.plot(u1, v1, w1, 'b', linewidth=0.8, alpha=0.6)
    ax3d.set_xlabel("u"); ax3d.set_ylabel("v"); ax3d.set_zlabel("w")
    ax3d.set_title("3D trajectory", fontsize=10)
    # Remove axes[1] and replace with 3D
    axes[1].remove()

    fig.suptitle("Guckenheimer-Holmes heteroclinic cycle", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "guckenheimer_holmes.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
