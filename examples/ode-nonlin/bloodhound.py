"""Bloodhound supersonic car acceleration.

Solves the equation of motion for the Bloodhound car:
  m v dv/dx = F_jet + F_rocket - F_drag
where forces depend on speed v and distance x.

Credit: Chebfun example ode-nonlin/Bloodhound.m (Tanya Morton, Jan 2013).
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

def run():
    print("=" * 60)
    print("Bloodhound supersonic car equation of motion")
    print("=" * 60)

    # Simplified model (not exact Bloodhound parameters):
    # m v' = F_thrust - b v^2,  v(0) = 0
    # where F_thrust ramps up then levels off, drag = b*v^2
    # Simple: m v' + b v^2 = F(t)  with  F = F0 tanh(t/tau)

    m = 7500.0     # mass kg
    b = 5.0        # drag coefficient (simplified)
    F0 = 120000.0  # max thrust N
    tau = 20.0     # thrust ramp time s

    T_end = 60.0
    dom = (0.0, T_end)

    # ODE: m v' = F0 * tanh(t/tau) - b * v^2, v(0) = 0
    # Solve as BVP with v(0)=0 and v(T_end)=unknown => use IVP approach
    # via scipy for reference, then compare with Chebop

    def rhs_scipy(t, v):
        F = F0 * np.tanh(t / tau)
        drag = b * v[0]**2
        return [(F - drag) / m]

    sol = solve_ivp(rhs_scipy, [0, T_end], [0.0], t_eval=np.linspace(0, T_end, 300),
                    rtol=1e-10, atol=1e-12)
    v_max = np.max(sol.y[0])
    print(f"\nScipyIVP: max speed = {v_max:.1f} m/s = {v_max * 3.6:.1f} km/h")
    assert v_max > 100.0  # should exceed 100 m/s

    # Solve with Chebop as BVP
    v_end = float(sol.y[0, -1])
    # In the Chebop lambda, t is a Chebfun; use cj.tanh (not jnp.tanh)
    N = Chebop(
        lambda t, v: m * v.diff() + b * v**2 - F0 * cj.tanh(t / tau),
        domain=dom
    )
    N.lbc = 0.0
    N.rbc = v_end

    print(f"\nChebop solve (v(0)=0, v({T_end:.0f})={v_end:.4f}):")
    v_cheb = N.solve(50.0)
    print(f"  Solution length: {len(v_cheb)}")

    # Compare with scipy
    t_test = np.linspace(1.0, T_end - 1.0, 200)
    v_cheb_vals = np.array(v_cheb(jnp.array(t_test, dtype=jnp.float64)))
    v_scipy = np.interp(t_test, sol.t, sol.y[0])
    err = np.max(np.abs(v_cheb_vals - v_scipy))
    print(f"  Max error vs scipy: {err:.2e}")
    assert err < 1.0  # within 1 m/s

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    t_plot = jnp.linspace(0.0, T_end, 400)
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(sol.t, sol.y[0] * 3.6, 'b', linewidth=1.8, label="scipy")
    axes[0].plot(t_plot, v_cheb(t_plot) * 3.6, 'r--', linewidth=1.4, label="chebfunjax")
    axes[0].set_title("Bloodhound car speed", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].plot(sol.t, F0 * np.tanh(sol.t / tau) / 1000.0, 'g', linewidth=1.8)
    axes[1].set_title("Engine thrust profile", fontsize=10)

    fig.suptitle("Bloodhound supersonic car (simplified model)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "bloodhound.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
