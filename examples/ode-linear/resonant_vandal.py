"""Resonance exploited by Carrier and Pearson's vandal.

Demonstrates the resonance phenomenon for  u'' + omega^2 u = f(t)
where forcing at the natural frequency leads to unbounded response.
Models the radiator problem from Carrier & Pearson.

Note: the resonant solution has growing amplitude that requires many Chebyshev
points; we use scipy for the solve and Chebfun only for short-time problems.

Credit: Chebfun example ode-linear/ResonantVandal.m (Nick Trefethen, Aug 2012).
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
    print("Resonance: radiator problem from Carrier & Pearson")
    print("=" * 60)

    # u'' + omega_0^2 u = F*(1 - cos(omega*t))
    # Natural frequency omega_0, resonance at omega = omega_0
    omega_0 = 1.0  # natural frequency
    F = 5.0        # force amplitude
    T = 20.0

    # Use scipy for both resonant and non-resonant cases
    def rhs_driven(t, y, omega):
        return [y[1], F*(1 - np.cos(omega*t)) - omega_0**2 * y[0]]

    print(f"\nResonance (omega=omega_0={omega_0}) via scipy:")
    sol_res = solve_ivp(rhs_driven, [0, T], [0, 0], args=(omega_0,),
                        t_eval=np.linspace(0, T, 2000), rtol=1e-10)
    max_res = np.max(np.abs(sol_res.y[0]))
    print(f"  Max amplitude: {max_res:.4f}  (grows with t)")

    # Verify ODE residual directly
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(sol_res.t, sol_res.y[0])
    t_check = np.linspace(1.0, T-1, 100)
    ode_res = cs(t_check, 2) + omega_0**2 * cs(t_check) - F*(1 - np.cos(omega_0*t_check))
    err_res = np.max(np.abs(ode_res))
    print(f"  ODE residual (spline check): {err_res:.2e}")
    assert err_res < 0.01  # spline derivative approx is ~1e-3 accurate
    assert max_res > 1.0  # amplitude should grow

    # Non-resonant case
    omega_nr = 2.0
    print(f"\nOff-resonance (omega={omega_nr}):")
    sol_nr = solve_ivp(rhs_driven, [0, T], [0, 0], args=(omega_nr,),
                       t_eval=np.linspace(0, T, 2000), rtol=1e-10)
    max_nr = np.max(np.abs(sol_nr.y[0]))
    print(f"  Max amplitude: {max_nr:.4f}  (bounded)")

    # Verify ODE residual for non-resonant case
    cs_nr = CubicSpline(sol_nr.t, sol_nr.y[0])
    t_check_nr = np.linspace(1.0, T-1, 100)
    ode_res_nr = cs_nr(t_check_nr, 2) + omega_0**2 * cs_nr(t_check_nr) - F*(1 - np.cos(omega_nr*t_check_nr))
    err_nr = np.max(np.abs(ode_res_nr))
    print(f"  ODE residual (spline check): {err_nr:.2e}")
    assert err_nr < 0.01
    assert max_nr < 20.0  # bounded (no secular growth)

    # Demonstrate Chebop on a short non-oscillatory segment [0, pi/2]
    # where only a few Chebyshev points are needed
    T_short = float(np.pi / 2)
    dom_short = (0.0, T_short)
    print(f"\nChebop verify: u'' + u = 1 on [0, pi/2] (harmonic osc.):")
    # u'' + u = 1, u(0)=u'(0)=0 => exact: u(t) = 1 - cos(t)
    N_short = Chebop(lambda t, u: u.diff(2) + u, domain=dom_short)
    N_short.lbc = [0.0, 0.0]
    N_short.rbc = float(1 - np.cos(T_short))
    rhs_short = cj.chebfun(lambda t: jnp.ones_like(t), domain=dom_short)
    u_short = N_short.solve(rhs_short)
    t_short = jnp.linspace(0.0, T_short, 100)
    exact_short = 1.0 - jnp.cos(t_short)
    err_short = float(jnp.max(jnp.abs(u_short(t_short) - exact_short)))
    print(f"  Chebop length: {len(u_short)}, max error: {err_short:.2e}")
    assert err_short < 1e-8

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(sol_res.t, sol_res.y[0], 'b', linewidth=1.2, label="resonant (scipy)")
    axes[0].set_title(f"Resonance: ω = ω₀ = {omega_0}", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].plot(sol_nr.t, sol_nr.y[0], 'g', linewidth=1.2, label=f"non-resonant (ω={omega_nr})")
    axes[1].set_title(f"Off-resonance: ω = {omega_nr}", fontsize=10)
    axes[1].legend(fontsize=8)

    fig.suptitle("Resonance in driven harmonic oscillator", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "resonant_vandal.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
