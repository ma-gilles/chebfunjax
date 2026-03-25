"""Resonance exploited by Carrier and Pearson's vandal.

Demonstrates the resonance phenomenon for  u'' + omega^2 u = f(t)
where forcing at the natural frequency leads to unbounded response.
Models the radiator problem from Carrier & Pearson.

Credit: Chebfun example ode-linear/ResonantVandal.m (Nick Trefethen, Aug 2012).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Resonance: radiator problem from Carrier & Pearson")
    print("=" * 60)

    # u'' + omega_0^2 u = F*(1 - cos(omega*t))
    # Natural frequency omega_0, resonance at omega = omega_0
    # At resonance: amplitude grows linearly in t
    omega_0 = 2.0  # natural frequency (sqrt(k/m))
    F = 50.0       # force amplitude

    # At resonance omega = omega_0, solve:
    # u'' + omega_0^2 u = F*(1 - cos(omega_0*t)), u(0) = u'(0) = 0
    # Exact solution: u = F/omega_0^2 - F/(2*omega_0) t sin(omega_0 t)
    # The amplitude grows as t * sin(omega_0 t)

    T = 10.0  # solve on [0, T]
    dom = (0.0, T)

    def exact_resonance(t):
        return F / omega_0**2 - (F / (2 * omega_0)) * t * np.sin(omega_0 * t)

    N_res = Chebop(lambda t, u: u.diff(2) + omega_0**2 * u, domain=dom)
    N_res.lbc = [0.0, 0.0]  # u(0)=0, u'(0)=0
    # rbc at t=T from exact
    N_res.rbc = float(exact_resonance(T))
    rhs_res = cj.chebfun(lambda t: F * (1.0 - jnp.cos(omega_0 * t)), domain=dom)
    u_res = N_res.solve(rhs_res)

    t_test = np.linspace(0.0, T, 300)
    err = float(np.max(np.abs(np.array(u_res(jnp.array(t_test, dtype=jnp.float64))) -
                               exact_resonance(t_test))))
    print(f"\nResonance (omega=omega_0={omega_0}): max error = {err:.2e}")
    assert err < 1e-6

    # Non-resonant case: omega ≠ omega_0
    omega_nr = 3.0  # off-resonance
    # Exact: particular solution u_p = F*(1 - cos(omega*t)) / (omega_0^2 - omega^2) is wrong
    # Full: u = A cos(omega_0 t) + B sin(omega_0 t) + F/(omega_0^2-omega^2)*(1-cos(omega t))
    # With u(0)=u'(0)=0:
    # u(0) = A + F/(omega_0^2-omega_nr^2) = 0 => A = -F/(omega_0^2-omega_nr^2)
    # u'(0) = B*omega_0 + F*omega_nr/(omega_0^2-omega_nr^2)*sin(0) = 0 => B=0
    D = omega_0**2 - omega_nr**2
    A = -F / D
    def exact_nonres(t):
        return A * np.cos(omega_0 * t) + F / D * (1 - np.cos(omega_nr * t))

    N_nr = Chebop(lambda t, u: u.diff(2) + omega_0**2 * u, domain=dom)
    N_nr.lbc = [0.0, 0.0]
    N_nr.rbc = float(exact_nonres(T))
    rhs_nr = cj.chebfun(lambda t: F * (1.0 - jnp.cos(omega_nr * t)), domain=dom)
    u_nr = N_nr.solve(rhs_nr)

    err_nr = float(np.max(np.abs(np.array(u_nr(jnp.array(t_test, dtype=jnp.float64))) -
                                  exact_nonres(t_test))))
    print(f"Non-resonance (omega={omega_nr}): max error = {err_nr:.2e}")
    assert err_nr < 1e-6

    print(f"\nMax displacement:")
    print(f"  Off-resonance (omega={omega_nr}): {float(jnp.max(jnp.abs(u_nr(jnp.array(t_test, dtype=jnp.float64))))):.2f}")
    print(f"  Resonance (omega={omega_0}): {float(jnp.max(jnp.abs(u_res(jnp.array(t_test, dtype=jnp.float64))))):.2f}")

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    t_plot = jnp.linspace(0.0, T, 500)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(t_plot, u_res(t_plot), 'b', linewidth=1.6, label="chebfunjax")
    axes[0].plot(t_test, exact_resonance(t_test), 'r--', linewidth=1.2, label="exact")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("u(t)")
    axes[0].set_title(f"Resonance: ω = ω₀ = {omega_0}", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_plot, u_nr(t_plot), 'b', linewidth=1.6, label="chebfunjax")
    axes[1].plot(t_test, exact_nonres(t_test), 'r--', linewidth=1.2, label="exact")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("u(t)")
    axes[1].set_title(f"Off-resonance: ω = {omega_nr}, ω₀ = {omega_0}", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Resonance in driven harmonic oscillator", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "resonant_vandal.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
