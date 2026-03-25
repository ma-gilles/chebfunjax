"""Variable-coefficient linear BVPs.

Solves linear BVPs with variable coefficients, including a Legendre-type
equation and an Airy-type equation, verifying against known exact solutions.

Credit: Inspired by Chebfun ode-linear examples.
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
from chebfunjax.plotting import plot
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Variable-coefficient linear BVPs")
    print("=" * 60)

    # --- BVP 1: (1-x^2)*u'' - 2x*u' + 2u = 0 (Legendre P_1 = x) -----
    # This is the Legendre equation with n=1: (1-x^2)*u'' - 2x*u' + n(n+1)*u = 0
    # For n=1: (1-x^2)*u'' - 2x*u' + 2u = 0; exact: u = x
    # Rewrite: u'' - (2x/(1-x^2))*u' + 2/(1-x^2)*u = 0
    # Use domain (-0.9, 0.9) to avoid singularity at x = ±1
    print("\n--- Legendre equation (n=1): (1-x^2)u'' - 2x*u' + 2u = 0 ---")
    dom1 = (-0.9, 0.9)
    exact1 = lambda x: x  # P_1(x) = x
    N1 = Chebop(
        lambda x, u: (1.0 - x**2) * u.diff(2) - 2.0 * x * u.diff() + 2.0 * u,
        domain=dom1
    )
    N1.lbc = float(exact1(jnp.array(-0.9)))
    N1.rbc = float(exact1(jnp.array(0.9)))
    u1 = N1.solve(0.0)
    x_test1 = jnp.linspace(-0.9, 0.9, 200)
    err1 = float(jnp.max(jnp.abs(u1(x_test1) - exact1(x_test1))))
    print(f"  Chebfun length: {len(u1)}")
    print(f"  Max error vs x: {err1:.2e}")
    assert err1 < 1e-8, f"Legendre error too large: {err1}"

    # --- BVP 2: u'' + (1+x)*u = 0, u(0)=1, u(1)=cos(1.5)*... --------
    # Exact: Not simple, use numerical reference
    # Better: u'' - x*u = 0 with u(0)=Ai(0), u(1)=Ai(1) (Airy equation)
    # Ai(x) satisfies u'' = x*u with Ai(0) = 3^(-2/3)/Gamma(2/3)
    from scipy.special import airy
    print("\n--- Airy equation: u'' - x*u = 0 ---")
    dom2 = (0.0, 3.0)
    # BCs from Airy function
    Ai0, _, _, _ = airy(0.0)
    Ai3, _, _, _ = airy(3.0)
    N2 = Chebop(lambda x, u: u.diff(2) - x * u, domain=dom2)
    N2.lbc = float(Ai0)
    N2.rbc = float(Ai3)
    u2 = N2.solve(0.0)
    x_test2 = jnp.linspace(0.0, 3.0, 200)
    Ai_exact = np.array([airy(float(xi))[0] for xi in x_test2])
    err2 = float(jnp.max(jnp.abs(u2(x_test2) - jnp.array(Ai_exact))))
    print(f"  Chebfun length: {len(u2)}")
    print(f"  Max error vs Ai(x): {err2:.2e}")
    assert err2 < 1e-8, f"Airy error too large: {err2}"

    # --- BVP 3: u'' + (2 + sin(x))*u = 0, u(0)=0, u(pi)=0 ----------
    # This is a variable-frequency oscillator BVP
    # No closed form, but we can check the ODE residual
    print("\n--- Variable-frequency oscillator ---")
    dom3 = (0.0, float(jnp.pi))
    # Note: inside Chebop, x is a Chebfun, so use x.sin() not jnp.sin(x)
    N3 = Chebop(lambda x, u: u.diff(2) + (2.0 + x.sin()) * u, domain=dom3)
    N3.lbc = 0.0
    N3.rbc = 0.0
    # Use a forcing: u'' + (2+sin(x))*u = sin(x), u(0)=u(pi)=0
    f3 = cj.chebfun(lambda x: jnp.sin(x), domain=dom3)
    u3 = N3.solve(f3)
    x_test3 = jnp.linspace(0.0, float(jnp.pi), 200)
    # Check residual directly
    res3 = u3.diff(2)(x_test3) + (2.0 + jnp.sin(x_test3)) * u3(x_test3) - jnp.sin(x_test3)
    max_res3 = float(jnp.max(jnp.abs(res3)))
    print(f"  Chebfun length: {len(u3)}")
    print(f"  Max ODE residual: {max_res3:.2e}")
    assert max_res3 < 1e-8, f"Residual too large: {max_res3}"

    # Check BCs
    assert abs(float(u3(jnp.array(0.0)))) < 1e-10
    assert abs(float(u3(jnp.array(float(jnp.pi))))) < 1e-10

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(u1, title="Bessel BVP: y″ + y/x − y·ν²/x² = 0",
                   label="u₁ (ν=0)")
    plot(u2, ax=ax, color="#E04040", label="u₂ (ν=1)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "bessel_bvp.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
