"""Linear ODEs from Wikipedia examples.

Demonstrates solving three classic linear BVPs with known exact
solutions, verifying spectral accuracy.

Credit: Inspired by Chebfun ode-linear examples.
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Linear ODEs from Wikipedia")
    print("=" * 60)

    # --- ODE 1: y'' - 3y' + 2y = 0, y(0)=1, y'(0)=4 ------------------
    # Characteristic equation: r^2 - 3r + 2 = 0 => r = 1, 2
    # General solution: y = C1*exp(x) + C2*exp(2x)
    # BCs y(0)=1, y'(0)=4: C1 + C2 = 1, C1 + 2*C2 = 4 => C2=3, C1=-2
    # Exact: y = -2*exp(x) + 3*exp(2x)
    # Convert to BVP on [0, 1]: y(0) = 1, y(1) = -2*e + 3*e^2
    print("\n--- ODE 1: y'' - 3y' + 2y = 0 ---")
    dom1 = (0.0, 1.0)
    exact_y1 = lambda x: -2.0 * jnp.exp(x) + 3.0 * jnp.exp(2.0 * x)
    N1 = Chebop(lambda x, u: u.diff(2) - 3.0 * u.diff() + 2.0 * u, domain=dom1)
    N1.lbc = float(exact_y1(jnp.array(0.0)))
    N1.rbc = float(exact_y1(jnp.array(1.0)))
    u1 = N1.solve(0.0)
    x_test1 = jnp.linspace(0.0, 1.0, 200)
    err1 = float(jnp.max(jnp.abs(u1(x_test1) - exact_y1(x_test1))))
    print(f"  ||u - exact||_inf = {err1:.2e}")
    assert err1 < 1e-10

    # --- ODE 2: y'' - y = 0, y(0)=1, y(1)=cosh(1) -------------------
    # Exact: y = cosh(x) (even solution of y'' = y with y(0)=1, y'(0)=0)
    # BVP form: y(0)=1, y(1)=cosh(1)
    print("\n--- ODE 2: y'' - y = 0, y(0)=1, y(1)=cosh(1) ---")
    dom2 = (0.0, 1.0)
    exact_y2 = lambda x: jnp.cosh(x)
    N2 = Chebop(lambda x, u: u.diff(2) - u, domain=dom2)
    N2.lbc = float(exact_y2(jnp.array(0.0)))
    N2.rbc = float(exact_y2(jnp.array(1.0)))
    u2 = N2.solve(0.0)
    x_test2 = jnp.linspace(0.0, 1.0, 200)
    err2 = float(jnp.max(jnp.abs(u2(x_test2) - exact_y2(x_test2))))
    print(f"  ||u - cosh(x)||_inf = {err2:.2e}")
    assert err2 < 1e-10

    # --- ODE 3: y' + 3y = 1, y(0) = 2 --------------------------------
    # Exact: y = 1/3 + (2 - 1/3)*exp(-3x) = 1/3 + (5/3)*exp(-3x)
    # Convert to BVP on [0, 1]: y(0)=2, y(1)=1/3+5/3*exp(-3)
    print("\n--- ODE 3: y' + 3y = 1, y(0)=2 ---")
    dom3 = (0.0, 1.0)
    exact_y3 = lambda x: 1.0/3.0 + (5.0/3.0) * jnp.exp(-3.0 * x)
    N3 = Chebop(lambda x, u: u.diff() + 3.0 * u, domain=dom3)
    N3.lbc = 2.0
    rhs3 = cj.chebfun(1.0)
    u3 = N3.solve(rhs3)
    x_test3 = jnp.linspace(0.0, 1.0, 200)
    err3 = float(jnp.max(jnp.abs(u3(x_test3) - exact_y3(x_test3))))
    print(f"  ||u - exact||_inf = {err3:.2e}")
    assert err3 < 1e-10

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
