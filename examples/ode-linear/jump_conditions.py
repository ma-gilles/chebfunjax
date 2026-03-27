"""Jump conditions in ODE BVPs.

Solves  0.01 u'' + sin(x) u = 0,  u(-1)=u(1)=1  with and without
a jump condition imposed at x=0.

Credit: Chebfun example ode-linear/JumpConditions.m (Nick Hale, Nov 2011).
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

from chebfunjax.operators.chebop import Chebop

def run():
    print("=" * 60)
    print("Jump conditions in ODE BVPs")
    print("=" * 60)

    dom = (-1.0, 1.0)
    eps = 1e-2

    # Case 1: no jump — standard BVP
    print("\nCase 1: 0.01 u'' + sin(x) u = 0, u(±1)=1 (no jump)")
    # In Chebop lambda, x is a Chebfun; use cj.sin (not jnp.sin)
    N1 = Chebop(lambda x, u: eps * u.diff(2) + cj.sin(x) * u, domain=dom)
    N1.lbc = 1.0
    N1.rbc = 1.0
    u1 = N1.solve(0.0)
    print(f"  Solution length: {len(u1)}")
    u1_left = float(u1(jnp.array(-1.0)))
    u1_right = float(u1(jnp.array(1.0)))
    print(f"  u(-1) = {u1_left:.8f}, u(1) = {u1_right:.8f}")
    assert abs(u1_left - 1.0) < 1e-8
    assert abs(u1_right - 1.0) < 1e-8

    # Case 2: with jump at x=0 in the derivative
    # Model the jump by splitting domain at x=0
    # Left piece: [-1, 0], right piece: [0, 1]
    # Join condition: u continuous at 0, u'(0+) - u'(0-) = 1 (jump in derivative)
    print("\nCase 2: same ODE with u'(0+) - u'(0-) = 1  (jump in derivative)")
    # Model the jump by setting up two sub-problems and matching at 0
    # Left: eps u'' + sin(x) u = 0, u(-1) = 1, u(0) = c
    # Right: eps u'' + sin(x) u = 0, u(0) = c, u(1) = 1
    # Matching: derivative jump = 1
    # We can parametrise by c and solve the system
    # For simplicity, demonstrate by solving left and right with fixed c
    # and checking residual of the jump condition
    from scipy.optimize import brentq

    def solve_piece(a, b, ua, ub):
        """Solve eps u'' + sin(x) u = 0 on [a,b] with u(a)=ua, u(b)=ub."""
        N = Chebop(lambda x, u: eps * u.diff(2) + cj.sin(x) * u, domain=(a, b))
        N.lbc = ua
        N.rbc = ub
        return N.solve(0.0)

    def jump_residual(c):
        ul = solve_piece(-1.0, 0.0, 1.0, c)
        ur = solve_piece(0.0, 1.0, c, 1.0)
        du_left = float(ul.diff()(jnp.array(0.0)))
        du_right = float(ur.diff()(jnp.array(0.0)))
        return du_right - du_left - 1.0  # want jump = 1

    c_opt = brentq(jump_residual, -2.0, 3.0)
    print(f"  Matching value at x=0: c = {c_opt:.6f}")

    ul = solve_piece(-1.0, 0.0, 1.0, c_opt)
    ur = solve_piece(0.0, 1.0, c_opt, 1.0)
    du_left = float(ul.diff()(jnp.array(0.0)))
    du_right = float(ur.diff()(jnp.array(0.0)))
    actual_jump = du_right - du_left
    print(f"  Jump u'(0+) - u'(0-) = {actual_jump:.6f}  (target: 1.0)")
    assert abs(actual_jump - 1.0) < 1e-6

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plt.subplots()

    x_left = np.linspace(-1.0, 0.0, 200)
    x_right = np.linspace(0.0, 1.0, 200)
    x_all = np.linspace(-1.0, 1.0, 400)

    ax.plot(x_all, u1(jnp.array(x_all, dtype=jnp.float64)), 'b', linewidth=1.8,
            label="no jump")
    ax.plot(x_left, ul(jnp.array(x_left, dtype=jnp.float64)), 'r', linewidth=1.8)
    ax.plot(x_right, ur(jnp.array(x_right, dtype=jnp.float64)), 'r', linewidth=1.8,
            label="jump u′(0)")
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_title("0.01 u″ + sin(x) u = 0 with/without jump", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "jump_conditions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
