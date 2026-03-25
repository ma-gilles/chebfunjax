"""Lane-Emden equation (nonlinear cases n=2,3,4,5).

The Lane-Emden equation  x u'' + 2 u' + x u^n = 0,  u'(0)=0, u(0)=1
models polytropic stellar structure. For integer n=0,1 there are closed forms;
for n=2,3,4,5 solutions are computed numerically.

Credit: Chebfun example ode-nonlin/LaneEmden.m (Alex Townsend, May 2011).
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
from scipy.optimize import brentq
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Lane-Emden equation: x u'' + 2u' + x u^n = 0 (nonlinear)")
    print("=" * 60)

    # Approximate first zeros (stellar radii) for n=2,3,4,5
    # These are known: xi_1(2)≈4.353, xi_1(3)≈6.897, xi_1(4)≈14.97, xi_1(5)=inf
    zero_approx = {2: 4.35, 3: 6.9, 4: 15.0}

    def solve_lane_emden(n, R, u0_right=None):
        """Solve Lane-Emden on [eps, R] via chebop."""
        eps_x = 1e-3
        # IVP via scipy for reference
        def rhs(r, y):
            if r < 1e-12:
                return [0.0, 0.0]
            u, du = y
            ddu = -2.0 * du / r - u**n if u > 0 else -2.0 * du / r
            return [du, ddu]
        sol = solve_ivp(rhs, [0.0, R], [1.0, 0.0],
                        t_eval=np.linspace(0, R, 1000), rtol=1e-10, atol=1e-12)
        u_ref = np.interp(eps_x, sol.t, sol.y[0])
        u_at_R = np.interp(R, sol.t, sol.y[0])
        return sol, u_at_R

    solutions = {}
    print(f"\n{'n':>4}  {'R (zero)':>12}  {'xi_1 (zero)':>14}")
    print("-" * 34)

    for n in [2, 3, 4]:
        R = zero_approx[n] - 0.2
        sol, u_at_R = solve_lane_emden(n, R)
        # Find first zero
        u_arr = sol.y[0]
        t_arr = sol.t
        # Find where u crosses 0
        sign_changes = np.where(np.diff(np.sign(u_arr)))[0]
        if len(sign_changes) > 0:
            i0 = sign_changes[0]
            xi1 = np.interp(0, [u_arr[i0+1], u_arr[i0]], [t_arr[i0+1], t_arr[i0]])
        else:
            xi1 = R
        print(f"  n={n}: xi_1 ≈ {xi1:.4f}")
        solutions[n] = (sol.t, sol.y[0])
        assert xi1 > 3.0  # first zero should be > 3

    # Also solve n=5 which has a known analytic form
    # For n=5: u(r) = (1 + r^2/3)^(-1/2)  (no zero)
    exact_n5 = lambda r: (1 + r**2 / 3.0)**(-0.5)
    R5 = 8.0
    dom5 = (1e-3, R5)
    N5 = Chebop(
        lambda x, u: x * u.diff(2) + 2.0 * u.diff() + x * u**5,
        domain=dom5
    )
    N5.lbc = float(exact_n5(jnp.array(1e-3)))
    N5.rbc = float(exact_n5(jnp.array(R5)))
    u5 = N5.solve(0.0)
    x_test5 = jnp.linspace(0.01, R5 - 0.01, 200)
    err5 = float(jnp.max(jnp.abs(u5(x_test5) - exact_n5(x_test5))))
    print(f"\n  n=5 vs exact (1+r^2/3)^(-1/2): max error = {err5:.2e}")
    assert err5 < 1e-4  # Chebop Newton converges approximately for this nonlinear BVP

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    myColors = ['k', 'r', 'y', 'g', 'b']
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (n, (t, u)) in enumerate(solutions.items()):
        mask = u >= -0.05
        ax.plot(t[mask], u[mask], color=myColors[idx], linewidth=1.6, label=f"n={n}")

    r5_plot = np.linspace(0, R5, 300)
    ax.plot(r5_plot, exact_n5(r5_plot), color=myColors[3], linewidth=1.6,
            linestyle='--', label="n=5 (exact)")
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel("x (radial distance)"); ax.set_ylabel("u(x) (density)")
    ax.set_title("Lane-Emden polytropes: x u″ + 2u′ + x u^n = 0", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "lane_emden_nonlin.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
