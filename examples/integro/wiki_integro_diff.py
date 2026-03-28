"""Wikipedia integro-differential equation example.

Solves the integro-differential equation from Wikipedia:
  u'(x) + 2*u(x) + 5*int_0^x u(t) dt = 1,  u(0) = 0

Exact solution: u(x) = (1/2)*exp(-x)*sin(2*x)

Following integro/WikiIntegroDiff.m by Mark Richardson (September 2010).

The IDE is converted to a second-order ODE by differentiation:
  u'' + 2u' + 5u = 0, with u(0) = 0, u'(0) = 1.

Original MATLAB: https://www.chebfun.org/examples/integro/WikiIntegroDiff.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chebfunjax.plotting import chebfun_style
chebfun_style()

import numpy as np
from scipy.integrate import solve_ivp
import os

def run():
    print("=" * 60)
    print("Wikipedia integro-differential equation")
    print("=" * 60)

    # Solve: u'(x) + 2*u(x) + 5*int_0^x u(t) dt = 1, u(0) = 0
    # Exact: u(x) = (1/2)*exp(-x)*sin(2*x)

    # Differentiating the IDE gives the equivalent ODE:
    # u'' + 2u' + 5u = 0, with u(0) = 0, u'(0) = 1
    print("\nEquivalent ODE: u'' + 2u' + 5u = 0, u(0)=0, u'(0)=1")
    print("Exact solution: u(x) = (1/2)*exp(-x)*sin(2*x)")

    domain = [0.0, 5.0]
    x_eval = np.linspace(0, 5, 500)

    # Solve as first-order system: y = [u, u']
    def rhs(x, y):
        u, up = y
        upp = -2 * up - 5 * u
        return [up, upp]

    sol = solve_ivp(rhs, domain, [0.0, 1.0], t_eval=x_eval, rtol=1e-10, atol=1e-12)
    u_vals = sol.y[0]

    # Exact solution
    exact_vals = 0.5 * np.exp(-x_eval) * np.sin(2 * x_eval)

    # Verify at several points (use exact grid points to avoid interpolation error)
    x_test = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])

    print("\nVerification:")
    max_err = 0.0
    for xi in x_test:
        idx = np.argmin(np.abs(x_eval - xi))
        x_actual = x_eval[idx]
        val = u_vals[idx]
        ex = 0.5 * np.exp(-x_actual) * np.sin(2 * x_actual)
        err = abs(val - ex)
        max_err = max(max_err, err)
        print(f"  u({xi}) = {val:.8f}, exact = {ex:.8f}, err = {err:.2e}")

    print(f"\nMax error: {max_err:.2e}")
    assert max_err < 1e-6, f"Error too large: {max_err}"
    print("PASS: matches exact solution to within tolerance")

    # Verify original IDE: u'(x) + 2*u(x) + 5*int_0^x u(t) dt = 1
    print("\nVerifying original IDE at several points...")
    u_deriv_vals = np.gradient(u_vals, x_eval)

    cumint = np.zeros(len(x_eval))
    for i in range(1, len(x_eval)):
        cumint[i] = np.trapezoid(u_vals[:i+1], x_eval[:i+1])

    ide_residual = u_deriv_vals + 2 * u_vals + 5 * cumint
    # At t=0 the equation should give 1
    ide_at_interior = ide_residual[5:-5]  # avoid boundary gradient artifacts
    rhs_vals = np.ones_like(ide_at_interior)
    err_ide = np.max(np.abs(ide_at_interior - rhs_vals))
    print(f"  IDE residual max (interior): {err_ide:.4f}")
    print(f"  (Some numerical differentiation error is expected)")
    assert err_ide < 0.1, f"IDE residual too large: {err_ide}"
    print("  PASS: IDE satisfied to within numerical differentiation error")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(x_eval, exact_vals, color='#D95319', linestyle='-', linewidth=2, label='Exact')
    axes[0].plot(x_eval[::20], u_vals[::20], color='#0072BD', marker='.', linestyle='none', markersize=8, label='solve_ivp')
    axes[0].set_title("u'(x)+2u+5∫u dt = 1, u(0)=0", fontsize=11)
    axes[0].legend()

    err_plot = np.abs(u_vals - exact_vals) + 1e-18
    axes[1].semilogy(x_eval, err_plot, color='#0072BD', linestyle='-', linewidth=2)
    axes[1].set_title("Error vs exact solution", fontsize=11)

    fig.suptitle("Wikipedia integro-differential equation", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "wiki_integro_diff.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True

if __name__ == "__main__":
    run()
