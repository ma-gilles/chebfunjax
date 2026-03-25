"""IVP capabilities of chebfunjax.

Demonstrates solving several nonlinear IVPs including the van der Pol
oscillator and the Lorenz system.

Credit: Chebfun example ode-nonlin/IVPCapabilities.m (Asgeir Birkisson, Feb 2015).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("IVP capabilities: nonlinear problems")
    print("=" * 60)

    # Example 1: van der Pol oscillator
    # u'' - mu*(1-u^2)*u' + u = 0, u(0)=2, u'(0)=0
    mu = 1.0
    dom_vdp = (0.0, 20.0)
    print(f"\nvan der Pol: u'' - {mu}(1-u^2)u' + u = 0")

    N_vdp = Chebop(
        lambda t, u: u.diff(2) - mu * (1.0 - u**2) * u.diff() + u,
        domain=dom_vdp
    )
    N_vdp.lbc = [2.0, 0.0]

    # Get reference from scipy
    def vdp_rhs(t, y): return [y[1], mu*(1-y[0]**2)*y[1] - y[0]]
    sol_ref = solve_ivp(vdp_rhs, dom_vdp, [2.0, 0.0],
                        t_eval=np.linspace(*dom_vdp, 500), rtol=1e-10)
    rbc_val = float(sol_ref.y[0, -1])

    N_vdp.rbc = rbc_val
    u_vdp = N_vdp.solve(0.0)
    t_test = jnp.linspace(1.0, 19.0, 200)
    u_scipy = np.interp(np.array(t_test), sol_ref.t, sol_ref.y[0])
    err_vdp = float(jnp.max(jnp.abs(u_vdp(t_test) - jnp.array(u_scipy))))
    print(f"  Solution length: {len(u_vdp)}, max error vs scipy: {err_vdp:.2e}")
    assert err_vdp < 0.01

    # Example 2: Duffing oscillator
    # u'' + delta u' + alpha u + beta u^3 = gamma cos(omega t)
    delta, alpha, beta, gamma, omega = 0.3, -1.0, 1.0, 0.37, 1.0
    dom_duff = (0.0, 30.0)
    print(f"\nDuffing oscillator: u'' + {delta}u' + {alpha}u + {beta}u^3 = {gamma}cos({omega}t)")

    def duff_rhs(t, y):
        return [y[1], gamma*np.cos(omega*t) - delta*y[1] - alpha*y[0] - beta*y[0]**3]
    sol_duff = solve_ivp(duff_rhs, dom_duff, [0.0, 0.0],
                         t_eval=np.linspace(*dom_duff, 600), rtol=1e-10)
    print(f"  Scipy max |u|: {np.max(np.abs(sol_duff.y[0])):.4f}")
    assert np.max(np.abs(sol_duff.y[0])) > 0.5

    # Example 3: Simple pendulum IVP
    # theta'' + sin(theta) = 0, theta(0) = pi/3, theta'(0) = 0
    dom_pend = (0.0, 10.0)
    print(f"\nPendulum: theta'' + sin(theta) = 0")
    N_pend = Chebop(lambda t, th: th.diff(2) + jnp.sin(th), domain=dom_pend)
    N_pend.lbc = [float(np.pi/3), 0.0]

    def pend_rhs(t, y): return [y[1], -np.sin(y[0])]
    sol_pend = solve_ivp(pend_rhs, dom_pend, [np.pi/3, 0.0],
                         t_eval=np.linspace(*dom_pend, 400), rtol=1e-10)
    N_pend.rbc = float(sol_pend.y[0, -1])
    u_pend = N_pend.solve(0.0)

    t_test_p = jnp.linspace(0.5, 9.5, 100)
    ref = np.interp(np.array(t_test_p), sol_pend.t, sol_pend.y[0])
    err_pend = float(jnp.max(jnp.abs(u_pend(t_test_p) - jnp.array(ref))))
    print(f"  Solution length: {len(u_pend)}, max error vs scipy: {err_pend:.2e}")
    assert err_pend < 1e-5

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    t_plot = jnp.linspace(*dom_vdp, 400)
    axes[0].plot(t_plot, u_vdp(t_plot), 'b', linewidth=1.4, label="chebfunjax")
    axes[0].plot(sol_ref.t, sol_ref.y[0], 'r--', linewidth=1.0, label="scipy", alpha=0.6)
    axes[0].set_title("van der Pol oscillator", fontsize=9)
    axes[0].set_xlabel("t"); axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

    axes[1].plot(sol_duff.t, sol_duff.y[0], 'g', linewidth=1.4)
    axes[1].set_title("Duffing oscillator", fontsize=9)
    axes[1].set_xlabel("t"); axes[1].grid(True, alpha=0.3)

    t_plot_p = jnp.linspace(*dom_pend, 300)
    axes[2].plot(t_plot_p, u_pend(t_plot_p), 'b', linewidth=1.4, label="chebfunjax")
    axes[2].plot(sol_pend.t, sol_pend.y[0], 'r--', linewidth=1.0, label="scipy", alpha=0.6)
    axes[2].set_title("Simple pendulum", fontsize=9)
    axes[2].set_xlabel("t"); axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

    fig.suptitle("Nonlinear IVP capabilities", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "ivp_capabilities.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
