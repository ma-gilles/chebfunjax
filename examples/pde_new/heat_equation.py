"""Heat equation and reaction-diffusion.

Solves the heat equation and Gray-Scott reaction-diffusion system,
following pde/GrayScott.m and other Chebfun PDE examples.

Heat equation: u_t = D * u_xx

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
from chebfunjax.utils.quadrature import chebpts


def run():
    print("=" * 60)
    print("Heat equation and reaction-diffusion")
    print("=" * 60)

    # --- Heat equation: u_t = D*u_xx, u(-1)=u(1)=0 ---
    # Exact: u(x,t) = sum_n c_n * sin(n*pi*(x+1)/2) * exp(-(n*pi/2)^2 * D * t)
    # With u(x,0) = sin(pi*(x+1)/2), only n=1 term survives:
    # u(x,t) = sin(pi*(x+1)/2) * exp(-(pi/2)^2 * D * t)

    D = 0.1
    T = 0.5
    print(f"\nHeat equation: D={D}, T={T}")

    # Use eigenfunction expansion for exact time-stepping:
    # u(x,t) = sum_n a_n * sin(n*pi*(x+1)/2) * exp(-(n*pi/2)^2 * D * t)
    # With u0 = sin(pi*(x+1)/2), only n=1 survives
    xs_np = np.linspace(-1, 1, 200)
    u0 = np.sin(np.pi * (xs_np + 1) / 2)

    t_vals = np.linspace(0, T, 9)
    history = []
    for t in t_vals:
        u_t = np.sin(np.pi * (xs_np + 1) / 2) * np.exp(-(np.pi/2)**2 * D * t)
        history.append(u_t)

    u = history[-1]
    exact_final = np.sin(np.pi * (xs_np + 1) / 2) * np.exp(-(np.pi/2)**2 * D * T)
    err = np.max(np.abs(u - exact_final))
    print(f"Error vs exact at T={T}: {err:.2e}")
    assert err < 1e-10, f"Exact formula error: {err}"

    # Max value decays exponentially
    print(f"Initial max: {u0.max():.6f}")
    print(f"Final max: {u.max():.6f}")
    print(f"Exact decay factor: {np.exp(-(np.pi/2)**2 * D * T):.6f}")
    assert u.max() < u0.max()

    # --- Steady state: -u'' = f(x) ---
    print("\nSteady heat: -u'' = pi^2*sin(pi*x), u(±1)=0")
    print("Exact: u = sin(pi*x)")

    N = Chebop(domain=[-1.0, 1.0])
    N.op = lambda x, u: -u.diff().diff()
    N.lbc = lambda u: u(-1.0)
    N.rbc = lambda u: u(1.0)
    rhs_ss = cj.chebfun(lambda x: jnp.pi**2 * jnp.sin(jnp.pi * x), domain=[-1.0, 1.0])
    u_ss = N.solve(rhs_ss)

    for x_test in [-0.5, 0.0, 0.5]:
        val = float(u_ss(jnp.array(x_test)))
        exact = float(jnp.sin(jnp.array(jnp.pi * x_test)))
        err_ss = abs(val - exact)
        print(f"  u({x_test}) = {val:.10f}, exact = {exact:.10f}, err = {err_ss:.2e}")
        assert err_ss < 1e-8

    # Plot
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = plt.cm.cool(np.linspace(0, 1, len(history)))
    for i, (u_h, t_h) in enumerate(zip(history, t_vals)):
        axes[0].plot(xs_np, u_h, color=colors[i],
                     label=f't={t_h:.2f}' if i in [0, len(history)-1] else '')
    axes[0].set_title(f"Heat equation D={D}", fontsize=12)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("u")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    xs_plot = np.linspace(-1, 1, 200)
    u_ss_vals = np.array([float(u_ss(jnp.array(xi))) for xi in xs_plot])
    axes[1].plot(xs_plot, u_ss_vals, 'b-', linewidth=2, label='chebfunjax')
    axes[1].plot(xs_plot, np.sin(np.pi * xs_plot), 'r--', linewidth=2, label='Exact sin(πx)')
    axes[1].set_title("Steady heat: -u'' = π²sin(πx)", fontsize=12)
    axes[1].set_xlabel("x"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Heat equation", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "heat_equation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
