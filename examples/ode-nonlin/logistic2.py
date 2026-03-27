"""The logistic map: varying initial conditions.

For fixed r = 3.7, plots the logistic orbit x_n as a function of x_0
to illustrate sensitivity to initial conditions (chaos).

Credit: Chebfun example ode-nonlin/Logistic2.m (Trefethen & Konecny, Aug 2014).
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



def run():
    print("=" * 60)
    print("Logistic map: x_n as a function of x_0 (fixed r)")
    print("=" * 60)

    r = 3.7
    N_iter = 10  # fewer iterations to avoid exponential growth in Chebfun length
    dom = (0.0, 1.0)

    # Compute x_N(x_0) = f^{(N)}(x_0) as Chebfun
    # Use repeated composition of f(x) = r*x*(1-x)
    # Start with the identity
    g = cj.chebfun(lambda x: x, domain=dom)

    print(f"\nComputing f^{{(N)}}(x_0) for r={r}, N={N_iter}...")
    for k in range(N_iter):
        g_new = cj.chebfun(lambda x: r * g(x) * (1.0 - g(x)), domain=dom)
        g = g_new
        if (k + 1) % 10 == 0:
            print(f"  iter {k+1}: length(g) = {len(g)}")

    # The function should be extremely oscillatory (chaotic)
    x_test = jnp.linspace(0.01, 0.99, 500)
    g_vals = g(x_test)
    print(f"\nAfter {N_iter} iterations:")
    print(f"  max|g| = {float(jnp.max(jnp.abs(g_vals))):.4f}")
    print(f"  g values in [0,1]: {float(jnp.min(g_vals)):.4f} to {float(jnp.max(g_vals)):.4f}")
    assert 0.0 <= float(jnp.min(g_vals)) and float(jnp.max(g_vals)) <= 1.0

    # For comparison: iterate from specific x0 values
    print("\nSample trajectories (length=10 iterations):")
    for x0 in [0.1, 0.1001, 0.5]:
        x = x0
        traj = [x]
        for _ in range(20):
            x = r * x * (1 - x)
            traj.append(x)
        print(f"  x0={x0}: x_20 = {traj[-1]:.8f}")

    # Assert that slightly different x0 give very different x_N (chaos)
    x_a = r * 0.1
    x_b = r * 0.1001
    for _ in range(20):
        x_a = r * x_a * (1 - x_a)
        x_b = r * x_b * (1 - x_b)
    print(f"\nDivergence: |x_20(0.1) - x_20(0.1001)| = {abs(x_a - x_b):.6f}")
    assert abs(x_a - x_b) > 0.01  # chaotic sensitivity

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot g = f^{(N)}(x_0) as function of x_0
    x_plot = jnp.linspace(0.0, 1.0, 1000)
    axes[0].plot(x_plot, g(x_plot), 'b', linewidth=0.8, alpha=0.8)
    axes[0].set_xlabel("x₀"); axes[0].set_ylabel(f"f^{{({N_iter})}}(x₀)")
    axes[0].set_title(f"Logistic iterate x_{N_iter}(x_0), r={r}", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Cobweb diagram for one trajectory
    x0_cob = 0.2
    n_cob = 50
    f = lambda x: r * x * (1 - x)
    x_cob = [x0_cob]
    for _ in range(n_cob):
        x_cob.append(f(x_cob[-1]))

    x_line = np.linspace(0, 1, 200)
    axes[1].plot(x_line, f(x_line), 'b', linewidth=1.4, label=f"f(x)=rx(1-x)")
    axes[1].plot(x_line, x_line, 'k--', linewidth=0.8, label="y=x")
    # Cobweb
    for k in range(n_cob - 1):
        axes[1].plot([x_cob[k], x_cob[k]], [x_cob[k], x_cob[k+1]], 'r', linewidth=0.8, alpha=0.5)
        axes[1].plot([x_cob[k], x_cob[k+1]], [x_cob[k+1], x_cob[k+1]], 'r', linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("x"); axes[1].set_ylabel("f(x)")
    axes[1].set_title(f"Cobweb diagram (r={r}, x0={x0_cob})", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Logistic map: chaos and sensitivity", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "logistic2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
