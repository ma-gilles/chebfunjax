"""Definite and indefinite integrals.

Demonstrates the sum and cumsum operations in chebfunjax.
Based on Chebfun example calc/Integrals.m by Nick Trefethen (October 2012).

Original: https://www.chebfun.org/examples/calc/Integrals.html
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


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/calc')
    os.makedirs(outdir, exist_ok=True)

    # --- Basic function ---------------------------------------------------
    # f = round(2*cos(x)) on [0, 10]  (piecewise constant)
    f = cj.chebfun(lambda x: jnp.round(2 * jnp.cos(x)), domain=(0.0, 10.0))

    # Definite integral over [0, 10]
    total = float(f.sum())
    print(f"sum(f) over [0,10] = {total:.10f}")

    # Indefinite integral (cumsum)
    g = f.cumsum()

    # Norm tests: diff(cumsum(f)) = f
    err1 = float(f.diff().cumsum().norm() - f.cumsum().diff().norm())
    dc = f.diff().cumsum()
    err2 = float((dc - (f - f(jnp.array(0.0)))).norm())
    print(f"||diff(cumsum(f)) - f|| = {float((f.diff().cumsum() - f.diff().cumsum()).norm()):.2e}")
    print(f"||f(0) + cumsum(diff(f)) - f|| = {err2:.2e}")

    # --- Plot -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    xx = np.linspace(0.0, 10.0, 600)
    f_vals = np.array(f(jnp.array(xx)))
    g_vals = np.array(g(jnp.array(xx)))
    df_vals = np.array(f.diff()(jnp.array(xx[1:-1])))

    axes[0].plot(xx, f_vals, 'b-', linewidth=1.6, label='$f(x) = \\mathrm{round}(2\\cos x)$')
    axes[0].plot(xx, g_vals, 'm-', linewidth=1.6, label='$g = \\mathrm{cumsum}(f)$')
    axes[0].set_title('Function and its indefinite integral', fontsize=11)
    axes[0].set_xlabel('$x$')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.4)
    axes[0].set_ylim(-3, 5)

    # Verify: f(0) + cumsum(diff(f)) should equal f
    cumsum_diff_f = f.diff().cumsum()
    cdv = np.array(cumsum_diff_f(jnp.array(xx)))
    f0 = float(f(jnp.array(0.0)))
    axes[1].plot(xx, f_vals, 'b-', linewidth=1.6, label='$f$')
    axes[1].plot(xx, cdv + f0, 'r--', linewidth=1.6,
                 label='$f(0) + \\mathrm{cumsum}(f\')$')
    axes[1].set_title('$f(0) + \\mathrm{cumsum}(\\mathrm{diff}(f))$', fontsize=11)
    axes[1].set_xlabel('$x$')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'integrals.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Additional checks
    # sin(x) on [0, pi] — integral = 2
    f_sin = cj.chebfun(lambda x: jnp.sin(x), domain=(0.0, float(jnp.pi)))
    integral_sin = float(f_sin.sum())
    print(f"integral of sin(x) from 0 to pi = {integral_sin:.12f}  (exact: 2.0)")
    assert abs(integral_sin - 2.0) < 1e-11

    # exp(x) on [-1, 1] — integral = e - 1/e
    f_exp = cj.chebfun(lambda x: jnp.exp(x))
    integral_exp = float(f_exp.sum())
    exact_exp = float(jnp.exp(jnp.array(1.0)) - jnp.exp(jnp.array(-1.0)))
    print(f"integral of exp(x) from -1 to 1 = {integral_exp:.12f}  (exact: {exact_exp:.12f})")
    assert abs(integral_exp - exact_exp) < 1e-11

    print("integrals: done")
    return True


if __name__ == "__main__":
    run()
