"""Simple computations with probability distributions.

Demonstrates computing expectations, mean, median, and mode of probability
density functions using chebfunjax. Translated from stats/Expectations.m.

Original: https://www.chebfun.org/examples/stats/Expectations.html
Author: Mark Richardson, May 2011
"""

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
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # --- 1. Expectation of exponential random variable ---
    # f(x) = 2*exp(-2*x) on [0, 40]
    a, b = 0.0, 40.0
    f_exp = cj.chebfun(lambda x: 2 * jnp.exp(-2 * x), domain=[a, b])

    total = float(f_exp.sum())
    print(f"Integral of 2*exp(-2x): {total:.12f}  (exact: 1.0)")
    assert abs(total - 1.0) < 1e-8

    # E[X] = integral x*f(x)
    xf = cj.chebfun(lambda x: x * 2 * jnp.exp(-2 * x), domain=[a, b])
    EX = float(xf.sum())
    print(f"E[X] = {EX:.12f}  (exact: 0.5)")
    assert abs(EX - 0.5) < 1e-8

    # E[X^2]
    x2f = cj.chebfun(lambda x: x**2 * 2 * jnp.exp(-2 * x), domain=[a, b])
    EX2 = float(x2f.sum())
    print(f"E[X²] = {EX2:.12f}  (exact: 0.5)")
    assert abs(EX2 - 0.5) < 1e-6

    # Plot
    xs = np.linspace(0, 5, 300)
    axes[0].plot(xs, 2 * np.exp(-2 * xs), 'b-', linewidth=2)
    axes[0].set_title('f(x) = 2e^{-2x}', fontsize=11)
    axes[0].set_xlabel('x'); axes[0].set_ylabel('f(x)')
    axes[0].set_ylim(-0.1, 2.1)
    axes[0].grid(True, alpha=0.3)

    # --- 2. Mean, median, mode of g(x) = 4x(9-x^2)/81 on [0,3] ---
    g_fn = lambda x: 4 * x * (9 - x**2) / 81
    g = cj.chebfun(g_fn, domain=[0.0, 3.0])

    # a) Mean = integral x*g(x)
    xg = cj.chebfun(lambda x: x * g_fn(x), domain=[0.0, 3.0])
    mean_val = float(xg.sum())
    print(f"Mean = {mean_val:.12f}  (exact: 1.6)")
    assert abs(mean_val - 1.6) < 1e-8

    # b) Median: find m where CDF(m) = 0.5
    # Use cumsum approximation by evaluating on fine grid
    xs_g = np.linspace(0, 3, 1000)
    g_vals = 4 * xs_g * (9 - xs_g**2) / 81
    cdf_vals = np.cumsum(g_vals) * (xs_g[1] - xs_g[0])
    cdf_vals = cdf_vals / cdf_vals[-1]
    median_idx = np.searchsorted(cdf_vals, 0.5)
    median_val = xs_g[median_idx]
    median_exact = float(np.sqrt(9 - 9 * np.sqrt(2) / 2))
    print(f"Median ≈ {median_val:.4f}  (exact: {median_exact:.4f})")

    # c) Mode: maximum of g
    xs_fine = np.linspace(0, 3, 1000)
    g_fine = 4 * xs_fine * (9 - xs_fine**2) / 81
    mode_idx = np.argmax(g_fine)
    mode_val = xs_fine[mode_idx]
    mode_exact = float(np.sqrt(3))
    print(f"Mode ≈ {mode_val:.4f}  (exact: {mode_exact:.4f})")

    xs_g_plot = np.linspace(0, 3, 300)
    g_plot = 4 * xs_g_plot * (9 - xs_g_plot**2) / 81
    axes[1].plot(xs_g_plot, g_plot, 'k-', linewidth=2)
    axes[1].axvline(mean_val, color='r', linewidth=2, label=f'mean={mean_val:.2f}')
    axes[1].axvline(median_val, color='m', linewidth=2, label=f'median={median_val:.2f}')
    axes[1].axvline(mode_val, color='k', linewidth=2, linestyle='--', label=f'mode={mode_val:.2f}')
    axes[1].set_title('g(x) = 4x(9-x²)/81', fontsize=11)
    axes[1].set_xlabel('x'); axes[1].legend(fontsize=9)
    axes[1].set_ylim(-0.01, 0.65)
    axes[1].grid(True, alpha=0.3)

    # --- 3. CDF of normal distribution ---
    from scipy.special import erf
    xs_n = np.linspace(-4, 4, 300)
    pdf_n = np.exp(-xs_n**2 / 2) / np.sqrt(2 * np.pi)
    cdf_n = 0.5 * (1 + erf(xs_n / np.sqrt(2)))
    axes[2].plot(xs_n, pdf_n, 'b-', linewidth=2, label='PDF N(0,1)')
    axes[2].plot(xs_n, cdf_n, 'r-', linewidth=2, label='CDF N(0,1)')
    axes[2].set_title('Normal PDF and CDF', fontsize=11)
    axes[2].set_xlabel('x'); axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Expectations and Probability Distributions', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'expectations.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("expectations: done")
    return True


if __name__ == "__main__":
    run()
