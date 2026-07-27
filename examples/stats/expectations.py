"""Simple computations with probability distributions.

Demonstrates computing expectations, mean, median, and mode of probability
density functions using chebfunjax. Translated from stats/Expectations.m.

Original: https://www.chebfun.org/examples/stats/Expectations.html
Author: Mark Richardson, May 2011
"""

import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3)

    # --- 1. Expectation of exponential random variable ---
    # MATLAB: x = chebfun('x',[0 40]); f = 2*exp(-2*x);
    #   sum(f); sum(x.*f); sum(x.^2.*f)
    x = cj.chebfun(lambda x: x, domain=(0.0, 40.0))
    f_exp = 2 * (-2 * x).exp()
    print(f"{float(f_exp.sum()):.15f}")
    print(f"{float((x * f_exp).sum()):.15f}")
    print(f"{float((x * x * f_exp).sum()):.15f}")

    xs = np.linspace(0, 5, 300)
    axes[0].plot(xs, 2 * np.exp(-2 * xs), color='#0072BD', lw=2)
    axes[0].set_title('f(x) = 2e^{-2x}', fontsize=11)
    axes[0].set_ylim(-0.1, 2.1)

    # --- 2. Mean, median, mode of g(x) = 4x(9-x^2)/81 on [0,3] ---
    # MATLAB: x = chebfun('x',[0 3]); g = 4*x.*(9-x.^2)/81;
    x3 = cj.chebfun(lambda x: x, domain=(0.0, 3.0))
    g = 4 * x3 * (9 - x3 * x3) / 81

    # Mean = sum(x.*g)
    mean_val = float((x3 * g).sum())
    print(f"mean = {mean_val:.15f}")

    # Median = roots(cumsum(g) - 0.5)
    G = g.cumsum()
    median_val = float(np.asarray((G - 0.5).roots()).ravel()[0])
    median_exact = float(np.sqrt(9 - 9 * np.sqrt(2) / 2))
    print(f"median = {median_val:.15f}")
    print(f"median_exact = {median_exact:.15f}")

    # Mode = argmax location.  chebfunjax max() returns (x_max, f_max),
    # so the mode is the first element (MATLAB's [gmax, mode] = max(g)).
    mode_val, _gmax = g.max()
    mode_exact = float(np.sqrt(3))
    print(f"mode = {mode_val:.15f}")
    print(f"mode_exact = {mode_exact:.15f}")

    xs_g_plot = np.linspace(0, 3, 300)
    g_plot = 4 * xs_g_plot * (9 - xs_g_plot**2) / 81
    axes[1].plot(xs_g_plot, g_plot, 'k-', linewidth=2)
    axes[1].axvline(mean_val, color='#D95319', linewidth=2, label=f'mean={mean_val:.2f}')
    axes[1].axvline(median_val, color='#7E2F8E', linewidth=2, label=f'median={median_val:.2f}')
    axes[1].axvline(mode_val, color='k', linewidth=2, linestyle='--', label=f'mode={mode_val:.2f}')
    axes[1].set_title('g(x) = 4x(9-x²)/81', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(-0.01, 0.65)

    # --- 3. CDF of normal distribution ---
    from scipy.special import erf
    xs_n = np.linspace(-4, 4, 300)
    pdf_n = np.exp(-xs_n**2 / 2) / np.sqrt(2 * np.pi)
    cdf_n = 0.5 * (1 + erf(xs_n / np.sqrt(2)))
    axes[2].plot(xs_n, pdf_n, color='#0072BD', linestyle='-', linewidth=2, label='PDF N(0,1)')
    axes[2].plot(xs_n, cdf_n, color='#D95319', linestyle='-', linewidth=2, label='CDF N(0,1)')
    axes[2].set_title('Normal PDF and CDF', fontsize=11)
    axes[2].legend(fontsize=9)

    fig.suptitle('Expectations and Probability Distributions', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'expectations.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("expectations: done")
    return True

if __name__ == "__main__":
    run()
