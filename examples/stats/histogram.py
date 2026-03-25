"""Histogram from function or data.

Demonstrates computing histograms of continuous functions by integrating
over bins using cumsum, and comparing with standard histogram methods.
Translated from stats/Histogram.m.

Original: https://www.chebfun.org/examples/stats/Histogram.html
Author: Nick Trefethen, May 2011
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj


def hist_chebfun(f_vals, xs, edges):
    """Compute histogram of continuous function by integrating over bins.

    For each bin [a_k, b_k], computes integral of f over [a_k, b_k]
    using the cumulative sum.
    """
    from scipy.interpolate import interp1d
    # Build CDF
    dx = xs[1] - xs[0]
    cdf = np.cumsum(f_vals) * dx
    cdf_interp = interp1d(xs, cdf, bounds_error=False, fill_value=(0, cdf[-1]))

    nbins = len(edges) - 1
    data = np.zeros(nbins)
    for k in range(nbins):
        a, b = edges[k], edges[k + 1]
        data[k] = cdf_interp(b) - cdf_interp(a)
    return data


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- 1. Histogram of a smooth function ---
    # f(x) = x/3 + cos(2x) + 0.5*sin(x^2) + 0.2*sin(27x) on [0, 10]
    xs = np.linspace(0, 10, 5000)
    f_vals = xs / 3 + np.cos(2 * xs) + 0.5 * np.sin(xs**2) + 0.2 * np.sin(27 * xs)

    edges = np.arange(0, 11, 1.0)  # bins of width 1
    hist_data = hist_chebfun(f_vals, xs, edges)

    axes[0].plot(xs, f_vals, 'b-', linewidth=1.5, label='f(x)')
    # Plot histogram as piecewise constant
    for k in range(len(edges) - 1):
        xbar = [edges[k], edges[k+1]]
        ybar = [hist_data[k], hist_data[k]]
        axes[0].plot(xbar, ybar, 'r-', linewidth=2.5)
    axes[0].set_title('Function and integral-histogram', fontsize=11)
    axes[0].set_xlabel('x'); axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    total_integral = np.sum(hist_data)
    direct_integral = np.trapezoid(f_vals, xs)
    print(f"Sum of histogram bins: {total_integral:.6f}")
    print(f"Direct integral: {direct_integral:.6f}")
    print(f"Difference: {abs(total_integral - direct_integral):.2e}")

    # --- 2. Histogram of data points ---
    rng = np.random.default_rng(42)
    n_pts = 50
    xpts = 5 + rng.standard_normal(n_pts)

    # Clamp to [0, 10]
    xpts = xpts[(xpts >= 0) & (xpts <= 10)]

    # Standard histogram
    edges2 = np.arange(0, 10.5, 0.5)
    counts, _ = np.histogram(xpts, bins=edges2)

    axes[1].bar(edges2[:-1], counts, width=0.5, align='edge',
                color='steelblue', edgecolor='white', alpha=0.8, label='Data histogram')
    axes[1].plot(xpts, np.zeros_like(xpts), '.k', markersize=8, label='Data points')
    axes[1].set_title('Histogram of data points', fontsize=11)
    axes[1].set_xlabel('x'); axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    print(f"Data points: {len(xpts)} points, histogram over [0,10]")

    fig.suptitle('Histogram from Function or Data', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'histogram.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("histogram: done")
    return True


if __name__ == "__main__":
    run()
