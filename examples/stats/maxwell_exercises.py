"""Maxwell distribution exercises.

Demonstrates mean, variance, and mode calculations for the Maxwell
distribution (relevant in statistical mechanics for particle speeds).
Translated from stats/MaxwellExercises.m.

Original: https://www.chebfun.org/examples/stats/MaxwellExercises.html
Author: Jie Gao, September 2013
"""

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



def maxwell_pdf(x, b):
    """Maxwell distribution PDF: f(x; b) = sqrt(2/pi) * x^2 * exp(-x^2/(2b^2)) / b^3."""
    return np.sqrt(2 / np.pi) * x**2 * np.exp(-x**2 / (2 * b**2)) / b**3


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- 1. Maxwell distribution with b=2.3 ---
    b = 2.3
    xs = np.linspace(0, 15, 1000)
    pdf_vals = maxwell_pdf(xs, b)

    # Mean = 2*sqrt(2/pi)*b
    mean_exact = 2 * np.sqrt(2 / np.pi) * b
    mean_numerical = np.trapezoid(xs * pdf_vals, xs)
    print(f"Maxwell(b={b}):")
    print(f"  Mean (numerical) = {mean_numerical:.10f}")
    print(f"  Mean (exact)     = {mean_exact:.10f}")
    assert abs(mean_numerical - mean_exact) < 1e-4

    # Variance = b^2 * (3*pi - 8) / pi
    mean_sq = np.trapezoid(xs**2 * pdf_vals, xs)
    var_numerical = mean_sq - mean_numerical**2
    var_exact = b**2 * (3 * np.pi - 8) / np.pi
    print(f"  Variance (numerical) = {var_numerical:.8f}")
    print(f"  Variance (exact)     = {var_exact:.8f}")

    # Mode = sqrt(2)*b
    mode_exact = np.sqrt(2) * b
    mode_idx = np.argmax(pdf_vals)
    mode_numerical = xs[mode_idx]
    print(f"  Mode (numerical) = {mode_numerical:.4f}")
    print(f"  Mode (exact)     = {mode_exact:.4f}")

    axes[0].plot(xs, pdf_vals, 'k-', linewidth=2)
    axes[0].axvline(mean_numerical, color='r', linewidth=2, linestyle='--',
                    label=f'mean={mean_numerical:.2f}')
    axes[0].axvline(mode_exact, color='b', linewidth=2, linestyle=':',
                    label=f'mode={mode_exact:.2f}')
    axes[0].set_title(f'Maxwell(b={b})', fontsize=11)
    axes[0].set_xlabel('x'); axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- 2. Maxwell distributions for different b values ---
    for b_val, color in [(1.0, 'b'), (2.0, 'r'), (3.0, 'g'), (4.0, 'm')]:
        pdf_b = maxwell_pdf(xs, b_val)
        axes[1].plot(xs, pdf_b, '-', color=color, linewidth=2,
                     label=f'b={b_val}')
    axes[1].set_title('Maxwell for various b values', fontsize=11)
    axes[1].set_xlabel('x'); axes[1].legend(fontsize=9)
    axes[1].set_xlim(0, 15); axes[1].grid(True, alpha=0.3)

    # --- 3. CDF and quantiles ---
    cdf_vals = np.cumsum(pdf_vals) * (xs[1] - xs[0])
    cdf_vals /= cdf_vals[-1]

    # 50th percentile (median)
    from scipy.interpolate import interp1d
    inv_cdf = interp1d(cdf_vals, xs, bounds_error=False, fill_value=(xs[0], xs[-1]))
    median_val = float(inv_cdf(0.5))
    p10 = float(inv_cdf(0.1))
    p90 = float(inv_cdf(0.9))
    print(f"\nMaxwell(b={b}): 10th={p10:.3f}, median={median_val:.3f}, 90th={p90:.3f}")

    axes[2].plot(xs, cdf_vals, 'k-', linewidth=2)
    for pct, val, color in [(0.1, p10, 'b'), (0.5, median_val, 'r'), (0.9, p90, 'g')]:
        axes[2].axhline(pct, color=color, linestyle='--', alpha=0.5)
        axes[2].axvline(val, color=color, linestyle='--', alpha=0.5,
                        label=f'Q{int(pct*100)}={val:.2f}')
    axes[2].set_title(f'Maxwell(b={b}): CDF and quantiles', fontsize=10)
    axes[2].set_xlabel('x'); axes[2].set_ylabel('CDF')
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    fig.suptitle('Maxwell Distribution Exercises', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'maxwell_exercises.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("maxwell_exercises: done")
    return True


if __name__ == "__main__":
    run()
