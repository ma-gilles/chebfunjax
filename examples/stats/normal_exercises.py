"""Normal distribution: exercises from a textbook.

Demonstrates computing probabilities, CDFs, and areas for the normal
distribution using chebfunjax. Translated from stats/NormalExercises.m.

Original: https://www.chebfun.org/examples/stats/NormalExercises.html
Author: Jie Gao and Nick Trefethen, April 2013
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



def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    # Standard normal PDF
    def normal_pdf(mu, sigma):
        return lambda x: jnp.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * jnp.sqrt(2 * jnp.pi))

    # Problem: X ~ N(2, 1), find P[|X-2| < 1]
    mu, sigma = 2.0, 1.0
    # Work on finite domain [-1, 6]
    a, b = -1.0, 6.0
    f = cj.chebfun(normal_pdf(mu, sigma), domain=[a, b])
    total = float(f.sum())
    print(f"Total probability (truncated domain): {total:.10f}")

    # Compute P[1 < X < 3] = P[|X-2| < 1] via numerical integration
    xs = np.linspace(a, b, 10000)
    pdf_vals = np.exp(-0.5 * ((xs - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    cdf = np.cumsum(pdf_vals) * (xs[1] - xs[0])

    idx1 = np.searchsorted(xs, 1.0)
    idx3 = np.searchsorted(xs, 3.0)
    prob = cdf[idx3] - cdf[idx1]
    from scipy.special import erf as scipy_erf
    print(f"P[|X-2| < 1] = {prob:.6f}  (exact via erf: {float(scipy_erf(1/np.sqrt(2))):.6f})")

    from scipy.special import erf
    exact_prob = float(erf(1.0 / np.sqrt(2)))
    print(f"Exact P[|X-2| < 1] = {exact_prob:.10f}")

    # Variant: modified distribution with |x|^(5/4) exponent
    # f2(x) ∝ exp(-|(x-mu)/sigma|^(5/4))
    def modified_pdf_unnorm(x, mu, sigma):
        return np.exp(-np.abs((x - mu) / sigma)**(5/4))

    xs_full = np.linspace(-5, 10, 10000)
    f2_vals = modified_pdf_unnorm(xs_full, mu, sigma)
    norm_const = np.trapezoid(f2_vals, xs_full)
    f2_vals_norm = f2_vals / norm_const

    cdf2 = np.cumsum(f2_vals_norm) * (xs_full[1] - xs_full[0])
    idx1_m = np.searchsorted(xs_full, 1.0)
    idx3_m = np.searchsorted(xs_full, 3.0)
    prob_mod = cdf2[idx3_m] - cdf2[idx1_m]
    print(f"Modified distribution P[1<X<3] = {prob_mod:.6f}")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    xs_plot = np.linspace(-1, 6, 400)
    pdf_plot = np.exp(-0.5 * ((xs_plot - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    axes[0].plot(xs_plot, pdf_plot, 'k-', linewidth=2)

    # Shade region [1, 3]
    mask = (xs_plot >= 1) & (xs_plot <= 3)
    axes[0].fill_between(xs_plot[mask], pdf_plot[mask], alpha=0.4, color='green',
                         label=f'P[|X-2|<1] = {exact_prob:.4f}')
    axes[0].set_title('N(2,1): P[|X-2|<1]', fontsize=11)
    axes[0].set_xlabel('x'); axes[0].set_ylabel('f(x)')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    # Modified distribution
    xs_m = np.linspace(-1, 6, 400)
    f2_m = modified_pdf_unnorm(xs_m, mu, sigma)
    f2_m_norm = f2_m / np.trapezoid(f2_m, xs_m)
    axes[1].plot(xs_m, f2_m_norm, 'k-', linewidth=2)
    mask2 = (xs_m >= 1) & (xs_m <= 3)
    axes[1].fill_between(xs_m[mask2], f2_m_norm[mask2], alpha=0.4, color='red',
                         label=f'P[|X-2|<1] = {prob_mod:.4f}')
    axes[1].set_title('Modified: exp(-|x-2|^{5/4})', fontsize=11)
    axes[1].set_xlabel('x'); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    fig.suptitle('Normal Distribution Exercises', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'normal_exercises.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("normal_exercises: done")
    return True


if __name__ == "__main__":
    run()
