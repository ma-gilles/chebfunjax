"""Sampling from a probability distribution.

Demonstrates how to sample from arbitrary distributions using the
inverse CDF method, illustrated with the von Mises and logit-normal
distributions. Translated from stats/ResamplingRandomVariables.m.

Original: https://www.chebfun.org/examples/stats/ResamplingRandomVariables.html
Author: Toby Driscoll, December 2011
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import interp1d
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

def compute_cdf_and_inverse(xs, pdf_vals):
    """Compute CDF and its inverse from a PDF sampled on xs."""
    dx = xs[1] - xs[0]
    cdf = np.cumsum(pdf_vals) * dx
    cdf = cdf / cdf[-1]  # normalize
    # Inverse CDF: map [0,1] -> support
    inv_cdf = interp1d(cdf, xs, bounds_error=False,
                       fill_value=(xs[0], xs[-1]))
    return cdf, inv_cdf

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2)

    # --- 1. von Mises distribution ---
    kappa = 1.5
    xs_vm = np.linspace(-np.pi, np.pi, 2000)
    density_unnorm = np.exp(kappa * np.cos(xs_vm))
    norm_vm = np.trapezoid(density_unnorm, xs_vm)
    density_vm = density_unnorm / norm_vm

    cdf_vm, inv_cdf_vm = compute_cdf_and_inverse(xs_vm, density_vm)

    # Sample
    u_vm = rng.uniform(0, 1, 10000)
    samples_vm = inv_cdf_vm(u_vm)

    # Plot
    axes[0].hist(samples_vm, bins=50, density=True, color='steelblue',
                 edgecolor='white', alpha=0.7, label='Samples')
    axes[0].plot(xs_vm, density_vm, color='#D95319', linestyle='-', linewidth=2, label='von Mises density')
    axes[0].set_title('von Mises distribution (κ=1.5)', fontsize=11)
    axes[0].legend(fontsize=9)

    print("von Mises:")
    print(f"  Sample mean: {np.mean(samples_vm):.4f}  (expected: 0)")
    print(f"  Distribution: E[cos(x)] = {np.mean(np.cos(samples_vm)):.4f}")

    # --- 2. Logit-normal distribution ---
    sig = 1.11
    xs_ln = np.linspace(1e-4, 1 - 1e-4, 2000)

    def logit_normal_pdf(x, sig):
        return np.exp(-(np.log(x / (1 - x)))**2 / (2 * sig**2)) / (x * (1 - x))

    density_ln_unnorm = logit_normal_pdf(xs_ln, sig)
    norm_ln = np.trapezoid(density_ln_unnorm, xs_ln)
    density_ln = density_ln_unnorm / norm_ln

    cdf_ln, inv_cdf_ln = compute_cdf_and_inverse(xs_ln, density_ln)

    # Sample using symmetry trick (support is (0,1))
    u_ln = rng.uniform(0, 1, 10000)
    flag = u_ln < 0.5
    u_ln_adj = np.where(flag, 1 - u_ln, u_ln)
    samples_ln = inv_cdf_ln(u_ln_adj)
    samples_ln = np.where(flag, 1 - samples_ln, samples_ln)

    axes[1].hist(samples_ln, bins=50, density=True, color='steelblue',
                 edgecolor='white', alpha=0.7, label='Samples')
    axes[1].plot(xs_ln, density_ln, color='#D95319', linestyle='-', linewidth=2, label='Logit-normal density')
    axes[1].set_title('Logit-normal distribution (σ=1.11)', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(0, 1)

    print("\nLogit-normal:")
    print(f"  Sample mean: {np.mean(samples_ln):.4f}  (expected: 0.5)")
    print(f"  Sample std: {np.std(samples_ln):.4f}")

    fig.suptitle('Inverse CDF Sampling from Probability Distributions', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'resampling_random_variables.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("resampling_random_variables: done")
    return True

if __name__ == "__main__":
    run()
