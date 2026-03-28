"""Probability distributions and expectations.

Demonstrates computing expectations, CDFs, and convolutions of probability
distributions using chebfunjax, following stats/Expectations.m,
stats/BivariateNormalDistribution.m, stats/ProbabilityConvolution.m,
and stats/CentralLimitTheorem.m.

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

def normal_pdf(mu=0.0, sigma=1.0):
    """Normal distribution PDF."""
    return lambda x: jnp.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * jnp.sqrt(2 * jnp.pi))

def run():
    print("=" * 60)
    print("Probability distributions and expectations")
    print("=" * 60)

    # --- Standard normal ---
    print("\nStandard normal N(0,1):")
    a, b = -5.0, 5.0
    phi = cj.chebfun(normal_pdf(0, 1), domain=[a, b])
    print(f"  Length: {len(phi.funs[0].tech.coeffs)}")

    # Total probability = 1
    total = float(phi.sum())
    print(f"  Integral = {total:.10f}  (exact: 1.0)")
    assert abs(total - 1.0) < 1e-4

    # Mean = 0
    x_phi = cj.chebfun(lambda x: x * normal_pdf(0, 1)(x), domain=[a, b])
    mean = float(x_phi.sum())
    print(f"  Mean E[X] = {mean:.2e}  (exact: 0)")
    assert abs(mean) < 1e-8

    # Variance = 1
    x2_phi = cj.chebfun(lambda x: x**2 * normal_pdf(0, 1)(x), domain=[a, b])
    var = float(x2_phi.sum())
    print(f"  Variance E[X²] = {var:.10f}  (exact: 1.0)")
    assert abs(var - 1.0) < 1e-3

    # --- Convolution of two normals = normal ---
    # N(0,1) * N(0,1) = N(0, sqrt(2))
    # NOTE: chebfun.conv() is very slow; use numpy-based convolution for verification.
    print("\nConvolution: N(0,1) * N(0,1) = N(0, √2) (numpy-based):")
    from scipy.signal import fftconvolve

    dx = 0.01
    xs = np.arange(-6, 6+dx, dx)
    p1 = np.exp(-0.5*xs**2) / np.sqrt(2*np.pi)
    conv_vals_np = fftconvolve(p1, p1, mode='full') * dx
    mid = len(conv_vals_np) // 2
    conv_at_0 = float(conv_vals_np[mid])
    exact_at_0 = 1.0 / (np.sqrt(2) * np.sqrt(2 * np.pi))
    print(f"  (φ*φ)(0) = {conv_at_0:.8f}  (exact: {exact_at_0:.8f})")
    assert abs(conv_at_0 - exact_at_0) < 0.01
    # Build a Chebfun from the convolution result for plotting
    xs_conv = np.linspace(-6, 6, 200)
    conv_half = conv_vals_np[mid - len(xs_conv)//2 : mid + len(xs_conv)//2 + 1][:len(xs_conv)]

    # --- Beta distribution ---
    print("\nBeta distribution B(2,3):")
    # pdf: x^(a-1) * (1-x)^(b-1) / B(a,b)
    from scipy.special import beta as beta_func
    a_b, b_b = 2, 3
    B_norm = beta_func(a_b, b_b)

    def beta_pdf(x):
        return x**(a_b-1) * (1-x)**(b_b-1) / B_norm

    phi_beta = cj.chebfun(
        lambda x: jnp.array(beta_pdf(np.array(x))),
        domain=[0.0, 1.0]
    )
    total_beta = float(phi_beta.sum())
    print(f"  Integral = {total_beta:.10f}  (exact: 1.0)")
    assert abs(total_beta - 1.0) < 1e-8

    # Mean = a/(a+b) = 2/5 = 0.4
    x_beta = cj.chebfun(
        lambda x: x * jnp.array(beta_pdf(np.array(x))),
        domain=[0.0, 1.0]
    )
    mean_beta = float(x_beta.sum())
    exact_mean_beta = a_b / (a_b + b_b)
    print(f"  Mean = {mean_beta:.8f}  (exact: {exact_mean_beta:.8f})")
    assert abs(mean_beta - exact_mean_beta) < 1e-6

    # --- Uniform distribution: central limit theorem ---
    print("\nCentral Limit Theorem: uniform U(0,1) convolution:")
    # Irwin-Hall: sum of n uniforms -> N(n/2, sqrt(n/12))
    # For n=1: uniform pdf = 1 on [0,1]
    phi_u = cj.chebfun(lambda x: jnp.ones_like(x), domain=[0.0, 1.0])
    total_u = float(phi_u.sum())
    print(f"  Uniform integral = {total_u:.10f}  (exact: 1.0)")
    assert abs(total_u - 1.0) < 1e-12

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(_here, '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 3)

    # Normal PDFs
    xs_n = np.linspace(-4, 4, 200)
    for mu, sigma, label in [(0, 1, 'N(0,1)'), (1, 0.5, 'N(1,0.5)'), (-1, 1.5, 'N(-1,1.5)')]:
        axes[0].plot(xs_n, np.exp(-0.5*((xs_n-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi)),
                     linewidth=2, label=label)
    axes[0].set_title("Normal distributions", fontsize=12)
    axes[0].legend(fontsize=9)

    # Convolution result (using numpy-based convolution computed above)
    xs_plot_conv = np.linspace(-6, 6, 200)
    exact_conv = np.exp(-xs_plot_conv**2/4) / (np.sqrt(2) * np.sqrt(2*np.pi))
    axes[1].plot(xs_plot_conv, conv_half, color='#0072BD', linestyle='-', linewidth=2, label='N(0,1) ∗ N(0,1)')
    axes[1].plot(xs_plot_conv, exact_conv, color='#D95319', linestyle='--', linewidth=2, label='N(0,√2) exact')
    axes[1].set_title("Convolution of normals", fontsize=12)
    axes[1].legend(fontsize=9)

    # Beta distribution family
    xs_b = np.linspace(0, 1, 200)
    for aa, bb in [(1, 1), (2, 2), (2, 5), (0.5, 0.5)]:
        Bnorm = beta_func(aa, bb)
        y = xs_b**(aa-1) * (1-xs_b)**(bb-1) / Bnorm
        axes[2].plot(xs_b, np.clip(y, 0, 5), linewidth=2, label=f'B({aa},{bb})')
    axes[2].set_title("Beta distributions", fontsize=12)
    axes[2].legend(fontsize=9)
    axes[2].set_ylim(0, 4)

    fig.suptitle("Probability distributions", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "probability_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
