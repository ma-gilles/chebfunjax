"""Smoothies: nowhere analytic functions.

Demonstrates C-infinity but nowhere analytic functions constructed via
random Fourier series with root-exponentially decaying coefficients.
Translated from stats/Smoothies.m.

Original: https://www.chebfun.org/examples/stats/Smoothies.html
Author: Nick Trefethen, February 2020
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

def make_smoothie(n_terms=200, seed=1, domain=(-1.0, 1.0)):
    """Construct a smoothie: C-inf but nowhere analytic function.

    Coefficients decay as C^{-sqrt(k)} (root-exponential).
    """
    rng = np.random.default_rng(seed)
    a, b = domain
    L = b - a
    ks = np.arange(1, n_terms + 1)
    # Root-exponential decay: amplitudes ~ exp(-sqrt(k)/2)
    C = 1.5
    amplitudes = C**(-np.sqrt(ks))
    c_cos = rng.standard_normal(n_terms) * amplitudes
    c_sin = rng.standard_normal(n_terms) * amplitudes

    n_pts = 1000
    xs = np.linspace(a, b, n_pts)
    f_vals = np.zeros(n_pts)
    for k, cc, cs in zip(ks, c_cos, c_sin):
        freq = 2 * np.pi * k / L
        f_vals += cc * np.cos(freq * (xs - a)) + cs * np.sin(freq * (xs - a))

    return xs, f_vals, ks, amplitudes

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(2, 2)

    # Smoothie function
    xs, f_vals, ks, amplitudes = make_smoothie(n_terms=200, seed=1)

    axes[0, 0].plot(xs, f_vals, color='#0072BD', linestyle='-', linewidth=1.5)
    axes[0, 0].set_title('A smoothie on [-1,1]', fontsize=11)
    axes[0, 0].set_ylim(-4, 4)

    # Coefficient magnitudes
    axes[0, 1].semilogy(ks, amplitudes, 'k-', linewidth=1.5, label='C^{-√k}')
    axes[0, 1].set_title('Fourier coefficient amplitudes', fontsize=11)
    axes[0, 1].legend(fontsize=9)

    # Compare: smoothie vs analytic (exponential decay)
    ks_exp = np.arange(1, 201)
    amp_exp = np.exp(-ks_exp / 10.0)  # exponential decay -> analytic
    xs2, f2_vals, _, amp2 = make_smoothie(n_terms=200, seed=2)

    axes[1, 0].plot(xs, f_vals, color='#0072BD', linestyle='-', linewidth=1.5, label='Smoothie (seed=1)')
    axes[1, 0].plot(xs2, f2_vals, color='#D95319', linestyle='-', linewidth=1.5, alpha=0.7, label='Smoothie (seed=2)')
    axes[1, 0].set_title('Two smoothies', fontsize=11)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].set_ylim(-4, 4)

    # First and second derivative (approximate numerically)
    dx = xs[1] - xs[0]
    df = np.gradient(f_vals, dx)
    d2f = np.gradient(df, dx)

    axes[1, 1].plot(xs, df, color='#0072BD', linestyle='-', linewidth=1.5, label='f′')
    axes[1, 1].plot(xs, d2f / 50, color='#D95319', linestyle='-', linewidth=1.5, label='f″/50')
    axes[1, 1].set_title('Derivatives grow rapidly in amplitude', fontsize=11)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].set_ylim(-80, 80)

    print(f"Smoothie: max |f| = {np.max(np.abs(f_vals)):.4f}")
    print(f"Root-exp decay: amplitude at k=100 is {amplitudes[99]:.2e}")
    print(f"Compare exp decay: amplitude at k=100 is {np.exp(-100/10):.2e}")

    fig.suptitle('Smoothies: C∞ but Nowhere Analytic Functions', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'smoothies.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("smoothies: done")
    return True

if __name__ == "__main__":
    run()
