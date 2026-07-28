"""Normal distribution: exercises from a textbook.

Demonstrates computing probabilities, CDFs, and areas for the normal
distribution using chebfunjax. Translated from stats/NormalExercises.m.

Original: https://www.chebfun.org/examples/stats/NormalExercises.html
Author: Jie Gao and Nick Trefethen, April 2013
"""

import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
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

    mu, sigma = 2.0, 1.0

    # Problem 1(d): X ~ N(2,1).  Build the density as a chebfun on the whole
    # real line, integrate to a CDF, and read off P[|X-2| < 1] = fint(3)-fint(1).
    f = cj.chebfun(
        lambda x: jnp.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * jnp.sqrt(2 * jnp.pi)),
        domain=(-np.inf, np.inf))
    fint = f.cumsum()
    prob = float(fint(3.0) - fint(1.0))
    print("p =")
    print(f"   {prob:.15f}")
    exact_prob = prob

    # Variant: a non-Gaussian density exp(-|x-2|^(5/4)) (corner at x=2 ->
    # splitting), normalized to unit mass, same probability region.
    g = cj.chebfun(lambda x: jnp.exp(-jnp.abs((x - mu) / sigma)**(5 / 4)),
                   domain=(-np.inf, np.inf), splitting=True)
    g = g / g.sum()
    gint = g.cumsum()
    prob_mod = float(gint(3.0) - gint(1.0))
    print("p =")
    print(f"   {prob_mod:.15f}")

    def modified_pdf_unnorm(x, mu, sigma):
        return np.exp(-np.abs((x - mu) / sigma)**(5 / 4))

    # Plotting
    fig, axes = plt.subplots(1, 2)

    xs_plot = np.linspace(-1, 6, 400)
    pdf_plot = np.exp(-0.5 * ((xs_plot - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    axes[0].plot(xs_plot, pdf_plot, 'k-', linewidth=2)

    # Shade region [1, 3]
    mask = (xs_plot >= 1) & (xs_plot <= 3)
    axes[0].fill_between(xs_plot[mask], pdf_plot[mask], alpha=0.4, color='#77AC30',
                         label=f'P[|X-2|<1] = {exact_prob:.4f}')
    axes[0].set_title('N(2,1): P[|X-2|<1]', fontsize=11)
    axes[0].legend(fontsize=9)

    # Modified distribution
    xs_m = np.linspace(-1, 6, 400)
    f2_m = modified_pdf_unnorm(xs_m, mu, sigma)
    f2_m_norm = f2_m / np.trapezoid(f2_m, xs_m)
    axes[1].plot(xs_m, f2_m_norm, 'k-', linewidth=2)
    mask2 = (xs_m >= 1) & (xs_m <= 3)
    axes[1].fill_between(xs_m[mask2], f2_m_norm[mask2], alpha=0.4, color='#D95319',
                         label=f'P[|X-2|<1] = {prob_mod:.4f}')
    axes[1].set_title('Modified: exp(-|x-2|^{5/4})', fontsize=11)
    axes[1].legend(fontsize=9)

    fig.suptitle('Normal Distribution Exercises', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'normal_exercises.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("normal_exercises: done")
    return True

if __name__ == "__main__":
    run()
