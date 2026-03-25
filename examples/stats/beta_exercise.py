"""Beta distribution exercises.

Demonstrates mode, median, and Bayesian inference using beta distributions.
Translated from stats/BetaExercise.m.

Original: https://www.chebfun.org/examples/stats/BetaExercise.html
Author: Jie Gao, July 2013
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.special import beta as beta_func
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj


def beta_pdf(xs, a, b):
    """Beta distribution PDF."""
    result = np.zeros_like(xs, dtype=float)
    mask = (xs > 0) & (xs < 1)
    result[mask] = xs[mask]**(a-1) * (1 - xs[mask])**(b-1) / beta_func(a, b)
    return result


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- 1. Beta(1.5, 3): mode ---
    a, b = 1.5, 3.0
    xs = np.linspace(0, 1, 1000)
    f = beta_pdf(xs, a, b)

    mode_idx = np.argmax(f)
    mode_val = xs[mode_idx]
    mode_exact = (a - 1) / (a + b - 2)
    print(f"Beta({a},{b}) mode = {mode_val:.6f}  (exact: {mode_exact:.6f})")
    assert abs(mode_val - mode_exact) < 0.005

    axes[0].plot(xs, f, 'k-', linewidth=2)
    axes[0].axvline(mode_val, color='r', linewidth=2, linestyle='--',
                    label=f'mode = {mode_val:.3f}')
    axes[0].set_title(f'Beta({a},{b}): mode', fontsize=11)
    axes[0].set_xlabel('x'); axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- 2. Modified distribution: mode and median ---
    xs2 = np.linspace(1e-6, 2 - 1e-6, 2000)
    g_unnorm = ((xs2/2)**(np.log(a))) * ((2 - xs2/2)**(np.exp(b-1) * np.sqrt(xs2)))
    g_unnorm = np.where(np.isfinite(g_unnorm), g_unnorm, 0)
    norm_g = np.trapezoid(g_unnorm, xs2)
    g = g_unnorm / norm_g

    gmode_idx = np.argmax(g)
    gmode = xs2[gmode_idx]

    cdf_g = np.cumsum(g) * (xs2[1] - xs2[0])
    cdf_g = cdf_g / cdf_g[-1]
    median_idx = np.searchsorted(cdf_g, 0.5)
    median_g = xs2[median_idx]

    print(f"Modified dist: mode = {gmode:.4f}, median = {median_g:.4f}")

    axes[1].plot(xs2, g, 'b-', linewidth=2)
    axes[1].axvline(gmode, color='g', linewidth=2, linestyle='--',
                    label=f'mode = {gmode:.3f}')
    axes[1].axvline(median_g, color='k', linewidth=2, linestyle='--',
                    label=f'median = {median_g:.3f}')
    axes[1].set_title('Modified distribution', fontsize=11)
    axes[1].set_xlabel('x'); axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # --- 3. Bayesian inference: Beta priors, update with data ---
    theta = np.linspace(0, 1, 1000)

    # Three priors
    priors = [
        (0.5, 0.5, 'Beta(0.5,0.5)\nJeffreys', 'k'),
        (1.0, 1.0, 'Beta(1,1)\nuniform', 'r'),
        (2.0, 2.0, 'Beta(2,2)\nskeptical', 'g'),
    ]

    # Data: x=13 successes from n=16
    y, n = 13, 16

    axes[2].set_prop_cycle(None)
    for (a_pr, b_pr, label, color) in priors:
        # Prior
        prior = beta_pdf(theta, a_pr, b_pr)
        # Posterior = Beta(y+a, n-y+b)
        posterior = beta_pdf(theta, y + a_pr, n - y + b_pr)

        axes[2].plot(theta, prior, '--', color=color, linewidth=1.5, alpha=0.7)
        axes[2].plot(theta, posterior, '-', color=color, linewidth=2, label=label)

    axes[2].set_title(f'Prior (dashed) and Posterior (solid)\n(x={y}, n={n})', fontsize=10)
    axes[2].set_xlabel('θ'); axes[2].legend(fontsize=8, loc='upper left')
    axes[2].grid(True, alpha=0.3)

    # Prior and posterior odds
    print("\nBayesian inference (H0: theta >= 0.6 vs H1: theta < 0.6):")
    for (a_pr, b_pr, label, color) in priors:
        from scipy.stats import beta as beta_dist
        P_H1_prior = beta_dist.cdf(0.6, a_pr, b_pr)
        P_H0_prior = 1 - P_H1_prior
        prior_odds = P_H0_prior / P_H1_prior

        P_H1_post = beta_dist.cdf(0.6, y + a_pr, n - y + b_pr)
        P_H0_post = 1 - P_H1_post
        post_odds = P_H0_post / P_H1_post
        bayes_factor = post_odds / prior_odds

        print(f"  {label.split(chr(10))[0]:20s}: prior_odds={prior_odds:.3f}, "
              f"post_odds={post_odds:.3f}, BF={bayes_factor:.3f}")

    fig.suptitle('Beta Distribution Exercises and Bayesian Inference', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'beta_exercise.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("beta_exercise: done")
    return True


if __name__ == "__main__":
    run()
