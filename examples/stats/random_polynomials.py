"""Random polynomials and their statistics.

Explores the distribution of roots of random polynomials with
chebfunjax. Based on Chebfun example stats and roots/RandomPolynomials.m.

Original: https://www.chebfun.org/examples/roots/RandomPolynomials.html
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



def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(42)

    # --- Random polynomials in the Chebyshev basis -----------------------
    # A "random Chebfun" with Gaussian coefficients decaying as exp(-k/10)
    n = 50
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    all_roots = []
    n_trials = 30
    for trial in range(n_trials):
        coeffs = rng.standard_normal(n) * np.exp(-np.arange(n) / 10.0)
        f = cj.chebfun.from_coeffs(jnp.array(coeffs))
        roots = np.array(f.roots())
        roots = roots[(roots >= -1.0) & (roots <= 1.0)]
        all_roots.extend(roots.tolist())

        if trial < 5:
            xx = np.linspace(-1, 1, 400)
            fv = np.array(f(jnp.array(xx)))
            axes[0].plot(xx, fv, 'b-', linewidth=0.5, alpha=0.4)

    axes[0].axhline(0, color='k', linewidth=0.8)
    axes[0].set_title(f'5 random Chebfuns (degree ~{n})', fontsize=11)
    axes[0].set_xlabel('$x$')
    axes[0].set_ylim(-2, 2)
    axes[0].grid(True, alpha=0.3)

    # Histogram of root locations
    axes[1].hist(all_roots, bins=30, density=True, color='steelblue',
                 edgecolor='white', alpha=0.8)
    # Arcsine distribution (Chebyshev equidistribution): rho(x) = 1/(pi*sqrt(1-x^2))
    xx_arc = np.linspace(-0.999, 0.999, 400)
    axes[1].plot(xx_arc, 1.0 / (np.pi * np.sqrt(1 - xx_arc**2)), 'r-',
                 linewidth=2, label='Arcsine density')
    axes[1].set_title(f'Root distribution over {n_trials} trials', fontsize=11)
    axes[1].set_xlabel('Root location')
    axes[1].set_ylabel('Density')
    axes[1].legend(fontsize=10)
    axes[1].set_xlim(-1.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'random_polynomials.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Verify arcsine equidistribution holds approximately
    # Integral of roots in (-0.5, 0.5) should be about arcsin(0.5)/pi + 0.5 = 1/3
    roots_arr = np.array(all_roots)
    fraction_center = np.mean((roots_arr > -0.5) & (roots_arr < 0.5))
    # Exact: integral from -0.5 to 0.5 of 1/(pi*sqrt(1-x^2)) dx = arcsin(0.5)*2/pi = 1/3
    expected = 2 * np.arcsin(0.5) / np.pi
    print(f"Fraction of roots in (-0.5,0.5): {fraction_center:.4f}  (expected ~{expected:.4f})")

    print("random_polynomials: done")
    return True


if __name__ == "__main__":
    run()
