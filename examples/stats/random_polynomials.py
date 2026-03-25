"""Random polynomials and random maxima.

Demonstrates the statistics of random polynomials (roots, maxima),
following stats/RandomPolynomials.m and stats/RandomMaxima.m.

Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj


def run():
    print("=" * 60)
    print("Random polynomials and random maxima")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # --- Random polynomials: distribution of roots on [-1,1] ---
    print("\nRandom Chebyshev expansions (degree 20):")
    n_poly = 21  # degree 20
    n_trials = 50
    all_roots = []
    n_roots_list = []

    for _ in range(n_trials):
        coeffs = rng.standard_normal(n_poly)
        coeffs[-1] = abs(coeffs[-1]) + 0.1  # ensure non-trivial leading term
        f = cj.chebfun.from_coeffs(jnp.array(coeffs))
        r = f.roots()
        real_roots = np.real(np.array(r))
        in_domain = real_roots[np.abs(real_roots) <= 1.0 + 1e-10]
        all_roots.extend(in_domain.tolist())
        n_roots_list.append(len(in_domain))

    print(f"  Average roots in [-1,1]: {np.mean(n_roots_list):.2f}")
    print(f"  Total root samples: {len(all_roots)}")
    assert len(all_roots) > 0

    # --- Random maxima: distribution ---
    print("\nDistribution of maxima of random polynomials:")
    maxima = []
    for _ in range(n_trials):
        coeffs = rng.standard_normal(n_poly)
        f = cj.chebfun.from_coeffs(jnp.array(coeffs))
        try:
            _, m_val = f.max()
            maxima.append(float(m_val))
        except Exception:
            pass

    if maxima:
        print(f"  Mean maximum: {np.mean(maxima):.4f}")
        print(f"  Std maximum:  {np.std(maxima):.4f}")
        assert all(np.isfinite(maxima))

    # --- A specific random polynomial ---
    key = jax.random.PRNGKey(0)
    coeffs_fixed = jax.random.normal(key, (11,))
    f_fixed = cj.chebfun.from_coeffs(coeffs_fixed)
    roots_fixed = f_fixed.roots()
    print(f"\nFixed random polynomial (degree 10):")
    print(f"  Number of real roots in [-1,1]: {len([r for r in np.real(np.array(roots_fixed)) if abs(r) <= 1+1e-10])}")
    x_max_v, max_v = f_fixed.max()
    x_min_v, min_v = f_fixed.min()
    print(f"  Max: {float(max_v):.6f} at x={float(x_max_v):.6f}")
    print(f"  Min: {float(min_v):.6f} at x={float(x_min_v):.6f}")
    assert float(max_v) > float(min_v)

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # A few random polynomials
    for i in range(5):
        c = rng.standard_normal(n_poly)
        f_i = cj.chebfun.from_coeffs(jnp.array(c))
        xs_p = np.linspace(-1, 1, 200)
        ys_p = [float(f_i(jnp.array(xi))) for xi in xs_p]
        axes[0].plot(xs_p, ys_p, alpha=0.7, linewidth=1)
    axes[0].set_title("5 random polynomials (deg 20)", fontsize=11)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("f(x)")
    axes[0].grid(True, alpha=0.3)

    # Root distribution
    if all_roots:
        axes[1].hist(all_roots, bins=25, color='steelblue', edgecolor='k', alpha=0.7, density=True)
        # Arcsine distribution density: 1/(pi*sqrt(1-x^2))
        xs_arc = np.linspace(-0.99, 0.99, 200)
        axes[1].plot(xs_arc, 1/(np.pi*np.sqrt(1-xs_arc**2)), 'r-', linewidth=2, label='Arcsine law')
        axes[1].set_title(f"Root distribution ({n_trials} polynomials)", fontsize=11)
        axes[1].set_xlabel("Root location"); axes[1].set_ylabel("Density")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Maximum distribution
    if maxima:
        axes[2].hist(maxima, bins=20, color='coral', edgecolor='k', alpha=0.7)
        axes[2].axvline(np.mean(maxima), color='r', linestyle='--', label=f'Mean={np.mean(maxima):.2f}')
        axes[2].set_title("Distribution of maxima", fontsize=11)
        axes[2].set_xlabel("Maximum value"); axes[2].set_ylabel("Count")
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

    fig.suptitle("Random polynomials statistics", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "random_polynomials.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
