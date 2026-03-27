"""Maxima of a random function.

Counts local maxima of random bandlimited functions and studies how
the number scales with interval length.
Translated from stats/RandomMaxima.m.

Original: https://www.chebfun.org/examples/stats/RandomMaxima.html
Author: Nick Trefethen, February 2017
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



def make_random_bandlimited(L, dx=1.0, seed=0):
    """Generate a random bandlimited function on [0, L] with scale dx."""
    rng = np.random.default_rng(seed)
    n_pts = max(int(L * 20), 200)
    xs = np.linspace(0, L, n_pts)
    n_modes = max(1, int(np.pi * L / dx))
    f_vals = np.zeros(n_pts)
    for k in range(1, n_modes + 1):
        freq = 2 * np.pi * k / L
        c = rng.standard_normal()
        s = rng.standard_normal()
        f_vals += (c * np.cos(freq * xs) + s * np.sin(freq * xs)) / np.sqrt(n_modes)
    return xs, f_vals


def count_local_maxima(f_vals):
    """Count strict local maxima (interior only)."""
    count = 0
    maxima_idx = []
    for i in range(1, len(f_vals) - 1):
        if f_vals[i] > f_vals[i-1] and f_vals[i] > f_vals[i+1]:
            count += 1
            maxima_idx.append(i)
    return count, maxima_idx


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2)

    # Plot random function on [0, 20] with local maxima marked
    dx = 1.0
    xs20, f20 = make_random_bandlimited(20, dx=dx, seed=0)
    n_max20, max_idx20 = count_local_maxima(f20)

    axes[0].plot(xs20, f20, 'k-', linewidth=2)
    axes[0].plot(xs20[max_idx20], f20[max_idx20], '.r', markersize=12)
    axes[0].set_title(f'{n_max20} local maxima on [0,20]', fontsize=12)
    axes[0].set_xlabel('x'); axes[0].grid(True, alpha=0.3)

    # Scaling experiment: how many maxima vs L?
    L_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    nmax_values = []
    for L in L_values:
        xs, f = make_random_bandlimited(L, dx=dx, seed=42)
        nmax, _ = count_local_maxima(f)
        nmax_values.append(nmax)

    axes[1].loglog(L_values, L_values, '-r', linewidth=2, label='O(L)')
    axes[1].loglog(L_values, nmax_values, '.b', markersize=12, label='# maxima')
    axes[1].set_xlabel('Length of interval', fontsize=11)
    axes[1].set_ylabel('Number of maxima', fontsize=11)
    axes[1].set_title('Maxima count scales linearly with L', fontsize=11)
    axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3)

    print(f"L=20: {n_max20} maxima")
    print("L vs #maxima (log scale):")
    for L, n in zip(L_values, nmax_values):
        print(f"  L={L:5d}: {n} maxima (ratio {n/L:.2f})")

    fig.suptitle('Maxima of a Random Function', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'random_maxima.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("random_maxima: done")
    return True


if __name__ == "__main__":
    run()
