"""Smooth random walk.

Explores smooth analogues of random walks via integration of random
Chebyshev functions, approaching Brownian motion.
Translated from stats/SmoothRandomWalk.m.

Original: https://www.chebfun.org/examples/stats/SmoothRandomWalk.html
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


def make_random_chebfun(dx, n_pts=1000, rng=None, domain=(-1.0, 1.0)):
    """Create a smooth random function with characteristic scale dx.

    Uses random Fourier modes with max wavenumber ~pi/dx.
    """
    if rng is None:
        rng = np.random.default_rng(1)
    a, b = domain
    L = b - a
    # Number of Fourier modes
    n_modes = max(1, int(np.pi * L / dx))
    ks = np.arange(1, n_modes + 1)
    # Random amplitudes normalized by sqrt(n_modes)
    c_cos = rng.standard_normal(n_modes) / np.sqrt(n_modes)
    c_sin = rng.standard_normal(n_modes) / np.sqrt(n_modes)

    xs = np.linspace(a, b, n_pts)
    f_vals = np.zeros(n_pts)
    for k, cc, cs in zip(ks, c_cos, c_sin):
        freq = 2 * np.pi * k / L
        f_vals += cc * np.cos(freq * (xs - a)) + cs * np.sin(freq * (xs - a))

    return xs, f_vals


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    rng = np.random.default_rng(1)
    dx_values = [0.1, 0.025, 0.00625, 0.0015625]
    titles = [f'dx = {dx}' for dx in dx_values]

    for idx, (ax, dx, title) in enumerate(zip(axes.flat, dx_values, titles)):
        xs, fx = make_random_chebfun(dx, n_pts=2000, rng=rng)
        fy = make_random_chebfun(dx, n_pts=2000, rng=rng)[1]

        # Scale by dx^{-1/2} and integrate (cumulative sum)
        scale = dx**(-0.5)
        # Numerical integration via cumsum
        dt = xs[1] - xs[0]
        path_x = np.cumsum(scale * fx) * dt
        path_y = np.cumsum(scale * fy) * dt

        ax.plot(path_x, path_y, 'k-', linewidth=1 - 0.1 * idx, alpha=0.8)
        ax.plot([path_x[0], path_x[-1]], [path_y[0], path_y[-1]],
                '.r', markersize=10)
        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Smooth Random Walk (approaching Brownian motion)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'smooth_random_walk.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("smooth_random_walk: done")
    return True


if __name__ == "__main__":
    run()
