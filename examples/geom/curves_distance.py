"""Distance between two curves.

Finds the minimum distance between two parametric curves by computing
the distance function and finding its global minimum.
Translated from geom/Curves.m.

Original: https://www.chebfun.org/examples/geom/Curves.html
Author: Nick Trefethen, November 2022
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj


def make_random_curve(rng, x_offset, n_pts=500):
    """Random smooth curve: x = x_offset + noise, y = t."""
    t = np.linspace(-1, 1, n_pts)
    # Smooth random perturbation
    n_modes = 10
    noise = np.zeros(n_pts)
    for k in range(1, n_modes + 1):
        c = rng.standard_normal() * 0.2 / k
        s = rng.standard_normal() * 0.2 / k
        noise += c * np.cos(k * np.pi * t) + s * np.sin(k * np.pi * t)
    x = x_offset + noise
    y = t
    return x + 1j * y


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(1)
    n_pts = 500

    # Two curves, separated by ~2 units
    f = make_random_curve(rng, x_offset=-1.0, n_pts=n_pts)
    g = make_random_curve(rng, x_offset=1.0, n_pts=n_pts)

    # Distance matrix: d(i,j) = |f(t_i) - g(t_j)|
    f_col = f[:, np.newaxis]
    g_row = g[np.newaxis, :]
    D = np.abs(f_col - g_row)

    # Find minimum distance
    min_idx = np.unravel_index(np.argmin(D), D.shape)
    min_dist = D[min_idx]
    f_closest = f[min_idx[0]]
    g_closest = g[min_idx[1]]
    print(f"Minimum distance between curves: {min_dist:.6f}")
    print(f"Closest points: f = {f_closest:.4f}, g = {g_closest:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot curves and closest points
    axes[0].plot(np.real(f), np.imag(f), 'b-', linewidth=2, label='Curve f')
    axes[0].plot(np.real(g), np.imag(g), 'r-', linewidth=2, label='Curve g')
    axes[0].plot([np.real(f_closest), np.real(g_closest)],
                 [np.imag(f_closest), np.imag(g_closest)],
                 '--k', linewidth=1.5)
    axes[0].plot([np.real(f_closest), np.real(g_closest)],
                 [np.imag(f_closest), np.imag(g_closest)],
                 '.k', markersize=10)
    axes[0].set_title(f'Minimum distance = {min_dist:.4f}', fontsize=11)
    axes[0].set_aspect('equal'); axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Contour plot of distance function
    t_vals = np.linspace(-1, 1, n_pts)
    im = axes[1].contourf(t_vals, t_vals, D.T, levels=20, cmap='viridis')
    plt.colorbar(im, ax=axes[1])
    axes[1].plot(t_vals[min_idx[0]], t_vals[min_idx[1]], '.w', markersize=12)
    axes[1].set_title('Distance d(x,y) = |f(x) - g(y)|', fontsize=11)
    axes[1].set_xlabel('x (parameter of f)')
    axes[1].set_ylabel('y (parameter of g)')

    fig.suptitle('Minimum Distance Between Two Curves', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'curves_distance.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("curves_distance: done")
    return True


if __name__ == "__main__":
    run()
