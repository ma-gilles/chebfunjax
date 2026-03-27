"""Random surfaces.

Explores smooth random functions on 2D domains and their level sets.
Translated from stats/RandomSurf.m.

Original: https://www.chebfun.org/examples/stats/RandomSurf.html
Author: Nick Trefethen, May 2019
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



def make_random_surface_disk(dx=0.1, seed=1, n_grid=100):
    """Random bandlimited function on unit disk via 2D Fourier series."""
    rng = np.random.default_rng(seed)
    n_modes = max(1, int(np.pi / dx))

    xs = np.linspace(-1, 1, n_grid)
    ys = np.linspace(-1, 1, n_grid)
    X, Y = np.meshgrid(xs, ys)

    f = np.zeros((n_grid, n_grid))
    for kx in range(-n_modes, n_modes + 1):
        for ky in range(-n_modes, n_modes + 1):
            if kx**2 + ky**2 <= n_modes**2:
                c = rng.standard_normal() + 1j * rng.standard_normal()
                c /= np.sqrt(kx**2 + ky**2 + 1)
                f += np.real(c * np.exp(1j * np.pi * (kx * X + ky * Y)))

    # Zero out outside unit disk
    mask = X**2 + Y**2 > 1
    f[mask] = 0
    # Normalize
    f /= (np.std(f[~mask]) + 1e-10)
    return xs, ys, X, Y, f


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3)

    xs, ys, X, Y, f = make_random_surface_disk(dx=0.2, seed=1, n_grid=150)
    mask = X**2 + Y**2 > 1

    # Paraboloid: 2 - 4*r^2
    R = np.sqrt(X**2 + Y**2)
    paraboloid = 2 - 4 * R**2
    paraboloid[mask] = np.nan

    total = f + paraboloid
    f_plot = f.copy(); f_plot[mask] = np.nan
    total[mask] = np.nan

    # Plot 1: random surface (contour)
    im = axes[0].contourf(X, Y, f_plot, levels=20, cmap='RdBu_r')
    circle = plt.Circle((0, 0), 1, fill=False, color='k', linewidth=2)
    axes[0].add_patch(circle)
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title('Random surface on disk', fontsize=11)
    axes[0].set_aspect('equal'); axes[0].axis('off')

    # Plot 2: paraboloid + random (zebra = alternating level sets)
    # We mimic 'zebra' by showing alternating filled contour bands
    n_levels = 15
    vmin, vmax = np.nanmin(total), np.nanmax(total)
    levels = np.linspace(vmin, vmax, n_levels + 1)
    cmap_zebra = matplotlib.colors.ListedColormap(['black', 'white'] * (n_levels // 2 + 1))
    axes[1].contourf(X, Y, total, levels=levels, cmap=cmap_zebra)
    circle2 = plt.Circle((0, 0), 1, fill=False, color='gray', linewidth=2)
    axes[1].add_patch(circle2)
    axes[1].set_title('Random + paraboloid (zebra)', fontsize=11)
    axes[1].set_aspect('equal'); axes[1].axis('off')

    # Plot 3: 3D surface
    ax3d = fig.add_subplot(1, 3, 3, projection='3d')
    fig.delaxes(axes[2])
    total_3d = total.copy()
    total_3d[mask] = np.nan
    ax3d.plot_surface(X, Y, total_3d, cmap='viridis', alpha=0.8,
                      rstride=3, cstride=3)
    ax3d.set_zlim(-10, 10)
    ax3d.set_title('Surface plot', fontsize=11)
    ax3d.set_xlabel('x'); ax3d.set_ylabel('y')

    print("random_surf: done")
    fig.suptitle('Random Surfaces on the Disk', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'random_surf.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    return True


if __name__ == "__main__":
    run()
