"""Lissajous curves.

Demonstrates Lissajous curves defined by sinusoidal parametric equations
with different frequencies. Translated from geom/Lissajous.m.

Original: https://www.chebfun.org/examples/geom/Lissajous.html
Author: Nick Trefethen, October 2010
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

def lissajous(m, n, d, n_pts=2000):
    """Generate Lissajous curve: x=sin(mt), y=sin(nt + d*pi)."""
    t = np.linspace(0, 2 * np.pi, n_pts)
    x = np.sin(m * t)
    y = np.sin(n * t + d * np.pi)
    return x + 1j * y

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(2, 4)

    # Row 1: m=5, n=6 with d=0 and d=1/2
    m, n = 5, 6
    z0 = lissajous(m, n, 0)
    z1 = lissajous(m, n, 0.5)

    axes[0, 0].plot(np.real(z0), np.imag(z0), color='#0072BD', linestyle='-', linewidth=1.5)
    axes[0, 0].set_title(f'm={m} n={n} d=0.0', fontsize=10)
    axes[0, 0].set_aspect('equal'); axes[0, 0].axis('off')

    axes[0, 1].plot(np.real(z1), np.imag(z1), color='#D95319', linestyle='-', linewidth=1.5)
    axes[0, 1].set_title(f'm={m} n={n} d=0.5', fontsize=10)
    axes[0, 1].set_aspect('equal'); axes[0, 1].axis('off')

    # Row 1, columns 3-4: random Lissajous
    rng = np.random.default_rng(2)
    colors = [[1, 0, 0], [0, 0.8, 0], [1, 0.75, 0], [0, 0, 0.75]]
    for k, (ax, color) in enumerate(zip(axes[0, 2:], colors[:2])):
        mi = rng.integers(1, 11)
        ni = rng.integers(1, 11)
        di = rng.uniform()
        z = lissajous(mi, ni, di)
        ax.plot(np.real(z), np.imag(z), '-', color=color, linewidth=1.5)
        ax.set_title(f'm={mi} n={ni} d={di:.2f}', fontsize=9)
        ax.set_aspect('equal'); ax.axis('off')

    # Row 2: more random Lissajous with diverse parameters
    rng2 = np.random.default_rng(42)
    for k, (ax, color) in enumerate(zip(axes[1, :], colors + [[1, 0, 1], [0.5, 0, 0.5]])):
        mi = rng2.integers(1, 15)
        ni = rng2.integers(1, 15)
        di = rng2.uniform()
        z = lissajous(mi, ni, di)
        ax.plot(np.real(z), np.imag(z), '-',
                color=color if k < len(colors) else 'k', linewidth=1.5)
        ax.set_title(f'm={mi} n={ni} d={di:.3f}', fontsize=9)
        ax.set_aspect('equal'); ax.axis('off')

    fig.suptitle('Lissajous Curves', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'lissajous.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Verify arc length
    t = np.linspace(0, 2 * np.pi, 10000)
    m, n, d = 3, 4, 0.25
    x = np.sin(m * t)
    y = np.sin(n * t + d * np.pi)
    dx = np.diff(x); dy = np.diff(y)
    arc_length = np.sum(np.sqrt(dx**2 + dy**2))
    print(f"Lissajous m=3,n=4,d=0.25: arc length ≈ {arc_length:.4f}")

    print("lissajous: done")
    return True

if __name__ == "__main__":
    run()
