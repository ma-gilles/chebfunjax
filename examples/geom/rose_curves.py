"""Rose curves.

Demonstrates rose curves r = sin(k*theta) in polar coordinates,
filling a grid for various rational values of k = m/n.
Translated from geom/RoseCurves.m.

Original: https://www.chebfun.org/examples/geom/RoseCurves.html
Author: Hrothgar, June 2014
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



def rose_curve(m, n, n_pts=5000):
    """Generate rose curve for k = m/n.

    Domain: [0, 2*pi*lcm(m,n)].
    Parametrize in Cartesian: x = cos(k*t)*cos(t), y = cos(k*t)*sin(t).
    """
    import math
    lcm_mn = m * n // math.gcd(m, n)
    t = np.linspace(0, 2 * np.pi * lcm_mn, n_pts)
    k = m / n
    r = np.cos(k * t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x + 1j * y


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.axis('off')

    N = 6
    for m in range(1, N + 1):
        for n in range(1, N + 1):
            z = rose_curve(m, n)
            offset = 2.5 * m + (-2.5j * n)
            ax.plot(np.real(z) + np.real(offset),
                    np.imag(z) + np.imag(offset),
                    'k-', linewidth=0.8, alpha=0.8)

    # Labels
    for m in range(1, N + 1):
        ax.text(2.5 * m, -2.5 * N - 1.5, f'm={m}', ha='center', fontsize=8)
    for n in range(1, N + 1):
        ax.text(0.5, -2.5 * n, f'n={n}', ha='center', fontsize=8)

    fig.suptitle('Rose Curves r = cos(m/n · θ)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'rose_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Verify: k=1 gives a circle
    z_circle = rose_curve(1, 1)
    radius = np.abs(z_circle)
    print(f"Rose(m=1,n=1): max radius = {np.max(radius):.4f}  (should be ≤ 1)")

    # k=2/1 gives 4 petals
    z_4petal = rose_curve(2, 1)
    print(f"Rose(m=2,n=1): max radius = {np.max(np.abs(z_4petal)):.4f}")

    print("rose_curves: done")
    return True


if __name__ == "__main__":
    run()
