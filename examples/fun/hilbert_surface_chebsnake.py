"""A Hilbert curve on a surface and Chebsnake.

Plots the Hilbert space-filling curve and demonstrates how it can be
used as a space-filling path on a 2D surface.
Translated from fun/HilbertSurfaceChebsnake2.m.

Original: https://www.chebfun.org/examples/fun/HilbertSurfaceChebsnake2.html
Author: Georges Klein, March 2013
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

def hilbert_curve(order):
    """Generate Hilbert curve of given order. Returns (x, y) arrays."""
    # Use L-system approach
    n = 2**order
    # Build the curve iteratively
    x, y = np.array([0]), np.array([0])

    for _ in range(order):
        nx, ny = len(x), len(y)
        # 4 quadrants transformation
        # Upper-left: reflect x
        # Lower-left: rotate 90 degrees
        # Upper-right: copy
        # Lower-right: reflect y, rotate -90 degrees
        x1 = y.copy()
        y1 = x.copy()
        x2 = x.copy()
        y2 = y.copy() + nx
        x3 = x.copy() + nx
        y3 = y.copy() + nx
        x4 = (nx - 1 - y[::-1]) + nx
        y4 = nx - 1 - x[::-1]

        x = np.concatenate([x1, x2, x3, x4])
        y = np.concatenate([y1, y2, y3, y4])

    return x / (n - 1), y / (n - 1)

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/fun')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3)

    # --- Panel 1: Hilbert curve orders 1, 2, 3 ---
    colors = ['b', 'g', 'r']
    for order, col in zip([1, 2, 3], colors):
        hx, hy = hilbert_curve(order)
        axes[0].plot(hx, hy, '-', color=col, linewidth=max(0.5, 2-0.5*order),
                     label=f'Order {order}', alpha=0.8)
    axes[0].set_aspect('equal')
    axes[0].set_title('Hilbert space-filling curve\norders 1, 2, 3', fontsize=10)
    axes[0].legend(fontsize=9)

    # --- Panel 2: Hilbert curve on a surface ---
    hx4, hy4 = hilbert_curve(4)
    # Map to [-1, 1] x [-1, 1]
    hx4 = 2 * hx4 - 1
    hy4 = 2 * hy4 - 1

    # Evaluate function on curve
    f_on_curve = np.sin(np.pi * hx4) * np.cos(np.pi * hy4)

    # Background surface
    x2 = np.linspace(-1, 1, 50)
    X2, Y2 = np.meshgrid(x2, x2)
    F2 = np.sin(np.pi * X2) * np.cos(np.pi * Y2)

    axes[1].contourf(X2, Y2, F2, levels=15, cmap='RdBu_r', alpha=0.6)
    sc = axes[1].scatter(hx4, hy4, c=f_on_curve, cmap='RdBu_r',
                          s=2, zorder=3)
    axes[1].set_aspect('equal')
    axes[1].set_title('Hilbert curve on\nsin(πx)cos(πy) surface', fontsize=10)
    plt.colorbar(sc, ax=axes[1])

    # --- Panel 3: "Chebsnake" — path that visits Chebyshev nodes ---
    N = 20
    # Chebyshev nodes in 2D
    k = np.arange(N+1)
    nodes_1d = np.cos(np.pi * k / N)
    x_nodes, y_nodes = np.meshgrid(nodes_1d, nodes_1d)

    # Snake path: traverse rows alternately left-right
    snake_x, snake_y = [], []
    for i in range(N+1):
        row_x = nodes_1d if i % 2 == 0 else nodes_1d[::-1]
        row_y = np.full(N+1, nodes_1d[i])
        snake_x.extend(row_x)
        snake_y.extend(row_y)
    snake_x = np.array(snake_x)
    snake_y = np.array(snake_y)

    axes[2].plot(snake_x, snake_y, 'b-', linewidth=0.8, alpha=0.6,
                 label='Chebsnake path')
    axes[2].plot(snake_x[::N+1], snake_y[::N+1], 'rs', markersize=6,
                 label='Row starts')
    axes[2].set_aspect('equal')
    axes[2].set_title(f'Chebsnake: {N}×{N} Chebyshev grid\nsnake traversal', fontsize=10)
    axes[2].legend(fontsize=9)

    print(f"Hilbert curve order 4: {len(hx4)} points")
    print(f"Chebsnake: {N}×{N} = {(N+1)**2} Chebyshev nodes")

    fig.suptitle('Hilbert Curve and Chebsnake on 2D Surfaces', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'hilbert_surface_chebsnake.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("hilbert_surface_chebsnake: done")
    return True

if __name__ == "__main__":
    run()
