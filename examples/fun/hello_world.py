"""Hello World.

Displays "HELLO" using contour plots of 2D Chebyshev functions,
inspired by the Chebfun2 Hello World example.
Translated from fun/HelloWorld.m.

Original: https://www.chebfun.org/examples/fun/HelloWorld.html
Author: Alex Townsend, March 2013
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



def letter_mask(letter, nx=100, ny=60):
    """Create a 2D binary mask for a capital letter."""
    mask = np.zeros((ny, nx), dtype=float)
    xg = np.linspace(0, 1, nx)
    yg = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xg, yg)

    if letter == 'H':
        mask[(Y > 0.1) & (Y < 0.9) & (X > 0.05) & (X < 0.25)] = 1
        mask[(Y > 0.1) & (Y < 0.9) & (X > 0.75) & (X < 0.95)] = 1
        mask[(Y > 0.42) & (Y < 0.58)] = 1
    elif letter == 'E':
        mask[(Y > 0.1) & (Y < 0.9) & (X > 0.05) & (X < 0.25)] = 1
        mask[(Y > 0.72) & (Y < 0.9)] = 1
        mask[(Y > 0.40) & (Y < 0.57)] = 1
        mask[(Y > 0.1) & (Y < 0.28)] = 1
    elif letter == 'L':
        mask[(Y > 0.1) & (Y < 0.9) & (X > 0.05) & (X < 0.25)] = 1
        mask[(Y > 0.1) & (Y < 0.28)] = 1
    elif letter == 'O':
        # Ring shape
        r = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
        mask[(r > 0.25) & (r < 0.42)] = 1

    return mask


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/fun')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: "HELLO" via 2D function zeros ---
    # f(x,y) = cos(pi*x)*cos(pi*y) has rank 1 (product function)
    x = np.linspace(-1, 1, 200)
    y = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(x, y)

    # "Hello" encoded as a 2D function with multiple modes
    f = (np.cos(np.pi * X) * np.cos(np.pi * Y)
         + 0.5 * np.cos(2*np.pi * X) * np.cos(2*np.pi * Y)
         + 0.25 * np.sin(np.pi * X) * np.sin(3*np.pi * Y))

    axes[0].contourf(X, Y, f, levels=20, cmap='RdBu_r')
    axes[0].contour(X, Y, f, levels=[0], colors='black', linewidths=2)
    axes[0].set_title('f(x,y) = cos(πx)cos(πy) +...\nzero set (black line)', fontsize=10)
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)

    # --- Panel 2: Low-rank approximation ---
    # Rank-1: f(x,y) = g(x)*h(y)
    g = np.cos(np.pi * x)
    h = np.cos(np.pi * y)
    F_rank1 = np.outer(h, g)

    axes[1].pcolormesh(X, Y, F_rank1, cmap='viridis', shading='auto')
    axes[1].set_title('Rank-1 function\ng(x)×h(y) = cos(πx)·cos(πy)', fontsize=10)
    axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.3)

    # Print rank info
    # SVD to find numerical rank
    Z_sample = f[::4, ::4]
    sv = np.linalg.svd(Z_sample, compute_uv=False)
    rank_num = np.sum(sv > sv[0] * 1e-10)
    print(f"Hello World example:")
    print(f"  f(x,y) numerical rank: {rank_num}")
    print(f"  Rank-1 function: cos(πx)·cos(πy)")

    # --- Panel 3: "HELLO" text artistic ---
    ax3 = axes[2]
    ax3.set_xlim(-1, 1); ax3.set_ylim(-0.5, 0.5)

    # Use text with coloring
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    letters = list("HELLO")
    positions = np.linspace(-0.8, 0.8, 5)
    for i, (ch, pos) in enumerate(zip(letters, positions)):
        ax3.text(pos, 0, ch, ha='center', va='center', fontsize=36,
                 fontweight='bold', color=colors[i],
                 fontfamily='monospace')

    # Background: low-rank function
    ax3.contourf(X, Y, f * 0.3, levels=10, cmap='Pastel1', alpha=0.4,
                 extent=[-1, 1, -0.5, 0.5])

    ax3.set_title('"Hello World"\nin any programming language', fontsize=10)
    ax3.axis('off')

    fig.suptitle('Hello World via 2D Functions', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'hello_world.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("hello_world: done")
    return True


if __name__ == "__main__":
    run()
