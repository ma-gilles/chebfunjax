"""Composition with multivariate chebfuns.

Demonstrates function composition in Chebfun: g(f) where f and g are
chebfuns of various types. We illustrate with 1D and 2D examples.
Translated from temp/ChebfunComposition.m.

Original: https://www.chebfun.org/examples/temp/ChebfunComposition.html
Author: Olivier Sete, February 2017
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

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/temp')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3)

    x_1d = np.linspace(-np.pi, np.pi, 500)
    x2 = np.linspace(-1, 1, 100)
    y2 = np.linspace(-1, 1, 100)
    X2, Y2 = np.meshgrid(x2, y2)

    # --- Panel 1: 1D composition h = g(f) where f=cos, g=exp ---
    f1 = np.cos(x_1d)
    g1 = np.exp  # g maps (-inf, inf) to (0, inf)
    h1 = g1(f1)  # exp(cos(t))

    axes[0].plot(x_1d, f1, 'b-', linewidth=2, label='f(t)=cos(t)')
    axes[0].plot(x_1d, h1, 'r-', linewidth=2, label='h(t)=exp(cos(t))')
    # Verify: h = exp(cos(t)) has range [exp(-1), exp(1)]
    assert abs(np.min(h1) - np.exp(-1)) < 0.01
    assert abs(np.max(h1) - np.exp(1)) < 0.01
    print(f"h = exp(cos(t)): range [{np.min(h1):.4f}, {np.max(h1):.4f}]")
    print(f"Expected: [{np.exp(-1):.4f}, {np.exp(1):.4f}]")

    axes[0].set_title('1D composition h=g(f)\nf=cos, g=exp', fontsize=10)
    axes[0].legend(fontsize=10)

    # --- Panel 2: 2D composition — restrict g(x,y) to a curve f(t) ---
    # g(x,y) = x^2 + y^2 (unit circle level set = 1)
    g2d = X2**2 + Y2**2

    # f(t) = (cos(t), sin(t)) is the unit circle
    t_circ = np.linspace(-np.pi, np.pi, 300)
    f_circ_x = np.cos(t_circ)
    f_circ_y = np.sin(t_circ)
    h_on_circ = f_circ_x**2 + f_circ_y**2  # should be ~1

    axes[1].contourf(X2, Y2, g2d, levels=20, cmap='viridis')
    axes[1].plot(f_circ_x, f_circ_y, 'r-', linewidth=3, label='Unit circle')
    axes[1].set_aspect('equal')
    axes[1].set_title('g(x,y)=x²+y² restricted to\ncircle → constant 1', fontsize=10)
    axes[1].legend(fontsize=9)
    print(f"g(cos(t),sin(t)) range: [{np.min(h_on_circ):.8f}, {np.max(h_on_circ):.8f}]")
    axes[1].text(0, 0, f'g=1 on circle', ha='center', color='white',
                 fontsize=12, fontweight='bold')

    # --- Panel 3: Restriction to non-rectangular domain ---
    # g(x,y) = x^2 + y, f maps [0,1]^2 to triangular region via (x+y, y)
    # Compose: h(x,y) = g(x+y, y) = (x+y)^2 + y
    x_tri = np.linspace(0, 1, 100)
    y_tri = np.linspace(0, 1, 100)
    X_tri, Y_tri = np.meshgrid(x_tri, y_tri)
    # Restrict to lower triangle: x+y <= 1
    mask = (X_tri + Y_tri <= 1)

    # Original g(x,y) = exp(-2*(x^2+y^2))*cos(3*pi*(x+y))
    def g_nice(x, y):
        return np.exp(-2*(x**2 + y**2)) * np.cos(3*np.pi*(x+y))

    # f: (x,y) -> (x+y, y) for (x,y) in triangle
    X_mapped = X_tri + Y_tri
    Y_mapped = Y_tri
    h_composed = g_nice(X_mapped, Y_mapped)
    h_composed[~mask] = np.nan

    im = axes[2].pcolormesh(X_tri, Y_tri, h_composed, cmap='RdBu_r',
                              shading='auto')
    # Triangle boundary
    axes[2].plot([0,1,0,0], [0,0,1,0], 'k-', linewidth=2, label='Triangle domain')
    axes[2].set_aspect('equal')
    axes[2].set_title('h(x,y)=g(x+y,y) on triangle\ng=exp(-2r²)cos(3π(x+y))', fontsize=9)
    axes[2].legend(fontsize=9)
    plt.colorbar(im, ax=axes[2])

    fig.suptitle('Composition with Multivariate Functions', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'chebfun_composition.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("chebfun_composition: done")
    return True

if __name__ == "__main__":
    run()
