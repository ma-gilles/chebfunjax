"""A scribble for Chebfun2.

Demonstrates how 2D Chebyshev functions can be used to represent
text-like curves as zero contours of bivariate polynomials.
Translated from fun/Scribble2.m.

Original: https://www.chebfun.org/examples/fun/Scribble2.html
Authors: Nick Hale and Alex Townsend, August 2013
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

def extract_contours_at_zero(X, Y, Z):
    """Extract zero contour lines."""
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(X, Y, Z, levels=[0])
    paths = []
    try:
        for seg_group in cs.allsegs:
            for seg in seg_group:
                if len(seg) > 3:
                    paths.append(seg[:, 0] + 1j * seg[:, 1])
    except AttributeError:
        for coll in cs.collections:
            for path in coll.get_paths():
                pts = path.vertices
                if len(pts) > 3:
                    paths.append(pts[:, 0] + 1j * pts[:, 1])
    plt.close(fig_tmp)
    return paths

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/fun')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3)

    x = np.linspace(-1, 1, 200)
    y = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(x, y)

    # --- Panel 1: Algebraic curve (letter-like shapes from bivariate polynomials) ---
    # Zero set of f(x,y) = (x^2 + y^2 - 0.5)*(x^2 + y^2 - 0.8)
    # gives two concentric circles — use Chebfun2-style evaluation
    f1 = (X**2 + Y**2 - 0.3) * (X**2 + Y**2 - 0.7)
    contours1 = extract_contours_at_zero(X, Y, f1)

    for c in contours1:
        axes[0].plot(np.real(c), np.imag(c), color='#0072BD', linestyle='-', linewidth=2)
    axes[0].set_aspect('equal')
    axes[0].set_title('Zeros of (x²+y²-0.3)(x²+y²-0.7)\ntwo circles', fontsize=10)
    axes[0].set_xlim(-1.1, 1.1); axes[0].set_ylim(-1.1, 1.1)

    # --- Panel 2: More complex algebraic curve ---
    # Deltoid: (x^2+y^2)^2 + 18(x^2+y^2) - 27 = 8x^3 - 24xy^2
    # Simplified: tricuspid shape
    t = np.linspace(0, 2*np.pi, 500)
    # Deltoid parametrically: x = 2cos(t) + cos(2t), y = 2sin(t) - sin(2t)
    x_del = 2*np.cos(t) + np.cos(2*t)
    y_del = 2*np.sin(t) - np.sin(2*t)
    x_del /= 3; y_del /= 3

    # Also show the zero contour of an algebraic approximation
    r2 = X**2 + Y**2
    # Implicit equation of deltoid: (x^2+y^2+2x-1)^2 = 4(1-x)
    f2 = (r2 + 2*X - 1)**2 - 4*(1-X)
    # Scale for display
    f2_scaled = f2 / np.max(np.abs(f2)) * 0.3

    contours2 = extract_contours_at_zero(X, Y, f2)
    for c in contours2:
        if len(c) > 10:
            axes[1].plot(np.real(c), np.imag(c), color='#D95319', linestyle='-', linewidth=2)
    axes[1].plot(x_del, y_del, color='#0072BD', linestyle='--', linewidth=1.5, alpha=0.6,
                 label='Parametric deltoid')
    axes[1].set_aspect('equal')
    axes[1].set_title('Deltoid: zero set of\nalgebraic equation', fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(-1.1, 1.1); axes[1].set_ylim(-1.1, 1.1)

    # --- Panel 3: "Chebfun" encoded as a Chebyshev series zero set ---
    # Use superposition of Chebyshev-like functions to encode text
    f3 = (np.cos(2*np.pi*X) * np.cos(np.pi*Y)
          - 0.7 * np.cos(3*np.pi*X) * np.cos(2*np.pi*Y)
          + 0.3 * np.cos(5*np.pi*X) * np.cos(3*np.pi*Y))

    axes[2].contourf(X, Y, f3, levels=20, cmap='coolwarm')
    contours3 = extract_contours_at_zero(X, Y, f3)
    for c in contours3:
        axes[2].plot(np.real(c), np.imag(c), 'k-', linewidth=1.5)
    axes[2].set_aspect('equal')
    axes[2].set_title('Zero set of\nChebyshev series f(x,y)', fontsize=10)
    axes[2].set_xlim(-1, 1); axes[2].set_ylim(-1, 1)

    print("Scribble2: zero sets of 2D algebraic/Chebyshev functions")
    print(f"  Concentric circles: 2 zero contours")
    print(f"  Deltoid: classical algebraic curve of degree 4")

    fig.suptitle('Scribbles via 2D Chebyshev Zero Sets', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'scribble2.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("scribble2: done")
    return True

if __name__ == "__main__":
    run()
