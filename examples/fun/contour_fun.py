"""Implicit functions and fun with contours.

Demonstrates how contour lines of a function f(x,y)=0 can be used to
define parametric curves, illustrated with a circle and other implicit curves.
Translated from fun/ContourFun.m.

Original: https://www.chebfun.org/examples/fun/ContourFun.html
Author: Stefan Guttel, July 2012
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



def extract_contours(X, Y, Z, level=0.0):
    """Extract contour lines at given level from a grid using matplotlib."""
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(X, Y, Z, levels=[level])
    paths = []
    # Support both old (cs.collections) and new (cs.allsegs) matplotlib APIs
    try:
        # New matplotlib >= 3.8 API
        for seg_group in cs.allsegs:
            for seg in seg_group:
                if len(seg) > 2:
                    paths.append(seg[:, 0] + 1j * seg[:, 1])
    except AttributeError:
        for collection in cs.collections:
            for path in collection.get_paths():
                pts = path.vertices
                if len(pts) > 2:
                    paths.append(pts[:, 0] + 1j * pts[:, 1])
    plt.close(fig_tmp)
    return paths


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/fun')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3)

    # --- 1. Circle from implicit equation x^2 + y^2 - r^2 = 0 ---
    r = 0.8
    x = np.linspace(-1, 1, 40)
    X, Y = np.meshgrid(x, x)
    Z = X**2 + Y**2 - r**2

    contours = extract_contours(X, Y, Z, level=0.0)

    # True circle
    t_true = np.linspace(0, 2*np.pi, 200)
    axes[0].plot(r*np.cos(t_true), r*np.sin(t_true), 'k-', linewidth=2,
                 label='Exact circle')
    axes[0].plot(X.ravel(), Y.ravel(), 'b+', markersize=4, alpha=0.4,
                 label='Grid points')
    for c in contours:
        axes[0].plot(np.real(c), np.imag(c), 'r-o', markersize=4,
                     linewidth=1.5, label='Contour approx')
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Circle x²+y²={r}²\nfrom 40×40 grid contours', fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(-1.1, 1.1); axes[0].set_ylim(-1.1, 1.1)

    # Verify residual
    if contours:
        c0 = contours[0]
        residual = np.max(np.abs(np.real(c0)**2 + np.imag(c0)**2 - r**2))
        print(f"Circle contour residual (max |f|): {residual:.6f}")

    # --- 2. Figure-eight / lemniscate: (x^2+y^2)^2 = x^2 - y^2 ---
    x2 = np.linspace(-1.2, 1.2, 80)
    X2, Y2 = np.meshgrid(x2, x2)
    Z2 = (X2**2 + Y2**2)**2 - (X2**2 - Y2**2)

    contours2 = extract_contours(X2, Y2, Z2, level=0.0)

    for c in contours2:
        axes[1].plot(np.real(c), np.imag(c), 'b-', linewidth=2)
    axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Lemniscate of Bernoulli\n(x²+y²)² = x²−y²', fontsize=10)
    axes[1].set_xlim(-1.3, 1.3); axes[1].set_ylim(-0.8, 0.8)

    # --- 3. Multiple level curves: Cassini ovals ---
    # (x^2 + y^2)^2 - 2c^2(x^2 - y^2) = a^4 - c^4
    # Equivalently: (x^2+y^2+c^2)^2 - 4c^2*x^2 = a^4
    c_val = 1.0
    x3 = np.linspace(-2.5, 2.5, 100)
    X3, Y3 = np.meshgrid(x3, x3)
    # Cassini oval: product of distances from two foci = b^2
    # foci at (±c, 0): sqrt((x-c)^2+y^2) * sqrt((x+c)^2+y^2) = b^2
    dist_sq = ((X3 - c_val)**2 + Y3**2) * ((X3 + c_val)**2 + Y3**2)

    colors = ['b', 'g', 'r', 'm', 'c']
    b_values = [0.5, 0.8, 1.0, 1.2, 1.8]
    for b_val, col in zip(b_values, colors):
        Z3 = dist_sq - b_val**4
        c_list = extract_contours(X3, Y3, Z3, level=0.0)
        for c in c_list:
            axes[2].plot(np.real(c), np.imag(c), '-', color=col,
                         linewidth=1.5, label=f'b={b_val}')
    # deduplicate legend
    handles, lbls = axes[2].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, lbls):
        if l not in seen:
            seen[l] = h
    axes[2].legend(seen.values(), seen.keys(), fontsize=8)
    axes[2].set_aspect('equal'); axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Cassini ovals (foci at ±1)\nfor various b values', fontsize=10)
    axes[2].set_xlim(-2.5, 2.5); axes[2].set_ylim(-2.0, 2.0)

    fig.suptitle('Implicit Curves from Contours', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'contour_fun.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("contour_fun: done")
    return True


if __name__ == "__main__":
    run()
