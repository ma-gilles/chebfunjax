"""Dual points, lines, polygons and curves.

Demonstrates the duality between points and lines in projective geometry,
and the dual of parametric curves. Translated from geom/DualCurves.m.

Original: https://www.chebfun.org/examples/geom/DualCurves.html
Author: Alex Townsend, August 2011
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



def dual_of_curve(f_re, f_im):
    """Compute the dual of a parametric curve (x(t), y(t)).

    Given curve C parameterized by (x(t), y(t)), the dual is the set of
    points dual to the tangent lines of C.
    Dual of line y = mx + c is point (m/c, -1/c).
    For parametric curve:
        p = -dy/dt / (dx/dt * y - x * dy/dt)
        q = dx/dt / (dy/dt * x - y * dx/dt)
    """
    df_re = np.gradient(f_re)
    df_im = np.gradient(f_im)

    denom_p = df_re * f_im - f_re * df_im
    denom_q = df_im * f_re - f_im * df_re

    # Avoid division by zero
    eps = 1e-10
    p = -df_im / (denom_p + eps)
    q = df_re / (denom_q + eps)

    return p + 1j * q


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- 1. Point and its dual line ---
    a, b = 1.0, np.pi / 3
    # Dual of point (a,b) is line y = -(a/b)*x - 1/b
    xs_line = np.linspace(-10, 10, 500)
    ys_dual_line = -(a / b) * xs_line - 1 / b

    axes[0].plot(a, b, 'xk', markersize=16, markeredgewidth=3, label=f'Point ({a:.2f},{b:.2f})')
    mask_y = np.abs(ys_dual_line) < 10
    axes[0].plot(xs_line[mask_y], ys_dual_line[mask_y], 'r-', linewidth=2, label='Dual line')
    axes[0].set_xlim(-10, 10); axes[0].set_ylim(-10, 10)
    axes[0].set_title('Point and its dual line', fontsize=11)
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    # --- 2. Heart curve and its dual ---
    t = np.linspace(0, 2 * np.pi, 2000)
    x_heart = 2 * np.sin(t)
    y_heart = (2 * np.cos(t) - 0.5 * np.cos(2 * t)
               - 0.25 * np.cos(3 * t) - 0.125 * np.cos(4 * t))

    dual = dual_of_curve(x_heart, y_heart)
    # Clip outliers
    max_r = 5
    mask_dual = np.abs(dual) < max_r

    axes[1].plot(x_heart, y_heart, 'b-', linewidth=2, label='Heart curve')
    axes[1].plot(np.real(dual[mask_dual]), np.imag(dual[mask_dual]),
                 'r-', linewidth=1.5, label='Dual curve')
    axes[1].set_aspect('equal')
    axes[1].set_title('Heart curve and its dual', fontsize=11)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-5, 5); axes[1].set_ylim(-5, 5)

    # --- 3. Ellipse and its dual ---
    a_e, b_e = 2.0, 1.0
    x_ell = a_e * np.cos(t)
    y_ell = b_e * np.sin(t)

    dual_ell = dual_of_curve(x_ell, y_ell)
    mask_ell = np.abs(dual_ell) < 3

    # Dual of ellipse x^2/a^2 + y^2/b^2 = 1 is ellipse X^2*a^2 + Y^2*b^2 = 1
    x_dual_exact = (1 / a_e**2) * np.cos(t)
    y_dual_exact = (1 / b_e**2) * np.sin(t)

    axes[2].plot(x_ell, y_ell, 'b-', linewidth=2, label=f'Ellipse (a={a_e},b={b_e})')
    axes[2].plot(np.real(dual_ell[mask_ell]), np.imag(dual_ell[mask_ell]),
                 'r-', linewidth=1.5, label='Dual (numerical)')
    axes[2].plot(x_dual_exact, y_dual_exact, 'g--', linewidth=2, label='Dual (exact)')
    axes[2].set_aspect('equal')
    axes[2].set_title('Ellipse and its dual', fontsize=11)
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(-3, 3); axes[2].set_ylim(-2, 2)

    fig.suptitle('Duality in Projective Geometry', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'dual_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("dual_curves: done")
    return True


if __name__ == "__main__":
    run()
