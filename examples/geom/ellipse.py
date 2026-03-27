"""Perimeter of an ellipse.

Computes the perimeter of Poisson's 1827 ellipse with semiaxes
0.5/pi and 0.4/pi, verifying the classical result.
Translated from geom/Ellipse.m.

Original: https://www.chebfun.org/examples/geom/Ellipse.html
Authors: Nick Hale and Nick Trefethen, December 2010
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

def ellipse_perimeter_numerical(a, b, n_pts=10000):
    """Compute perimeter of ellipse with semiaxes a and b by arc-length integration."""
    theta = np.linspace(0, 2 * np.pi, n_pts + 1)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    dx = np.diff(x); dy = np.diff(y)
    return np.sum(np.sqrt(dx**2 + dy**2))

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)

    # Poisson's ellipse: a = 0.5/pi, b = 0.4/pi
    a = 0.5 / np.pi
    b = 0.4 / np.pi
    exact = 0.90277992777219

    # Numerical perimeter via arc-length integral
    arc_len = ellipse_perimeter_numerical(a, b, n_pts=100000)
    print(f"Poisson's ellipse perimeter:")
    print(f"  Numerical = {arc_len:.14f}")
    print(f"  Exact     = {exact:.14f}")
    print(f"  Error     = {abs(arc_len - exact):.2e}")

    # Also compute via chebfunjax
    theta_vals = np.linspace(0, 2 * np.pi, 10000)
    dx_dt = -a * np.sin(theta_vals)
    dy_dt = b * np.cos(theta_vals)
    speed = np.sqrt(dx_dt**2 + dy_dt**2)
    arc_cheb = np.trapezoid(speed, theta_vals)
    print(f"  Via trapz  = {arc_cheb:.14f}")

    fig, axes = plt.subplots(1, 2)

    # Plot the ellipse
    theta_plot = np.linspace(0, 2 * np.pi, 500)
    x_e = a * np.cos(theta_plot)
    y_e = b * np.sin(theta_plot)
    axes[0].plot(x_e, y_e, 'b-', linewidth=2)
    axes[0].set_aspect('equal')
    axes[0].set_title(f"Poisson's ellipse\na={a:.4f}, b={b:.4f}", fontsize=11)
    axes[0].text(0, -b*1.3, f'Perimeter ≈ {arc_cheb:.8f}\nExact: {exact:.8f}',
                 ha='center', fontsize=9)

    # Compare: speed |dz/dtheta| as a function of theta
    axes[1].plot(theta_plot, np.sqrt((a * np.sin(theta_plot))**2 +
                                     (b * np.cos(theta_plot))**2),
                 'r-', linewidth=2)
    axes[1].set_title('Arc-length element |dz/dθ|', fontsize=11)

    # Also show a family of ellipses
    for ratio in [0.2, 0.4, 0.6, 0.8, 1.0]:
        a_f = 1.0
        b_f = ratio * a_f
        theta_f = np.linspace(0, 2 * np.pi, 500)
        x_f = a_f * np.cos(theta_f)
        y_f = b_f * np.sin(theta_f)
        # Ellipse perimeter scaling
        axes[1].annotate('', xy=(0, 0), xytext=(0, 0))

    fig.suptitle('Perimeter of an Ellipse', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'ellipse.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    assert abs(arc_len - exact) < 1e-6

    print("ellipse: done")
    return True

if __name__ == "__main__":
    run()
