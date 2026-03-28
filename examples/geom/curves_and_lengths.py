"""Curves, arclengths, and geometric computations.

Demonstrates parametric curves, perimeter computations, and geometric properties
using chebfunjax, following geom/Ellipse.m by Hale & Trefethen (December 2010),
geom/Lissajous.m by Trefethen (October 2010), geom/RoseCurves.m by Hrothgar (June 2014),
and geom/Area.m by Stefan Guettel (October 2010).

Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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

def ellipse_perimeter(a, b, n=100000):
    """Compute ellipse perimeter numerically via high-resolution integration."""
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    dx = -a * np.sin(t)
    dy = b * np.cos(t)
    ds = np.sqrt(dx**2 + dy**2)
    return np.trapezoid(ds, t)

def run():
    print("=" * 60)
    print("Curves, arclengths, and geometric computations")
    print("=" * 60)

    # --- Ellipse perimeter ---
    a, b = 2.0, 1.0
    # Parametric: x(t) = a*cos(t), y(t) = b*sin(t), t in [0, 2*pi]
    # Arc length = integral sqrt((dx/dt)^2 + (dy/dt)^2) dt
    # dx/dt = -a*sin(t), dy/dt = b*cos(t)
    domain = [0.0, 2.0 * float(jnp.pi)]
    integrand = cj.chebfun(
        lambda t: jnp.sqrt((a * jnp.sin(t))**2 + (b * jnp.cos(t))**2),
        domain=domain
    )
    perimeter = float(integrand.sum())
    perimeter_ref = ellipse_perimeter(a, b)
    print(f"\nEllipse (a={a}, b={b}) perimeter:")
    print(f"  chebfunjax: {perimeter:.10f}")
    print(f"  trapz ref:  {perimeter_ref:.10f}")
    print(f"  Error:      {abs(perimeter - perimeter_ref):.2e}")
    assert abs(perimeter - perimeter_ref) < 1e-4

    # Circle: perimeter = 2*pi
    circ_int = cj.chebfun(lambda t: jnp.ones_like(t), domain=domain)
    circle_perim = float(circ_int.sum())
    print(f"\nCircle perimeter (a=b=1): {circle_perim:.10f}  (exact: {2*np.pi:.10f})")
    assert abs(circle_perim - 2*np.pi) < 1e-10

    # --- Lissajous curves ---
    # x = sin(at + delta), y = sin(bt)
    print("\nLissajous curves (a=3, b=2, delta=pi/2):")
    a_l, b_l = 3, 2
    delta = float(jnp.pi) / 2
    # Arclength of Lissajous
    t_dom = [0.0, 2.0 * float(jnp.pi)]
    ds_l = cj.chebfun(
        lambda t: jnp.sqrt((a_l * jnp.cos(a_l*t + delta))**2 + (b_l * jnp.cos(b_l*t))**2),
        domain=t_dom
    )
    length_l = float(ds_l.sum())
    print(f"  Arclength: {length_l:.6f}")
    assert length_l > 0

    # --- Rose curves ---
    # r = cos(n*theta), area = pi*n/4 for even n
    print("\nRose curve r = cos(2θ):")
    # Area = (1/2) * integral r^2 dθ = (1/2) * integral cos^2(2θ) dθ over [0, 2π]
    theta_dom = [0.0, 2.0 * float(jnp.pi)]
    r2_integrand = cj.chebfun(lambda t: 0.5 * jnp.cos(2*t)**2, domain=theta_dom)
    area_rose = float(r2_integrand.sum())
    print(f"  Area (rose r=cos(2θ)): {area_rose:.8f}  (exact: {np.pi/2:.8f})")
    assert abs(area_rose - np.pi/2) < 1e-10

    # --- Plot ---
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 3)

    t = np.linspace(0, 2*np.pi, 500)

    # Ellipse
    axes[0].plot(a * np.cos(t), b * np.sin(t), color='#0072BD', linestyle='-', linewidth=2)
    axes[0].set_aspect('equal')
    axes[0].set_title(f"Ellipse a={a}, b={b}\nPerimeter ≈ {perimeter:.4f}", fontsize=11)

    # Lissajous
    x_l = np.sin(a_l * t + delta)
    y_l = np.sin(b_l * t)
    axes[1].plot(x_l, y_l, color='#D95319', linestyle='-', linewidth=1.5)
    axes[1].set_aspect('equal')
    axes[1].set_title(f"Lissajous a={a_l}, b={b_l}\nLength ≈ {length_l:.4f}", fontsize=11)

    # Rose curve
    r_rose = np.cos(2 * t)
    x_rose = r_rose * np.cos(t)
    y_rose = r_rose * np.sin(t)
    axes[2].plot(x_rose, y_rose, color='#77AC30', linestyle='-', linewidth=2)
    axes[2].set_aspect('equal')
    axes[2].set_title(f"Rose r=cos(2θ)\nArea ≈ {area_rose:.4f}", fontsize=11)

    fig.suptitle("Geometric curves with chebfunjax", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "curves_and_lengths.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
