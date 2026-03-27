"""Integration of scalar functions over 3D curves.

Demonstrates computing the line integral of a Chebfun3 scalar field over a
3D parametric curve, with two examples: a sine-wave curve on the unit sphere
and a spherical helix (loxodrome).

Original MATLAB Chebfun: approx3/LineIntegral3D.m by Behnam Hashemi, June 2016.
See https://www.chebfun.org/examples/approx3/LineIntegral3D.html
Copyright 2016 by The University of Oxford and The Chebfun Developers.
"""

import matplotlib
matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt
from chebfunjax.plotting import chebfun_style
chebfun_style()

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)), "docs", "images", "approx3"
)
os.makedirs(_IMG_DIR, exist_ok=True)


def line_integral_3d(f, curve_x, curve_y, curve_z, t_range, n=5000):
    """Compute int_C f ds where C is a 3D parametric curve.

    Parameters
    ----------
    f : Chebfun3
        Scalar field.
    curve_x, curve_y, curve_z : callables
        Parametrization: (x(t), y(t), z(t)).
    t_range : (float, float)
        Parameter range.
    n : int
        Number of quadrature points.

    Returns
    -------
    float
        Line integral value.
    """
    t = np.linspace(t_range[0], t_range[1], n)
    dt = t[1] - t[0]

    x = curve_x(t)
    y = curve_y(t)
    z = curve_z(t)

    # Arc length element |ds/dt|
    dx_dt = np.gradient(x, t)
    dy_dt = np.gradient(y, t)
    dz_dt = np.gradient(z, t)
    ds_dt = np.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2)

    # Evaluate f along the curve
    f_vals = np.array(f(jnp.array(x), jnp.array(y), jnp.array(z)))

    return float(np.trapezoid(f_vals * ds_dt, t))


def run():
    print("=" * 60)
    print("Line integrals over 3D curves (LineIntegral3D)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Example 1: cos(x + yz) over a sine-wave curve on the unit sphere
    # C: x = cos(t)*sqrt(1 - r^2*cos^2(pt))
    #    y = sin(t)*sqrt(1 - r^2*cos^2(pt))
    #    z = r*cos(pt)
    # with p=10, q=1, r=0.3, t in [0, 2pi]
    # ------------------------------------------------------------------
    print("\n--- Example 1: cos(x+yz) over sine-wave curve on sphere ---")
    p, q, r = 10, 1, 0.3

    def Cx1(t):
        return np.cos(t) * np.sqrt(q**2 - r**2 * np.cos(p * t)**2)

    def Cy1(t):
        return np.sin(t) * np.sqrt(q**2 - r**2 * np.cos(p * t)**2)

    def Cz1(t):
        return r * np.cos(p * t)

    # Verify curve lies on unit sphere
    t_test = np.linspace(0, 2 * np.pi, 1000)
    sphere_check = Cx1(t_test)**2 + Cy1(t_test)**2 + Cz1(t_test)**2
    print(f"  Curve on unit sphere: max|x^2+y^2+z^2-1| = {np.max(np.abs(sphere_check - 1)):.2e}")
    assert np.max(np.abs(sphere_check - 1)) < 1e-10

    f1 = chebfun3(lambda x, y, z: jnp.cos(x + y * z))
    I1 = line_integral_3d(f1, Cx1, Cy1, Cz1, (0, 2 * np.pi))
    print(f"  I = integral of cos(x+yz) = {I1:.10f}")

    # Reference: compute same integral via direct substitution (higher n)
    I1_ref = line_integral_3d(f1, Cx1, Cy1, Cz1, (0, 2 * np.pi), n=20000)
    print(f"  Reference (n=20000):       {I1_ref:.10f}")
    print(f"  Difference: {abs(I1 - I1_ref):.2e}")
    assert abs(I1 - I1_ref) / (abs(I1_ref) + 1e-10) < 1e-3

    # ------------------------------------------------------------------
    # Example 2: x + yz over spherical helix (loxodrome)
    # C: x = sin(t/(2r))*cos(t)
    #    y = sin(t/(2r))*sin(t)
    #    z = cos(t/(2r))
    # r=5, t in [0, 10pi]
    # ------------------------------------------------------------------
    print("\n--- Example 2: x+yz over spherical helix ---")
    r2 = 5

    def Cx2(t):
        return np.sin(t / (2 * r2)) * np.cos(t)

    def Cy2(t):
        return np.sin(t / (2 * r2)) * np.sin(t)

    def Cz2(t):
        return np.cos(t / (2 * r2))

    # Verify curve lies on unit sphere
    sphere_check2 = Cx2(t_test)**2 + Cy2(t_test)**2 + Cz2(t_test)**2
    print(f"  Helix on unit sphere: max|r^2-1| = {np.max(np.abs(sphere_check2 - 1)):.2e}")
    assert np.max(np.abs(sphere_check2 - 1)) < 1e-10

    f2 = chebfun3(lambda x, y, z: x + y * z)
    I2 = line_integral_3d(f2, Cx2, Cy2, Cz2, (0, 10 * np.pi), n=10000)
    print(f"  I = integral of x+yz over helix = {I2:.10f}")

    # Reference
    I2_ref = line_integral_3d(f2, Cx2, Cy2, Cz2, (0, 10 * np.pi), n=50000)
    print(f"  Reference (n=50000):             {I2_ref:.10f}")
    print(f"  Difference: {abs(I2 - I2_ref):.2e}")
    assert abs(I2 - I2_ref) / (abs(I2_ref) + 1e-10) < 1e-2

    # ------------------------------------------------------------------
    # Exact check: f = 1 over a circle of radius 1 => integral = 2*pi
    # ------------------------------------------------------------------
    print("\n--- Exact check: constant 1 over unit circle ---")
    f_const = chebfun3(lambda x, y, z: jnp.ones_like(x + y + z))
    I_circle = line_integral_3d(
        f_const,
        lambda t: np.cos(t),
        lambda t: np.sin(t),
        lambda t: np.zeros_like(t),
        (0, 2 * np.pi), n=5000
    )
    print(f"  integral(1) over unit circle = {I_circle:.8f}  (exact: {2*np.pi:.8f})")
    assert abs(I_circle - 2 * np.pi) / (2 * np.pi) < 1e-4

    # ------------------------------------------------------------------
    # Plot: both curves and fields
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 5))

    # Plot 1: sine-wave curve on sphere
    ax1 = fig.add_subplot(131, projection="3d")
    t_plot = np.linspace(0, 2 * np.pi, 1000)
    xc = Cx1(t_plot); yc = Cy1(t_plot); zc = Cz1(t_plot)
    # Color by f value
    f_along = np.cos(xc + yc * zc)
    sc = ax1.scatter(xc, yc, zc, c=f_along, cmap="coolwarm", s=3)
    # Draw unit sphere (transparent)
    phi_s = np.linspace(0, np.pi, 30)
    theta_s = np.linspace(0, 2 * np.pi, 60)
    Phi_s, Th_s = np.meshgrid(phi_s, theta_s)
    ax1.plot_surface(np.sin(Phi_s) * np.cos(Th_s),
                     np.sin(Phi_s) * np.sin(Th_s),
                     np.cos(Phi_s), alpha=0.1, color="gray")
    fig.colorbar(sc, ax=ax1, shrink=0.6, label="cos(x+yz)")
    ax1.set_title(f"Sine-wave on sphere\nI={I1:.4f}", fontsize=10)
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

    # Plot 2: spherical helix
    ax2 = fig.add_subplot(132, projection="3d")
    t_plot2 = np.linspace(0, 10 * np.pi, 3000)
    xh = Cx2(t_plot2); yh = Cy2(t_plot2); zh = Cz2(t_plot2)
    f_helix = xh + yh * zh
    sc2 = ax2.scatter(xh, yh, zh, c=f_helix, cmap="plasma", s=2)
    ax2.plot_surface(np.sin(Phi_s) * np.cos(Th_s),
                     np.sin(Phi_s) * np.sin(Th_s),
                     np.cos(Phi_s), alpha=0.1, color="gray")
    fig.colorbar(sc2, ax=ax2, shrink=0.6, label="x+yz")
    ax2.set_title(f"Spherical helix\nI={I2:.4f}", fontsize=10)
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")

    # Plot 3: f = cos(x+yz) as slice at z=0
    ax3 = fig.add_subplot(133)
    x_p = np.linspace(-1, 1, 80)
    y_p = np.linspace(-1, 1, 80)
    Xp, Yp = np.meshgrid(x_p, y_p)
    f_slice = np.cos(Xp + Yp * 0)  # z=0
    im = ax3.contourf(Xp, Yp, f_slice, levels=20, cmap="coolwarm")
    ax3.set_title("f = cos(x+y·z) at z=0", fontsize=10)
    ax3.set_xlabel("x"); ax3.set_ylabel("y")
    fig.colorbar(im, ax=ax3)

    fig.suptitle("Line integrals of 3D scalar fields over curves", fontsize=12)
    fig.tight_layout()
    fig.savefig(
        os.path.join(_IMG_DIR, "LineIntegral3D.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
