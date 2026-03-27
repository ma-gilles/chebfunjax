"""Triple integrals in spherical, cylindrical and other coordinate systems.

Demonstrates how Chebfun3 handles changes of variables (spherical, cylindrical,
toroidal) to compute triple integrals over non-rectangular 3D domains using
the Jacobian determinant.

Original MATLAB Chebfun: approx3/ChangeVar3D.m by Rodrigo Platte, November 2016.
See https://www.chebfun.org/examples/approx3/ChangeVar3D.html
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

from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)), "docs", "images", "approx3"
)
os.makedirs(_IMG_DIR, exist_ok=True)


def jacobian_det(x: Chebfun3, y: Chebfun3, z: Chebfun3) -> Chebfun3:
    """Compute the absolute value of the Jacobian determinant.

    Given x(u,v,w), y(u,v,w), z(u,v,w) as Chebfun3 objects, approximate
    the Jacobian determinant via finite differences on the Chebyshev grid.

    We construct the Jacobian as a scalar Chebfun3 by sampling the partial
    derivatives numerically and interpolating the determinant.
    """
    xa, xb, ya, yb, za, zb = x.domain
    eps = 1e-6

    def _jac_func(u, v, w):
        # Evaluate x,y,z and their finite-difference partial derivatives
        xv = x(u, v, w)
        yv = y(u, v, w)
        zv = z(u, v, w)

        xu = (x(jnp.clip(u + eps, xa, xb), v, w) - x(jnp.clip(u - eps, xa, xb), v, w)) / (2 * eps)
        xv_ = (x(u, jnp.clip(v + eps, ya, yb), w) - x(u, jnp.clip(v - eps, ya, yb), w)) / (2 * eps)
        xw = (x(u, v, jnp.clip(w + eps, za, zb)) - x(u, v, jnp.clip(w - eps, za, zb))) / (2 * eps)

        yu = (y(jnp.clip(u + eps, xa, xb), v, w) - y(jnp.clip(u - eps, xa, xb), v, w)) / (2 * eps)
        yv_ = (y(u, jnp.clip(v + eps, ya, yb), w) - y(u, jnp.clip(v - eps, ya, yb), w)) / (2 * eps)
        yw = (y(u, v, jnp.clip(w + eps, za, zb)) - y(u, v, jnp.clip(w - eps, za, zb))) / (2 * eps)

        zu = (z(jnp.clip(u + eps, xa, xb), v, w) - z(jnp.clip(u - eps, xa, xb), v, w)) / (2 * eps)
        zv_ = (z(u, jnp.clip(v + eps, ya, yb), w) - z(u, jnp.clip(v - eps, ya, yb), w)) / (2 * eps)
        zw = (z(u, v, jnp.clip(w + eps, za, zb)) - z(u, v, jnp.clip(w - eps, za, zb))) / (2 * eps)

        det = (xu * (yv_ * zw - yw * zv_)
               - xv_ * (yu * zw - yw * zu)
               + xw * (yu * zv_ - yv_ * zu))
        return jnp.abs(det)

    return chebfun3(_jac_func, domain=x.domain)


def run():
    print("=" * 60)
    print("Triple integrals via coordinate changes (ChangeVar3D)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Section 1: Spherical coordinates — ice-cream cone
    # ------------------------------------------------------------------
    print("\n--- Spherical coordinates (ice-cream cone) ---")
    dom_sph = (0.0, 1.0, 0.0, 2 * np.pi, np.pi / 4, np.pi / 2)

    r_f = chebfun3(lambda r, t, p: r, domain=dom_sph)
    t_f = chebfun3(lambda r, t, p: t, domain=dom_sph)
    p_f = chebfun3(lambda r, t, p: p, domain=dom_sph)

    x_sph = chebfun3(lambda r, t, p: r * jnp.cos(t) * jnp.cos(p), domain=dom_sph)
    y_sph = chebfun3(lambda r, t, p: r * jnp.sin(t) * jnp.cos(p), domain=dom_sph)
    z_sph = chebfun3(lambda r, t, p: r * jnp.sin(p), domain=dom_sph)

    # density = sin(10t)*cos(10r) + 1; Jacobian = r^2 * cos(p)
    # For spherical: |J| = r^2 * cos(p)
    # mass = integral3( density * r^2 * cos(p) )
    # Exact for density=r^2: int_0^1 r^4 dr * int_0^{2pi} dt * int_{pi/4}^{pi/2} cos(p) dp
    #       = 1/5 * 2pi * (1 - 1/sqrt(2)) = pi*(2-sqrt(2))/5
    M_simple_func = chebfun3(
        lambda r, t, p: r**2 * r**2 * jnp.cos(p),
        domain=dom_sph
    )
    M_simple = float(M_simple_func.sum3())
    exact_simple = np.pi * (2 - np.sqrt(2)) / 5
    print(f"  integral of r^2 * |J_sph| = {M_simple:.10f}")
    print(f"  Exact:                      {exact_simple:.10f}")
    print(f"  Error: {abs(M_simple - exact_simple):.2e}")

    # ------------------------------------------------------------------
    # Section 2: Cylindrical coordinates — center of mass
    # ------------------------------------------------------------------
    print("\n--- Cylindrical coordinates (cylinder sector) ---")
    dom_cyl = (0.0, 1.0, 0.0, np.pi, 0.0, 1.0)
    r_c = chebfun3(lambda r, t, z: r, domain=dom_cyl)
    t_c = chebfun3(lambda r, t, z: t, domain=dom_cyl)
    z_c = chebfun3(lambda r, t, z: z, domain=dom_cyl)

    # density = y*sin(10t)+1, Jacobian = r
    # Mass: integral3(density * r)
    M_cyl = float(chebfun3(
        lambda r, t, z: (r * jnp.sin(t) * jnp.sin(10 * t) + 1) * r,
        domain=dom_cyl
    ).sum3())
    print(f"  Mass of cylinder sector: {M_cyl:.10f}")

    # Center of mass z-component (density=1): zc = int(z*r) / int(r) = (1/2)/(1) = 0.5
    z_cm_num = float(chebfun3(lambda r, t, z: z * r, domain=dom_cyl).sum3())
    r_vol = float(chebfun3(lambda r, t, z: r, domain=dom_cyl).sum3())
    zc = z_cm_num / r_vol
    print(f"  z-center of mass (uniform density): {zc:.10f}  (exact: 0.5)")
    assert abs(zc - 0.5) < 1e-8

    # ------------------------------------------------------------------
    # Section 3: Torus
    # ------------------------------------------------------------------
    print("\n--- Torus integral ---")
    dom_tor = (0.0, 1.0, 0.0, 2 * np.pi, 0.0, 2 * np.pi)
    # x = (4+r*cos(t))*cos(p), y = (4+r*cos(t))*sin(p), z = r*sin(t)
    # |J| = r*(4+r*cos(t))
    # Volume = integral3(|J|) = integral_{0}^{1} integral_{0}^{2pi} integral_{0}^{2pi}
    #          r*(4+r*cos(t)) dr dt dp
    # = 2pi * int_0^{2pi} int_0^1 r*(4+r*cos(t)) dr dt
    # = 2pi * [int_0^1 4r dr * 2pi + int_0^1 r^2 dr * int_0^{2pi} cos(t) dt]
    # = 2pi * [4pi + 0] = 8pi^2
    vol_torus = float(chebfun3(
        lambda r, t, p: r * (4 + r * jnp.cos(t)),
        domain=dom_tor
    ).sum3())
    exact_vol_torus = 8 * np.pi**2
    print(f"  Torus volume: {vol_torus:.10f}")
    print(f"  Exact:        {exact_vol_torus:.10f}")
    print(f"  Error: {abs(vol_torus - exact_vol_torus):.2e}")
    assert abs(vol_torus - exact_vol_torus) / exact_vol_torus < 1e-8

    # ------------------------------------------------------------------
    # Plot: surface of ice-cream cone and cylinder sector
    # ------------------------------------------------------------------
    fig = plt.figure()

    # Plot 1: ice-cream cone surface
    ax1 = fig.add_subplot(131, projection="3d")
    phi_vals = np.linspace(np.pi / 4, np.pi / 2, 30)
    theta_vals = np.linspace(0, 2 * np.pi, 60)
    Phi, Theta = np.meshgrid(phi_vals, theta_vals)
    X_c = np.cos(Theta) * np.cos(Phi)
    Y_c = np.sin(Theta) * np.cos(Phi)
    Z_c = np.sin(Phi)
    ax1.plot_surface(X_c, Y_c, Z_c, alpha=0.7, cmap="viridis")
    ax1.set_title("Ice-cream cone\n(r=1 surface)", fontsize=9)
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

    # Plot 2: cylinder sector
    ax2 = fig.add_subplot(132, projection="3d")
    r_vals = np.linspace(0, 1, 20)
    t_vals = np.linspace(0, np.pi, 40)
    R, T = np.meshgrid(r_vals, t_vals)
    Xcyl = R * np.cos(T)
    Ycyl = R * np.sin(T)
    Zcyl_bot = np.zeros_like(R)
    Zcyl_top = np.ones_like(R)
    ax2.plot_surface(Xcyl, Ycyl, Zcyl_bot, alpha=0.4, color="steelblue")
    ax2.plot_surface(Xcyl, Ycyl, Zcyl_top, alpha=0.4, color="steelblue")
    # Curved surface
    t2 = np.linspace(0, np.pi, 40)
    z2 = np.linspace(0, 1, 20)
    T2, Z2 = np.meshgrid(t2, z2)
    ax2.plot_surface(np.cos(T2), np.sin(T2), Z2, alpha=0.4, color="orange")
    ax2.set_title("Cylinder sector\n(r=1, 0≤θ≤π)", fontsize=9)
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")

    # Plot 3: torus
    ax3 = fig.add_subplot(133, projection="3d")
    p_vals = np.linspace(0, 2 * np.pi, 60)
    t_vals2 = np.linspace(0, 2 * np.pi, 30)
    P, T3 = np.meshgrid(p_vals, t_vals2)
    R0 = 4
    r0 = 1
    Xtor = (R0 + r0 * np.cos(T3)) * np.cos(P)
    Ytor = (R0 + r0 * np.cos(T3)) * np.sin(P)
    Ztor = r0 * np.sin(T3)
    ax3.plot_surface(Xtor, Ytor, Ztor, alpha=0.7, cmap="plasma")
    ax3.set_title("Torus (R=4, r=1)", fontsize=9)
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")

    fig.suptitle("Coordinate transformations for 3D integration", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG_DIR, "ChangeVar3D.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
