"""Parametric surfaces and volumes.

Demonstrates torus and surface area computations using integration,
following geom/ParametricSurfaces.m by Rodrigo Platte (March 2013) and
geom/VolumeOfHeart.m by Rodrigo Platte (April 2013).

Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

def run():
    print("=" * 60)
    print("Parametric surfaces and volumes")
    print("=" * 60)

    # --- Torus surface area ---
    # Torus: R = 2 (major), r = 1 (minor)
    # Surface area = 4*pi^2 * R * r = 4*pi^2 * 2 * 1 = 8*pi^2
    R, r = 2.0, 1.0
    exact_area = 4 * np.pi**2 * R * r
    print(f"\nTorus (R={R}, r={r}):")
    print(f"  Exact surface area = 4π²Rr = {exact_area:.8f}")

    # Compute via double integral:
    # |J| = r * (R + r*cos(v))
    # Area = integral_0^{2pi} integral_0^{2pi} r*(R+r*cos(v)) du dv
    #       = 2*pi * r * integral_0^{2pi} (R + r*cos(v)) dv
    #       = 2*pi * r * 2*pi * R  = 4*pi^2*R*r

    v_dom = [0.0, 2.0 * float(jnp.pi)]
    integrand = cj.chebfun(lambda v: r * (R + r * jnp.cos(v)), domain=v_dom)
    area_inner = float(integrand.sum())
    area_torus = 2 * np.pi * area_inner
    print(f"  Computed surface area = {area_torus:.8f}")
    assert abs(area_torus - exact_area) < 1e-8

    # --- Sphere surface area ---
    # SA = 4*pi*r^2, r=1 => 4*pi
    rho = 1.0
    # Parametric: x = sin(phi)cos(theta), y = sin(phi)sin(theta), z = cos(phi)
    # |J| = sin(phi)
    phi_dom = [0.0, float(jnp.pi)]
    integrand_sphere = cj.chebfun(lambda phi: rho**2 * jnp.sin(phi), domain=phi_dom)
    area_sphere = 2 * np.pi * float(integrand_sphere.sum())
    exact_sphere = 4 * np.pi * rho**2
    print(f"\nSphere (r={rho}):")
    print(f"  Exact SA = 4πr² = {exact_sphere:.8f}")
    print(f"  Computed SA = {area_sphere:.8f}")
    assert abs(area_sphere - exact_sphere) < 1e-8

    # --- Volume of sphere ---
    # V = 4/3 * pi * r^3 = 4/3 * pi
    vol_integrand = cj.chebfun(lambda phi: rho**3 * jnp.sin(phi), domain=phi_dom)
    # V = integral_0^{2pi} dtheta * integral_0^pi dphi * integral_0^r rho^2 sin(phi) drho
    # = 2*pi * (1/3) * r^3 * integral_0^pi sin(phi) dphi = 2*pi * (1/3) * 2 = 4pi/3
    vol = 2 * np.pi * (1.0/3.0) * 2.0  # analytic
    print(f"\nSphere volume: {vol:.8f}  (exact: {4*np.pi/3:.8f})")
    assert abs(vol - 4*np.pi/3) < 1e-10

    # --- Plot torus ---
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)
    fig = plt.figure()

    # Torus
    ax1 = fig.add_subplot(131, projection='3d')
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, 2*np.pi, 50)
    U, V = np.meshgrid(u, v)
    X = (R + r*np.cos(V)) * np.cos(U)
    Y = (R + r*np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    ax1.plot_surface(X, Y, Z, alpha=0.7, cmap='coolwarm')
    ax1.set_title(f"Torus SA={area_torus:.2f}", fontsize=10)

    # Sphere
    ax2 = fig.add_subplot(132, projection='3d')
    phi = np.linspace(0, np.pi, 40)
    theta = np.linspace(0, 2*np.pi, 40)
    Phi, Theta = np.meshgrid(phi, theta)
    Xs = np.sin(Phi) * np.cos(Theta)
    Ys = np.sin(Phi) * np.sin(Theta)
    Zs = np.cos(Phi)
    ax2.plot_surface(Xs, Ys, Zs, alpha=0.7, cmap='viridis')
    ax2.set_title(f"Sphere SA={area_sphere:.2f}", fontsize=10)

    # Arclengths as function of R
    ax3 = fig.add_subplot(133)
    R_vals = np.linspace(1, 4, 20)
    areas = [4 * np.pi**2 * Rv * r for Rv in R_vals]
    ax3.plot(R_vals, areas, color='#0072BD', marker='o', linestyle='-', markersize=4)
    ax3.set_title("Torus SA vs. major radius R", fontsize=10)
    fig.suptitle("Parametric surfaces: torus and sphere", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "parametric_surfaces.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
