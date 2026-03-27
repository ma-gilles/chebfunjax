"""Surfaces of revolution and arc-length integrals.

Demonstrates computing volume, surface area, centre of gravity, and moment
of inertia of solids of revolution using Chebfun integration.

Credit: Inspired by Chebfun example calc/SurfaceRevolution.m
(Georges Klein, March 2013).
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

from chebfunjax.plotting import plot


def run():
    print("=" * 60)
    print("Surfaces of revolution")
    print("=" * 60)

    pi = float(jnp.pi)

    # --- 1. Sphere: rotate sqrt(1 - x^2) about x-axis -----------------
    # Volume of unit sphere = (4/3)*pi
    f_sphere = cj.chebfun(lambda x: jnp.sqrt(jnp.maximum(1.0 - x**2, 0.0)))
    V_sphere = pi * float(f_sphere.inner(f_sphere))   # pi * int f^2 dx
    print(f"\nUnit sphere (rotate sqrt(1-x^2) about x-axis):")
    print(f"  V = pi * int_{{-1}}^1 f^2 dx = {V_sphere:.10f}")
    print(f"  Exact (4/3)*pi = {4.0/3.0*pi:.10f}")
    assert abs(V_sphere - 4.0/3.0*pi) < 1e-4, f"Sphere volume error: {V_sphere - 4.0/3.0*pi}"

    # --- 2. Cone: rotate (-3x + 3) about x-axis over [0, 1] ----------
    f_cone = cj.chebfun(lambda x: -3.0*x + 3.0, domain=(0.0, 1.0))
    V_cone = pi * float((f_cone * f_cone).sum())
    exact_cone = pi * 3.0  # pi * int_0^1 (3-3x)^2 dx = pi*9*[x - x^2 + x^3/3]_0^1 = pi*3
    print(f"\nCone (rotate -3x+3 over [0,1]):")
    print(f"  V = pi * int f^2 dx = {V_cone:.10f}")
    print(f"  Exact pi*3 = {exact_cone:.10f}")
    assert abs(V_cone - exact_cone) < 1e-8

    # --- 3. f(x) = sqrt(4 + 2*sin(2x)) on [0, 2*pi] ------------------
    dom = (0.0, 2.0 * pi)
    x_cf = cj.chebfun(lambda x: x, domain=dom)
    f = cj.chebfun(lambda x: jnp.sqrt(4.0 + 2.0 * jnp.sin(2.0 * x)), domain=dom)
    df = f.diff()

    # Volume: V = pi * int_a^b f^2(x) dx
    V = pi * float((f * f).sum())
    exact_V = 8.0 * pi**2   # Exact: pi * int_0^{2pi} (4 + 2*sin(2x)) dx = pi*8*pi
    print(f"\nf(x) = sqrt(4 + 2*sin(2x)) on [0, 2*pi]:")
    print(f"  Volume V = {V:.10f}")
    print(f"  Exact 8*pi^2 = {exact_V:.10f}")
    error_V = abs(V - exact_V)
    print(f"  Error = {error_V:.2e}")
    assert error_V < 1e-8, f"Volume error {error_V}"

    # Surface area: A = 2*pi * int_a^b f(x) * sqrt(1 + |f'(x)|^2) dx
    integrand_A = f * (cj.chebfun(lambda x: jnp.ones_like(x), domain=dom) +
                       df * df).sqrt()
    A = 2.0 * pi * float(integrand_A.sum())
    print(f"  Surface area A = {A:.10f}")
    assert 80 < A < 120, f"Surface area {A} out of expected range"

    # Centre of gravity: z_G = (pi/V) * int_a^b x * f^2(x) dx
    zG = (pi / V) * float((x_cf * f * f).sum())
    print(f"  Centre of gravity z_G = {zG:.6f}  (in [0, 2*pi])")
    # zG should lie inside the domain [0, 2*pi]
    assert 0.0 < zG < 2.0 * pi, f"Centre of gravity {zG} out of domain"

    # Moment of inertia: J = (pi/2) * int_a^b f^4(x) dx
    f2 = f * f
    f4 = f2 * f2
    J = (pi / 2.0) * float(f4.sum())
    print(f"  Moment of inertia J = {J:.10f}")
    assert J > 0

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    xs = np.linspace(0.0, 2.0 * pi, 400)
    f_vals = np.array(f(jnp.array(xs)))

    # Left: generator curve
    axes[0].plot(xs, f_vals, color="#1e77b4", linewidth=2)
    axes[0].axhline(0, color="k", linewidth=0.5)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].set_title(r"Generator curve $f(x)=\sqrt{4+2\sin 2x}$")
    axes[0].grid(True, alpha=0.4)

    # Right: 3D surface of revolution (top-down cross-section)
    theta = np.linspace(0, 2 * pi, 80)
    xs3d = np.linspace(0, 2 * pi, 80)
    XX, TT = np.meshgrid(xs3d, theta)
    R = np.array(f(jnp.array(xs3d)))  # radii
    RR = np.tile(R, (80, 1))
    XX_plot = XX
    YY_plot = RR * np.cos(TT)
    ZZ_plot = RR * np.sin(TT)
    ax3 = fig.add_subplot(122, projection='3d', position=[0.55, 0.05, 0.42, 0.9])
    ax3.plot_surface(XX_plot, YY_plot, ZZ_plot, alpha=0.6, cmap='viridis', linewidth=0)
    ax3.set_xlabel("x")
    ax3.set_title("Surface of revolution")
    ax3.tick_params(labelsize=7)
    axes[1].set_visible(False)

    fig.suptitle("Surface of revolution: volumes and areas via Chebfun", fontsize=11)
    fig.savefig(os.path.join(_here, "surface_revolution.png"), dpi=150, bbox_inches="tight")
    _docs = os.path.join(_here, "..", "..", "docs", "images", "calc")
    os.makedirs(_docs, exist_ok=True)
    fig.savefig(os.path.join(_docs, "surface_revolution.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
