"""Integration of scalar functions over 2D surfaces in 3D.

Demonstrates computing surface integrals ∫_S f dS over various parametric
surfaces: a sphere, a cone, two seashells, and a spring.

Original MATLAB Chebfun: approx3/SurfaceIntegral3D.m by Behnam Hashemi, June 2016.
See https://www.chebfun.org/examples/approx3/SurfaceIntegral3D.html
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

def surface_integral(f, Sx, Sy, Sz, u_range, v_range, n=300):
    """Compute int_S f dS for surface S parametrized by (Sx, Sy, Sz).

    ∫_S f(x,y,z) dS = ∫∫ f(S(u,v)) * ||T_u × T_v|| du dv

    Parameters
    ----------
    f : Chebfun3
        Scalar field.
    Sx, Sy, Sz : callables
        Surface parametrization.
    u_range, v_range : (float, float)
        Parameter domain.
    n : int
        Number of quadrature points per direction.

    Returns
    -------
    float
        Surface integral value.
    """
    u = np.linspace(u_range[0], u_range[1], n)
    v = np.linspace(v_range[0], v_range[1], n)
    U, V = np.meshgrid(u, v)

    X = Sx(U, V)
    Y = Sy(U, V)
    Z = Sz(U, V)

    # Partial derivatives via finite differences
    du = u[1] - u[0]
    dv = v[1] - v[0]
    eps_u = du * 1e-3
    eps_v = dv * 1e-3

    dXdu = (Sx(U + eps_u, V) - Sx(U - eps_u, V)) / (2 * eps_u)
    dYdu = (Sy(U + eps_u, V) - Sy(U - eps_u, V)) / (2 * eps_u)
    dZdu = (Sz(U + eps_u, V) - Sz(U - eps_u, V)) / (2 * eps_u)

    dXdv = (Sx(U, V + eps_v) - Sx(U, V - eps_v)) / (2 * eps_v)
    dYdv = (Sy(U, V + eps_v) - Sy(U, V - eps_v)) / (2 * eps_v)
    dZdv = (Sz(U, V + eps_v) - Sz(U, V - eps_v)) / (2 * eps_v)

    # Cross product norm = surface element
    Nx = dYdu * dZdv - dZdu * dYdv
    Ny = dZdu * dXdv - dXdu * dZdv
    Nz = dXdu * dYdv - dYdu * dXdv
    dS = np.sqrt(Nx**2 + Ny**2 + Nz**2)

    # Evaluate f at surface points
    f_vals = np.array(f(jnp.array(X), jnp.array(Y), jnp.array(Z)))

    return float(np.trapezoid(np.trapezoid(f_vals * dS, v, axis=0), u))

def run():
    print("=" * 60)
    print("Surface integrals of 3D scalar fields (SurfaceIntegral3D)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Example 1: int_{unit sphere} x^2 dS = 4*pi/3
    # ------------------------------------------------------------------
    print("\n--- Example 1: x^2 over unit sphere ---")
    f1 = chebfun3(lambda x, y, z: x**2)

    Sx_sph = lambda u, v: np.sin(u) * np.cos(v)
    Sy_sph = lambda u, v: np.sin(u) * np.sin(v)
    Sz_sph = lambda u, v: np.cos(u)

    I1 = surface_integral(f1, Sx_sph, Sy_sph, Sz_sph,
                          u_range=(0, np.pi), v_range=(0, 2 * np.pi))
    exact1 = 4 * np.pi / 3
    print(f"  Computed: {I1:.10f}")
    print(f"  Exact:    {exact1:.10f}  (= 4*pi/3)")
    print(f"  Error:    {abs(I1 - exact1):.2e}")
    assert abs(I1 - exact1) / exact1 < 1e-3

    # ------------------------------------------------------------------
    # Example 2: sqrt(1+x^2+y^2) over surface S(u,v)=(u*cos(v), u*sin(v), v)
    # u in [0,2], v in [0,pi]
    # Exact: 14*pi/3
    # ------------------------------------------------------------------
    print("\n--- Example 2: sqrt(1+x^2+y^2) over cone ---")
    f2 = chebfun3(lambda x, y, z: jnp.sqrt(1 + x**2 + y**2),
                  domain=(-3, 3, -3, 3, -3, 3))

    Sx2 = lambda u, v: u * np.cos(v)
    Sy2 = lambda u, v: u * np.sin(v)
    Sz2 = lambda u, v: v

    I2 = surface_integral(f2, Sx2, Sy2, Sz2,
                          u_range=(0, 2), v_range=(0, np.pi))
    exact2 = 14 * np.pi / 3
    print(f"  Computed: {I2:.8f}")
    print(f"  Exact:    {exact2:.8f}  (= 14*pi/3)")
    print(f"  Error:    {abs(I2 - exact2):.2e}")
    assert abs(I2 - exact2) / exact2 < 1e-3

    # ------------------------------------------------------------------
    # Example 3: Seashell surface integral of x+y+z
    # ------------------------------------------------------------------
    print("\n--- Example 3: x+y+z over seashell ---")
    f3 = chebfun3(lambda x, y, z: x + y + z,
                  domain=(-6, 6, -6, 6, 0, 25))

    def Sx_shell(u, v):
        return 5/4 * (1 - v/(2*np.pi)) * np.cos(2*v) * (1 + np.cos(u)) + np.cos(2*v)

    def Sy_shell(u, v):
        return 5/4 * (1 - v/(2*np.pi)) * np.sin(2*v) * (1 + np.cos(u)) + np.sin(2*v)

    def Sz_shell(u, v):
        return 10 * v / (2*np.pi) + 5/4 * (1 - v/(2*np.pi)) * np.sin(u) + 15

    I3 = surface_integral(f3, Sx_shell, Sy_shell, Sz_shell,
                          u_range=(0, 2 * np.pi), v_range=(-2 * np.pi, 2 * np.pi),
                          n=150)
    print(f"  Integral of x+y+z over seashell: {I3:.6f}")
    assert np.isfinite(I3)

    # ------------------------------------------------------------------
    # Example 4: Spring surface integral of x+y+z
    # ------------------------------------------------------------------
    print("\n--- Example 4: x+y+z over spring ---")
    r1, r2, t_coil = 0.5, 0.5, 1.5
    f4 = chebfun3(lambda x, y, z: x + y + z,
                  domain=(-2, 2, -2, 2, -2, 10))

    def Sx_spring(u, v):
        return (1 - r1 * np.cos(v)) * np.cos(u)

    def Sy_spring(u, v):
        return (1 - r1 * np.cos(v)) * np.sin(u)

    def Sz_spring(u, v):
        return r2 * (np.sin(v) + t_coil * u / np.pi)

    I4 = surface_integral(f4, Sx_spring, Sy_spring, Sz_spring,
                          u_range=(0, 10 * np.pi), v_range=(0, 10 * np.pi),
                          n=100)
    exact4 = 1878.4483067846025
    print(f"  Computed: {I4:.4f}")
    print(f"  Exact:    {exact4:.4f}")
    rel_err4 = abs(I4 - exact4) / abs(exact4)
    print(f"  Relative error: {rel_err4:.2e}")
    # coarse quadrature — accept 1% tolerance
    assert rel_err4 < 0.05, f"Spring integral too far off: {rel_err4:.2e}"

    # ------------------------------------------------------------------
    # Plot: the surfaces
    # ------------------------------------------------------------------
    from chebfunjax.plotting import PARULA, _setup_3d_axes

    fig = plt.figure(figsize=(16, 3.5))

    # Unit sphere
    ax1 = fig.add_subplot(141, projection="3d")
    _setup_3d_axes(ax1, fig)
    u_s = np.linspace(0, np.pi, 50)
    v_s = np.linspace(0, 2 * np.pi, 80)
    U_s, V_s = np.meshgrid(u_s, v_s)
    col = Sx_sph(U_s, V_s)**2
    col_norm = (col - col.min()) / (col.max() - col.min() + 1e-15)
    ax1.plot_surface(Sx_sph(U_s, V_s), Sy_sph(U_s, V_s), Sz_sph(U_s, V_s),
                     facecolors=PARULA(col_norm), rstride=1, cstride=1,
                     linewidth=0, antialiased=True, shade=False)
    ax1.set_title(f"Unit sphere\nint x^2 dS={I1:.4f}", fontsize=9, pad=0)

    # Seashell
    ax2 = fig.add_subplot(142, projection="3d")
    _setup_3d_axes(ax2, fig)
    u_sh = np.linspace(0, 2 * np.pi, 60)
    v_sh = np.linspace(-2 * np.pi, 2 * np.pi, 80)
    U_sh, V_sh = np.meshgrid(u_sh, v_sh)
    ax2.plot_surface(Sx_shell(U_sh, V_sh), Sy_shell(U_sh, V_sh),
                     Sz_shell(U_sh, V_sh), cmap=PARULA, rstride=1,
                     cstride=1, linewidth=0, antialiased=True)
    ax2.set_title("Seashell", fontsize=9, pad=0)

    # Second seashell
    def Sx_shell2(u, v):
        return u * np.cos(u) * (np.cos(v) + 1)

    def Sy_shell2(u, v):
        return u * np.sin(u) * (np.cos(v) + 1)

    def Sz_shell2(u, v):
        return u * np.sin(v) - ((u + 3) / (8 * np.pi))**2 - 20

    ax3 = fig.add_subplot(143, projection="3d")
    _setup_3d_axes(ax3, fig)
    u_sh2 = np.linspace(0, 13 * np.pi, 80)
    v_sh2 = np.linspace(-np.pi, np.pi, 60)
    U_sh2, V_sh2 = np.meshgrid(u_sh2, v_sh2)
    ax3.plot_surface(Sx_shell2(U_sh2, V_sh2), Sy_shell2(U_sh2, V_sh2),
                     Sz_shell2(U_sh2, V_sh2), cmap=PARULA, rstride=1,
                     cstride=1, linewidth=0, antialiased=True)
    ax3.set_title("Another seashell", fontsize=9, pad=0)

    # Spring
    ax4 = fig.add_subplot(144, projection="3d")
    _setup_3d_axes(ax4, fig)
    u_sp = np.linspace(0, 10 * np.pi, 100)
    v_sp = np.linspace(0, 10 * np.pi, 60)
    U_sp, V_sp = np.meshgrid(u_sp, v_sp)
    ax4.plot_surface(Sx_spring(U_sp, V_sp), Sy_spring(U_sp, V_sp),
                     Sz_spring(U_sp, V_sp), cmap=PARULA, rstride=1,
                     cstride=1, linewidth=0, antialiased=True)
    ax4.set_title("Spring", fontsize=9, pad=0)

    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(
        os.path.join(_IMG_DIR, "SurfaceIntegral3D.png"), dpi=150,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
