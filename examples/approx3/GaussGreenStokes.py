"""The theorems of Gauss, Green, and Stokes verified with Chebfun3.

Uses chebfunjax to verify the divergence theorem (Gauss), Green's identities,
and Stokes' theorem numerically for concrete functions on the unit cube.

Original MATLAB Chebfun: approx3/GaussGreenStokes.m by Olivier Sète, June 2016.
See https://www.chebfun.org/examples/approx3/GaussGreenStokes.html
Copyright 2016 by The University of Oxford and The Chebfun Developers.
"""

import matplotlib
matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)), "docs", "images", "approx3"
)
os.makedirs(_IMG_DIR, exist_ok=True)


def face_integral_2d(g_func, u_range, v_range, n=200):
    """Compute double integral of g over a rectangular face."""
    u = np.linspace(u_range[0], u_range[1], n)
    v = np.linspace(v_range[0], v_range[1], n)
    U, V = np.meshgrid(u, v)
    vals = g_func(U, V)
    return float(np.trapezoid(np.trapezoid(vals, v, axis=0), u))


def run():
    print("=" * 60)
    print("Gauss, Green, and Stokes theorems (GaussGreenStokes)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Gauss's (Divergence) Theorem
    # v = (x^2 - y, y^2, z) on K = [-1,1]^3
    # div(v) = 2x + 2y + 1
    # int_K div(v) dV = 0 + 0 + 8 = 8   (exact)
    # ------------------------------------------------------------------
    print("\n--- 1. Gauss's Theorem ---")
    # div(v) = 2x + 2y + 1
    div_v = chebfun3(lambda x, y, z: 2 * x + 2 * y + 1)
    I1 = float(div_v.sum3())
    print(f"  int div(v) over [-1,1]^3 = {I1:.10f}  (exact: 8)")
    assert abs(I1 - 8.0) < 1e-8, f"Divergence theorem failed: {I1}"

    # Flux through boundary of cube: 6 faces
    # v1 = x^2 - y, v2 = y^2, v3 = z
    # Face x=+1: int_{-1}^1 int_{-1}^1 v1(1,y,z) dy dz = int (1-y) dy dz
    #          = (1*2*2 - 0) = 4 (since int_{-1}^1 y dy = 0)
    # Face x=-1: int v1(-1,y,z)(-1) dy dz = -int (1-y)(-1) dy dz = -(-4) = ... wait
    # Actually outward normal on x=+1 is +x, on x=-1 is -x
    # Flux on x=+1: int_{-1}^1^2 v1(+1,y,z) dydz = int (1-y) dydz = 1*4 - 0 = 4
    # Flux on x=-1: int_{-1}^1^2 (-1)*v1(-1,y,z) dydz = -int (1-y) dydz = -4
    # Wait: v1(-1,y,z) = (-1)^2 - y = 1-y, same as v1(1,y,z)=1-y. But outward normal is -x.
    # Flux on x=-1 face: int v1*(-1) = -(1-y) integrated = -4
    # Actually, MATLAB sums: sum2(v1(1,:,:)) - sum2(v1(-1,:,:))
    # = int (1-y)dydz - int (1-y)dydz = 0 ... hmm

    # Let me use direct computation
    # v1(1,y,z) = 1-y, integrated = 4
    # v1(-1,y,z) = 1-y, integrated = 4, outward normal is (-1), so contribution = -4
    # For MATLAB: I2 = sum2(v1(1,:,:)) - sum2(v1(-1,:,:)) = 4 - 4 = 0
    # v2(x,1,z) = 1, int = 4; v2(x,-1,z) = 1, outward normal (-1), contribution = -4... no
    # MATLAB: sum2(v2(:,1,:)) - sum2(v2(:,-1,:)) = 4 - 4 = 0? But exact is 8?
    # Wait, v2 = y^2. v2(x,1,z) = 1, v2(x,-1,z) = 1. sum2(v2(:,1,:))=4, sum2(v2(:,-1,:))=4
    # Contribution = 4 - 4 = 0? But int_{-1}^1 2y dy = 0 not 8.
    # Oh wait: int div(v) = int (2x + 2y + 1) = 0 + 0 + 8 = 8 ✓
    # Surface integral: I need contributions from all 6 faces
    # x-faces: v1(1,y,z) - v1(-1,y,z) = (1-y)-(1-y) = 0 -> wait no
    # Outward flux on x=+1 face: int v1(1,y,z) dydz = int (1-y) dydz = 4
    # Outward flux on x=-1 face: -int v1(-1,y,z) dydz = -int (1-y) dydz = -4
    # So x-face total = 4 - 4 = 0  ...
    # y-faces: v2(x,1,z) - v2(x,-1,z):
    #   outward on y=+1: int v2(x,1,z) dxdz = int 1 dxdz = 4
    #   outward on y=-1: -int v2(x,-1,z) dxdz = -int 1 dxdz = -4
    #   total = 0
    # z-faces: v3(x,y,1) - v3(x,y,-1):
    #   outward on z=+1: int v3(x,y,1) dxdy = int 1 dxdy = 4
    #   outward on z=-1: -int v3(x,y,-1) dxdy = -int (-1) dxdy = 4
    #   total = 8
    # Sum = 0 + 0 + 8 = 8 ✓

    # Face integrals via trapz
    v1 = lambda x, y, z: x**2 - y
    v2 = lambda x, y, z: y**2
    v3 = lambda x, y, z: z

    # x=+1 face
    I_xp = face_integral_2d(lambda y, z: v1(np.ones_like(y), y, z),
                             (-1, 1), (-1, 1))
    # x=-1 face (outward normal is -x, so subtract)
    I_xm = face_integral_2d(lambda y, z: v1(-np.ones_like(y), y, z),
                             (-1, 1), (-1, 1))
    # y=+1 face
    I_yp = face_integral_2d(lambda x, z: v2(x, np.ones_like(x), z),
                             (-1, 1), (-1, 1))
    # y=-1 face
    I_ym = face_integral_2d(lambda x, z: v2(x, -np.ones_like(x), z),
                             (-1, 1), (-1, 1))
    # z=+1 face
    I_zp = face_integral_2d(lambda x, y: v3(x, y, np.ones_like(x)),
                             (-1, 1), (-1, 1))
    # z=-1 face
    I_zm = face_integral_2d(lambda x, y: v3(x, y, -np.ones_like(x)),
                             (-1, 1), (-1, 1))

    I2 = (I_xp - I_xm) + (I_yp - I_ym) + (I_zp - I_zm)
    print(f"  Boundary flux integral     = {I2:.10f}  (exact: 8)")
    print(f"  Divergence theorem: |I1-I2| = {abs(I1 - I2):.2e}")
    assert abs(I1 - I2) < 1e-3, f"Surface integral mismatch: {I2}"

    # ------------------------------------------------------------------
    # 2. Green's First Identity
    # f = 1 + x*exp(y+z), g = x^2 + y^2 + z^2
    # lap(g) = 6; grad(f) dot grad(g) = 2x*exp(y+z) + (1+x*exp(y+z))*(2y+2z)
    # int_K (f*lap(g) + grad(f).grad(g)) dV should equal boundary flux
    # Exact value: 48 (given in the MATLAB example)
    # ------------------------------------------------------------------
    print("\n--- 2. Green's First Identity ---")
    # f * lap(g) = (1 + x*exp(y+z)) * 6
    f_lap_g = chebfun3(lambda x, y, z: (1 + x * jnp.exp(y + z)) * 6)
    # grad(f) = (exp(y+z), x*exp(y+z), x*exp(y+z))
    # grad(g) = (2x, 2y, 2z)
    # dot product = 2x*exp(y+z) + 2y*x*exp(y+z) + 2z*x*exp(y+z)
    gradf_gradg = chebfun3(
        lambda x, y, z: (
            2 * x * jnp.exp(y + z)
            + 2 * y * x * jnp.exp(y + z)
            + 2 * z * x * jnp.exp(y + z)
        )
    )
    I3 = float(f_lap_g.sum3()) + float(gradf_gradg.sum3())
    print(f"  int (f*lap(g) + grad(f)·grad(g)) = {I3:.6f}  (exact: 48)")
    assert abs(I3 - 48.0) < 1e-4, f"Green's first identity: {I3}"

    # ------------------------------------------------------------------
    # 3. Stokes' Theorem
    # v = (x^2-y, y^2, z) on unit disk in z=0 plane
    # curl(v) = (0-0, 0-0, 0-(-1)) = (0, 0, 1)
    # Flux of curl(v) through disk = area = pi
    # Line integral around boundary circle: int_{circle} v.ds
    #   parametrize: (cos t, sin t, 0), ds = (-sin t, cos t, 0) dt
    #   v.ds = (cos^2 t - sin t)(-sin t) + sin^2 t * cos t
    #        = -cos^2 t sin t + sin^2 t + sin^2 t cos t
    #   int_0^{2pi} = 0 + pi + 0 = pi  ✓
    # ------------------------------------------------------------------
    print("\n--- 3. Stokes' Theorem ---")
    # curl(v) = (dv3/dy - dv2/dz, dv1/dz - dv3/dx, dv2/dx - dv1/dy)
    # v1=x^2-y, v2=y^2, v3=z
    # curl_z = d(y^2)/dx - d(x^2-y)/dy = 0 - (-1) = 1
    # Flux of curl through unit disk (z=0):
    # int_{disk} (0,0,1).n dS where n=(0,0,1) => int_{disk} 1 dA = pi

    # Line integral: gamma(t) = (cos t, sin t, 0), t in [0, 2pi]
    # v(cos t, sin t, 0) = (cos^2 t - sin t, sin^2 t, 0)
    # ds/dt = (-sin t, cos t, 0)
    # integrand = (cos^2 t - sin t)(-sin t) + sin^2 t * cos t
    t_line = np.linspace(0, 2 * np.pi, 10000)
    integrand_stokes = (
        (np.cos(t_line)**2 - np.sin(t_line)) * (-np.sin(t_line))
        + np.sin(t_line)**2 * np.cos(t_line)
    )
    I8 = float(np.trapezoid(integrand_stokes, t_line))
    exact_stokes = np.pi
    print(f"  Line integral (Stokes boundary) = {I8:.8f}")
    print(f"  Exact (= pi):                    {exact_stokes:.8f}")
    print(f"  Error: {abs(I8 - exact_stokes):.2e}")
    assert abs(I8 - exact_stokes) / exact_stokes < 1e-5

    # Verify flux of curl = pi via numeric integration over disk
    # curl_z = 1 everywhere, disk area = pi
    r_d = np.linspace(0, 1, 200)
    t_d = np.linspace(0, 2 * np.pi, 400)
    R_d, T_d = np.meshgrid(r_d, t_d)
    I7 = float(np.trapezoid(np.trapezoid(R_d, r_d, axis=1), t_d))  # = pi
    print(f"  Flux of curl(v) through disk    = {I7:.8f}  (exact: pi)")
    assert abs(I7 - np.pi) / np.pi < 1e-4

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Gauss theorem: div v on a slice
    ax1 = axes[0]
    xs = np.linspace(-1, 1, 80)
    ys = np.linspace(-1, 1, 80)
    X2, Y2 = np.meshgrid(xs, ys)
    div_slice = 2 * X2 + 2 * Y2 + 1  # div v at z=0
    im1 = ax1.contourf(X2, Y2, div_slice, levels=20, cmap="RdBu_r")
    ax1.set_title("div(v) = 2x+2y+1 at z=0\nGauss: ∫div = 8", fontsize=10)
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    fig.colorbar(im1, ax=ax1)

    # Green identity: f(x,y,z) at z=0
    ax2 = axes[1]
    f_slice = 1 + X2 * np.exp(Y2 + 0)
    im2 = ax2.contourf(X2, Y2, f_slice, levels=20, cmap="viridis")
    ax2.set_title("f = 1 + x·exp(y+z) at z=0\nGreen: ∫(f·Δg+∇f·∇g)=48", fontsize=10)
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    fig.colorbar(im2, ax=ax2)

    # Stokes: unit disk with boundary circle
    ax3 = axes[2]
    t_c = np.linspace(0, 2 * np.pi, 200)
    r_disk = np.linspace(0, 1, 50)
    theta_disk = np.linspace(0, 2 * np.pi, 200)
    R_disk, T_disk = np.meshgrid(r_disk, theta_disk)
    # Color by v1 = x^2 - y
    Xd = R_disk * np.cos(T_disk)
    Yd = R_disk * np.sin(T_disk)
    v1_disk = Xd**2 - Yd
    im3 = ax3.contourf(Xd, Yd, v1_disk, levels=20, cmap="plasma")
    ax3.plot(np.cos(t_c), np.sin(t_c), "w-", lw=2, label="boundary")
    ax3.set_title(f"Stokes: unit disk (z=0)\nLine integral ≈ {I8:.4f} ≈ π", fontsize=10)
    ax3.set_xlabel("x"); ax3.set_ylabel("y")
    ax3.legend(loc="upper right")
    fig.colorbar(im3, ax=ax3)

    fig.suptitle("Gauss, Green, and Stokes Theorems", fontsize=13)
    fig.tight_layout()
    fig.savefig(
        os.path.join(_IMG_DIR, "GaussGreenStokes.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
