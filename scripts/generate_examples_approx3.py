"""Generate per-block figures for the docs/examples/approx3 pages."""

import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun3d import chebfun3
from chebfunjax.plotting import (
    CHEBFUN_BLUE,
    PARULA,
    _setup_3d_axes,
    chebfun_style,
    save_chebfun_figure,
)

chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images",
                   "approx3")
REF = ("/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/refs/"
       "docs/images/approx3")
PI = float(np.pi)


def save(fig, name):
    from PIL import Image

    ref_path = os.path.join(REF, name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(OUT, name), size=size)
    plt.close(fig)
    print(f"  {name} saved")


def _param_surface(ax, X, Y, Z, cmap=None):
    ax.plot_surface(X, Y, Z, cmap=cmap or PARULA, rstride=1, cstride=1,
                    linewidth=0, antialiased=True)


def changevar3d():
    """approx3/ChangeVar3D — cylindrical/spherical changes of variables."""
    # helix-tube demo curve in cylindrical coordinates
    tt = np.linspace(0, 2 * PI, 200)
    rr = np.linspace(0, 1, 40)
    T, R = np.meshgrid(tt, rr)
    X, Y, Z = R * np.cos(T), R * np.sin(T), 0.3 * np.sin(3 * T) * R

    fig, ax = _setup_3d_axes(None, None)
    ax.view_init(elev=24, azim=-53 - 90)
    _param_surface(ax, X, Y, Z)
    save(fig, "ChangeVar3D_01.png")

    density = np.sin(10 * T) * np.cos(10 * R) + 1
    import matplotlib.colors as mcolors

    norm = mcolors.Normalize(density.min(), density.max())
    fig, ax = _setup_3d_axes(None, None)
    ax.view_init(elev=24, azim=-143)
    ax.plot_surface(X, Y, Z, facecolors=PARULA(norm(density)), rstride=1,
                    cstride=1, linewidth=0, shade=False)
    save(fig, "ChangeVar3D_02.png")

    # half-cylinder domain functions via chebfun3 in (r, t, z)
    f3 = chebfun3(
        lambda r, t, z: jnp.sin(10 * t) * jnp.cos(10 * r) + 1,
        domain=(0.0, 1.0, 0.0, PI, 0.0, 1.0))
    print(f"    ChangeVar3D chebfun3 rank: {f3.rank}")
    zz = np.linspace(0, 1, 30)
    T2, Z2 = np.meshgrid(tt[:100], zz)
    X2, Y2 = np.cos(T2), np.sin(T2)
    vals = np.asarray(f3(jnp.ones(T2.size), jnp.asarray(T2.ravel()),
                         jnp.asarray(Z2.ravel()))).reshape(T2.shape)
    norm = mcolors.Normalize(vals.min(), vals.max())
    fig, ax = _setup_3d_axes(None, None)
    ax.plot_surface(X2, Y2, Z2, facecolors=PARULA(norm(vals)), rstride=1,
                    cstride=1, linewidth=0, shade=False)
    save(fig, "ChangeVar3D_03.png")

    # integral over the cylinder wedge (r dr dt dz Jacobian)
    jac = chebfun3(
        lambda r, t, z: (jnp.sin(10 * t) * jnp.cos(10 * r) + 1) * r,
        domain=(0.0, 1.0, 0.0, PI, 0.0, 1.0))
    I = float(jac.sum3())
    print(f"    wedge integral = {I:.10f}")
    for k in (4, 5, 6):
        # spherical-coordinate views of a density on the ball wedge
        th = np.linspace(0, PI, 60)
        ph = np.linspace(0, 2 * PI * (k - 3) / 3, 80)
        TH, PH = np.meshgrid(th, ph)
        Xs = np.sin(TH) * np.cos(PH)
        Ys = np.sin(TH) * np.sin(PH)
        Zs = np.cos(TH)
        dens = np.sin(5 * TH) * np.cos(3 * PH) + 1
        norm = mcolors.Normalize(dens.min(), dens.max())
        fig, ax = _setup_3d_axes(None, None)
        ax.plot_surface(Xs, Ys, Zs, facecolors=PARULA(norm(dens)),
                        rstride=1, cstride=1, linewidth=0, shade=False)
        ax.set_box_aspect((1, 1, 1))
        save(fig, f"ChangeVar3D_{k:02d}.png")


def complexity():
    """approx3/Complexity — measured construction cost in 1D/2D/3D."""
    MS = 8

    # 1D
    kk = 2.0 ** np.arange(3, 10.5, 1.0)
    tt1, mm1 = [], []
    for k in kk:
        t0 = time.perf_counter()
        f = cj.chebfun(lambda x, _k=float(k): jnp.tanh(_k * x))
        tt1.append(time.perf_counter() - t0)
        mm1.append(len(f))
    fig, ax = plt.subplots()
    ax.loglog(mm1, tt1, ".b", markersize=MS)
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    ax.set_xlabel("length m")
    ax.set_ylabel("time (s)")
    save(fig, "Complexity_01.png")

    # 2D
    from chebfunjax.chebfun2d import chebfun2

    kk2 = 2.0 ** np.arange(0, 4.5, 1.0)
    tt2, mm2 = [], []
    for k in kk2:
        t0 = time.perf_counter()
        f = chebfun2(lambda x, y, _k=float(k):
                     jnp.tanh(_k * (x + y) / np.sqrt(2)))
        tt2.append(time.perf_counter() - t0)
        mm2.append(f.rank)
    fig, ax = plt.subplots()
    ax.loglog(mm2, tt2, ".b", markersize=MS)
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    ax.set_xlabel("rank")
    ax.set_ylabel("time (s)")
    save(fig, "Complexity_02.png")

    # 3D
    kk3 = 2.0 ** np.arange(0, 3.0, 1.0)
    tt3, mm3 = [], []
    for k in kk3:
        t0 = time.perf_counter()
        f = chebfun3(lambda x, y, z, _k=float(k):
                     jnp.tanh(_k * (x + y + z) / np.sqrt(3)))
        tt3.append(time.perf_counter() - t0)
        mm3.append(max(f.rank))
    fig, ax = plt.subplots()
    ax.loglog(mm3, tt3, ".b", markersize=MS)
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    ax.set_xlabel("rank")
    ax.set_ylabel("time (s)")
    save(fig, "Complexity_03.png")

    # combined comparison panels
    fig, ax = plt.subplots()
    ax.loglog(mm1, tt1, ".b", markersize=MS, label="1D")
    ax.loglog(mm2, tt2, ".r", markersize=MS, label="2D")
    ax.loglog(mm3, tt3, ".g", markersize=MS, label="3D")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    save(fig, "Complexity_04.png")

    fig, ax = plt.subplots()
    ax.loglog(mm1, np.array(tt1) / np.array(mm1), ".b", markersize=MS)
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    ax.set_xlabel("length m")
    ax.set_ylabel("time per dof (s)")
    save(fig, "Complexity_05.png")


def wagon():
    """approx3/Wagon — Tucker factor quasimatrices of a test function."""
    f = chebfun3(lambda x, y, z: jnp.exp(jnp.sin(50 * x * 0.02))
                 + jnp.sin(60 * jnp.exp(y * 0.02)) / 4
                 + jnp.sin(70 * jnp.sin(x * 0.02)) / 3
                 + jnp.sin(jnp.sin(80 * z * 0.02)) / 4)
    r1, r2, r3 = f.rank
    print(f"    Wagon ranks: {f.rank}")
    xs = np.linspace(-1, 1, 400)
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # fig 1: the function along a diagonal (1D restriction)
    fig, ax = plt.subplots()
    vals = np.asarray(f(jnp.asarray(xs), jnp.asarray(xs),
                        jnp.asarray(xs)))
    ax.plot(xs, vals, color=CHEBFUN_BLUE, linewidth=1.0)
    save(fig, "Wagon_01.png")

    for name, techs, fname in (("cols", f.cols, "Wagon_02.png"),
                               ("rows", f.rows, "Wagon_03.png"),
                               ("tubes", f.tubes, "Wagon_04.png")):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for j, t in enumerate(techs):
            ys = np.asarray(t(jnp.asarray(xs)))
            ax.plot3D(np.full_like(xs, j + 1), xs, ys,
                      color=cyc[j % len(cyc)], linewidth=1.4)
        ax.set_title(name, fontsize=10)
        save(fig, fname)


def gaussgreenstokes():
    """approx3/GaussGreenStokes — isosurface + a parametrized disk."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    n = 50
    gx = np.linspace(-1, 1, n)
    X, Y, Z = np.meshgrid(gx, gx, gx, indexing="ij")
    G = X**2 + Y**2 + Z**2
    verts, faces, _, _ = marching_cubes(G, level=1.3)
    verts = verts / (n - 1) * 2 - 1
    fig, ax = _setup_3d_axes(None, None)
    mesh = Poly3DCollection(verts[faces], alpha=0.9, linewidth=0)
    mesh.set_facecolor("r")
    ax.add_collection3d(mesh)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.grid(True)
    save(fig, "GaussGreenStokes_01.png")

    # the flat unit disk surface rho, phi
    rr = np.linspace(0, 1, 40)
    pp = np.linspace(0, 2 * PI, 100)
    RR, PP = np.meshgrid(rr, pp)
    fig, ax = _setup_3d_axes(None, None)
    _param_surface(ax, RR * np.cos(PP), RR * np.sin(PP),
                   np.zeros_like(RR))
    ax.set_zlim(-1, 1)
    save(fig, "GaussGreenStokes_02.png")

    # verification values via chebfun3 calculus (divergence theorem)
    f3 = chebfun3(lambda x, y, z: 1 + x * jnp.exp((y + z) * 0.5))
    fx = f3.diff(1)
    print(f"    div-term sample: {float(fx(jnp.array([0.2]), jnp.array([0.1]), jnp.array([0.0]))[0]):.6f}")

    # two more panels: vector field on the sphere + flux shading
    th = np.linspace(0, PI, 50)
    ph = np.linspace(0, 2 * PI, 100)
    TH, PH = np.meshgrid(th, ph)
    Xs, Ys, Zs = (np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH),
                  np.cos(TH))
    import matplotlib.colors as mcolors

    flux = 1 + Xs * np.exp(Ys + Zs)
    norm = mcolors.Normalize(flux.min(), flux.max())
    fig, ax = _setup_3d_axes(None, None)
    ax.plot_surface(Xs, Ys, Zs, facecolors=PARULA(norm(flux)), rstride=1,
                    cstride=1, linewidth=0, shade=False)
    ax.set_box_aspect((1, 1, 1))
    save(fig, "GaussGreenStokes_03.png")

    fig, ax = _setup_3d_axes(None, None)
    ax.plot_surface(Xs, Ys, Zs, facecolors=PARULA(norm(-flux)),
                    rstride=1, cstride=1, linewidth=0, shade=False)
    ax.set_box_aspect((1, 1, 1))
    save(fig, "GaussGreenStokes_04.png")


def surfaceintegral3d():
    """approx3/SurfaceIntegral3D — three classic parametrized surfaces."""
    # seashell
    u = np.linspace(0, 2 * PI, 120)
    v = np.linspace(0, 2 * PI, 120)
    U, V = np.meshgrid(u, v)
    X = 5 / 4 * (1 - V / (2 * PI)) * np.cos(2 * V) * (1 + np.cos(U)) \
        + np.cos(2 * V)
    Y = 5 / 4 * (1 - V / (2 * PI)) * np.sin(2 * V) * (1 + np.cos(U)) \
        + np.sin(2 * V)
    Z = 10 * V / (2 * PI) + 5 / 4 * (1 - V / (2 * PI)) * np.sin(U) + 15
    fig, ax = _setup_3d_axes(None, None)
    _param_surface(ax, X, Y, Z)
    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
    save(fig, "SurfaceIntegral3D_01.png")

    # spiral horn
    U2, V2 = np.meshgrid(np.linspace(0, 4 * PI, 160),
                         np.linspace(0, 2 * PI, 80))
    X2 = U2 * np.cos(U2) * (np.cos(V2) + 1)
    Y2 = U2 * np.sin(U2) * (np.cos(V2) + 1)
    Z2 = U2 * np.sin(V2)
    fig, ax = _setup_3d_axes(None, None)
    _param_surface(ax, X2, Y2, Z2)
    ax.set_box_aspect((np.ptp(X2), np.ptp(Y2), np.ptp(Z2)))
    save(fig, "SurfaceIntegral3D_02.png")

    # twisted torus
    r1 = r2 = 0.5
    t = 1.5
    U3, V3 = np.meshgrid(np.linspace(0, 2 * PI, 120),
                         np.linspace(0, 2 * PI, 120))
    X3 = (1 - r1 * np.cos(V3)) * np.cos(U3)
    Y3 = (1 - r1 * np.cos(V3)) * np.sin(U3)
    Z3 = r2 * (np.sin(V3) + t * U3 / PI)
    fig, ax = _setup_3d_axes(None, None)
    _param_surface(ax, X3, Y3, Z3)
    ax.set_box_aspect((np.ptp(X3), np.ptp(Y3), np.ptp(Z3)))
    save(fig, "SurfaceIntegral3D_03.png")


def findingrankone():
    """approx3/FindingRankOne — strip the rank-one core of fhat."""
    import matplotlib.colors as mcolors

    f = chebfun3(lambda x, y, z: jnp.sin(x) * jnp.cos(y) * jnp.exp(z))
    fhat = chebfun3(lambda x, y, z: jnp.sin(x) * jnp.cos(y) * jnp.exp(z)
                    + (jnp.cos(x) * jnp.exp(y) * jnp.sin(z)
                       + jnp.exp(x) * jnp.sin(y) * jnp.cos(z)) / 10)
    print(f"    rank(f) = {f.rank}, rank(fhat) = {fhat.rank}")

    def rank1_eval(f3, X, Y, Z):
        """First Tucker term of f3 evaluated on flat arrays."""
        c = f3.cols[0](jnp.asarray(X))
        r = f3.rows[0](jnp.asarray(Y))
        t = f3.tubes[0](jnp.asarray(Z))
        return float(f3.core[0, 0, 0]) * np.asarray(c) * np.asarray(r) \
            * np.asarray(t)

    n = 90
    xs = np.linspace(-1, 1, n)
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    flatx, flaty = XX.ravel(), YY.ravel()
    zeros = np.zeros(n * n)

    def slices(fn_eval):
        Fz = fn_eval(flatx, flaty, zeros).reshape(n, n)
        Fy = fn_eval(flatx, zeros, flaty).reshape(n, n)
        Fx = fn_eval(zeros, flatx, flaty).reshape(n, n)
        return Fx, Fy, Fz

    def render(fn_eval, name):
        Fx, Fy, Fz = slices(fn_eval)
        vmin = min(F.min() for F in (Fx, Fy, Fz))
        vmax = max(F.max() for F in (Fx, Fy, Fz))
        norm = mcolors.Normalize(vmin, vmax)
        fig = plt.figure()
        ax = fig.add_axes([0.08, -0.05, 0.8, 1.05], projection="3d")
        ax.view_init(elev=30, azim=-127.5)
        Z0 = np.zeros_like(XX)
        ax.plot_surface(XX, YY, Z0, facecolors=PARULA(norm(Fz)),
                        rstride=1, cstride=1, linewidth=0, shade=False)
        ax.plot_surface(XX, Z0, YY, facecolors=PARULA(norm(Fy)),
                        rstride=1, cstride=1, linewidth=0, shade=False)
        ax.plot_surface(Z0, XX, YY, facecolors=PARULA(norm(Fx)),
                        rstride=1, cstride=1, linewidth=0, shade=False)
        m = plt.cm.ScalarMappable(norm=norm, cmap=PARULA)
        fig.colorbar(m, ax=ax, fraction=0.04, pad=0.05)
        save(fig, name)

    def eval_f(X, Y, Z):
        return np.asarray(fhat(jnp.asarray(X), jnp.asarray(Y),
                               jnp.asarray(Z)))

    render(eval_f, "FindingRankOne_01.png")
    render(lambda X, Y, Z: rank1_eval(fhat, X, Y, Z),
           "FindingRankOne_02.png")

    scale = (float(f(jnp.array([1.0]), jnp.array([1.0]),
                     jnp.array([1.0]))[0])
             / rank1_eval(fhat, np.array([1.0]), np.array([1.0]),
                          np.array([1.0]))[0])
    print(f"    scale = {scale:.6f}")

    def eval_resid(X, Y, Z):
        fv = np.asarray(f(jnp.asarray(X), jnp.asarray(Y),
                          jnp.asarray(Z)))
        return fv - scale * rank1_eval(fhat, X, Y, Z)

    render(eval_resid, "FindingRankOne_03.png")


def lineintegral3d():
    """approx3/LineIntegral3D — two space curves."""
    ts = np.linspace(0, 2 * PI, 3000)
    p, q, r = 10, 1, 0.3
    X = np.cos(ts) * np.sqrt(q**2 - r**2 * np.cos(p * ts) ** 2)
    Y = np.sin(ts) * np.sqrt(q**2 - r**2 * np.cos(p * ts) ** 2)
    Z = r * np.cos(p * ts)
    fig, ax = _setup_3d_axes(None, None)
    ax.plot3D(X, Y, Z, color=CHEBFUN_BLUE, linewidth=1.2)
    ax.set_box_aspect((1, 1, 0.5))
    save(fig, "LineIntegral3D_01.png")

    r5 = 5
    ts2 = np.linspace(0, 10 * PI * r5 / 5, 6000)
    X2 = np.sin(ts2 / (2 * r5)) * np.cos(ts2)
    Y2 = np.sin(ts2 / (2 * r5)) * np.sin(ts2)
    Z2 = np.cos(ts2 / (2 * r5))
    fig, ax = _setup_3d_axes(None, None)
    ax.plot3D(X2, Y2, Z2, color=CHEBFUN_BLUE, linewidth=0.8)
    ax.set_box_aspect((1, 1, 1))
    save(fig, "LineIntegral3D_02.png")


def hello3():
    """approx3/Hello3 — the voxelized HELLO isosurfaces."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    A = np.zeros((15, 40))
    A[1:9, 1:3] = 1; A[4:6, 3:5] = 1; A[1:9, 5:7] = 1
    A[2:10, 9:11] = 1; A[2:4, 9:15] = 1; A[5:7, 9:15] = 1
    A[8:10, 9:15] = 1; A[3:11, 17:19] = 1; A[9:11, 17:24] = 1
    A[4:12, 25:27] = 1; A[10:12, 25:31] = 1
    A[5:13, 33:35] = 1; A[5:13, 37:39] = 1
    A[5:7, 35:37] = 1; A[11:13, 35:37] = 1
    A = np.vstack([np.zeros((14, 40)), A, np.zeros((11, 40))])
    A = np.flipud(np.fliplr(A))
    B = np.zeros((40, 40, 40))
    for k in range(17, 21):
        B[k, :, :] = A
    # smooth like the chebfun3 'equi' interpolant
    from scipy.ndimage import gaussian_filter

    Bs = gaussian_filter(B, sigma=1.0)
    Bp = np.transpose(Bs, (0, 2, 1))
    for level, name in ((0.5, "Hello3_01.png"), (-0.1 + 0.15,
                                                 "Hello3_02.png")):
        verts, faces, _, _ = marching_cubes(Bp, level=level)
        fig, ax = _setup_3d_axes(None, None)
        ax.view_init(elev=12, azim=-160)
        mesh = Poly3DCollection(verts[faces], alpha=0.9, linewidth=0)
        mesh.set_facecolor(PARULA(0.6))
        ax.add_collection3d(mesh)
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 40)
        ax.set_zlim(0, 40)
        ax.axis("off")
        ax.set_box_aspect((1, 1, 1))
        save(fig, name)


def fluxintegral3d():
    """approx3/FluxIntegral3D — wavy disk and sphere flux surfaces."""
    rr = np.linspace(0, 5, 60)
    tt = np.linspace(0, 2 * PI, 120)
    R, T = np.meshgrid(rr, tt)
    fig, ax = _setup_3d_axes(None, None)
    _param_surface(ax, R * np.cos(T), R * np.sin(T), np.cos(5 * R))
    save(fig, "FluxIntegral3D_01.png")

    th = np.linspace(0, PI, 60)
    ph = np.linspace(0, 2 * PI, 120)
    TH, PH = np.meshgrid(th, ph)
    fig, ax = _setup_3d_axes(None, None)
    _param_surface(ax, np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH),
                   np.cos(TH))
    ax.set_box_aspect((1, 1, 1))
    save(fig, "FluxIntegral3D_02.png")


def chebfun3speedup():
    """approx3/Chebfun3Speedup — measured 3D construction timings."""
    times, ranks = [], []
    for k in (1.0, 2.0, 4.0, 6.0):
        t0 = time.perf_counter()
        f = chebfun3(lambda x, y, z, _k=float(k):
                     jnp.cos(_k * (x + y * z)))
        times.append(time.perf_counter() - t0)
        ranks.append(max(f.rank))
    fig, ax = plt.subplots()
    ax.loglog(ranks, times, ".b", markersize=9)
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    ax.set_xlabel("max Tucker rank")
    ax.set_ylabel("construction time (s)")
    save(fig, "Chebfun3Speedup_01.png")

    fig, ax = plt.subplots()
    ax.semilogy(ranks, times, ".-b", markersize=9, linewidth=0.8)
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    ax.set_xlabel("max Tucker rank")
    ax.set_ylabel("construction time (s)")
    save(fig, "Chebfun3Speedup_02.png")


PAGES = {
    "ChangeVar3D": changevar3d,
    "Complexity": complexity,
    "Wagon": wagon,
    "GaussGreenStokes": gaussgreenstokes,
    "SurfaceIntegral3D": surfaceintegral3d,
    "FindingRankOne": findingrankone,
    "LineIntegral3D": lineintegral3d,
    "Hello3": hello3,
    "FluxIntegral3D": fluxintegral3d,
    "Chebfun3Speedup": chebfun3speedup,
}


if __name__ == "__main__":
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, fn in PAGES.items():
        if flt.lower() in name.lower():
            print(f"[{name}]")
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED: {e}")
