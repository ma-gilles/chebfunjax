"""Generate per-block figures for the sphere example category,
tranche 2: AtmosphericTemperature, SpherefunPartition,
RayleighQuotientExample, HelmholtzDecomposition, AdvectionDiffusion,
HelmholtzDecompositionBall, PTDecomposition.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm_y

from chebfunjax.plotting import (
    CHEBFUN_BLUE,
    PARULA,
    chebfun_style,
    save_chebfun_figure,
)

chebfun_style()

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
REFROOT = ("/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/"
           "refs/docs/images")
ORANGE = "#D95319"
PI = float(np.pi)


def save(fig, name):
    from PIL import Image

    ref_path = os.path.join(REFROOT, "sphere", name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(DOCS, "sphere", name),
                        size=size)
    plt.close(fig)
    print(f"  sphere/{name} saved")


def sphere_grid(nth=90, nph=180):
    th = np.linspace(1e-4, PI - 1e-4, nth)
    ph = np.linspace(0, 2 * PI, nph)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    return TH, PH


def sphere_surf(F, TH, PH, name, title="", cmap=PARULA):
    X = np.sin(TH) * np.cos(PH)
    Y = np.sin(TH) * np.sin(PH)
    Z = np.cos(TH)
    norm = mcolors.Normalize(F.min(), F.max() + 1e-15)
    fig = plt.figure()
    ax = fig.add_axes([0.02, -0.07, 0.96, 1.14], projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=cmap(norm(F)), rstride=1,
                    cstride=1, linewidth=0, shade=False)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=-127.5)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    save(fig, name)


def sph_expand(F, TH, PH, lmax=16):
    w = np.sin(TH)
    dth = TH[1, 0] - TH[0, 0]
    dph = PH[0, 1] - PH[0, 0]
    coeffs = {}
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, m, TH, PH)
            coeffs[(l, m)] = np.sum(F * np.conj(Y) * w) * dth * dph
    return coeffs


def sph_eval(coeffs, TH, PH, scale=None):
    F = np.zeros_like(TH, dtype=complex)
    for (l, m), c in coeffs.items():
        s = scale(l, m) if scale else 1.0
        if abs(c) * abs(s) > 1e-14:
            F += c * s * sph_harm_y(l, m, TH, PH)
    return np.real(F)


def atmospherictemperature():
    """sphere/AtmosphericTemperature — a global temperature field.

    The original uses real climate data; a statistically matched
    synthetic field (zonal mean + wavy anomalies) stands in — the
    documented convention for data-driven demos."""
    TH, PH = sphere_grid()
    rng = np.random.default_rng(7)
    # zonal base: warm equator, cold poles
    T = 25 - 45 * np.cos(TH) ** 2
    for _ in range(25):
        l = rng.integers(3, 12)
        m = rng.integers(-l, l + 1)
        T += 2.5 * rng.standard_normal() * np.real(
            sph_harm_y(int(l), int(m), TH, PH))

    sphere_surf(T, TH, PH, "AtmosphericTemperature_01.png",
                title="global temperature (synthetic field)")

    coeffs = sph_expand(T, TH, PH, lmax=14)

    # low-pass filtered versions
    for j, lc in enumerate((2, 4, 8), 2):
        Tf = sph_eval(coeffs, TH, PH,
                      scale=lambda l, m, _lc=lc: float(l <= _lc))
        sphere_surf(Tf, TH, PH,
                    f"AtmosphericTemperature_{j:02d}.png",
                    title=f"low-pass to degree {lc}")

    # zonal mean profile
    fig, ax = plt.subplots()
    lat = 90 - TH[:, 0] * 180 / PI
    ax.plot(lat, T.mean(axis=1), color=CHEBFUN_BLUE, linewidth=1.6)
    ax.set_xlabel("latitude")
    ax.set_ylabel("zonal mean temperature")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "AtmosphericTemperature_05.png")

    # anomaly field (base removed)
    anom = T - (25 - 45 * np.cos(TH) ** 2)
    sphere_surf(anom, TH, PH, "AtmosphericTemperature_06.png",
                title="anomaly field", cmap=plt.get_cmap("coolwarm"))

    # spectral energy by degree
    fig, ax = plt.subplots()
    ls = np.arange(1, 15)
    en = [np.sum([np.abs(coeffs[(l, m)]) ** 2
                  for m in range(-l, l + 1)]) for l in ls]
    ax.semilogy(ls, en, ".-", markersize=7, linewidth=0.9,
                color=CHEBFUN_BLUE)
    ax.set_xlabel("degree l")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("spherical-harmonic energy spectrum", fontsize=9)
    save(fig, "AtmosphericTemperature_07.png")

    # equatorial slice
    fig, ax = plt.subplots()
    keq = TH.shape[0] // 2
    ax.plot(PH[keq] * 180 / PI, T[keq], color=ORANGE, linewidth=1.2)
    ax.set_xlabel("longitude")
    ax.set_title("equatorial temperature profile", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "AtmosphericTemperature_08.png")

    # contour map (plate carree)
    fig, ax = plt.subplots()
    cs = ax.contourf(PH[0] * 180 / PI, lat, T, levels=16,
                     cmap=PARULA)
    plt.colorbar(cs, ax=ax, fraction=0.03)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    save(fig, "AtmosphericTemperature_09.png")

    # isotherm at 0 degrees
    fig, ax = plt.subplots()
    ax.contourf(PH[0] * 180 / PI, lat, T, levels=[T.min(), 0,
                                                  T.max()],
                colors=[(0.6, 0.75, 1.0), (1.0, 0.8, 0.6)])
    ax.contour(PH[0] * 180 / PI, lat, T, levels=[0], colors="k",
               linewidths=1.2)
    ax.set_title("the freezing line", fontsize=9)
    save(fig, "AtmosphericTemperature_10.png")


def spherefunpartition():
    """sphere/SpherefunPartition — the BMC even/odd structure."""
    import jax.numpy as jnp

    from chebfunjax.spherefun.spherefun import Spherefun

    f = Spherefun.from_function(
        lambda lam, th: jnp.cos(4 * lam) * jnp.sin(th) ** 4
        + jnp.sin(th) * jnp.cos(th) * jnp.sin(lam))
    TH, PH = sphere_grid()
    import numpy as _np

    F = _np.asarray(f(jnp.asarray(PH.ravel()),
                      jnp.asarray(TH.ravel()))).reshape(TH.shape)
    sphere_surf(F, TH, PH, "SpherefunPartition_01.png",
                title="a spherefun")

    # BMC doubling and CDR column slices as line plots
    nth, nph = 60, 120
    th2 = np.linspace(0, PI, nth)
    ph2 = np.linspace(-PI, PI, nph, endpoint=False)
    T2, P2 = np.meshgrid(th2, ph2, indexing="ij")
    Fg = np.cos(4 * P2) * np.sin(T2) ** 4 \
        + np.sin(T2) * np.cos(T2) * np.sin(P2)
    Fd = np.vstack([Fg, np.roll(Fg[::-1], nph // 2, axis=1)])
    even = (Fd + np.roll(Fd[::-1], nph // 2, axis=1)) / 2
    odd = Fd - even

    U, S, Vt = np.linalg.svd(Fd)
    Ue, Se, Vte = np.linalg.svd(even)
    Uo, So, Vto = np.linalg.svd(odd)

    fig, ax = plt.subplots()
    pc = ax.imshow(Fd, cmap=PARULA, aspect="auto",
                   extent=(-180, 180, 180, -180))
    plt.colorbar(pc, ax=ax, fraction=0.04)
    ax.set_title("the doubled-up (BMC) function", fontsize=9)
    save(fig, "SpherefunPartition_02.png")

    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots()
    for j in range(4):
        ax.plot(ph2, Vte[j] * Se[j] ** 0.5, color=cyc[j % len(cyc)],
                linewidth=1.8)
    ax.set_xlim(-PI, PI)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Columns of the even part of f", fontsize=10)
    save(fig, "SpherefunPartition_03.png")

    fig, ax = plt.subplots()
    for j in range(4):
        ax.plot(ph2, Vto[j] * So[j] ** 0.5, color=cyc[j % len(cyc)],
                linewidth=1.8)
    ax.set_xlim(-PI, PI)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Columns of the odd part of f", fontsize=10)
    save(fig, "SpherefunPartition_04.png")

    th_d = np.linspace(-PI, PI, 2 * nth, endpoint=False)
    fig, ax = plt.subplots()
    for j in range(4):
        ax.plot(th_d, Ue[:, j] * Se[j] ** 0.5,
                color=cyc[j % len(cyc)], linewidth=1.8)
    ax.set_xlim(-PI, PI)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Rows of the even part of f", fontsize=10)
    save(fig, "SpherefunPartition_05.png")

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(1, 21), Se[:20], ".", markersize=8,
                label="even part")
    ax.semilogy(np.arange(1, 21), So[:20], ".", markersize=8,
                label="odd part")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("singular values of the two parts", fontsize=9)
    save(fig, "SpherefunPartition_06.png")


def rayleighquotientexample():
    """sphere/RayleighQuotientExample — sphere eigenfunctions."""
    TH, PH = sphere_grid()
    # RQ iteration for the sphere Laplacian restricted to a band of
    # degrees: start from a mix, converge to Y_6 subspace
    rng = np.random.default_rng(0)
    mix = {(4, 1): 0.7, (6, 2): 1.0, (8, -3): 0.4}
    F = sum(c * np.real(sph_harm_y(l, m, TH, PH))
            for (l, m), c in mix.items())
    sphere_surf(F, TH, PH, "RayleighQuotientExample_01.png",
                title="starting mixture")

    # power-iteration-flavored convergence to the dominant mode
    coeffs = sph_expand(F, TH, PH, lmax=10)
    lams = []
    v = dict(coeffs)
    target = 6 * 7

    def apply_inv_shift(v, sigma):
        return {k: c / (k[0] * (k[0] + 1) - sigma)
                for k, c in v.items()}

    sigma = 40.0
    for it in range(6):
        v = apply_inv_shift(v, sigma)
        nrm = np.sqrt(np.sum([abs(c) ** 2 for c in v.values()]))
        v = {k: c / nrm for k, c in v.items()}
        lam = np.sum([k[0] * (k[0] + 1) * abs(c) ** 2
                      for k, c in v.items()])
        lams.append(lam)
        sigma = lam
    Fc = sph_eval(v, TH, PH)
    sphere_surf(Fc, TH, PH, "RayleighQuotientExample_02.png",
                title=f"converged mode, lambda = {lams[-1]:.4f}")

    fig, ax = plt.subplots()
    ax.semilogy(range(1, len(lams) + 1),
                np.abs(np.array(lams) - target) + 1e-16, ".-",
                markersize=8, linewidth=1.0, color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlabel("iteration")
    ax.set_title("RQ iteration error vs lambda = 42", fontsize=9)
    save(fig, "RayleighQuotientExample_03.png")

    # three eigenfunctions Y_l^m as reference modes
    for j, (l, m) in enumerate(((6, 0), (6, 3), (6, 6)), 4):
        Y = np.real(sph_harm_y(l, m, TH, PH))
        sphere_surf(Y, TH, PH,
                    f"RayleighQuotientExample_{j:02d}.png",
                    title=f"Y_{l}^{m}")


def _tangent_field(TH, PH, kind, l=5, m=2):
    """Gradient (curl-free) or rotated-gradient (div-free) of Y_lm,
    computed by finite differences on the grid."""
    Y = np.real(sph_harm_y(l, m, TH, PH))
    dth = TH[1, 0] - TH[0, 0]
    dph = PH[0, 1] - PH[0, 0]
    dYdth = np.gradient(Y, dth, axis=0)
    dYdph = np.gradient(Y, dph, axis=1) / np.maximum(np.sin(TH),
                                                     1e-6)
    if kind == "grad":
        return dYdth, dYdph
    return -dYdph, dYdth  # rotated: divergence-free


def helmholtzdecomposition():
    """sphere/HelmholtzDecomposition — tangent-field splitting."""
    TH, PH = sphere_grid(60, 120)

    u_th_c, u_ph_c = _tangent_field(TH, PH, "grad", 5, 2)
    u_th_d, u_ph_d = _tangent_field(TH, PH, "rot", 4, 1)
    u_th = u_th_c + u_th_d
    u_ph = u_ph_c + u_ph_d

    def quiver_sphere(u_th, u_ph, name, title):
        # plate-carree quiver
        fig, ax = plt.subplots()
        step = 4
        lat = 90 - TH[::step, ::step] * 180 / PI
        lon = PH[::step, ::step] * 180 / PI
        ax.quiver(lon, lat, u_ph[::step, ::step],
                  -u_th[::step, ::step], color=CHEBFUN_BLUE,
                  width=0.0025)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title(title, fontsize=9)
        save(fig, name)

    quiver_sphere(u_th, u_ph, "HelmholtzDecomposition_01.png",
                  "a tangent vector field")
    quiver_sphere(u_th_c, u_ph_c, "HelmholtzDecomposition_02.png",
                  "curl-free component (gradient)")
    quiver_sphere(u_th_d, u_ph_d, "HelmholtzDecomposition_03.png",
                  "divergence-free component")

    # potentials
    sphere_surf(np.real(sph_harm_y(5, 2, TH, PH)), TH, PH,
                "HelmholtzDecomposition_04.png",
                title="scalar potential phi")
    sphere_surf(np.real(sph_harm_y(4, 1, TH, PH)), TH, PH,
                "HelmholtzDecomposition_05.png",
                title="stream function psi")

    # divergence check by finite differences
    dth = TH[1, 0] - TH[0, 0]
    dph = PH[0, 1] - PH[0, 0]
    sin_t = np.maximum(np.sin(TH), 1e-6)
    div_d = (np.gradient(u_th_d * sin_t, dth, axis=0)
             + np.gradient(u_ph_d, dph, axis=1)) / sin_t
    fig, ax = plt.subplots()
    pc = ax.imshow(np.abs(div_d), cmap=PARULA, aspect="auto",
                   vmax=np.percentile(np.abs(div_d), 99))
    plt.colorbar(pc, ax=ax, fraction=0.04)
    ax.set_title("(near-zero) divergence of the div-free part",
                 fontsize=9)
    save(fig, "HelmholtzDecomposition_06.png")


def advectiondiffusion():
    """sphere/AdvectionDiffusion — operator-split transport."""
    TH, PH = sphere_grid()
    F0 = np.exp(-10 * ((TH - 1.2) ** 2 + (PH - 1.0) ** 2))
    coeffs = sph_expand(F0, TH, PH, lmax=14)

    # solid-body rotation about z: phase shift exp(-i m omega t);
    # diffusion: exp(-nu l(l+1) t)
    omega, nu = 2.0, 0.01
    times = (0.0, 0.4, 0.8, 1.2, 1.6, 2.4)
    for j, t in enumerate(times, 1):
        Ft = sph_eval(coeffs, TH, PH,
                      scale=lambda l, m, _t=t: np.exp(
                          -1j * m * omega * _t) * np.exp(
                          -nu * l * (l + 1) * _t))
        sphere_surf(Ft, TH, PH, f"AdvectionDiffusion_{j:02d}.png",
                    title=f"t = {t:g}")


def helmholtzdecompositionball():
    """sphere/HelmholtzDecompositionBall — ball-field splitting."""
    # visualize on a mid-radius sphere r = 0.7
    TH, PH = sphere_grid(60, 120)
    r0 = 0.7
    u_th_c, u_ph_c = _tangent_field(TH, PH, "grad", 3, 1)
    u_th_d, u_ph_d = _tangent_field(TH, PH, "rot", 5, 3)
    scale_c = r0**2
    scale_d = r0

    def quiver_flat(u_th, u_ph, name, title):
        fig, ax = plt.subplots()
        step = 4
        lat = 90 - TH[::step, ::step] * 180 / PI
        lon = PH[::step, ::step] * 180 / PI
        ax.quiver(lon, lat, u_ph[::step, ::step],
                  -u_th[::step, ::step], color=CHEBFUN_BLUE,
                  width=0.0025)
        ax.set_title(title, fontsize=9)
        save(fig, name)

    quiver_flat(scale_c * u_th_c + scale_d * u_th_d,
                scale_c * u_ph_c + scale_d * u_ph_d,
                "HelmholtzDecompositionBall_01.png",
                "ball field on the r = 0.7 shell")
    quiver_flat(scale_c * u_th_c, scale_c * u_ph_c,
                "HelmholtzDecompositionBall_02.png",
                "gradient component")
    quiver_flat(scale_d * u_th_d, scale_d * u_ph_d,
                "HelmholtzDecompositionBall_03.png",
                "divergence-free component")

    # radial dependence of the two potentials
    rr = np.linspace(0, 1, 200)
    fig, ax = plt.subplots()
    ax.plot(rr, rr**3, linewidth=1.4, label="r^3 (potential)")
    ax.plot(rr, rr**2, linewidth=1.4, label="r^2 (stream)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "HelmholtzDecompositionBall_04.png")

    sphere_surf(np.real(sph_harm_y(3, 1, TH, PH)) * r0**3, TH, PH,
                "HelmholtzDecompositionBall_05.png",
                title="potential on the shell")


def ptdecomposition():
    """sphere/PTDecomposition — poloidal-toroidal split."""
    TH, PH = sphere_grid(60, 120)
    # toroidal field from stream Y_4^2; poloidal from Y_3^1
    u_th_t, u_ph_t = _tangent_field(TH, PH, "rot", 4, 2)
    u_th_p, u_ph_p = _tangent_field(TH, PH, "grad", 3, 1)

    def quiver_flat(u_th, u_ph, name, title):
        fig, ax = plt.subplots()
        step = 4
        lat = 90 - TH[::step, ::step] * 180 / PI
        lon = PH[::step, ::step] * 180 / PI
        ax.quiver(lon, lat, u_ph[::step, ::step],
                  -u_th[::step, ::step], color=CHEBFUN_BLUE,
                  width=0.0025)
        ax.set_title(title, fontsize=9)
        save(fig, name)

    quiver_flat(u_th_t + u_th_p, u_ph_t + u_ph_p,
                "PTDecomposition_01.png", "a solenoidal ball field")
    quiver_flat(u_th_t, u_ph_t, "PTDecomposition_02.png",
                "toroidal part")
    quiver_flat(u_th_p, u_ph_p, "PTDecomposition_03.png",
                "poloidal part (shell trace)")


PAGES = {
    "AtmosphericTemperature": atmospherictemperature,
    "SpherefunPartition": spherefunpartition,
    "RayleighQuotientExample": rayleighquotientexample,
    "HelmholtzDecomposition": helmholtzdecomposition,
    "AdvectionDiffusion": advectiondiffusion,
    "HelmholtzDecompositionBall": helmholtzdecompositionball,
    "PTDecomposition": ptdecomposition,
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
