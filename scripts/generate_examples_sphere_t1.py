"""Generate per-block figures for the sphere example category,
tranche 1: SphereHeatConduction, SpherefunRotate, SolidHarmonics,
LaplaceBall, Gravity.

Spectral computations use spherical harmonics (scipy sph_harm_y);
the spherefun calculus layer itself is task #25.
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

from chebfunjax.plotting import PARULA, chebfun_style, save_chebfun_figure

chebfun_style()

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
REFROOT = ("/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/"
           "refs/docs/images")
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


def sphere_surf(F, TH, PH, name, title="", elev=25, azim=-127.5,
                cmap=PARULA):
    X = np.sin(TH) * np.cos(PH)
    Y = np.sin(TH) * np.sin(PH)
    Z = np.cos(TH)
    norm = mcolors.Normalize(F.min(), F.max() + 1e-15)
    fig = plt.figure()
    ax = fig.add_axes([0.02, -0.07, 0.96, 1.14], projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=cmap(norm(F)), rstride=1,
                    cstride=1, linewidth=0, shade=False)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    save(fig, name)


def sph_expand(F, TH, PH, lmax=20):
    """Least-squares projection of F(theta, phi) onto Y_lm."""
    w = np.sin(TH)
    coeffs = {}
    dth = TH[1, 0] - TH[0, 0]
    dph = PH[0, 1] - PH[0, 0]
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, m, TH, PH)
            coeffs[(l, m)] = np.sum(F * np.conj(Y) * w) * dth * dph
    return coeffs


def sph_eval(coeffs, TH, PH, scale=None):
    F = np.zeros_like(TH, dtype=complex)
    for (l, m), c in coeffs.items():
        s = scale(l) if scale else 1.0
        if abs(c) * abs(s) > 1e-14:
            F += c * s * sph_harm_y(l, m, TH, PH)
    return np.real(F)


def sphereheatconduction():
    """sphere/SphereHeatConduction — Y_lm spectral heat flow."""
    TH, PH = sphere_grid()
    # initial condition: two hot blobs
    F0 = (np.exp(-8 * ((TH - 1.0) ** 2 + (PH - 1.2) ** 2))
          + 0.8 * np.exp(-10 * ((TH - 2.1) ** 2 + (PH - 4.0) ** 2)))
    coeffs = sph_expand(F0, TH, PH, lmax=16)

    times = (0.0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    for j, t in enumerate(times, 1):
        Ft = sph_eval(coeffs, TH, PH,
                      scale=lambda l, _t=t: np.exp(-l * (l + 1) * _t))
        sphere_surf(Ft, TH, PH, f"SphereHeatConduction_{j:02d}.png",
                    title=f"t = {t:g}")

    # mean temperature (conserved) and mode decay
    fig, ax = plt.subplots()
    ls = np.arange(1, 17)
    for t in (0.01, 0.05, 0.2):
        energy = [np.sum([np.abs(coeffs[(l, m)]) ** 2
                          * np.exp(-2 * l * (l + 1) * t)
                          for m in range(-l, l + 1)]) for l in ls]
        ax.semilogy(ls, np.maximum(energy, 1e-20), ".-",
                    markersize=5, linewidth=0.8, label=f"t = {t:g}")
    ax.legend(fontsize=7)
    ax.set_xlabel("degree l")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("modal energy decay exp(-2l(l+1)t)", fontsize=9)
    save(fig, "SphereHeatConduction_09.png")

    fig, ax = plt.subplots()
    ts = np.linspace(0, 1, 200)
    maxT = [np.max(sph_eval(coeffs, TH[::6, ::6], PH[::6, ::6],
                            scale=lambda l, _t=t: np.exp(
                                -l * (l + 1) * _t)))
            for t in ts[::10]]
    ax.plot(ts[::10], maxT, ".-", markersize=6, linewidth=0.9)
    ax.set_xlabel("t")
    ax.set_title("maximum temperature decays to the mean",
                 fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "SphereHeatConduction_10.png")


def spherefunrotate():
    """sphere/SpherefunRotate — rotating a function on the sphere."""
    TH, PH = sphere_grid()

    def f_np(th, ph):
        x = np.sin(th) * np.cos(ph)
        y = np.sin(th) * np.sin(ph)
        z = np.cos(th)
        return np.exp(-6 * ((x - 0.7) ** 2 + y**2
                            + (z - 0.7) ** 2)) + 0.5 * np.cos(3 * x)

    def rotate_eval(alpha, beta, gamma):
        # rotate coordinates by Euler angles (z-y-z), evaluate f
        X = np.sin(TH) * np.cos(PH)
        Y = np.sin(TH) * np.sin(PH)
        Z = np.cos(TH)
        P = np.stack([X.ravel(), Y.ravel(), Z.ravel()])

        def Rz(a):
            return np.array([[np.cos(a), -np.sin(a), 0],
                             [np.sin(a), np.cos(a), 0], [0, 0, 1]])

        def Ry(a):
            return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                             [-np.sin(a), 0, np.cos(a)]])

        R = Rz(alpha) @ Ry(beta) @ Rz(gamma)
        Q = R.T @ P
        thr = np.arccos(np.clip(Q[2], -1, 1)).reshape(TH.shape)
        phr = np.mod(np.arctan2(Q[1], Q[0]), 2 * PI).reshape(PH.shape)
        return f_np(thr, phr)

    cases = [(0, 0, 0), (PI / 4, 0, 0), (0, PI / 4, 0),
             (0, PI / 2, 0), (PI / 4, PI / 4, 0),
             (PI / 2, PI / 2, PI / 4), (PI, PI / 3, PI / 6)]
    for j, (a, b, g) in enumerate(cases, 1):
        F = rotate_eval(a, b, g)
        sphere_surf(F, TH, PH, f"SpherefunRotate_{j:02d}.png",
                    title=f"rotation ({a:.2f}, {b:.2f}, {g:.2f})")


def solidharmonics():
    """sphere/SolidHarmonics — r^l Y_lm inside the ball."""
    TH, PH = sphere_grid()
    for j, (l, m) in enumerate(((4, 2), (6, 3)), 1):
        Y = np.real(sph_harm_y(l, m, TH, PH))
        sphere_surf(Y, TH, PH, f"SolidHarmonics_{j:02d}.png",
                    title=f"solid harmonic (l, m) = ({l}, {m})")


def laplaceball():
    """sphere/LaplaceBall — Poisson solve in the ball via harmonics."""
    TH, PH = sphere_grid()
    # boundary data g = Y_3^2 + 0.5 Y_1^0; harmonic extension is
    # r^l Y_lm; plot boundary data and two interior slices
    G = (np.real(sph_harm_y(3, 2, TH, PH))
         + 0.5 * np.real(sph_harm_y(1, 0, TH, PH)))
    sphere_surf(G, TH, PH, "LaplaceBall_01.png",
                title="Dirichlet boundary data")

    # interior slice z = 0 plane: u(r, phi) in polar coordinates
    rr = np.linspace(0, 1, 100)
    pp = np.linspace(0, 2 * PI, 200)
    RR, PP = np.meshgrid(rr, pp, indexing="ij")
    TH0 = PI / 2 * np.ones_like(PP)
    U = (RR**3 * np.real(sph_harm_y(3, 2, TH0, PP))
         + 0.5 * RR * np.real(sph_harm_y(1, 0, TH0, PP)))
    fig, ax = plt.subplots()
    cs = ax.contourf(RR * np.cos(PP), RR * np.sin(PP), U, levels=20,
                     cmap=PARULA)
    plt.colorbar(cs, ax=ax, fraction=0.045)
    ax.set_aspect("equal")
    ax.set_title("harmonic extension on the plane z = 0",
                 fontsize=9)
    save(fig, "LaplaceBall_02.png")

    # radial profile along a chosen direction
    th0, ph0 = 1.1, 0.7
    prof = (rr**3 * np.real(sph_harm_y(3, 2, np.full_like(rr, th0),
                                       np.full_like(rr, ph0)))
            + 0.5 * rr * np.real(sph_harm_y(
                1, 0, np.full_like(rr, th0), np.full_like(rr, ph0))))
    fig, ax = plt.subplots()
    ax.plot(rr, prof, linewidth=1.4)
    ax.set_xlabel("r")
    ax.set_title("radial profile of the harmonic extension",
                 fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "LaplaceBall_03.png")


def gravity():
    """sphere/Gravity — gravitational potential of a bumpy planet."""
    TH, PH = sphere_grid()
    # a lumpy density on the sphere -> potential smooths it: 1/(2l+1)
    F = (np.exp(-6 * ((TH - 1.1) ** 2 + (PH - 1.0) ** 2))
         + 0.7 * np.exp(-8 * ((TH - 2.0) ** 2 + (PH - 4.2) ** 2))
         + 0.4 * np.cos(2 * PH) * np.sin(TH) ** 2)
    coeffs = sph_expand(F, TH, PH, lmax=14)
    V = sph_eval(coeffs, TH, PH, scale=lambda l: 1.0 / (2 * l + 1))
    sphere_surf(V, TH, PH, "Gravity_01.png",
                title="gravitational potential of a lumpy planet")


PAGES = {
    "SphereHeatConduction": sphereheatconduction,
    "SpherefunRotate": spherefunrotate,
    "SolidHarmonics": solidharmonics,
    "LaplaceBall": laplaceball,
    "Gravity": gravity,
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
