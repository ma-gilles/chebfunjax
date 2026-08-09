"""Atmospheric temperature data on the sphere.

Faithful replica of sphere/AtmosphericTemperature.m: the 529 x 1024
global temperature dataset (AtmosphericData.mat, fetched from the
chebfun examples repository) as a spherefun -- mean temperature, pole
values, equator slice, zonal mean, the steady-heat Poisson solve, and
Gaussian filtering at sigma = 2, 10, 20 degrees (implemented
spectrally: the Gauss-Weierstrass filter scales harmonic band l by
exp(-l(l+1) sigma^2 / 2)).

Original: https://www.chebfun.org/examples/sphere/AtmosphericTemperature.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import urllib.request
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from scipy.io import loadmat

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'sphere')
_MAT_URL = ("https://raw.githubusercontent.com/chebfun/examples/"
            "master/sphere/AtmosphericData.mat")
FIG = [0]


def _load_temp():
    cache = os.path.join(_HERE, '..', '..', '.atmospheric_data.mat')
    if not os.path.exists(cache):
        urllib.request.urlretrieve(_MAT_URL, cache)
    return np.asarray(loadmat(cache)["Temp"], dtype=float)


def _dfs_coeffs(V):
    """2D Fourier coefficients of the DFS extension of the data.

    V is (n_th, n_lam) with theta 0..pi inclusive, lambda [-pi,pi).
    """
    nth, nlam = V.shape
    # drop the duplicated theta = pi row for the doubled grid
    Vh = V[:-1, :]
    V2 = np.vstack([Vh, np.roll(V[::-1][:-1, :], nlam // 2, axis=1)])
    return np.fft.fft2(V2) / V2.size


def _dfs_eval(C, lam, th):
    n2, nlam = C.shape
    kt = np.fft.fftfreq(n2, d=1.0 / n2)
    kl = np.fft.fftfreq(nlam, d=1.0 / nlam)
    Et = np.exp(1j * np.outer(np.asarray(th).ravel(), kt))
    El = np.exp(1j * np.outer(np.asarray(lam).ravel() + np.pi, kl))
    out = np.real(np.einsum("pk,kl,pl->p", Et, C, El))
    return out.reshape(np.shape(lam))


def _sphere_plot(fun, title="", view=(0, 50), n=280, cmap="jet",
                 clim=None):
    FIG[0] += 1
    lam = np.linspace(-np.pi, np.pi, n)
    th = np.linspace(0, np.pi, n)
    L, T = np.meshgrid(lam, th)
    V = fun(L, T)
    fig, ax = plt.subplots(figsize=(7.2, 5.6),
                           subplot_kw={"projection": "3d"})
    X, Y, Z = (np.cos(L) * np.sin(T), np.sin(L) * np.sin(T), np.cos(T))
    lo, hi = clim if clim else (V.min(), V.max())
    W = np.clip((V - lo) / (hi - lo + 1e-300), 0, 1)
    ax.plot_surface(X, Y, Z, facecolors=plt.get_cmap(cmap)(W),
                    rstride=1, cstride=1, linewidth=0,
                    antialiased=False)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(*view)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"AtmosphericTemperature_repl_{FIG[0]:02d}.png"),
        dpi=140, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    Temp = _load_temp()
    C = _dfs_coeffs(Temp)
    print("dataset:", Temp.shape)

    f = lambda L, T: _dfs_eval(C, L, T)  # noqa: E731
    _sphere_plot(f, view=(0, 50))

    # Celsius; mean over the sphere (integrate f sin(theta)).
    C0 = C.copy()
    Cc = C0.copy()
    Cc[0, 0] -= 273.15

    fc = lambda L, T: _dfs_eval(Cc, L, T)  # noqa: E731
    nq = 180
    from numpy.polynomial.legendre import leggauss
    xg, wg = leggauss(nq)
    thq = np.arccos(xg)
    lamq = -np.pi + 2 * np.pi * np.arange(2 * nq) / (2 * nq)
    LQ, TQ = np.meshgrid(lamq, thq)
    FV = fc(LQ, TQ)
    mean2 = float(np.sum(FV * wg[:, None]) * (2 * np.pi / (2 * nq))
                  / (4 * np.pi))
    print("mean2(f) =")
    print(f"  {mean2:.15f}")
    print("f(North pole) =")
    print(f"   {float(fc(0.0, 0.0)):.15f}")
    print("f(South pole) =")
    print(f" {float(fc(0.0, np.pi)):.15f}")

    # Equator slice.
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    lamg = np.linspace(-np.pi, np.pi, 1200)
    ax.plot(lamg, fc(lamg, np.pi / 2 * np.ones_like(lamg)), lw=1.4)
    ax.set_xlabel(r"Longitude, $\lambda$")
    ax.set_ylabel("Temperature (Celsius)")
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"AtmosphericTemperature_repl_{FIG[0]:02d}.png"),
        dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Zonal mean (mean over lambda as a function of theta).
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    thg = np.linspace(0, np.pi, 800)
    zon = np.real(np.array(
        [_dfs_eval(Cc[:, :1] * 0 + Cc, np.full(1, 0.0), np.array([t]))
         for t in thg]).ravel())
    # zonal mean = lambda-DC Fourier mode
    Cz = np.zeros_like(Cc)
    Cz[:, 0] = Cc[:, 0]
    zon = _dfs_eval(Cz, np.zeros_like(thg), thg)
    ax.plot(thg, zon, lw=1.6)
    ax.set_xlim(0, np.pi)
    ax.set_xlabel(r"Co-latitude, $\theta$")
    ax.set_ylabel("Temperature (Celsius)")
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"AtmosphericTemperature_repl_{FIG[0]:02d}.png"),
        dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Steady heat: Poisson solve on the mean-free data (harmonic space).
    from chebfunjax.spherefun.spherefun import _real_ylm_values
    LMAXP = 60
    co = {}
    for l in range(1, LMAXP + 1):
        for m in range(-l, l + 1):
            Y = np.asarray(_real_ylm_values(
                l, m, LQ.ravel(), TQ.ravel())).reshape(LQ.shape)
            c = float(np.sum((FV - mean2) * Y * wg[:, None])
                      * (2 * np.pi / (2 * nq)))
            if abs(c) > 1e-10:
                co[(l, m)] = c / (l * (l + 1))   # -lap u = rhs

    def heat(Lg, Tg):
        out = np.zeros(np.shape(Lg))
        for (l, m), cc in co.items():
            out = out + cc * np.asarray(_real_ylm_values(l, m, Lg, Tg))
        return out

    _sphere_plot(fc, "Original dataset", view=(0, 50))
    _sphere_plot(heat, "Steady Heat", view=(0, 50))

    # Gaussian filtering at sigma = 2, 10, 20 degrees.
    for sig in np.array([2, 10, 20]) * np.pi / 180:
        LMAXF = min(int(6 / sig), 200)
        cof = {}
        for l in range(0, LMAXF + 1):
            damp = np.exp(-l * (l + 1) * sig**2 / 2)
            if damp < 1e-8:
                break
            for m in range(-l, l + 1):
                Y = np.asarray(_real_ylm_values(
                    l, m, LQ.ravel(), TQ.ravel())).reshape(LQ.shape)
                c = float(np.sum(FV * Y * wg[:, None])
                          * (2 * np.pi / (2 * nq)))
                if abs(c) > 1e-10:
                    cof[(l, m)] = c * damp

        def fsm(Lg, Tg, _co=cof):
            out = np.zeros(np.shape(Lg))
            for (l, m), cc in _co.items():
                out = out + cc * np.asarray(
                    _real_ylm_values(l, m, Lg, Tg))
            return out

        _sphere_plot(fsm,
                     f"Smoothed Temp., $\\sigma$="
                     f"{sig * 180 / np.pi:g} degrees", view=(0, 50))
        print(f"sigma={sig * 180 / np.pi:g} deg done", flush=True)


if __name__ == "__main__":
    run()
