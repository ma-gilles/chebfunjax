"""Solving the heat equation on the unit sphere.

Faithful replica of sphere/SphereHeatConduction.m by Alex Townsend and
Grady Wright (May 2016): u_t = alpha lap(u) integrated with BDF2,
each step a spectral Helmholtz solve.  First the "soccer ball"
eigenfunction initial condition (analytic solution
exp(-42 alpha t) u0), then five random Gaussian bumps whose mean is
conserved.

The BDF2 iteration runs in spherical-harmonic COEFFICIENT space --
identical to the sequence of spectral Helmholtz solves (each is
diagonal there: a -> (K^2/3)(4a - a_prev)/(K^2 - l(l+1))), with one
projection and a handful of reconstructions.  (200 successive
spherefun constructions in one process exhaust the JIT compiler --
the known LLVM-OOM class.)

Original: https://www.chebfun.org/examples/sphere/SphereHeatConduction.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from numpy.polynomial.legendre import leggauss

from chebfunjax.plotting import chebfun_style
from chebfunjax.spherefun.spherefun import Spherefun, _real_ylm_values, _sph_harmonic_eval_sum

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'sphere')
FIG = [0]
ALPHA = 1 / 42


def _plot_fn(F, title="", clim=None, n=220):
    FIG[0] += 1
    lam = np.linspace(-np.pi, np.pi, n)
    th = np.linspace(0, np.pi, n)
    L, T = np.meshgrid(lam, th)
    V = np.asarray(F(L.ravel(), T.ravel())).reshape(L.shape)
    fig, ax = plt.subplots(figsize=(7.0, 5.4),
                           subplot_kw={"projection": "3d"})
    X, Y, Z = (np.cos(L) * np.sin(T), np.sin(L) * np.sin(T), np.cos(T))
    lo, hi = clim if clim else (V.min(), V.max())
    W = np.clip((V - lo) / (hi - lo + 1e-300), 0, 1)
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.jet(W), rstride=1,
                    cstride=1, linewidth=0, antialiased=False)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"SphereHeatConduction_repl_{FIG[0]:02d}.png"),
        dpi=140, bbox_inches="tight")
    plt.close(fig)


def _project(u0_fn, lmax, nq=96):
    """Spherical-harmonic coefficients of a callable via quadrature."""
    xg, wg = leggauss(nq)
    thq = np.arccos(xg)
    lamq = -np.pi + 2 * np.pi * np.arange(2 * nq) / (2 * nq)
    LQ, TQ = np.meshgrid(lamq, thq)
    wl = 2 * np.pi / (2 * nq)
    FV = np.asarray(u0_fn(LQ.ravel(), TQ.ravel())).reshape(LQ.shape)
    pairs, a = [], []
    for el in range(lmax + 1):
        for m in range(-el, el + 1):
            YV = np.asarray(_real_ylm_values(
                el, m, LQ.ravel(), TQ.ravel())).reshape(LQ.shape)
            c = float(np.sum(FV * YV * wg[:, None]) * wl)
            if abs(c) > 1e-13:
                pairs.append((el, m))
                a.append(c)
    return pairs, np.array(a)


def _bdf2_run(u0_fn, dt, tfinal, clim, lmax):
    pairs, a0 = _project(u0_fn, lmax)
    eig = np.array([el * (el + 1) for el, _ in pairs], dtype=float)

    def plot_from(a, title):
        cmap = dict(zip(pairs, a))
        _plot_fn(lambda lam_, th_: _sph_harmonic_eval_sum(
            cmap, lmax + 1, lam_, th_), title, clim=clim)

    nsteps = int(np.ceil(tfinal / dt))
    K2 = -1 / (dt * ALPHA)                     # BDF1 first step
    up_c = a0
    u_c = K2 * a0 / (K2 - eig)
    K2 = -3 / (2 * dt * ALPHA)                 # BDF2
    for n in range(2, nsteps + 1):
        rhs = K2 / 3 * (4 * u_c - up_c)
        up_c = u_c
        u_c = rhs / (K2 - eig)
        if n % 25 == 0:
            plot_from(u_c, f"Time {n * dt:1.2f}")
    return pairs, u_c, a0


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # 4. The soccer ball function (analytic solution).
    u0 = (Spherefun.sphharm(6, 0)
          + np.sqrt(14 / 11) * Spherefun.sphharm(6, 5))
    _plot_fn(lambda lam, th: np.asarray(u0(lam, th)), clim=(-0.5, 1))
    pairs, u_c, a0 = _bdf2_run(
        lambda lam, th: np.asarray(u0(lam, th)), 0.01, 1.0,
        clim=(-0.5, 1), lmax=8)
    # Analytic solution: coefficients decay by exp(-42*alpha*t).
    err = np.linalg.norm(u_c - np.exp(-42 * ALPHA * 1.0) * a0)
    print("ans =")
    print(f"     {err:.15e}")

    # 5. Five random Gaussian bumps: the mean is conserved.
    rng = np.random.default_rng(10)
    centers = []
    for _ in range(5):
        x0 = 2 * rng.random() - 1
        y0 = np.sqrt(1 - x0**2) * (2 * rng.random() - 1)
        z0 = np.sqrt(1 - x0**2 - y0**2)
        centers.append((x0, y0, z0))

    def u0_fn(lam, th):
        x = np.cos(lam) * np.sin(th)
        y = np.sin(lam) * np.sin(th)
        z = np.cos(th)
        out = np.zeros(np.shape(np.asarray(lam)))
        for (x0, y0, z0) in centers:
            out = out + np.exp(-30 * ((x - x0)**2 + (y - y0)**2
                                      + (z - z0)**2))
        return out

    _plot_fn(u0_fn, clim=(0, 1))
    pairs, u_c, a0 = _bdf2_run(u0_fn, 0.01, 1.0, clim=(0, 1), lmax=40)
    # Mean over the sphere = a_00 / sqrt(4 pi) / (4 pi) * 4 pi.
    i00 = pairs.index((0, 0))
    mean0 = a0[i00] / np.sqrt(4 * np.pi)
    meanT = u_c[i00] / np.sqrt(4 * np.pi)
    print("ans =")
    print(f"     {abs(mean0 - meanT):.15e}")


if __name__ == "__main__":
    run()
