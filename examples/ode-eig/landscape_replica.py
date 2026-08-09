"""Landscape function and localization of eigenfunctions.

Faithful replica of ode-eig/Landscape.m by Nick Trefethen (August
2021): a random sequence of square wells, the first six localized
eigenfunctions of the periodic Schroedinger operator, the landscape
function u = H\\1, and the effective potential W = 1/u whose local
minima order the lowest eigenvalues.

The well edges xk come from MATLAB's rng(2) randn sequence (dumped
from MATLAB R2025b, since MATLAB's and numpy's randn streams differ
even for equal seeds).

Original: https://www.chebfun.org/examples/ode-eig/Landscape.html
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

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-eig')

# MATLAB: d = 80; xk = 1:3.7:d; rng(2), xk = xk + .5*randn(1,length(xk));
XK = np.array([
    0.937886426047406, 3.429238311174607, 8.538604054973048,
    12.002010330902774, 15.701893882209564, 19.347135877128942,
    22.635534939475189, 26.997123146831470, 30.296467249108129,
    33.885776937693173, 38.267918560101165, 41.754734106232810,
    44.838955029732873, 49.123020769850207, 52.180715418830296,
    56.819089487692096, 60.772610562725454, 63.892040873720042,
    67.930481872227247, 70.027234668746672, 75.006243503157719,
    78.191872141839397])
D = 80.0


def _V_vals(t):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.zeros_like(t)
    for k in range(0, len(XK), 2):
        out += (t >= XK[k]).astype(float) - (t >= XK[k + 1]).astype(float)
    return out


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    V = chebfun(_V_vals, domain=tuple([0.0] + list(np.sort(XK)) + [D]))
    H = Chebop(lambda phi: -phi.diff(2) + V * phi, domain=(0, D))
    H.bc = "periodic"
    lam, F = H.eigs(k=6, sigma="SR", return_eigenfunctions=True)
    lam = np.asarray(lam).real
    idx = np.argsort(lam)
    e, F = lam[idx], [F[i] for i in idx]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    xx = np.linspace(0, D, 600)
    grey = (0.9, 0.9, 0.9)

    # Potential + first 6 eigenfunctions, cut off below |f| = 0.1.
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.fill_between(xx, _V_vals(xx), color=grey, zorder=0)
    ax.set_ylim(0, 1.19)
    ax.set_xlim(0, D)
    ax.grid(True)
    for k in range(6):
        ff = np.asarray(F[k](xx)).real
        if np.mean(ff) < 0:
            ff = -ff
        big = np.where(np.abs(ff) > 0.1)[0]
        i1 = max(big[0] - 5, 0)
        i2 = min(big[-1] + 5, len(ff))
        ffk = e[k] + 0.3 * ff
        ffk[:i1] = np.nan
        ffk[i2:] = np.nan
        ax.plot(xx, ffk, lw=2, color=colors[k % len(colors)])
        pos = np.nanargmax(ffk)
        ax.text(xx[pos], ffk[pos] + .05, str(k + 1),
                color=colors[k % len(colors)], ha="center", fontsize=12)
    ax.set_title("potential V and first 6 eigenfunctions")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Landscape_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("e =")
    for v in e:
        print(f"   {v:.15f}")

    # The landscape function u = H\1 and scaled eigenfunctions under it.
    u = H.solve(1.0)
    uu = np.asarray(u(xx)).real
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.4))
    for k, ax in enumerate(axes.ravel()):
        ff = np.asarray(F[k](xx)).real
        ff = ff / (e[k] * np.max(np.abs(ff)))
        ax.plot(xx, uu, 'k', lw=0.7)
        ax.plot(xx, np.abs(ff), lw=1, color=colors[k % len(colors)])
        ax.grid(True)
        ax.set_title(f"k = {k + 1}", color=colors[k % len(colors)])
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Landscape_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Effective potential W = 1/u: its local minima order the eigenvalues.
    xf = np.linspace(0, D, 8000)
    Wv = 1.0 / np.asarray(u(xf)).real
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.fill_between(xx, _V_vals(xx), color=grey, zorder=0)
    ax.set_ylim(0, 1.19)
    ax.set_xlim(0, D)
    ax.grid(True)
    ax.plot(xf, Wv, 'k', lw=1.2)
    ax.set_title("effective potential W")
    # Local minima of W (interior sign changes of the discrete gradient).
    dW = np.diff(Wv)
    imin = np.where((dW[:-1] < 0) & (dW[1:] >= 0))[0] + 1
    vals, poss = Wv[imin], xf[imin]
    order = np.argsort(vals)
    for k in range(6):
        ax.text(poss[order[k]], vals[order[k]] - .05, str(k + 1),
                color=colors[k % len(colors)], ha="center", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Landscape_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
