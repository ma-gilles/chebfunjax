"""Optimization of the Rosenbrock function.

Translation of opt/Rosenbrock.m by Nick Trefethen
(September 2010): 2D minimization by nested 1D chebfun
minimizations, for the Rosenbrock function and for a wiggly
two-variable function.

Original: https://www.chebfun.org/examples/opt/Rosenbrock.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'opt')

FIG = [0]


def _fmt_scalar(x):
    """MATLAB format-long scalar display: bare '0' for exact zero,
    else %.15f in [1e-5, 1e5) and %.15e outside it."""
    if x == 0:
        return "0"
    if 1e-5 <= abs(x) < 1e5:
        return f"{x:.15f}"
    return f"{x:.15e}"


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"Rosenbrock_{FIG[0]:02d}.png"))
    plt.close(fig)


def _inner_min(f, ylo, yhi, x0):
    """Exact inner minimum over y for fixed x (grid + refine)."""
    yg = np.linspace(ylo, yhi, 800)
    vals = f(x0, yg)
    j = int(np.argmin(vals))
    lo = yg[max(0, j - 2)]
    hi = yg[min(len(yg) - 1, j + 2)]
    res = minimize_scalar(lambda t: float(f(x0, t)),
                          bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-14})
    return float(res.fun), float(res.x)


def _nested_min(f, xdom, ydom):
    def fminx0(x_arr):
        x_arr = np.atleast_1d(np.asarray(x_arr, dtype=float))
        out = np.empty_like(x_arr)
        for i, x0 in enumerate(x_arr.ravel()):
            out.ravel()[i] = _inner_min(f, *ydom, x0)[0]
        return out.reshape(np.shape(x_arr))

    fminx = cj.chebfun(lambda t: jnp.asarray(fminx0(np.asarray(t))),
                       domain=xdom, splitting=True)
    minx, minf = fminx.min()
    res = minimize_scalar(lambda t: float(fminx0(np.array([t]))[0]),
                          bounds=(max(xdom[0], float(minx) - 0.05),
                                  min(xdom[1], float(minx) + 0.05)),
                          method="bounded",
                          options={"xatol": 1e-14})
    minx, minf = float(res.x), float(res.fun)
    _, miny = _inner_min(f, *ydom, minx)
    return fminx, minf, minx, miny


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # MATLAB: figure, contour(...), hold on -- fig1 stays open (image 1,
    # no dot yet) while the slice-plot figure (image 2) is drawn; only
    # after that one is closed does MATLAB's "close, plot(dot)" land
    # back on fig1, producing a third snapshot (image 3) with the dot.
    f = lambda x, y: (1 - x)**2 + 100 * (y - x**2)**2  # noqa: E731
    x = np.linspace(-1.5, 1.5, 100)
    y = np.linspace(-1, 3, 100)
    xx, yy = np.meshgrid(x, y)
    fig1, ax1 = plt.subplots(figsize=(7.6, 7.0))
    cs = ax1.contour(x, y, f(xx, yy), levels=np.arange(10, 301, 10))
    fig1.colorbar(cs, ax=ax1)
    ax1.axis([-1.5, 1.5, -1, 3])
    ax1.set_title("Rosenbrock function f(x,y)", fontsize=12)
    fig1.set_facecolor("white")
    fig1.tight_layout()
    _savefig(fig1, os.path.join(_IMG, "Rosenbrock_01.png"))
    FIG[0] = 1

    fminx, minf, minx, miny = _nested_min(f, (-1.5, 1.5), (-1, 3))

    xs = np.linspace(-1.5, 1.5, 600)
    fig, ax2 = plt.subplots(figsize=(9.0, 4.4))
    ax2.plot(xs, np.asarray(fminx(xs)), lw=1.4)
    ax2.set_xlabel("x")
    ax2.set_ylabel("min_y(f(x,y))")
    ax2.set_title("minimum of f(x,y) along vertical slices",
                  fontsize=12)
    ax2.grid(True)
    _save(fig)
    print("minf =")
    print(f"    {_fmt_scalar(minf)}")
    print("minx =")
    print(f"   {minx:.15f}")

    # [minf,miny] = min(chebfun(@(y) f(minx,y), [-1 3])) -- a second,
    # separate MATLAB display re-minimizing over y at the fixed minx.
    minf_y, miny = _inner_min(f, -1.0, 3.0, minx)
    print("minf =")
    print(f"    {_fmt_scalar(minf_y)}")
    print("miny =")
    print(f"   {miny:.15f}")

    # MATLAB: close, plot(minx,miny,'.k',MS,14) -- closes the slice
    # figure, so the dot lands back on fig1 (still open with hold on).
    ax1.plot(minx, miny, '.k', ms=12)
    _save(fig1)

    # a wigglier function
    f2 = lambda x, y: (np.exp(x - 2 * x**2 - y**2)  # noqa: E731
                       * np.sin(6 * (x + y + x * y**2)))
    x = np.linspace(-1, 1, 100)
    xx, yy = np.meshgrid(x, x)

    # MATLAB draws the plain contour (image 4) in its own cell, before
    # fminx2/minx2/miny2 are even computed.
    fig2, ax2c = plt.subplots(figsize=(7.6, 7.0))
    cs = ax2c.contour(x, x, f2(xx, yy), 30)
    fig2.colorbar(cs, ax=ax2c)
    ax2c.axis([-1, 1, -1, 1])
    ax2c.set_title("f(x,y)", fontsize=12)
    _save(fig2)

    t0 = time.time()
    fminx2, minf2, minx2, miny2 = _nested_min(f2, (-1.0, 1.0),
                                              (-1, 1))
    xs = np.linspace(-1, 1, 600)
    fig, ax2 = plt.subplots(figsize=(9.0, 4.4))
    ax2.plot(xs, np.asarray(fminx2(xs)), lw=1.4)
    ax2.set_xlabel("x")
    ax2.set_ylabel("min_y(f(x,y))")
    ax2.grid(True)
    _save(fig)
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")

    ends = [float(b) for b in fminx2.domain.breakpoints]
    print("ans =")
    print("  Columns 1 through 3")
    print("".join(f"{v:20.15f}" for v in ends[:3]))
    print("  Column 4")
    print(f"{ends[3]:20.15f}")

    # [minf,minx] = min(fminx) | [minf,miny] = min(chebfun(@(y)
    # f(minx,y), [-1 3])) -- a single MATLAB cell with two
    # semicolon-free statements, so all four values print together.
    print("minf =")
    print(f"  {minf2:.15f}")
    print("minx =")
    print(f"   {minx2:.15f}")
    minf2_y, miny2 = _inner_min(f2, -1.0, 3.0, minx2)
    print("minf =")
    print(f"  {minf2_y:.15f}")
    print("miny =")
    print(f"   {miny2:.15f}")

    # MATLAB: close, plot(minx2,miny2,'.k',MS,14) -- lands the dot back
    # on the (still open) f2 contour figure.
    ax2c.plot(minx2, miny2, '.k', ms=12)
    _save(fig2)


if __name__ == "__main__":
    run()
