"""Optimization of the Rosenbrock function.

Faithful replica of opt/Rosenbrock.m by Nick Trefethen
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

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'opt')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Rosenbrock_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
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

    f = lambda x, y: (1 - x)**2 + 100 * (y - x**2)**2  # noqa: E731
    x = np.linspace(-1.5, 1.5, 100)
    y = np.linspace(-1, 3, 100)
    xx, yy = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    cs = ax.contour(x, y, f(xx, yy), levels=np.arange(10, 301, 10))
    fig.colorbar(cs, ax=ax)
    ax.axis([-1.5, 1.5, -1, 3])
    ax.set_title("Rosenbrock function f(x,y)", fontsize=12)

    fminx, minf, minx, miny = _nested_min(f, (-1.5, 1.5), (-1, 3))
    ax.plot(minx, miny, '.k', ms=12)
    _save(fig)

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
    print(f"    {minf:.15e}")
    print("minx =")
    print(f"   {minx:.15f}")
    print("miny =")
    print(f"   {miny:.15f}")

    # a wigglier function
    f2 = lambda x, y: (np.exp(x - 2 * x**2 - y**2)  # noqa: E731
                       * np.sin(6 * (x + y + x * y**2)))
    x = np.linspace(-1, 1, 100)
    xx, yy = np.meshgrid(x, x)
    t0 = time.time()
    fminx2, minf2, minx2, miny2 = _nested_min(f2, (-1.0, 1.0),
                                              (-1, 1))
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")
    print("breakpoints =")
    for v in [float(b) for b in fminx2.domain.breakpoints]:
        print(f"  {v:.15f}")

    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    cs = ax.contour(x, x, f2(xx, yy), 30)
    fig.colorbar(cs, ax=ax)
    ax.axis([-1, 1, -1, 1])
    ax.set_title("f(x,y)", fontsize=12)
    ax.plot(minx2, miny2, '.k', ms=12)
    _save(fig)

    xs = np.linspace(-1, 1, 600)
    fig, ax2 = plt.subplots(figsize=(9.0, 4.4))
    ax2.plot(xs, np.asarray(fminx2(xs)), lw=1.4)
    ax2.set_xlabel("x")
    ax2.set_ylabel("min_y(f(x,y))")
    ax2.grid(True)
    _save(fig)
    print("minf =")
    print(f"  {minf2:.15f}")
    print("minx =")
    print(f"   {minx2:.15f}")
    print("miny =")
    print(f"   {miny2:.15f}")


if __name__ == "__main__":
    run()
