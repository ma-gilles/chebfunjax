"""The six-hump camel function of Dixon and Szego.

Faithful replica of opt/DixonSzego.m by Nick Trefethen
(September 2010, revised 2016): global minimization of the six-hump
camel function, first by nested 1D chebfun minimization, then in one
step with chebfun2 min2.

Original: https://www.chebfun.org/examples/opt/DixonSzego.html
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'opt')


def f_np(x, y):
    return ((4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y
            + 4 * (y**2 - 1) * y**2)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1.25, 1.25, 100)
    xx, yy = np.meshgrid(x, y)
    ff = f_np(xx, yy)

    t0 = time.time()

    def fminx0(x0_arr):
        # inner minimum over y on [-1.25, 1.25]: df/dy = x + 16y^3 - 8y
        # is a cubic, solved exactly (same minimum as the 1D chebfun)
        x0_arr = np.atleast_1d(np.asarray(x0_arr, dtype=float))
        out = np.empty_like(x0_arr)
        for i, x0 in enumerate(x0_arr.ravel()):
            r = np.roots([16.0, 0.0, -8.0, x0])
            cand = [(-1.25), (1.25)] + [float(v.real) for v in r
                                        if abs(v.imag) < 1e-12
                                        and -1.25 <= v.real <= 1.25]
            out.ravel()[i] = min(f_np(x0, np.asarray(c))
                                 for c in cand)
        return out.reshape(np.shape(x0_arr))

    fminx = cj.chebfun(lambda t: jnp.asarray(fminx0(np.asarray(t))),
                       domain=(-2.0, 2.0), splitting=True)
    minx, minf = fminx.min()
    minx, minf = float(minx), float(minf)
    # polish the outer minimizer on the exact inner-min function
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(
        lambda t: float(fminx0(np.array([t]))[0]),
        bounds=(minx - 0.05, minx + 0.05), method="bounded",
        options={"xatol": 1e-14})
    minx, minf = float(res.x), float(res.fun)
    print("minf =")
    print(f"  {minf:.15f}")
    print("minx =")
    print(f"  {minx:.15f}")
    g = cj.chebfun(lambda t: jnp.asarray(f_np(minx, np.asarray(t))),
                   domain=(-1.0, 3.0))
    miny, minf2 = g.min()
    print("minf =")
    print(f"  {float(minf2):.15f}")
    print("miny =")
    print(f"   {float(miny):.15f}")
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    cs = ax.contour(x, y, ff, 30, linewidths=1.2)
    fig.colorbar(cs, ax=ax)
    ax.axis([-2, 2, -1.25, 1.25])
    ax.plot(minx, float(miny), '.k', ms=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "DixonSzego_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    t0 = time.time()
    F = cj.chebfun2(lambda a, b: (4 - 2.1 * a**2 + a**4 / 3) * a**2
                    + a * b + 4 * (b**2 - 1) * b**2,
                    domain=(-2, 2, -1.25, 1.25))
    v2, loc2 = F.min2()
    print("minf =")
    print(f"  {float(v2):.15f}")
    print("minx =")
    print(f"  {float(loc2[0]):.15f}   {float(loc2[1]):.15f}")
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    cs = ax.contour(x, y, ff, 30, linewidths=1.2)
    fig.colorbar(cs, ax=ax)
    ax.axis([-2, 2, -1.25, 1.25])
    ax.plot(float(loc2[0]), float(loc2[1]), '.k', ms=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "DixonSzego_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
