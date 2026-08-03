"""The lowest position of a resting needle.

Faithful replica of opt/Needle.m by Nick Trefethen (October 2012):
a needle of length 1 rests on a bumpy landscape h(s); for each
horizontal position x and angle theta the resting height is
max(h - needle-line); the overall lowest resting position is found
by optimization over (x, theta).

Original: https://www.chebfun.org/examples/opt/Needle.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'opt')

FIG = [0]


def h_of(s):
    return (0.1 * s**2 + 0.1 * np.sin(6 * s)
            + 0.03 * np.sin(12 * s))


def minfun(x, theta):
    r = 0.5 * np.cos(theta)
    s = np.linspace(x - r, x + r, 2001)
    v = h_of(s) - np.tan(theta) * (s - x)
    j = int(np.argmax(v))
    if 0 < j < len(s) - 1:
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(
            lambda t: -(h_of(t) - np.tan(theta) * (t - x)),
            bounds=(s[j - 1], s[j + 1]), method="bounded",
            options={"xatol": 1e-14})
        return float(-res.fun)
    return float(v[j])


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Needle_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plotneedle(ax, x, theta, title):
    y = minfun(x, theta)
    r = 0.5 * np.cos(theta)
    ss = np.linspace(-4, 4, 1500)
    ax.plot(ss, h_of(ss), 'b', lw=1)
    sn = np.linspace(x - r, x + r, 100)
    ax.plot(sn, y + np.tan(theta) * (sn - x), 'k', lw=1.6)
    ax.set_aspect("equal")
    ax.axis([-4, 4, -0.4, 2])
    ax.set_title(title, fontsize=12)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    ss = np.linspace(-4, 4, 1500)
    ax.plot(ss, h_of(ss), 'b', lw=1)
    ax.set_aspect("equal")
    ax.axis([-4, 4, -0.4, 2])
    _save(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.6))
    _plotneedle(axes[0], -0.6, -0.2,
                "needle with (x,theta) = (-0.6, -0.2)")
    _plotneedle(axes[1], 1.7, 1.0,
                "needle with (x,theta) = (1.7, 1)")
    _save(fig)

    npts = 25
    t0 = time.time()
    xg = np.linspace(-2, 2, npts)
    tg = np.linspace(-1.5, 1.5, npts)
    yy = np.array([[minfun(x, t) for x in xg] for t in tg])
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    cs = ax.contour(xg, tg, yy, 80)
    fig.colorbar(cs, ax=ax)
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("theta")
    ax.set_title(f"min value on grid: {yy.min():g}", fontsize=12)
    _save(fig)
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")

    xg2 = np.linspace(-0.8, 0.6, npts)
    tg2 = np.linspace(-0.5, 0, npts)
    yy2 = np.array([[minfun(x, t) for x in xg2] for t in tg2])
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    cs = ax.contour(xg2, tg2, yy2,
                    levels=np.arange(0.06, 0.121, 0.003))
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("theta")
    ax.set_title(f"min value on grid: {yy2.min():g}", fontsize=12)
    _save(fig)

    t0 = time.time()
    res = fmin(lambda v: minfun(v[0], v[1]), [0.41, -0.2],
               xtol=1e-14, ftol=1e-14, disp=False)
    yval = minfun(res[0], res[1])
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")
    print("yval =")
    print(f"   {yval:.15f}")

    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    _plotneedle(ax, res[0], res[1], "")
    ax.plot(res[0], yval, '.k', ms=10)
    ax.axis([-2, 2, -0.4, 1.2])
    ax.grid(True)
    _save(fig)


if __name__ == "__main__":
    run()
