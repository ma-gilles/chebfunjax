"""An ellipse rolling around another ellipse.

Faithful replica of geom/Ellipses.m by Nick Trefethen
(December 2015): two ellipses parametrized by arc length via ODEs;
one rolls without slipping around the other, and the center of
contact traces the curve w.

Original: https://www.chebfun.org/examples/geom/Ellipses.html
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
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'geom')

L1, L2 = 3.0, 2.0
TMAX = 7.5
FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Ellipses_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _ode1(t, y):
    z1 = y[0] + 1j * y[1]
    th = np.arctan2(z1.imag, z1.real / L1)
    d = (-L1 * np.sin(th) + 1j * np.cos(th)) / np.sqrt(
        L1**2 * np.sin(th)**2 + np.cos(th)**2)
    return [d.real, d.imag]


def _ode2(t, y):
    z2 = y[0] + 1j * y[1]
    th = np.arctan2(z2.imag, z2.real / L2)
    d = (L2 * np.sin(th) - 1j * np.cos(th)) / np.sqrt(
        L2**2 * np.sin(th)**2 + np.cos(th)**2)
    return [d.real, d.imag]


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()

    ts = np.linspace(0, TMAX, 4001)
    s1 = solve_ivp(_ode1, (0, TMAX), [L1 / 2, 0.0], t_eval=ts,
                   rtol=1e-13, atol=1e-13, method="DOP853")
    s2 = solve_ivp(_ode2, (0, TMAX), [-L2 / 2, 0.0], t_eval=ts,
                   rtol=1e-13, atol=1e-13, method="DOP853")
    z1 = s1.y[0] + 1j * s1.y[1]
    z2 = s2.y[0] + 1j * s2.y[1]
    d1 = np.array([_ode1(0, [v.real, v.imag]) for v in z1])
    d1 = d1[:, 0] + 1j * d1[:, 1]
    d2 = np.array([_ode2(0, [v.real, v.imag]) for v in z2])
    d2 = d2[:, 0] + 1j * d2[:, 1]
    w = z1 - z2 * d1 / d2

    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    ax.plot(w.real, w.imag, 'k', lw=1.2)
    ax.grid(True)
    ax.axis([-3, 3, -3, 3])
    ax.set_aspect("equal")
    _save(fig)

    wim = CubicSpline(ts, w.imag)
    mask = (ts >= 5) & (ts <= 7.5)
    tt = ts[mask]
    vv = w.imag[mask]
    tfinal = None
    for i in range(len(tt) - 1):
        if vv[i] * vv[i + 1] < 0:
            tfinal = brentq(wim, tt[i], tt[i + 1], xtol=1e-14)
            break
    print("tfinal =")
    print(f"   {tfinal:.15f}")

    wre = CubicSpline(ts, w.real)
    wimc = CubicSpline(ts, w.imag)
    tq = np.linspace(0, tfinal, 20000)
    speed = np.sqrt(wre(tq, 1)**2 + wimc(tq, 1)**2)
    traj = np.trapezoid(speed, tq)
    print("trajectory_length =")
    print(f"  {traj:.15f}")
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")

    # the rolling ellipse at several times
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    ax.fill(z1.real, z1.imag, 'b')
    ax.axis([-3, 3, -3, 3])
    ax.set_aspect("equal")
    zs2 = CubicSpline(ts, np.column_stack([z2.real, z2.imag]))
    zs1 = CubicSpline(ts, np.column_stack([z1.real, z1.imag]))
    ws = CubicSpline(ts, np.column_stack([w.real, w.imag]))
    for tv in range(0, 7):
        z1t = complex(*zs1(tv))
        z2t = complex(*zs2(tv))
        wt = complex(*ws(tv))
        ell2 = wt + z2 * (z1t - wt) / z2t
        ax.plot(ell2.real, ell2.imag, 'r', lw=0.9)
        ax.plot(wt.real, wt.imag, '.k', ms=9)
    ax.plot(w.real, w.imag, 'k', lw=1.2)
    _save(fig)


if __name__ == "__main__":
    run()
