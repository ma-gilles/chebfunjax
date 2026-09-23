"""An ellipse rolling around another ellipse.

Translation of geom/Ellipses.m by Nick Trefethen
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

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun1d.chebfun import ode113
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'geom')
FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(_IMG, f"Ellipses_{FIG[0]:02d}.png"))
    plt.close(fig)


def _axes():
    fig, ax = plt.subplots()
    return fig, ax


def _square(ax):
    ax.axis([-3, 3, -3, 3])
    ax.set_aspect("equal")        # axis square


def run():
    os.makedirs(_IMG, exist_ok=True)

    tic = time.time()
    L1 = 3
    L2 = 2
    theta1 = lambda z1: np.arctan2(z1.imag, z1.real / L1)  # noqa: E731
    theta2 = lambda z2: np.arctan2(z2.imag, z2.real / L2)  # noqa: E731

    def ode1(t, z1):
        th = theta1(z1)
        return ((-L1 * np.sin(th) + 1j * np.cos(th))
                / np.sqrt(L1**2 * np.sin(th)**2 + np.cos(th)**2))

    def ode2(t, z2):
        th = theta2(z2)
        return ((L2 * np.sin(th) - 1j * np.cos(th))
                / np.sqrt(L2**2 * np.sin(th)**2 + np.cos(th)**2))

    tmax = 7.5
    opts = dict(atol=1e-13, rtol=1e-13)
    z1 = ode113(ode1, (0, tmax), np.array([L1 / 2 + 0j]), **opts)
    z2 = ode113(ode2, (0, tmax), np.array([-L2 / 2 + 0j]), **opts)

    w = z1 - z2 * z1.diff() / z2.diff()
    fig, ax = _axes()
    matlab_plot(w, 'k', ax=ax)
    ax.grid(True)
    _square(ax)
    _save(fig)

    tfinal = float(np.asarray(w.restrict(5, 7.5).imag().roots())[0])
    print("tfinal =")
    print(f"   {tfinal:.15f}")

    trajectory_length = float(w.restrict(0, tfinal).diff().norm(1))
    print("trajectory_length =")
    print(f"  {trajectory_length:.15f}")

    print(f"Elapsed time is {time.time() - tic:.6f} seconds.")

    def ell2(t):
        return w(t) + z2 * (z1(t) - w(t)) / z2(t)

    tt = np.linspace(0, tmax, 2001)
    z1v = np.asarray(z1(tt))

    fig, ax = _axes()
    ax.fill(z1v.real, z1v.imag, 'b', edgecolor='k')
    _square(ax)
    for t in range(0, 7):
        matlab_plot(ell2(float(t)), 'r', ax=ax)
        wt = complex(np.asarray(w(float(t))))
        ax.plot(wt.real, wt.imag, '.k', markersize=12 * 0.6)
    matlab_plot(w, 'k', ax=ax)
    _square(ax)
    _save(fig)

    # The animation loop deletes every frame but the last (t = tmax).
    fig, ax = _axes()
    ax.fill(z1v.real, z1v.imag, 'b', edgecolor='k')
    _square(ax)
    matlab_plot(w, 'k', ax=ax)
    matlab_plot(ell2(tmax), 'r', ax=ax)
    wt = complex(np.asarray(w(tmax)))
    ax.plot(wt.real, wt.imag, '.k', markersize=18 * 0.6)
    _square(ax)
    _save(fig)


if __name__ == "__main__":
    run()
