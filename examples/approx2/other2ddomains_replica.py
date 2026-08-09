"""Chebfun2 objects on non-rectangular domains.

Faithful replica of approx2/Other2DDomains.m (Townsend, 2013):
integral over a curve-bounded region via Green's theorem
(integral2(f, c)), a surface of revolution, functions on (warped)
sector domains in polar coordinates with Jacobian-weighted
integrals, and the shadow of the Klein bottle immersion where the
Jacobian degenerates.

Original: https://www.chebfun.org/examples/approx2/Other2DDomains.html
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
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx2')
FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG,
                             f"Other2DDomains_repl_{FIG[0]:02d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Volume over a curve-bounded region (Green's theorem).
    f = Chebfun2.from_function(lambda x, y: x**2 + y**2)
    c = chebfun(lambda t: np.cos(np.pi * t) + 1j * np.sin(2 * np.pi * t),
                domain=(-.5, .5))
    ts = np.linspace(-.5, .5, 800)
    z = np.asarray(c(ts))
    g = np.linspace(-1, 1, 300)
    X, Y = np.meshgrid(g, g)
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.contour(X, Y, np.asarray(f(X, Y)), 10)
    ax.plot(np.real(z), np.imag(z), 'k', lw=2)
    ax.set_aspect("equal")
    _save(fig)
    print(f"Volume enclosed by curve = {float(f.integral2(c)):.3f}")

    # Surface of revolution of a chebfun (MATLAB cylinder(f)).
    fr = chebfun(lambda x: x * (5 - x) + np.sin(np.pi * x),
                 domain=(0, 5))
    zz = np.linspace(0, 5, 300)
    th = np.linspace(0, 2 * np.pi, 200)
    ZZ, TH = np.meshgrid(zz, th)
    RR = np.asarray(fr(ZZ))
    fig = plt.figure(figsize=(7.6, 5.8))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(RR * np.cos(TH), RR * np.sin(TH), ZZ,
                    cmap="viridis", rstride=2, cstride=2, linewidth=0)
    _save(fig)

    # Sector domain.
    t1, t2, r1, r2 = np.pi / 6, 7 / 4 * np.pi, 1 / 2, 2
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    tt = np.linspace(t1, t2, 500)
    rr = np.linspace(r1, r2, 100)
    ax.plot(r2 * np.cos(tt), r2 * np.sin(tt), 'k', lw=2)
    ax.plot(r1 * np.cos(tt), r1 * np.sin(tt), 'k', lw=2)
    for tb in (t1, t2):
        ax.plot(rr * np.cos(tb), rr * np.sin(tb), 'k', lw=2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_title("Sector domain", fontsize=14)
    _save(fig)

    # Function on the sector in polar coordinates.
    dom = (r1, r2, t1, t2)
    x = Chebfun2.from_function(lambda r, t: r * np.cos(t), domain=dom)
    y = Chebfun2.from_function(lambda r, t: r * np.sin(t), domain=dom)
    RG, TG = np.meshgrid(np.linspace(r1, r2, 160),
                         np.linspace(t1, t2, 320))
    XG = RG * np.cos(TG)
    YG = RG * np.sin(TG)
    FG = np.cos(5 * XG * YG) + YG
    fig = plt.figure(figsize=(7.6, 5.8))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(XG, YG, FG, cmap="viridis", rstride=2, cstride=2,
                    linewidth=0)
    ax.set_zlim(-5, 5)
    ax.set_title("Function on sector domain", fontsize=14)
    _save(fig)

    # integral of f over the sector: jacobian of (x, y) is r.
    fpol = Chebfun2.from_function(
        lambda r, t: (np.cos(5 * (r * np.cos(t)) * (r * np.sin(t)))
                      + r * np.sin(t)) * r, domain=dom)
    print("ans =")
    print(f"   {float(fpol.sum2()):.15f}")

    # Warped sector domain.
    def rw(r, t):
        return r + .1 * np.cos(5 * t)

    XW = rw(RG, TG) * np.cos(TG)
    YW = rw(RG, TG) * np.sin(TG)
    FW = np.cos(5 * XW * YW) + YW
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    pc = ax.pcolormesh(XW, YW, FW, cmap="viridis", shading="auto")
    ax.set_aspect("equal")
    ax.set_title("Function on warped sector domain", fontsize=14)
    _save(fig)

    # Jacobian of the warped map (well-behaved).
    xw = Chebfun2.from_function(
        lambda r, t: rw(r, t) * np.cos(t), domain=dom)
    yw = Chebfun2.from_function(
        lambda r, t: rw(r, t) * np.sin(t), domain=dom)
    jac = (xw.diff(dim=2) * yw.diff(dim=1)
           - xw.diff(dim=1) * yw.diff(dim=2))
    JV = np.asarray(jac(RG, TG))
    fig = plt.figure(figsize=(7.6, 5.4))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(RG, TG, JV, cmap="viridis", rstride=2, cstride=2,
                    linewidth=0)
    ax.set_title("Jacobian", fontsize=14)
    _save(fig)

    # Shadow of the Klein bottle immersion.
    U = np.linspace(0, np.pi, 220)
    V = np.linspace(0, 2 * np.pi, 440)
    UU, VV = np.meshgrid(U, V)
    cu, cv, su = np.cos(UU), np.cos(VV), np.sin(UU)
    XK = -(2 / 15) * cu * (3 * cv - 30 * su + 90 * cu**4 * su
                           - 60 * cu**6 * su + 5 * cu * cv * su)
    YK = -(1 / 15) * su * (3 * cv - 3 * cu**2 * cv - 48 * cu**4 * cv
                           + 48 * cu**6 * cv - 60 * su
                           + 5 * cu * cv * su - 5 * cu**3 * cv * su
                           - 80 * cu**5 * cv * su + 80 * cu**7 * cv * su)
    FK = np.cos(5 * XK * YK) + YK
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    pc = ax.pcolormesh(XK, YK, FK, cmap="viridis", shading="auto")
    ax.set_aspect("equal")
    _save(fig)

    # Its Jacobian is singular.
    xk = Chebfun2.from_function(
        lambda u, v: -(2 / 15) * np.cos(u)
        * (3 * np.cos(v) - 30 * np.sin(u)
           + 90 * np.cos(u)**4 * np.sin(u)
           - 60 * np.cos(u)**6 * np.sin(u)
           + 5 * np.cos(u) * np.cos(v) * np.sin(u)),
        domain=(0, np.pi, 0, 2 * np.pi))
    yk = Chebfun2.from_function(
        lambda u, v: -(1 / 15) * np.sin(u)
        * (3 * np.cos(v) - 3 * np.cos(u)**2 * np.cos(v)
           - 48 * np.cos(u)**4 * np.cos(v)
           + 48 * np.cos(u)**6 * np.cos(v) - 60 * np.sin(u)
           + 5 * np.cos(u) * np.cos(v) * np.sin(u)
           - 5 * np.cos(u)**3 * np.cos(v) * np.sin(u)
           - 80 * np.cos(u)**5 * np.cos(v) * np.sin(u)
           + 80 * np.cos(u)**7 * np.cos(v) * np.sin(u)),
        domain=(0, np.pi, 0, 2 * np.pi))
    jk = (xk.diff(dim=2) * yk.diff(dim=1)
          - xk.diff(dim=1) * yk.diff(dim=2))
    JK = np.asarray(jk(UU, VV))
    fig = plt.figure(figsize=(7.6, 5.4))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(UU, VV, JK, cmap="viridis", rstride=2, cstride=2,
                    linewidth=0)
    ax.set_title("Jacobian", fontsize=14)
    _save(fig)


if __name__ == "__main__":
    run()
