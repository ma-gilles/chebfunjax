"""The heat equation on the unit disk.

Faithful replica of disk/HeatEqn.m by Heather Wilber, January 2017:
BDF1/BDF2 timestepping of the heat equation via complex-frequency
Helmholtz solves, an exact-decay validation on harmonic initial data,
and a Dirichlet steady-state comparison with the Poisson solve.

Original: https://www.chebfun.org/examples/disk/HeatEqn.html
Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'disk')


def _contour(u, title, stem, zmax=2.0):
    th = np.linspace(-np.pi, np.pi, 180)
    r = np.linspace(0, 1, 90)
    T, R = np.meshgrid(th, r)
    V = np.asarray(u(jnp.asarray(T.ravel()), jnp.asarray(R.ravel()))
                   ).reshape(T.shape)
    X, Y = R * np.cos(T), R * np.sin(T)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    h = ax.contourf(X, Y, np.real(V), levels=24, cmap="hot",
                    vmin=-zmax, vmax=zmax)
    fig.colorbar(h, ax=ax)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, stem + ".png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # -- Exact-decay validation on harmonic initial data -------------
    u0 = Diskfun.harmonic(8, 2) + Diskfun.harmonic(4, 4)
    _contour(u0, "initial condition", "HeatEqn_repl_01")

    lam1 = float(np.asarray(cj.chebfun(
        lambda x: jnp.asarray(jv(8, np.asarray(x))),
        domain=[15, 17]).roots())[0])
    lam2 = float(np.asarray(cj.chebfun(
        lambda x: jnp.asarray(jv(4, np.asarray(x))),
        domain=[16, 18]).roots())[0])
    alpha = 1.0 / (lam1 ** 2 + lam2 ** 2)
    bc = lambda t: 0 * t
    dt, tfinal = 0.01, 2.0
    nsteps = int(np.ceil(tfinal / dt))
    m = 20
    up = u0
    K = np.sqrt(1 / (dt * alpha)) * 1j
    u = Diskfun.helmholtz(up * (K ** 2), K, bc, m, m)
    K = np.sqrt(3 / (2 * dt * alpha)) * 1j
    fignum = 2
    for n in range(2, nsteps + 1):
        rhs = (u * 4.0 - up) * (K ** 2 / 3.0)
        up = u
        u = Diskfun.helmholtz(rhs, K, bc, m, m)
        if n % 50 == 0:
            _contour(u, f"Time {n * dt:1.2f}",
                     f"HeatEqn_repl_{fignum:02d}")
            fignum += 1

    utrue = (Diskfun.harmonic(8, 2) * np.exp(-lam1 ** 2 * alpha * tfinal)
             + Diskfun.harmonic(4, 4)
             * np.exp(-lam2 ** 2 * alpha * tfinal))
    # L2 norm of the error by tensor quadrature (Diskfun.norm of a
    # difference currently NaNs on the evaluation grid -- ledgered bug;
    # this computes the same integral quantity directly).
    th = np.linspace(-np.pi, np.pi, 256, endpoint=False)
    from numpy.polynomial.legendre import leggauss

    xg, wg = leggauss(64)
    rq = 0.5 * (xg + 1.0)   # Gauss nodes on [0, 1]
    wq = 0.5 * wg
    T, R = np.meshgrid(th, rq)
    diff2 = np.abs(np.asarray(u(jnp.asarray(T.ravel()),
                                jnp.asarray(R.ravel()))
                              - utrue(jnp.asarray(T.ravel()),
                                      jnp.asarray(R.ravel()))
                              ).reshape(T.shape)) ** 2
    val = np.sqrt((2 * np.pi / len(th))
                  * float(wq @ (diff2 * R).sum(axis=1)))
    print("ans =")
    print(f"     {val:.15e}")

    # -- Dirichlet steady state --------------------------------------
    xs = [-1.0, 0.0, 1.0, 0.0]
    ys = [0.0, -1.0, 0.0, 1.0]
    ms = [6.0, 5.0, 8.0, 30.0]
    def _cart(g):
        # our Diskfun constructor takes (theta, r); wrap Cartesian g(x,y)
        return Diskfun.from_function(
            lambda t, r, _g=g: _g(r * jnp.cos(t), r * jnp.sin(t)))

    u0 = _cart(lambda x, y: 60 * jnp.exp(-5 * (x + 0.2) ** 2
                                         - 5 * (y + 0.2) ** 2))
    for xi, yi, mi in zip(xs, ys, ms):
        u0 = u0 + _cart(
            lambda x, y, _x=xi, _y=yi, _m=mi:
            _m * jnp.exp(-20 * (x - _x) ** 2 - 20 * (y - _y) ** 2))
    g = u0.restrict_boundary() if hasattr(u0, "restrict_boundary") else None
    if g is None:
        # boundary trace g(theta) = u0 at r = 1
        g = cj.chebfun(
            lambda t: u0(t, jnp.ones_like(t)),
            domain=[-np.pi, np.pi], trig=True)
    zero = Diskfun.from_function(lambda t, r: 0 * t)
    u = Diskfun.poisson(zero, g, 128)
    print("u =")
    print(repr(u))
    # max2 returns the max value; locate the maximiser on a dense grid
    maxu = float(u.max2())
    thq = np.linspace(-np.pi, np.pi, 2000)
    rq2 = np.linspace(0, 1, 400)
    Tq, Rq = np.meshgrid(thq, rq2)
    Vq = np.asarray(u(jnp.asarray(Tq.ravel()),
                      jnp.asarray(Rq.ravel()))).reshape(Tq.shape)
    iq = np.unravel_index(int(np.argmax(Vq)), Vq.shape)
    mxth = float(Tq[iq])
    print("mxu =")
    print(f"  {maxu:.15f}   {mxth:.15f}")
    (xming, ming), (xmaxg, maxg) = g.minandmax()
    print("mxg =")
    print(f"  {float(maxg):.15f}   {float(xmaxg):.15f}")
    return True


if __name__ == "__main__":
    run()
