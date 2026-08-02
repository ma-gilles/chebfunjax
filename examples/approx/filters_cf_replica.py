"""Digital filters via CF approximation.

Faithful replica of approx/FiltersCF.m by Nick Trefethen (September
2014): high-degree polynomial CF approximations of a square-wave
filter shape, and of its mollified (triangular-kernel) smoothing.

Original: https://www.chebfun.org/examples/approx/FiltersCF.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.cfpade import cf

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

XS = np.linspace(-1, 1, 6000)
FIG = [0]


def _plot(curves, fname_note="", ylim=(-1, 2)):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    for ys, kw in curves:
        ax.plot(XS, ys, **kw)
    ax.axis([-1, 1, *ylim])
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"FiltersCF_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    def fop(x):
        return (jnp.where(jnp.abs(x) < 0.3, 1.0, 0.0)
                + jnp.where(jnp.abs(x - 0.7) < 0.1, 1.0, 0.0)
                + jnp.where(jnp.abs(x + 0.65) < 0.2, 1.0, 0.0))

    f = cj.chebfun(fop, domain=[-1.0, -0.85, -0.45, -0.3, 0.3, 0.6,
                                0.8, 1.0])
    fv = np.asarray(fop(jnp.asarray(XS)))
    _plot([(fv, dict(color='k', lw=1.2))])

    t0 = time.time()
    for m in (100, 1000):
        p, q, rh, s = cf(f, m, 0, max(100, 2 * m))
        pv = np.asarray(rh(jnp.asarray(XS)))
        _plot([(fv, dict(color='k', lw=1.2)),
               (pv, dict(color='r', lw=1.2))])
        print(f"m={m}: max|f-p| = {np.max(np.abs(fv - pv)):.3f}")
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")

    # Mollified filter: convolution with the triangular kernel
    # phi = 50 - 2500|s| on [-0.02, 0.02].  (Chebfun.conv currently
    # chokes on this tiny-kernel piecewise case -- ledgered; the
    # convolution of a step function with phi has the exact closed
    # form below via the kernel's CDF Phi.)
    d = 0.02

    def Phi(u):
        u = jnp.asarray(u)
        neg = 50 * u + 1250 * u**2 + 0.5
        pos = 50 * u - 1250 * u**2 + 0.5
        val = jnp.where(u <= 0, neg, pos)
        return jnp.where(u < -d, 0.0, jnp.where(u > d, 1.0, val))

    intervals = [(-0.85, -0.45), (-0.3, 0.3), (0.6, 0.8)]

    def f2op(x):
        out = 0.0
        for a_, b_ in intervals:
            out = out + Phi(x - a_) - Phi(x - b_)
        return out

    brk = sorted({-1.0, 1.0} | {v + s0 * d for a_, b_ in intervals
                                for v in (a_, b_) for s0 in (-1, 1)})
    f2 = cj.chebfun(f2op, domain=brk)
    f2v = np.asarray(f2op(jnp.asarray(XS)))
    for m in (100, 200):
        p, q, rh, s = cf(f2, m, 0, max(100, 2 * m))
        pv = np.asarray(rh(jnp.asarray(XS)))
        _plot([(f2v, dict(color='k', lw=1.2)),
               (pv, dict(color='r', lw=1.2))])
        print(f"mollified m={m}: max|f2-p| = "
              f"{np.max(np.abs(f2v - pv)):.2e}")

    _plot([(f2v - pv, dict(lw=1.2))], ylim=(-0.02, 0.02))


if __name__ == "__main__":
    run()
