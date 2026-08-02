"""Best L1 polynomial approximation.

Faithful replica of approx/BestL1.m by Yuji Nakatsukasa (June 2019):
L-infinity, L2, and L1 polynomial fits of a wiggly function and of
|x - 1/4| — the L1 error localizes near the singularity.

Original: https://www.chebfun.org/examples/approx/BestL1.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.minimax import minimax

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

FIG = [0]


def _plot(xs, curves, title, fname_stub, ylim):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    for ys, kw in curves:
        ax.plot(xs, ys, **kw)
    ax.set_ylim(*ylim)
    ax.grid(True)
    ax.set_title(title, fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"BestL1_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # Wiggly function on [0, 14], degree 100
    dom = (0.0, 14.0)
    deg = 100
    fop = lambda x: jnp.sin(x)**2 + jnp.sin(x**2)  # noqa: E731
    f = cj.chebfun(fop, domain=dom)
    xs = np.linspace(0, 14, 4000)
    fv = np.asarray(f(jnp.asarray(xs)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimax(fop, deg, domain=dom, tol=1e-8)
    pinf = cj.chebfun(jnp.asarray(res.coeffs), coeffs=True, domain=dom)
    pv = np.asarray(pinf(jnp.asarray(xs)))
    _plot(xs, [(fv, dict(lw=1.1)), (pv, dict(lw=1.1))],
          "f and Linfty approximant", "", (-3, 3))
    _plot(xs, [(fv - pv, dict(color='k', lw=1.1))],
          "error of Linfty approximant", "", (-3, 3))

    p2 = f.polyfit(deg)
    p2v = np.asarray(p2(jnp.asarray(xs)))
    _plot(xs, [(fv, dict(lw=1.1)), (p2v, dict(lw=1.1))],
          "f and L2 approximant", "", (-3, 3))
    _plot(xs, [(fv - p2v, dict(color='k', lw=1.1))],
          "error of L2 approximant", "", (-3, 3))

    p1 = f.polyfitL1(deg)
    p1v = np.asarray(p1(jnp.asarray(xs)))
    _plot(xs, [(fv, dict(lw=1.1)), (p1v, dict(lw=1.1))],
          "f and L1 approximant", "", (-3, 3))
    _plot(xs, [(fv - p1v, dict(color='k', lw=1.1))],
          "error of L1 approximant", "", (-3, 3))
    print(f"wiggly: |f-pinf| {np.max(np.abs(fv-pv)):.3f}  "
          f"|f-p2| {np.max(np.abs(fv-p2v)):.3f}  "
          f"|f-p1| {np.max(np.abs(fv-p1v)):.3f}")

    # |x - 1/4| on [-1, 1], degree 80
    g = cj.chebfun(lambda t: jnp.abs(t - 0.25),
                   domain=[-1.0, 0.25, 1.0])
    gop = lambda t: jnp.abs(t - 0.25)  # noqa: E731
    xs = np.linspace(-1, 1, 6000)
    gv = np.abs(xs - 0.25)
    deg = 80

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimax(gop, deg, breakpoints=[0.25])
    pinf = cj.chebfun(jnp.asarray(res.coeffs), coeffs=True)
    _plot(xs, [(gv - np.asarray(pinf(jnp.asarray(xs))),
                dict(color='k', lw=1.0))],
          "Linf error", "", (-1e-2, 1e-2))

    p2 = g.polyfit(deg)
    _plot(xs, [(gv - np.asarray(p2(jnp.asarray(xs))),
                dict(color='k', lw=1.0))],
          "L2 error", "", (-1e-2, 1e-2))

    p1 = g.polyfitL1(deg)
    e1 = gv - np.asarray(p1(jnp.asarray(xs)))
    _plot(xs, [(e1, dict(color='k', lw=1.0))],
          "L1 error", "", (-1e-2, 1e-2))
    _plot(xs, [(e1, dict(color='k', lw=1.0))],
          "closeup", "", (-1e-4, 1e-4))
    print(f"absx deg80: sup Linf-err "
          f"{np.max(np.abs(gv - np.asarray(pinf(jnp.asarray(xs))))):.2e}"
          f"  sup L1-err {np.max(np.abs(e1)):.2e}")


if __name__ == "__main__":
    run()
