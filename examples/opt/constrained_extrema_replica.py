"""Constrained extrema via composition.

Faithful replica of opt/ConstrainedExtrema.m by Hrothgar
(October 2013): extrema of multivariate functions along constraint
curves and surfaces, computed by composing with parametrizations —
no Lagrange multipliers required.

Original: https://www.chebfun.org/examples/opt/ConstrainedExtrema.html
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
from chebfunjax.utils.gallery2 import gallery2

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'opt')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"ConstrainedExtrema_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # extrema of x^2 - y^2 on the unit circle
    g = cj.chebfun2(lambda x, y: x**2 - y**2)
    h = cj.chebfun(lambda t: jnp.asarray(
        g(jnp.cos(t), jnp.sin(t))), domain=(0.0, 2 * np.pi))
    poss, vals = h.minandmax(flag="local")
    poss, vals = np.asarray(poss), np.asarray(vals)
    print("Y =")
    for v in vals:
        print(f"   {v:.15f}")
    print("X =")
    for p in poss:
        print(f"   {p:.15f}")
    print("X (on circle) =")
    for p in poss:
        print(f"   {np.cos(p):>18.15f}   {np.sin(p):>18.15f}")

    # the SIAM 100-digit challenge function on the circle
    gch = gallery2("challenge")
    h2 = cj.chebfun(lambda t: jnp.asarray(
        gch(jnp.cos(t), jnp.sin(t))), domain=(0.0, 2 * np.pi))
    pmin, vmin = h2.min()
    pmax, vmax = h2.max()
    print("Y =")
    print(f"  {float(vmin):.15f}")
    print(f"   {float(vmax):.15f}")
    print("Xh =")
    print(f"   {float(pmin):.15f}")
    print(f"   {float(pmax):.15f}")
    print("X =")
    for p in (float(pmin), float(pmax)):
        print(f"   {np.cos(p):>18.15f}   {np.sin(p):>18.15f}")

    xs = np.linspace(-1, 1, 240)
    X, Y = np.meshgrid(xs, xs)
    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    Z = np.asarray(gch(jnp.asarray(X), jnp.asarray(Y)))
    ax.contourf(X, Y, Z, 4)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), 'k-', lw=3)
    for p, lab in ((float(pmin), "min"), (float(pmax), "max")):
        ax.plot(np.cos(p), np.sin(p), 'ko', ms=9)
        ax.text(np.cos(p), np.sin(p), "  " + lab, color='w',
                fontweight='bold', fontsize=16)
    ax.set_aspect("equal")
    _save(fig)

    ts = np.linspace(0, 2 * np.pi, 700)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(ts, np.asarray(h2(ts)), lw=1.4)
    ax.plot([float(pmin), float(pmax)],
            [float(vmin), float(vmax)], 'ko', mfc='k')
    ax.grid(True)
    _save(fig)

    # extrema of x+y+z on the surface (x, y, x^3+y^2)
    hsurf = cj.chebfun2(lambda x, y: x + y + (x**3 + y**2))
    (vmin2, vmax2), locs = hsurf.minandmax2()
    print("Y =")
    print(f"  {float(vmin2):.15f}   {float(vmax2):.15f}")
    print("X =")
    for k in range(2):
        xk, yk = float(locs[k][0]), float(locs[k][1])
        print(f"  {xk:>18.15f} {yk:>18.15f} "
              f"{xk**3 + yk**2:>18.15f}")

    # extrema of x^3 + cos(5x) - y^2 on a rotated square
    def fmap(x, y):
        return x - y, x + y

    gsq = cj.chebfun2(
        lambda x, y: (x - y)**3 + jnp.cos(5 * (x - y))
        - (x + y)**2, domain=(-0.5, 0.5, -0.5, 0.5))
    (vmin3, vmax3), locs3 = gsq.minandmax2()
    print("Y =")
    print(f"  {float(vmin3):.15f}   {float(vmax3):.15f}")
    print("Xmin/Xmax (in x-y coords) =")
    for k in range(2):
        u, v = float(locs3[k][0]), float(locs3[k][1])
        a, b = fmap(u, v)
        print(f"  {a:>18.15f} {b:>18.15f}")


if __name__ == "__main__":
    run()
