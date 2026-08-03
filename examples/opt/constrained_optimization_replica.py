"""Constrained optimization.

Faithful replica of opt/ConstrainedOptimization.m by Alex Townsend
(March 2013): maximizing objectives subject to set constraints via
indicator functions, and a 2D objective on a heart-shaped region via
gradient critical points plus boundary maximization.

Original: https://www.chebfun.org/examples/opt/ConstrainedOptimization.html
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
from matplotlib.path import Path

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
        _IMG, f"ConstrainedOptimization_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # objective on prime-indexed unit intervals
    obj = lambda t: np.sin(t)**2 + np.sin(t**2)  # noqa: E731
    primes = [2, 3, 5, 7]
    ts = np.linspace(0, 10, 4000)
    mask = np.zeros_like(ts)
    for p in primes:
        mask += ((ts > p) & (ts < p + 1))
    gv = obj(ts) * mask
    # constrained max: per-interval chebfun max
    best = (-np.inf, None)
    for p in primes:
        g = cj.chebfun(lambda t: jnp.sin(t)**2 + jnp.sin(t**2),
                       domain=(float(p), float(p + 1)))
        pos, val = g.max()
        if float(val) > best[0]:
            best = (float(val), float(pos))
    mx, loc = best
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    ax.plot(ts, gv, lw=1.4)
    ax.plot(loc, mx, 'r.', ms=14)
    ax.set_ylim(-2, 3)
    ax.set_title(f"constrained maximum = {mx:1.3f}", fontsize=13)
    _save(fig)
    print(f"prime-interval max = {mx:.15f} at {loc:.15f}")

    # constraint |sin(10x)| < 1/2
    allowed = np.abs(np.sin(10 * ts)) < 0.5
    gv2 = obj(ts) * allowed
    # boundaries where |sin(10x)| = 1/2
    from scipy.optimize import brentq
    fbnd = lambda t: np.abs(np.sin(10 * t)) - 0.5  # noqa: E731
    edges = [0.0]
    grid = np.linspace(0, 10, 20001)
    fb = fbnd(grid)
    for i in range(len(grid) - 1):
        if fb[i] * fb[i + 1] < 0:
            edges.append(brentq(fbnd, grid[i], grid[i + 1]))
    edges.append(10.0)
    best2 = (-np.inf, None)
    for a, b in zip(edges[:-1], edges[1:]):
        mid = (a + b) / 2
        if np.abs(np.sin(10 * mid)) < 0.5:
            g = cj.chebfun(lambda t: jnp.sin(t)**2 + jnp.sin(t**2),
                           domain=(float(a), float(b)))
            pos, val = g.max()
            if float(val) > best2[0]:
                best2 = (float(val), float(pos))
    mx2, loc2 = best2
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    ax.plot(ts, gv2, lw=1.0)
    ax.plot(loc2, mx2, 'r.', ms=14)
    ax.set_ylim(-2, 3)
    ax.set_title(f"constrained maximum = {mx2:1.3f}", fontsize=13)
    _save(fig)
    print(f"sine-constraint max = {mx2:.15f} at {loc2:.15f}")

    # 2D objective on a heart-shaped region
    tt = np.linspace(0, 2 * np.pi, 1200)
    cx = 2 * np.sin(tt)
    cy = (2 * np.cos(tt) - 0.5 * np.cos(2 * tt)
          - 0.25 * np.cos(3 * tt) - 0.125 * np.cos(4 * tt))
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    ax.plot(cx, cy, 'k-', lw=1.6)
    ax.axis([-3, 3, -3, 3])
    ax.set_aspect("equal")
    ax.set_title("Constraint", fontsize=13)
    _save(fig)

    F = cj.chebfun2(lambda x, y: jnp.cos((x - 0.1) * y)**2
                    + x * jnp.sin(3 * x + y),
                    domain=(-3, 3, -3, 3))
    Fx = F.diff(dim=2)
    Fy = F.diff(dim=1)
    r = np.atleast_2d(np.asarray(Fx.roots(Fy, method="ms")))
    path = Path(np.column_stack([cx, cy]))
    inside = path.contains_points(r)
    r = r[inside]
    vals_in = np.asarray(F(jnp.asarray(r[:, 0]),
                           jnp.asarray(r[:, 1])))
    max_inside = float(np.max(vals_in))
    bvals = np.asarray(F(jnp.asarray(cx), jnp.asarray(cy)))
    max_boundary = float(np.max(bvals))
    max_overall = max(max_inside, max_boundary)
    print("max_overall =")
    print(f"   {max_overall:.15f}")

    xs = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(xs, xs)
    Z = np.asarray(F(jnp.asarray(X), jnp.asarray(Y)))
    fig, ax = plt.subplots(figsize=(7.8, 7.0))
    cs = ax.contour(X, Y, Z, 20)
    fig.colorbar(cs, ax=ax)
    ax.plot(cx, cy, 'k-', lw=1.6)
    ax.plot(r[:, 0], r[:, 1], '.k', ms=10)
    k = int(np.argmax(vals_in))
    ax.plot(r[k, 0], r[k, 1], 'r.', ms=20)
    ax.set_aspect("equal")
    ax.axis([-3, 3, -3, 3])
    ax.set_title(f"Overall maximum = {max_overall:1.3f}",
                 fontsize=13)
    _save(fig)


if __name__ == "__main__":
    run()
