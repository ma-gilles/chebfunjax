"""B-splines and convolution.

Faithful replica of approx/BSplineConv.m by Nick Trefethen (July
2012): B-splines of orders 0-4 built by iterated convolution of the
box function.

Original: https://www.chebfun.org/examples/approx/BSplineConv.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def _plot(B, pts, title, fname):
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    brk = [float(b) for b in B.domain.breakpoints]
    for lo, hi in zip(brk[:-1], brk[1:]):
        xs = np.linspace(lo + 1e-10, hi - 1e-10, 300)
        ax.plot(xs, np.asarray(B(jnp.asarray(xs))), 'C0', lw=1.6)
    vals = np.asarray(B(jnp.asarray(np.clip(pts, brk[0] + 1e-10,
                                            brk[-1] - 1e-10))))
    ax.plot(pts, vals, '.k', ms=12)
    ax.axis([-3, 3, -0.2, 1.2])
    ax.grid(True)
    ax.set_title(title, fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    B0 = cj.chebfun(lambda x: 1.0 + 0 * x, domain=(-0.5, 0.5))
    pts = np.array([-0.5, 0.5])
    _plot(B0, pts, "B-spline of order 0", "BSplineConv_repl_01.png")

    B = B0
    for n in range(1, 5):
        B = B0.conv(B)
        pts = np.concatenate([pts - 0.5, [np.max(pts) + 0.5]])
        _plot(B, np.sort(pts), f"B-spline of order {n}",
              f"BSplineConv_repl_{n+1:02d}.png")
        print(f"order {n}: support [{float(B.domain.a):g}, "
              f"{float(B.domain.b):g}], sum = {float(B.sum()):.15f}")


if __name__ == "__main__":
    run()
