"""Scaling and squaring for the exponential.

Faithful replica of approx/ScalingAndSquaring.m by Nick Trefethen
(April 2019): a type (8,8) Pade approximant of exp combined with s=2
scaling-and-squaring steps, with contour plots of absolute and
relative error in the complex plane.

Original: https://www.chebfun.org/examples/approx/ScalingAndSquaring.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.ratapprox import padeapprox

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_IMG, exist_ok=True)

    s, m = 2, 8
    f = np.concatenate([[1.0], 1.0 / np.cumprod(np.arange(1, 51))])
    r, *_ = padeapprox(f, m, m, tol=0.0)

    xgrid = np.linspace(-100, 100, 140)
    x, y = np.meshgrid(xgrid, xgrid)
    z = x + 1j * y
    rz = np.asarray(r(z.ravel() / 2**s)).reshape(z.shape) ** (2**s)
    eps = np.finfo(float).eps

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    cs = ax.contourf(x, y, np.log10(np.abs(np.exp(z) - rz) + eps),
                     levels=np.arange(-16, 1, 2))
    fig.colorbar(cs, ax=ax)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ScalingAndSquaring_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.log10(np.abs(np.exp(z) - rz) / np.abs(np.exp(z)))
    cs = ax.contourf(x, y, rel, levels=np.arange(-16, 17, 2))
    fig.colorbar(cs, ax=ax)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ScalingAndSquaring_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
