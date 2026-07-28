"""Procrustes shape analysis.

Faithful port of geom/Procrustes.m by Alex Townsend (August 2011).  Compares
two closed planar curves (represented as complex-valued periodic chebfuns) by
Procrustes shape analysis: translate the mean to 0, scale to unit L2 norm,
align the major axis (rotate + reparametrize so the point of maximum modulus
is at angle 0 and parameter 0), then report the Procrustes distance
``norm(f-g)``.

Original: https://www.chebfun.org/examples/geom/Procrustes.html
Copyright 2011 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): both published Procrustes distances reproduce
to ~13-14 significant figures (frisbee vs pebble 0.072347575424997; pebble vs
its reflection 0.097593759012228), using complex chebfun arithmetic, the
continuous L2 norm, ``max(abs(.))`` with its location, and periodic
reparametrization ``f(mod(x+x_max, 2*pi))``.  The prior port used discrete
sample arrays and computed different numbers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()

_DOM = (0.0, 2 * np.pi)
_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'geom')
os.makedirs(_OUTDIR, exist_ok=True)


def _cf(fn):
    return cj.chebfun(fn, domain=_DOM)


def _shape_analysis(f, g):
    """Translate to mean 0, scale to unit norm, align major axis (MATLAB
    ShapeAnalysis subfunction)."""
    out = []
    for h in (f, g):
        h = h - h.mean()          # translate mean to 0
        h = h / h.norm()          # scale so L2 norm is 1
        x_max, _ = h.__abs__().max()   # location of max modulus
        rot = float(np.angle(complex(h(x_max))))
        # reparametrize so the parameter starts at the major axis, and
        # rotate so that point lies on the positive real axis.
        shifted = cj.chebfun(
            lambda x, h=h, c=x_max: np.asarray(h(np.mod(x + c, 2 * np.pi))),
            domain=_DOM)
        out.append(shifted * complex(np.exp(-1j * rot)))
    return out[0], out[1]


def run():
    # ------------------------------------------------------------------
    # Frisbee and pebble.
    # ------------------------------------------------------------------
    f = _cf(lambda t: 3 * (1.5 * np.cos(t) + 1j * np.sin(t)))
    g = _cf(lambda t: np.exp(1j * np.pi / 3) * (
        1 + np.cos(t) + 1.5j * np.sin(t)
        + 0.125 * (1 + 1.5j) * np.sin(3 * t)**2))

    fa, ga = _shape_analysis(f, g)
    print("ans =")
    print(f"   {float((fa - ga).norm()):.15f}")

    # ------------------------------------------------------------------
    # Pebble and its reflection.
    # ------------------------------------------------------------------
    f2 = _cf(lambda t: np.exp(1j * np.pi / 3) * (
        1 + np.cos(t) + 1.5j * np.sin(t)
        + 0.125 * (1 + 1.5j) * np.sin(3 * t)**2))
    g2 = _cf(lambda t: np.exp(-1j * np.pi / 3) * (
        1 + np.cos(2 * np.pi - t) - 1.5j * np.sin(2 * np.pi - t)
        + 0.125 * (1 - 1.5j) * np.sin(3 * (2 * np.pi - t))**2))

    fb, gb = _shape_analysis(f2, g2)
    print("ans =")
    print(f"   {float((fb - gb).norm()):.15f}")

    # ------------------------------------------------------------------
    # Plot: original curves and aligned curves for both cases.
    # ------------------------------------------------------------------
    tt = np.linspace(0, 2 * np.pi, 600)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, (h1, h2, ttl) in zip(axes, [
            (f, g, "frisbee and pebble"),
            (f2, g2, "pebble and its reflection")]):
        z1 = np.asarray(h1(tt))
        z2 = np.asarray(h2(tt))
        ax.plot(z1.real, z1.imag, "r", lw=2)
        ax.plot(z2.real, z2.imag, "k", lw=2)
        ax.set_aspect("equal")
        ax.set_title(ttl, fontsize=11)

    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, "procrustes.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return True


if __name__ == "__main__":
    run()
