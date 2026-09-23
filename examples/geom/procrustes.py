"""Procrustes shape analysis.

Translation of geom/Procrustes.m by Alex Townsend (August 2011):
translating, scaling and rotating two closed curves (a frisbee and a
pebble, then the pebble and its reflection) so that their Procrustes
distance ``norm(f - g)`` can be measured.

Original: https://www.chebfun.org/examples/geom/Procrustes.html
Copyright by The University of Oxford and The Chebfun Developers.
"""

from __future__ import annotations

import os
import sys
import warnings

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj  # noqa: E402
from chebfunjax.plotting import chebfun_style  # noqa: E402
from chebfunjax.plotting import save_chebfun_figure as _savefig  # noqa: E402

jax.config.update("jax_enable_x64", True)
chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'geom')
FIG = [0]
_SIZES = {1: (600, 270), 2: (600, 270), 3: (600, 270), 4: (600, 270)}
_TT = np.linspace(0, 2 * np.pi, 2001)


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"Procrustes_{FIG[0]:02d}.png"),
             size=_SIZES.get(FIG[0], (600, 270)))
    plt.close(fig)


def _plot(ax, f, color):
    """plot(f) of a complex chebfun: the curve (real f, imag f)."""
    v = np.asarray(f(jnp.asarray(_TT)))
    ax.plot(np.real(v), np.imag(v), color=color, lw=2)


def shape_analysis(f, g):
    """SHAPEANALYSIS(F,G): plots the parameterised curves before and after
    each stage of translating, scaling and aligning; outputs are
    parameterised curves ready for Procrustes shape analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.6))
    ax = axes[0, 0]
    _plot(ax, f, 'r'), _plot(ax, g, 'k')
    ax.set_aspect('equal'), ax.set_title('Original', fontsize=14)
    # Translate mean to 0.
    f = f - f.mean()
    g = g - g.mean()
    ax = axes[0, 1]
    _plot(ax, f, 'r'), _plot(ax, g, 'k')
    ax.set_aspect('equal'), ax.set_title('After translation', fontsize=14)
    # Scale so RMSD is 1.
    f = f / float(f.norm())
    g = g / float(g.norm())
    ax = axes[1, 0]
    _plot(ax, f, 'r'), _plot(ax, g, 'k')
    ax.set_aspect('equal'), ax.set_title('After scaling', fontsize=14)
    # Align major axis: find argument of major axis.
    fxmax, _ = f.abs().max()
    gxmax, _ = g.abs().max()
    rotf = float(np.angle(complex(np.asarray(f(jnp.asarray(fxmax))))))
    rotg = float(np.angle(complex(np.asarray(g(jnp.asarray(gxmax))))))
    # Rotate both so major axis lies on the positive real axis.
    x = cj.chebfun(lambda t: t, domain=(0.0, 2 * np.pi))
    f = np.exp(-1j * rotf) * f((x + float(fxmax)).rem(2 * np.pi))
    g = np.exp(-1j * rotg) * g((x + float(gxmax)).rem(2 * np.pi))
    ax = axes[1, 1]
    _plot(ax, f, 'r'), _plot(ax, g, 'k')
    ax.set_aspect('equal'), ax.set_title('After aligning', fontsize=14)
    _save(fig)
    return f, g


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t = cj.chebfun(lambda s: s, domain=(0.0, 2 * np.pi))
    f = 3 * (1.5 * t.cos() + 1j * t.sin())                           # frisbee
    g = np.exp(1j * np.pi / 3) * (1 + t.cos() + 1.5j * t.sin()
                                  + .125 * (1 + 1.5j) * (3 * t).sin() ** 2)  # pebble
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    _plot(ax, f, 'r'), _plot(ax, g, 'k')
    ax.set_aspect('equal'), ax.set_title('Frisbee and pebble', fontsize=14)
    _save(fig)

    f, g = shape_analysis(f, g)
    print("ans =")
    print(f"   {float((f - g).norm()):.15f}")

    f = np.exp(1j * np.pi / 3) * (1 + t.cos() + 1.5j * t.sin()
                                  + .125 * (1 + 1.5j) * (3 * t).sin() ** 2)
    g = np.exp(-1j * np.pi / 3) * (1 + (2 * np.pi - t).cos() - 1.5j * (2 * np.pi - t).sin()
                                   + .125 * (1 - 1.5j) * (3 * (2 * np.pi - t)).sin() ** 2)
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    _plot(ax, f, 'r'), _plot(ax, g, 'k')
    ax.set_aspect('equal'), ax.set_title('Pebble and its reflection', fontsize=14)
    _save(fig)

    f, g = shape_analysis(f, g)
    print("ans =")
    print(f"   {float((f - g).norm()):.15f}")


if __name__ == "__main__":
    run()
