"""Happy Valentine's Day.

Faithful replica of fun/ValentinesDay.m by Nick Hale
(February 2012): heart curves and their areas — the classic heart,
the cardioid, and a lumpy variant.

Original: https://www.chebfun.org/examples/fun/ValentinesDay.html
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
from chebfunjax.utils.scribble import scribble

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'fun')

FIG = [0]
DOM = (-np.pi, np.pi)


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"ValentinesDay_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scribble(ax, cf, scale=1.0, color='k'):
    bps = [float(v) for v in cf.domain.breakpoints]
    for a, b in zip(bps[:-1], bps[1:]):
        t = np.linspace(a, b, 12)
        z = scale * np.asarray(cf(t))
        ax.plot(z.real, z.imag, color, lw=1.2)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    x = cj.chebfun(lambda t: 16 * jnp.sin(t)**3, domain=DOM)
    y = cj.chebfun(lambda t: 13 * jnp.cos(t) - 5 * jnp.cos(2 * t)
                   - 2 * jnp.cos(3 * t) - jnp.cos(4 * t),
                   domain=DOM)
    ts = np.linspace(*DOM, 700)
    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    ax.fill(np.asarray(x(ts)), np.asarray(y(ts)), 'r')
    _plot_scribble(ax, scribble("Happy Valentine's Day!"),
                   scale=12.5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    _save(fig)

    A = abs(float((x * y.diff()).sum()))
    print("A =")
    print(f"     {A:.15e}")
    print("err =")
    print(f"     {A - 180*np.pi:.15e}")

    # the cardioid
    r1 = cj.chebfun(lambda t: 1 - jnp.sin(t), domain=DOM)
    x1 = cj.chebfun(lambda t: (1 - jnp.sin(t)) * jnp.cos(t),
                    domain=DOM)
    y1 = cj.chebfun(lambda t: (1 - jnp.sin(t)) * jnp.sin(t),
                    domain=DOM)
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    ax.plot(np.asarray(x1(ts)), np.asarray(y1(ts)), '.-r', ms=3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    _save(fig)
    A1 = abs(float((x1 * y1.diff()).sum()))
    print("A1 =")
    print(f"   {A1:.15f}")
    print("err =")
    print(f"    {A1 - 3*np.pi/2:.15e}")

    # a lumpy heart, with an abs() kink handled by splitting
    def rop(t):
        return (2 - 2 * jnp.sin(t) + jnp.sin(t)
                * jnp.sqrt(jnp.abs(jnp.cos(t)) + 0.1)
                / (jnp.sin(t) + 1.4))

    x5 = cj.chebfun(lambda t: rop(t) * jnp.cos(t), domain=DOM,
                    splitting=True)
    y5 = cj.chebfun(lambda t: rop(t) * jnp.sin(t), domain=DOM,
                    splitting=True)
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    ax.plot(np.asarray(x5(ts)), np.asarray(y5(ts)), '.-r', ms=3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    _save(fig)
    A5 = abs(float((x5 * y5.diff()).sum()))
    print("A5 =")
    print(f"  {A5:.15f}")


if __name__ == "__main__":
    run()
