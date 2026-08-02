"""Computing complex roots with contour integrals.

Faithful replica of roots/ComplexRoots.m by Nick Trefethen
(December 2011): counting and locating roots inside the unit disk
via contour integrals sum(z^k f'/f)/(2 pi i) over the unit circle,
following Delves & Lyness (1967).

Original: https://www.chebfun.org/examples/roots/ComplexRoots.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')

FIG = [0]


def _fmt(v):
    sign = "+" if v.imag >= 0 else "-"
    return f"{v.real:.15f} {sign} {abs(v.imag):.15f}i"


def _z():
    return cj.chebfun(lambda t: jnp.exp(1j * jnp.pi * t))


def _moment(z, f, k):
    return complex(np.asarray((z**k * f.diff() / f).sum())) / (2j * np.pi)


def roots3(ff):
    """Find 3 roots of ff in the unit disk via Delves-Lyness."""
    z = _z()
    f = ff(z)
    s1 = _moment(z, f, 1)
    s2 = _moment(z, f, 2)
    s3 = _moment(z, f, 3)
    p = [1, -s1, (s1**2 - s2) / 2,
         -(s1**3 - 3 * s1 * s2 + 2 * s3) / 6]
    return np.roots(p)


def _portrait(ff, fname_idx):
    FIG[0] += 1
    xs = np.linspace(-1.15, 1.15, 480)
    X, Y = np.meshgrid(xs, xs)
    with np.errstate(all="ignore"):
        V = ff(X + 1j * Y)
    H = (np.angle(V) + np.pi) / (2 * np.pi)
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.imshow(plt.cm.hsv(H), origin="lower",
              extent=(-1.15, 1.15, -1.15, 1.15))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), 'k-', lw=1.5)
    ax.set_aspect("equal")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"ComplexRoots_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    z = _z()
    f = (z - 0.5j) * z.exp()
    print("s1 =")
    print(f" {_fmt(_moment(z, f, 1))}")

    f = (np.pi * z).cosh()
    s0 = _moment(z, f, 0)
    s1 = _moment(z, f, 1)
    s2 = _moment(z, f, 2)
    print("s0 =")
    print(f"  {_fmt(s0)}")
    print("s1 =")
    print(f"      {s1.real:.15e} {'+' if s1.imag >= 0 else '-'} "
          f"{abs(s1.imag):.15e}i")
    print("s2 =")
    print(f" {_fmt(s2)}")
    p = [1, -s1, (s1**2 - s2) / 2]
    print("ans =")
    for v in np.roots(p):
        print(f" {_fmt(v)}")

    ff1 = lambda w: np.cosh(np.exp(w)) * (w - 0.3) * (1 + 4 * w**2)  # noqa: E731

    def ff1_cf(zc):
        return (zc.exp()).cosh() * (zc - 0.3) * (1 + 4 * zc**2)

    print("ans =")
    for v in np.sort_complex(roots3(ff1_cf)):
        print(f" {_fmt(v)}")
    _portrait(ff1, 1)

    ff2 = lambda w: (w**3 - 1 / 8) * np.exp((-1 - 2j) * w)  # noqa: E731

    def ff2_cf(zc):
        return (zc**3 - 1 / 8) * ((-1 - 2j) * zc).exp()

    print("ans =")
    for v in np.sort_complex(roots3(ff2_cf)):
        print(f" {_fmt(v)}")
    _portrait(ff2, 2)


if __name__ == "__main__":
    run()
