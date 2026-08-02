"""Zeros of rational harmonic functions.

Faithful replica of complex/RationalHarmonic.m by Olivier Sete
(December 2015): zeros of r(z) - conj(z) (gravitational-lensing
harmonic mappings), located as common roots of the real and imaginary
parts, with phase portraits.

Original: https://www.chebfun.org/examples/complex/RationalHarmonic.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'complex')

N = 3
A = 0.7
DOM = tuple(1.4 * np.array([-1, 1, -1, 1]))
FIG = [0]


def _common_roots(expr):
    """Common zeros of Re(expr) and Im(expr) where expr maps (x, y) to
    a complex value."""
    fre = cj.chebfun2(lambda x, y: jnp.real(expr(x, y)), domain=DOM)
    fim = cj.chebfun2(lambda x, y: jnp.imag(expr(x, y)), domain=DOM)
    pts = np.asarray(fre.roots(fim))
    return np.atleast_2d(pts)


def smash(v):
    with np.errstate(all="ignore"):
        g = v / (1 + np.abs(v) ** 2)
    return np.where(np.isnan(g), 0.0, g)


def _portrait(F, zeros_xy, poles_xy, fname):
    FIG[0] += 1
    xs = np.linspace(DOM[0], DOM[1], 480)
    X, Y = np.meshgrid(xs, xs)
    with np.errstate(all="ignore"):
        V = F(X + 1j * Y)
    H = (np.angle(V) + np.pi) / (2 * np.pi)
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.imshow(plt.cm.hsv(H), origin="lower",
              extent=(DOM[0], DOM[1], DOM[2], DOM[3]), aspect="equal")
    if poles_xy is not None and len(poles_xy):
        ax.plot(poles_xy[:, 0], poles_xy[:, 1], 'ws', ms=4, mfc='w')
    if zeros_xy is not None and len(zeros_xy):
        ax.plot(zeros_xy[:, 0], zeros_xy[:, 1], 'ko', ms=4, mfc='k')
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"RationalHarmonic_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    def zc(x, y):
        return x + 1j * y

    p = lambda z: z ** (N - 1)          # noqa: E731
    q = lambda z: z**N - A**N           # noqa: E731

    poles = _common_roots(lambda x, y: q(zc(x, y)))
    zeros = _common_roots(
        lambda x, y: p(zc(x, y)) - q(zc(x, y)) * jnp.conj(zc(x, y)))
    print(f"n_zeros = {len(zeros)}  n_poles = {len(poles)}")

    ff = lambda z: z ** (N - 1) / (z**N - A**N) - np.conj(z)  # noqa: E731
    _portrait(lambda z: smash(ff(z)), zeros, poles, "")

    # epsilon-perturbed: a pole at the origin creates additional zeros
    eps_ = 0.01
    zeros_eps = _common_roots(
        lambda x, y: (p(zc(x, y)) * zc(x, y) + eps_ * q(zc(x, y))
                      - q(zc(x, y)) * zc(x, y) * jnp.conj(zc(x, y))))
    print("ans =")
    print(f"    {len(zeros_eps)}")
    poles_eps = np.vstack([poles, [[0.0, 0.0]]])
    ffe = lambda z: ff(z) + eps_ / z  # noqa: E731
    _portrait(lambda z: smash(ffe(z) * np.abs(z * q(z)) ** 2),
              zeros_eps, poles_eps, "")


if __name__ == "__main__":
    run()
