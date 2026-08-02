"""A double-keyhole contour of Ablowitz and Fokas.

Faithful replica of complex/KeyholeAblowitzFokas.m by Nick Trefethen
(December 2019): integrating a branch-cut function around a contour
enclosing the cut [-1,1], giving sqrt(2)/2, with two equivalent
contours.

Original: https://www.chebfun.org/examples/complex/KeyholeAblowitzFokas.html
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

FIG = [0]


def ff(z):
    sgn = jnp.where(jnp.real(z) > 0, -1.0, 1.0)
    return (0.5j / jnp.pi) * jnp.sqrt(z**2 - 1) * sgn / (1 + z**2)


def _arc(center, radius, sign, a, b):
    """center + radius*exp(sign*1i*pi*s) on s in [a, b], mapped to
    [0,1], with derivative."""
    w = sign * 1j * np.pi

    def zf(t):
        s = a + t * (b - a)
        return center + radius * jnp.exp(w * s)

    def dzf(t):
        s = a + t * (b - a)
        return radius * w * (b - a) * jnp.exp(w * s)

    return zf, dzf


def _line(p, q):
    return (lambda t: p + t * (q - p),
            lambda t: (q - p) * jnp.ones_like(t))


def _integrate(segs, fname, title):
    FIG[0] += 1
    ss = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for zf, _ in segs:
        zv = np.asarray(zf(jnp.asarray(ss)))
        ax.plot(zv.real, zv.imag, 'k', lw=1.2)
    ax.plot([-1, 1], [0, 0], '.r', ms=10)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"KeyholeAblowitzFokas_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)

    I = 0j
    for zf, dzf in segs:
        g = cj.chebfun(lambda t, _z=zf, _d=dzf: ff(_z(t)) * _d(t),
                       domain=(0.0, 1.0))
        I += complex(np.asarray(g.sum()))
    sign = "+" if I.imag >= 0 else "-"
    print("I =")
    print(f"  {I.real:.15f} {sign} {abs(I.imag):.15f}i")


def _contour(c0_lim, c1_c, c1_r, c1_lim, c2_c, c2_r, c2_lim):
    z0, d0 = _arc(0.0, 1.5, 1, *c0_lim)
    z1, d1 = _arc(c1_c, c1_r, -1, *c1_lim)
    z2, d2 = _arc(c2_c, c2_r, -1, *c2_lim)
    p1 = complex(np.asarray(z0(jnp.asarray(0.0))))
    p2 = complex(np.asarray(z0(jnp.asarray(1.0))))
    p4 = complex(np.asarray(z1(jnp.asarray(0.0))))
    p5 = complex(np.asarray(z1(jnp.asarray(1.0))))
    p6 = complex(np.asarray(z2(jnp.asarray(0.0))))
    p7 = complex(np.asarray(z2(jnp.asarray(1.0))))
    p3 = p2.real + 1j * p4.imag
    p8 = p1.real + 1j * p7.imag
    return [
        (z0, d0), _line(p2, p3), _line(p3, p4), (z1, d1),
        _line(p5, p6), (z2, d2), _line(p7, p8), _line(p8, p1),
    ]


def run():
    os.makedirs(_IMG, exist_ok=True)

    # c2 = -c1 = -1 - 0.2 exp(-i pi s): a NEGATIVE radius (phase pi)
    segs = _contour((0.51, 2.49), 1.0, 0.2, (-0.93, 0.93),
                    -1.0, -0.2, (-0.93, 0.93))
    _integrate(segs, "", "Ablowitz-Fokas contour")
    print("Iexact =")
    print(f"   {np.sqrt(2)/2:.15f}")

    # a distorted but homotopic contour gives the same answer
    segs = _contour((0.51, 2.38), 1.0 - 0.03j, 0.2, (-0.91, 0.80),
                    -1.0 + 0.07j, -0.3, (-0.89, 0.82))
    _integrate(segs, "", "Another equivalent contour")


if __name__ == "__main__":
    run()
