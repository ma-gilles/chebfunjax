"""A keyhole contour integral.

Faithful replica of complex/KeyholeContour.m by Nick Trefethen and
Nick Hale (October 2010): integrating log(x)tanh(x) around a keyhole
contour to get 4i*pi*log(pi/2).

Original: https://www.chebfun.org/examples/complex/KeyholeContour.html
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


def run():
    os.makedirs(_IMG, exist_ok=True)

    f = lambda z: jnp.log(z) * jnp.tanh(z)  # noqa: E731
    r, R, e = 0.2, 2.0, 0.1
    c = [complex(-R, e), complex(-r, e), complex(-r, -e),
         complex(-R, -e)]

    # The four pieces of the keyhole, each parametrized on [0, 1],
    # with analytic derivatives:
    segs = []
    # 1. top of the keyhole (line c1 -> c2)
    segs.append((lambda s: c[0] + s * (c[1] - c[0]),
                 lambda s: (c[1] - c[0]) * jnp.ones_like(s)))
    # 2. inner circle: c2 * (c3/c2)^s
    # c2*c3^s/c2^s with separate principal logs: the LONG way
    # around (through the positive real axis), avoiding the cut
    w2 = np.log(c[2]) - np.log(c[1])
    segs.append((lambda s: c[1] * jnp.exp(s * w2),
                 lambda s: c[1] * w2 * jnp.exp(s * w2)))
    # 3. bottom of the keyhole (line c3 -> c4)
    segs.append((lambda s: c[2] + s * (c[3] - c[2]),
                 lambda s: (c[3] - c[2]) * jnp.ones_like(s)))
    # 4. outer circle: c4 * (c1/c4)^s
    w4 = np.log(c[0]) - np.log(c[3])
    segs.append((lambda s: c[3] * jnp.exp(s * w4),
                 lambda s: c[3] * w4 * jnp.exp(s * w4)))

    # Plot the contour
    ss = np.linspace(0, 1, 600)
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    for zfun, _ in segs:
        zv = np.asarray(zfun(jnp.asarray(ss)))
        ax.plot(zv.real, zv.imag, 'C0', lw=1.4)
    ax.plot([-2.6, 0], [0, 0], '-r', lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-2.6, 2.6)
    ax.set_title("A keyhole contour in the complex plane", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "KeyholeContour_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # I = sum over segments of int f(z(s)) z'(s) ds
    I = 0.0 + 0.0j
    for zfun, dzfun in segs:
        g = cj.chebfun(lambda s, zf=zfun, dzf=dzfun:
                       f(zf(s)) * dzf(s), domain=(0.0, 1.0))
        I += complex(np.asarray(g.sum()))
    Iexact = 4j * np.pi * np.log(np.pi / 2)
    print("I =")
    print(f"  {I.real:.15f} + {I.imag:.15f}i")
    print("Iexact =")
    print(f"  {Iexact.real:.15f} + {Iexact.imag:.15f}i")
    print("error =")
    print(f"     {abs(I - Iexact):.15e}")


if __name__ == "__main__":
    run()
