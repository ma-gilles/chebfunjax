"""Arc length of complex paths.

Faithful replica of complex/ComplexArcLength.m by Kuan Xu (October
2012): arc lengths of the keyhole contour (whole and per piece) and of
a flower curve, with equal-arclength points found by rootfinding.

Original: https://www.chebfun.org/examples/complex/ComplexArcLength.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

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

    r, R, e = 0.2, 2.0, 0.1
    c = [complex(-R, e), complex(-r, e), complex(-r, -e),
         complex(-R, -e)]
    w2 = np.log(c[2]) - np.log(c[1])
    w4 = np.log(c[0]) - np.log(c[3])
    segs = [
        cj.chebfun(lambda t: c[0] + t * (c[1] - c[0]),
                   domain=(0.0, 1.0)),
        cj.chebfun(lambda t: c[1] * jnp.exp(t * w2),
                   domain=(0.0, 1.0)),
        cj.chebfun(lambda t: c[2] + t * (c[3] - c[2]),
                   domain=(0.0, 1.0)),
        cj.chebfun(lambda t: c[3] * jnp.exp(t * w4),
                   domain=(0.0, 1.0)),
    ]

    ss = np.linspace(0, 1, 500)
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for z in segs:
        zv = np.asarray(z(jnp.asarray(ss)))
        ax.plot(zv.real, zv.imag, 'C0', lw=1.6)
    ax.set_aspect("equal")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ComplexArcLength_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    lengths = [z.arc_length() for z in segs]
    print("L =")
    print(f"  {sum(lengths):.15f}")
    print("L (per piece) =")
    print("   " + "   ".join(f"{v:.15f}" for v in lengths))

    # A flower curve, and N equal-arclength points on it
    s_fun = lambda t: (jnp.exp(1j * 2 * jnp.pi * t)  # noqa: E731
                       * (0.5 * jnp.sin(8 * jnp.pi * t) ** 2 + 0.5))
    s = cj.chebfun(s_fun, domain=(0.0, 1.0))
    L = s.arc_length()
    print("L =")
    print(f"   {L:.15f}")

    N = 64
    h = L / N
    t0 = time.time()
    # cumulative arclength: chebfun integral of |s'(t)|
    sp = s.diff()
    speed = cj.chebfun(lambda t: jnp.abs(sp(t)), domain=(0.0, 1.0))
    cum = speed.cumsum()
    T = [0.0]
    for k in range(1, N):
        rts = np.atleast_1d(np.asarray((cum - k * h).roots()))
        T.append(float(rts[0]))
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")

    P = np.asarray(s(jnp.asarray(np.asarray(T))))
    ts = np.linspace(0, 1, 3000)
    sv = np.asarray(s(jnp.asarray(ts)))
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    ax.plot(sv.real, sv.imag, lw=1.6)
    ax.plot(P.real, P.imag, '.r', ms=12)
    ax.set_aspect("equal")
    ax.set_title(f"{N} points equally spaced by arc length",
                 fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ComplexArcLength_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
