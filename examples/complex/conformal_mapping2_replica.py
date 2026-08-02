"""Conformal maps to an annulus.

Faithful replica of complex/ConformalMapping2.m by Nick Trefethen
(March 2020): the `conformal2` command mapping doubly-connected
regions to a circular annulus, whose conformal modulus rho is
determined as part of the computation.

Original: https://www.chebfun.org/examples/complex/ConformalMapping2.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.conformal2 import conformal2
from chebfunjax.utils.quadrature import chebpts

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'complex')

FIG = [0]


def _plots(C1v, C2v, f, finv, rho, pol, polinv):
    """The two-panel 'plots' figure of conformal2.m."""
    FIG[0] += 1
    pol, polinv = np.asarray(pol), np.asarray(polinv)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 5.6))
    circ = np.exp(2j * np.pi * np.arange(0, 301) / 300)
    axL.plot(C1v.real, C1v.imag, 'b', lw=1)
    axL.plot(C2v.real, C2v.imag, 'b', lw=1)
    lim = 1.2 * np.max(np.abs(C1v))
    axL.axis([-lim, lim, -lim, lim])
    axL.plot(pol.real, pol.imag, '.r', ms=7)
    axL.set_title(f"{len(pol)} poles", fontsize=11)
    axR.plot(circ.real, circ.imag, 'b', lw=1)
    axR.plot((rho * circ).real, (rho * circ).imag, 'b', lw=1)
    axR.axis([-1.5, 1.5, -1.5, 1.5])
    axR.plot(polinv.real, polinv.imag, '.r', ms=7)
    axR.set_title(f"{len(polinv)} poles", fontsize=11)
    ncirc = 8
    for r in rho + (1 - rho) * np.arange(1, ncirc) / ncirc:
        v = np.asarray(finv(jnp.asarray(r * circ)))
        axL.plot(v.real, v.imag, '-k', lw=0.5)
        axR.plot((r * circ).real, (r * circ).imag, '-k', lw=0.5)
    ray = rho + (1 - rho) * (np.asarray(chebpts(101)) + 1) / 2
    for k in range(1, 17):
        w = ray * np.exp(2j * np.pi * k / 16)
        v = np.asarray(finv(jnp.asarray(w)))
        axL.plot(v.real, v.imag, '-k', lw=0.5)
        axR.plot(w.real, w.imag, '-k', lw=0.5)
    for ax in (axL, axR):
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"ConformalMapping2_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    t = np.linspace(-1, 1, 1200, endpoint=False)
    circle = np.exp(1j * np.pi * t)
    ellipse = circle.real + 0.6j * circle.imag
    C1 = 3 * ellipse - 1
    C2 = np.exp(0.5j) * ellipse

    t0 = time.time()
    f, finv, rho, pol, polinv = conformal2(jnp.asarray(C1),
                                           jnp.asarray(C2))
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")
    _plots(C1, C2, f, finv, rho, pol, polinv)
    print("rho =")
    print(f"   {rho:.15f}")

    t0 = time.time()
    f, finv, rho, pol, polinv = conformal2(jnp.asarray(C1),
                                           jnp.asarray(C2), tol=1e-12)
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")
    _plots(C1, C2, f, finv, rho, pol, polinv)
    print("rho =")
    print(f"   {rho:.15f}")

    z = np.array([1.0, 1j])
    z2 = np.asarray(finv(f(jnp.asarray(z))))
    print("ans =")
    for v in z2:
        sign = "+" if v.imag >= 0 else "-"
        print(f"  {v.real:.15f} {sign} {abs(v.imag):.15f}i")

    rs = np.random.RandomState(5489)
    z = 1 + 0.1 * rs.random_sample(10**6) \
        + 0.1j * rs.random_sample(10**6)
    t0 = time.time()
    finv(f(jnp.asarray(z)))
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")

    # wavy boundaries
    C1 = circle * (2 + 0.1 * np.cos(8 * np.pi * t))
    C2 = circle * (1 + 0.1 * np.cos(5 * np.pi * t))
    t0 = time.time()
    f, finv, rho, pol, polinv = conformal2(jnp.asarray(C1),
                                           jnp.asarray(C2))
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")
    _plots(C1, C2, f, finv, rho, pol, polinv)
    print("rho =")
    print(f"   {rho:.15f}")

    # outer boundary = zero contour of a chebfun2
    F = cj.chebfun2(lambda x, y: x**8 + y**8)
    curves = (F - 0.5).roots()
    c1 = curves[0]
    bps = list(c1.domain.breakpoints)
    tt = np.linspace(bps[0], bps[-1], 1200, endpoint=False)
    C1 = np.asarray(c1(tt))
    C2 = 0.5 * circle
    t0 = time.time()
    f, finv, rho, pol, polinv = conformal2(jnp.asarray(C1),
                                           jnp.asarray(C2))
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")
    _plots(C1, C2, f, finv, rho, pol, polinv)
    print("rho =")
    print(f"   {rho:.15f}")


if __name__ == "__main__":
    run()
