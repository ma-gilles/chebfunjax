"""Sumdisk for integration over a disk.

Faithful replica of quad/SumdiskDemo.m by Klaus Wang and Nick
Trefethen (June 2016): the chebfun2 `sumdisk` command integrates a
chebfun2 over the disk inscribed in its square of definition, using
closed-form disk integrals of Chebyshev products T_j(x) T_k(y).

Original: https://www.chebfun.org/examples/quad/SumdiskDemo.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import numpy as np
from matplotlib import cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import matplotlib.pyplot as plt

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'quad')


def _disp(v, name="ans"):
    print(f"{name} =")
    if v == int(v) and abs(v) < 1e15:
        print(f"     {int(v)}")
    elif v < 0:
        print(f"  {v:.15f}")
    else:
        print(f"   {v:.15f}")


def run():
    os.makedirs(_IMG, exist_ok=True)

    # For a trivial example, suppose our chebfun2 is the constant 1.
    # Its integral over the square is 4,
    f = cj.chebfun2(lambda x, y: 1.0 + 0 * x + 0 * y)
    _disp(float(f.sum2()))

    # but its integral over the disk is pi,
    _disp(f.sumdisk())

    # Bivariate Gaussian exp(-(x^2+y^2)/2): integral over the unit disk.
    f = cj.chebfun2(lambda x, y: jnp.exp(-(x**2 + y**2) / 2))
    _disp(f.sumdisk())

    # Polar coordinates give the integral exactly: 2*pi*(1 - 1/sqrt(e)).
    exact = 2 * np.pi * (1 - np.exp(-0.5))
    _disp(exact, "exact")

    # Direct polar-coordinates quadrature as in examples/quad/TjTkDisk:
    def fr(r):
        circ = cj.chebfun(
            lambda t: f(r * jnp.cos(t), r * jnp.sin(t)),
            domain=(0.0, 2 * np.pi), trig=True)
        return r * float(circ.sum())

    def radial_vals(r):
        arr = np.atleast_1d(np.asarray(r, dtype=np.float64))
        vals = [fr(float(ri)) for ri in arr.ravel()]
        return jnp.asarray(vals, dtype=jnp.float64).reshape(arr.shape)

    radial = cj.chebfun(radial_vals, domain=(0.0, 1.0))
    I = float(radial.sum())
    _disp(I, "I")

    # A harmonic function: the real part of an analytic function.
    fcomplex = cj.chebfun2(lambda z: jnp.cos(2 * jnp.cosh(z)))
    f = fcomplex.real()

    xs = np.linspace(-1, 1, 121)
    X, Y = np.meshgrid(xs, xs)
    Z = np.asarray(f(jnp.asarray(X), jnp.asarray(Y)))
    fig = plt.figure(figsize=(6.8, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, rstride=2, cstride=2,
                    linewidth=0, antialiased=True)
    ax.view_init(elev=30, azim=-37.5)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "SumdiskDemo_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Mean of f over the unit disk via sumdisk:
    _disp(f.sumdisk() / np.pi)

    # Since f is harmonic, this equals the value of f at the origin:
    _disp(float(f(0.0, 0.0)))


if __name__ == "__main__":
    run()
