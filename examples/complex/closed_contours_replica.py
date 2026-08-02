"""Integrals over closed contours.

Faithful replica of complex/ClosedContours.m by Mohsin Javed (October
2013): residue computations by integrating trig-parametrized circles
in the complex plane.

Original: https://www.chebfun.org/examples/complex/ClosedContours.html
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


def _cdisp(v, name="s"):
    print(f"{name} =")
    sign = "+" if v.imag >= 0 else "-"
    print(f"  {v.real:.15f} {sign} {abs(v.imag):.15f}i")


def _contour_integral(ff, radius):
    z = cj.chebfun(lambda t: radius * jnp.exp(2j * jnp.pi * t),
                   domain=(0.0, 1.0), trig=True)
    f = cj.chebfun(lambda t: ff(radius * jnp.exp(2j * jnp.pi * t)),
                   domain=(0.0, 1.0), trig=True)
    dz = z.diff()
    return f, complex(np.asarray((f * dz).sum()))


def run():
    os.makedirs(_IMG, exist_ok=True)

    ff = lambda zz: (1 - 2 * zz) / (zz * (zz - 1) * (zz - 3))  # noqa: E731
    f, s = _contour_integral(ff, 2.0)
    print("f ="); print(repr(f))

    ts = np.linspace(0, 1, 2000)
    fv = np.asarray(f(jnp.asarray(ts)))
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.plot(fv.real, fv.imag, lw=1.3)
    ax.set_aspect("equal")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ClosedContours_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    axes[0].plot(ts, fv.real, lw=1.2)
    axes[0].set_title("real part", fontsize=12)
    axes[1].plot(ts, fv.imag, lw=1.2)
    axes[1].set_title("imaginary part", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ClosedContours_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    _cdisp(s)
    print("ans =")
    print(f"     {abs(s - 5/3*np.pi*1j):.15e}")

    # sin(5z)/(5z): analytic, integral 0
    ff2 = lambda zz: jnp.sin(5 * zz) / (5 * zz)  # noqa: E731
    f2, s2 = _contour_integral(ff2, 1.0)
    fv2 = np.asarray(f2(jnp.asarray(ts)))
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    ax.plot(fv2.real, fv2.imag, lw=1.3)
    ax.set_aspect("equal")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ClosedContours_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("s =")
    print(f"     {s2.real:.15e} + {s2.imag:.15e}i")

    # essential singularity: exp(1/z) sin(1/z), residue 1
    ff3 = lambda zz: jnp.exp(1 / zz) * jnp.sin(1 / zz)  # noqa: E731
    _, s3 = _contour_integral(ff3, 1.0)
    _cdisp(s3)
    _cdisp(2j * np.pi, name="exact")


if __name__ == "__main__":
    run()
