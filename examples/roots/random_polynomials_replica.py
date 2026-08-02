"""Roots of random polynomials.

Faithful replica of roots/RandomPolynomials.m by Nick Trefethen
(March 2016): roots of degree-n polynomials with random coefficients
in the monomial, Chebyshev, and Legendre bases — clustering on the
unit circle for monomials and on [-1,1] with a circle at radius ~1
for the orthogonal bases.

randn draws are not bit-reproducible between MATLAB and numpy; the
pictures are statistical.

Original: https://www.chebfun.org/examples/roots/RandomPolynomials.html
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
from chebfunjax.utils.transforms import leg2cheb

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')

NN = [50, 200]
FIG = [0]


def _pair_plot(rootfun, basis_name):
    FIG[0] += 1
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.9))
    for ax, n in zip(axes, NN):
        r = rootfun(n)
        ax.plot(r.real, r.imag, '.k', ms=3)
        ax.axis([-1.5, 1.5, -1.5, 1.5])
        ax.set_aspect("equal")
        ax.set_title(f"{basis_name}, n={n}", fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"RandomPolynomials_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    rs = np.random.RandomState(1)

    def monomial_roots(n):
        a = np.concatenate([[1.0], rs.randn(n)])
        return np.roots(a)

    _pair_plot(monomial_roots, "monomial")

    def cheb_roots(n):
        a = np.concatenate([[1.0], rs.randn(n)])
        p = cj.Chebfun.from_coeffs(jnp.asarray(a[::-1]))
        return np.asarray(p.roots(all_roots=True))

    _pair_plot(cheb_roots, "Chebyshev")

    def leg_roots(n):
        a = np.concatenate([[1.0], rs.randn(n)])
        ccheb = np.asarray(leg2cheb(jnp.asarray(a[::-1])))
        p = cj.Chebfun.from_coeffs(jnp.asarray(ccheb))
        return np.asarray(p.roots(all_roots=True))

    _pair_plot(leg_roots, "Legendre")
    print("figures:", FIG[0])


if __name__ == "__main__":
    run()
