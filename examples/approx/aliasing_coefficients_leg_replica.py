"""Accuracy of Legendre coefficients via aliasing.

Faithful replica of approx/AliasingCoefficientsLeg.m by Yuji
Nakatsukasa (April 2016): the Legendre analogue of the aliasing
example — coefficients of low-degree Legendre interpolants err by
aliased tails, without the ultra-accurate nth coefficient peculiar to
Chebyshev.

Original: https://www.chebfun.org/examples/approx/AliasingCoefficientsLeg.html
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
from chebfunjax.utils.quadrature import legpts
from chebfunjax.utils.transforms import cheb2leg, legvals2legcoeffs

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

EPS = float(np.finfo(np.float64).eps)
GREEN = (0.0, 0.7, 0.0)


def _plot(fc, pc, err, fname):
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.semilogy(np.arange(len(fc)), np.abs(fc) + EPS, '.', color=GREEN,
                ms=10, label='f')
    ax.semilogy(np.arange(len(pc)), np.abs(pc) + EPS, 'b.', ms=10,
                label='p')
    ax.semilogy(np.arange(len(err)), np.abs(err) + EPS, '.r', ms=10,
                label='f-p')
    ax.legend(fontsize=16)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # Analytic function
    fori = lambda x: jnp.log(jnp.sin(10 * x) + 2)  # noqa: E731
    f = cj.chebfun(fori)
    fc = np.asarray(cheb2leg(f.coeffs))
    k = round(len(f) / 3)
    s = np.asarray(legpts(k)[0] if isinstance(legpts(k), tuple) else legpts(k))
    pc = np.asarray(legvals2legcoeffs(fori(jnp.asarray(s))))
    print(f"length(f) = {len(f)}, k = {k}")
    _plot(fc, pc, pc - fc[:len(pc)], "AliasingCoefficientsLeg_repl_01.png")

    # Non-analytic function
    fori2 = lambda x: jnp.abs((x - 0.5) ** 3)  # noqa: E731
    f2 = cj.chebfun(fori2)
    fc2 = np.asarray(cheb2leg(f2.coeffs))
    k2 = round(len(f2) / 5)
    s2 = np.asarray(legpts(k2)[0] if isinstance(legpts(k2), tuple)
                    else legpts(k2))
    pc2 = np.asarray(legvals2legcoeffs(fori2(jnp.asarray(s2))))
    print(f"length(f2) = {len(f2)}, k2 = {k2}")
    _plot(fc2, pc2, pc2[:k2] - fc2[:k2],
          "AliasingCoefficientsLeg_repl_02.png")

    # Two dimensions
    def _cols(op, M):
        # apply a vector transform to each column, as MATLAB does
        return np.column_stack([np.asarray(op(jnp.asarray(M[:, j])))
                                for j in range(M.shape[1])])

    fori2d = lambda x, y: jnp.sin(x + y) + jnp.cos(x - y)  # noqa: E731
    f2d = cj.chebfun2(fori2d)
    C = np.real(np.asarray(f2d.chebcoeffs2()))
    fcl = _cols(cheb2leg, _cols(cheb2leg, C).T).T
    k = 6
    sk = np.asarray(legpts(k)[0] if isinstance(legpts(k), tuple)
                    else legpts(k))
    XX, YY = np.meshgrid(sk, sk)   # XX(i,j)=s(j), YY(i,j)=s(i), as in MATLAB
    vals = np.asarray(fori2d(jnp.asarray(XX), jnp.asarray(YY)))
    ptc = _cols(legvals2legcoeffs, _cols(legvals2legcoeffs, vals).T).T
    D = np.abs(fcl[:k, :k] - ptc)
    print("ans =")
    for row in D:
        print("".join(f"{v:13.4e}" for v in row))


if __name__ == "__main__":
    run()
