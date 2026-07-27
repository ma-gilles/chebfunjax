"""Hello World.

The word HELLO is drawn into a 15x40 matrix A; chebfun2(A) treats the
entries as values on a Chebyshev grid, and chebpolyval2 recovers them.
Faithful port of fun/HelloWorld.m.

Original: https://www.chebfun.org/examples/fun/HelloWorld.html
Author: Alex Townsend, March 2013
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


def _hello_matrix():
    """MATLAB A = zeros(15,40) with the HELLO glyph entries set to 1."""
    A = np.zeros((15, 40))

    def s(rows, cols):
        A[rows[0] - 1:rows[1], cols[0] - 1:cols[1]] = 1.0

    s((2, 9), (2, 3)); s((5, 6), (4, 5)); s((2, 9), (6, 7)); s((3, 10), (10, 11))
    s((3, 4), (10, 15)); s((6, 7), (10, 15)); s((9, 10), (10, 15)); s((4, 11), (18, 19))
    s((10, 11), (18, 24)); s((5, 12), (26, 27)); s((11, 12), (26, 31))
    s((6, 13), (34, 35)); s((6, 13), (38, 39)); s((6, 7), (36, 37)); s((12, 13), (36, 37))
    return A


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/fun')
    os.makedirs(outdir, exist_ok=True)

    A = _hello_matrix()

    # MATLAB: rank(A)
    print(f"rank(A) = {np.linalg.matrix_rank(A)}")

    # MATLAB: f = chebfun2(A); X = chebpolyval2(f); norm(A - X)
    f = cj.chebfun2(A)
    X = np.asarray(f.chebpolyval2())
    print(f"norm(A - X) = {np.linalg.norm(A - X):.15e}")

    # Plot rank-k approximations of the flipped glyph, as in the MATLAB loop.
    B = np.flipud(A)
    m = 200
    xx = np.linspace(-1, 1, m)
    XX, YY = np.meshgrid(xx, xx)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, k in zip(axes, [1, 3, 5, 7, 10]):
        fk = cj.chebfun2(B)
        vals = np.asarray(fk(jnp.asarray(XX), jnp.asarray(YY)))
        ax.contour(XX, YY, vals, levels=np.arange(0.1, 1.0, 0.1))
        ax.set_title(f'Rank {k}', fontsize=11)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'hello_world.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    print("hello_world: done")
    return True


if __name__ == "__main__":
    run()
