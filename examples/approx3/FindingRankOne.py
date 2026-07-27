"""Finding a rank-one trivariate function inside a rank-3 chebfun3.

f, g, h are rank-1; fhat = f + (g+h)/10 has trilinear rank 3.  A rank-(1,1,1)
truncation of fhat, rescaled to agree with f at (1,1,1), recovers f up to a
small residual.  Faithful port of approx3/FindingRankOne.m by Yuji
Nakatsukasa, June 2016.

See https://www.chebfun.org/examples/approx3/FindingRankOne.html
Copyright 2016 by The University of Oxford and The Chebfun Developers.
"""

import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.plotting import chebfun_style

chebfun_style()

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                        "docs", "images", "approx3")
os.makedirs(_IMG_DIR, exist_ok=True)


def run():
    # MATLAB: f = chebfun3(@(x,y,z) sin(x).*cos(y).*exp(z)); rank(f)
    f = chebfun3(lambda x, y, z: jnp.sin(x) * jnp.cos(y) * jnp.exp(z))
    print(f"rank(f) = {max(f.rank)}")

    # g, h; fhat = f + (g+h)/10; rank(fhat)
    def fhat_fn(x, y, z):
        return (jnp.sin(x) * jnp.cos(y) * jnp.exp(z)
                + (jnp.cos(x) * jnp.exp(y) * jnp.sin(z)
                   + jnp.exp(x) * jnp.sin(y) * jnp.cos(z)) / 10.0)

    fhat = chebfun3(fhat_fn)
    print(f"rank(fhat) = {max(fhat.rank)}")

    # ftmp = chebfun3(@(x,y,z) fhat(x,y,z), 'rank', [1 1 1]);
    ftmp = chebfun3(fhat_fn, rank=(1, 1, 1))
    one = jnp.array(1.0)
    scale = float(f(one, one, one)) / float(ftmp(one, one, one))
    print(f"norm(f - ftmp*scale) = {float((f - ftmp * scale).norm()):.15f}")

    # --- Plot: a 1-D slice of f and fhat -------------------------------
    fig, ax = plt.subplots()
    xl = np.linspace(-1, 1, 100)
    yv, zv = jnp.array(0.5), jnp.array(0.3)
    ax.plot(xl, [float(f(jnp.array(xi), yv, zv)) for xi in xl],
            color='#0072BD', lw=1.5, label='f')
    ax.plot(xl, [float(fhat(jnp.array(xi), yv, zv)) for xi in xl],
            '--', color='#D95319', lw=1.5, label='fhat = f+(g+h)/10')
    ax.legend(fontsize=9)
    ax.set_title('Rank-one f inside rank-3 fhat', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG_DIR, 'FindingRankOne.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    print("FindingRankOne: done")
    return True


if __name__ == "__main__":
    run()
