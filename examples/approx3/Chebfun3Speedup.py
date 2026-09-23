"""Chebfun3 construction speedup: Tucker fiber algorithm vs classic slice-Tucker.

Demonstrates the improved complexity of the Chebfun3f algorithm (fiber-based
Tucker construction) compared to the classic slice-Tucker approach, for both
hard (not low-rank) and easy (low-rank) functions.

Original MATLAB Chebfun: approx3/Chebfun3Speedup.m
by Behnam Hashemi, Christoph Strössner, and Nick Trefethen, March 2023.
See https://www.chebfun.org/examples/approx3/Chebfun3Speedup.html
Copyright 2023 by The University of Oxford and The Chebfun Developers.

chebfunjax implements only the Chebfun3f constructor (the MATLAB default
since March 2023); the legacy ``chebfun3(ff,'classic')`` slice-Tucker
constructor (@chebfun3/chebfun3classic.m) is not ported, so the "classic"
timings of the original are omitted here.
"""

import matplotlib

matplotlib.use("Agg")
import os
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.plotting import CHEBFUN_RED, chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)), "docs", "images", "approx3"
)
os.makedirs(_IMG_DIR, exist_ok=True)


def _timings(ff, kk):
    t2, m2 = [], []
    for k in kk:
        tic = time.time()
        f2 = chebfun3(lambda x, y, z, _k=k: ff(_k, x, y, z))
        t2.append(time.time() - tic)
        m2.append(max(f2.length()))   # MATLAB length(f) = max([m n p])
    return np.array(m2), np.array(t2)


def _figure(m2, t2, guide, text, title, fname):
    fig, ax = plt.subplots()
    ax.loglog(m2, t2, '.', color=CHEBFUN_RED, ms=16 * 0.6)
    ax.loglog(m2, guide(m2), '--k')
    ax.text(*text)
    ax.set_xlabel('length m')
    ax.set_ylabel('time (s)')
    ax.legend(['new'], loc='upper left')
    ax.grid(True)
    ax.set_title(title)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG_DIR, fname))
    plt.close(fig)


def run():
    kk = 2.0 ** np.arange(0, 2.7 + 1e-12, 1 / 3)
    m2, t2 = _timings(lambda k, x, y, z: jnp.tanh(k * (x + y + z)), kk)
    _figure(m2, t2, lambda m: 2.5e-6 * m**3, (110, 1.5, 'O(m^3)'),
            'Hard functions (not low rank)', 'Chebfun3Speedup_01.png')

    kk = 2.0 ** np.arange(0, 7 + 1e-12, 1 / 3)
    m2, t2 = _timings(
        lambda k, x, y, z: 1.0 / (1 + k * (x**2 + y**2 + z**2)), kk)
    _figure(m2, t2, lambda m: 4e-4 * m, (235, 0.22, 'O(m)'),
            'Easy functions (low rank)', 'Chebfun3Speedup_02.png')

    tic = time.time()
    chebfun3(lambda x, y, z: jnp.tanh(10 * (x + y)) * jnp.cos(z))
    print(f"Elapsed time is {time.time() - tic:.6f} seconds.")
    # tic, f = chebfun3(ff,'classic'); toc -- classic constructor not ported.


if __name__ == "__main__":
    run()
