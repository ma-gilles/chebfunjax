"""Inpainting in one dimension.

Faithful replica of approx/Inpainting1D.m by Yuji Nakatsukasa and Nick
Trefethen (November 2019): recovering a smooth function corrupted on
part of its domain — L1 fitting recovers it to nearly machine
precision where L2 and Linf fail.

Original: https://www.chebfun.org/examples/approx/Inpainting1D.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.minimax import minimax
from chebfunjax.utils.randnfun import randnfun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

XS = np.linspace(-1, 1, 3000)


def _plot(vals, title, fname, color='C0'):
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(XS, vals, color, lw=1.4)
    ax.grid(True)
    ax.set_title(title, fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    t0 = time.time()

    x = cj.chebfun(lambda t: t)
    smooth = 0.3 + x**2 + (0.3 * x).exp()
    # MATLAB rng(1) randnfun noise: randn streams are not reproducible
    # outside MATLAB, so this replica uses its own smooth random noise.
    noise = randnfun(0.1, key=jax.random.PRNGKey(1))
    corrupted = smooth.maximum(noise)
    _plot(np.asarray(corrupted(jnp.asarray(XS))),
          "corrupted smooth function", "Inpainting1D_repl_01.png")

    n = len(smooth) - 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p1 = corrupted.polyfitL1(n)
    _plot(np.asarray(p1(jnp.asarray(XS))), "L1 fit",
          "Inpainting1D_repl_02.png")
    err1 = float((p1 - smooth).norm(np.inf))
    print("err1 =")
    print(f"     {err1:.15e}")

    p2 = corrupted.polyfit(n - 2)
    _plot(np.asarray(p2(jnp.asarray(XS))), "L2 fit",
          "Inpainting1D_repl_03.png")
    err2 = float((p2 - smooth).norm(np.inf))
    print("err2 =")
    print(f"   {err2:.15f}")
    _plot(np.asarray((p2 - smooth)(jnp.asarray(XS))), "L2 error",
          "Inpainting1D_repl_04.png", color='k')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimax(lambda t: corrupted(t), n - 2)
    pinf = cj.chebfun(jnp.asarray(res.coeffs), coeffs=True)
    _plot(np.asarray(pinf(jnp.asarray(XS))), "Linf fit",
          "Inpainting1D_repl_05.png")
    errinf = float((pinf - smooth).norm(np.inf))
    print("errinf =")
    print(f"   {errinf:.15f}")
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")


if __name__ == "__main__":
    run()
