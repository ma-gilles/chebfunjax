"""Checkmark functions and their best approximations.

Faithful replica of approx/Checkmark.m by Nick Trefethen (February
2020): the degree-n minimax error E_n(alpha) of |x-alpha| as a
function of the kink location alpha.

Original: https://www.chebfun.org/examples/approx/Checkmark.html
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
from chebfunjax.utils.minimax import minimax

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

RED = (0.9, 0, 0)
BLUE = (0, 0, 0.9)


def e_of(a, n):
    if a >= 1.0 - 1e-12:
        a = 1.0 - 1e-12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = minimax(lambda x, _a=a: jnp.abs(x - _a), n,
                    breakpoints=[float(a)])
    return float(r.err)


def run():
    os.makedirs(_IMG, exist_ok=True)
    t0 = time.time()

    E = {}
    for n in range(1, 8):
        def en_vals(a, _n=n):
            arr = np.atleast_1d(np.asarray(a, dtype=np.float64))
            out = [e_of(float(v), _n) for v in arr.ravel()]
            return jnp.asarray(out, dtype=jnp.float64).reshape(arr.shape)

        # fixed 33-point construction (E_n is smooth; MATLAB used
        # adaptive eps=1e-6 -- 33 Chebyshev points give ~1e-8 here)
        en = cj.chebfun(en_vals, domain=(0.0, 1.0), n=33)
        # newDomain(join(flipud(en), en), [-1, 1]): E_n is even in alpha
        E[n] = en
        print(f"n={n}: len {len(en)}")

    alphas = np.linspace(-1, 1, 801)

    def En(n, al):
        return np.asarray(E[n](jnp.asarray(np.abs(al))))

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(alphas, En(2, alphas), color=RED, lw=1.4, label="n = 2")
    ax.plot(alphas, En(3, alphas), color=BLUE, lw=1.4, label="n = 3")
    ax.grid(True)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$E_n(\alpha)$")
    ax.set_title("n = 2 and 3", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Checkmark_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # local minima of E_3 on [-1, 1].  The 33-point interpolant carries
    # ~1e-4 Remez-convergence noise, so refine the interior minimum by
    # direct scalar minimization of e_of.
    from scipy.optimize import minimize_scalar
    opt = minimize_scalar(lambda a: e_of(a, 3), bounds=(0.4, 0.6),
                          method="bounded",
                          options={"xatol": 1e-8})
    a_min, v_min = float(opt.x), float(opt.fun)
    print("val =")
    for v in (e_of(1.0, 3), v_min, v_min, e_of(1.0, 3)):
        print(f"   {v:.15f}")
    print("pos =")
    for p in (-1.0, -a_min, a_min, 1.0):
        print(f"  {p:.15f}")

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for n in range(1, 8):
        ax.plot(alphas, En(n, alphas),
                color=BLUE if n % 2 == 1 else RED, lw=1.4)
    ax.grid(True)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$E_n(\alpha)$")
    ax.set_title("n = 1,2,...,7", fontsize=12)
    ax.set_ylim(0, 0.5)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Checkmark_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")


if __name__ == "__main__":
    run()
