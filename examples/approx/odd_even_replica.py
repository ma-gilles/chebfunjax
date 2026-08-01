"""Odd and even best approximations.

Faithful replica of approx/OddEven.m by Mohsin Javed and Nick
Trefethen (March 2015): best approximation is nonlinear — the sum of
best approximations to the even and odd parts of f is not the best
approximation to f.

Original: https://www.chebfun.org/examples/approx/OddEven.html
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
from chebfunjax.utils.minimax import minimax

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

GREEN = (0.0, 0.7, 0.0)
AX = (-1, 1, -1.2, 1.2)
XS = np.linspace(-1, 1, 2000)


def _p_eval(res):
    p_cf = cj.chebfun(jnp.asarray(res.coeffs), coeffs=True)
    return lambda x: np.asarray(p_cf(jnp.asarray(x)))


def _pair_plot(fvals, pvals, err, title1, fname):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    axes[0].plot(XS, fvals, 'b', lw=1.3)
    axes[0].plot(XS, pvals, 'r', lw=1.3)
    axes[0].axis(AX)
    axes[0].grid(True)
    axes[0].set_title(title1, fontsize=11)
    axes[1].plot(XS, pvals - fvals, color=GREEN, lw=1.3)
    axes[1].grid(True)
    axes[1].axis(AX)
    axes[1].set_title(f"error = {err:.5g}", fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # Degree-0 example: a Gaussian
    fh = lambda x: jnp.exp(-150 * (x - 0.5) ** 2)  # noqa: E731
    res = minimax(fh, 0)
    fv = np.asarray(fh(jnp.asarray(XS)))
    pv = _p_eval(res)(XS)
    print(f"err = {res.err:.15f}")
    _pair_plot(fv, pv, res.err, "f and its best approximation",
               "OddEven_repl_01.png")

    feh = lambda x: (fh(x) + fh(-x)) / 2  # noqa: E731
    res_e = minimax(feh, 0)
    print(f"erreven = {res_e.err:.15f}")
    _pair_plot(np.asarray(feh(jnp.asarray(XS))), _p_eval(res_e)(XS),
               res_e.err, "Approximation of the even part",
               "OddEven_repl_02.png")

    foh = lambda x: (fh(x) - fh(-x)) / 2  # noqa: E731
    res_o = minimax(foh, 0)
    print(f"errodd = {res_o.err:.15f}")
    _pair_plot(np.asarray(foh(jnp.asarray(XS))), _p_eval(res_o)(XS),
               res_o.err, "Approximation of the odd part",
               "OddEven_repl_03.png")

    psum = _p_eval(res_e)(XS) + _p_eval(res_o)(XS)
    errsum = float(np.max(np.abs(fv - psum)))
    print(f"errsum = {errsum:.15f}")
    _pair_plot(fv, psum, errsum, "combined", "OddEven_repl_04.png")

    # Degree-1 example: a bactrian camel
    fh = lambda x: (jnp.exp(-300 * (x - 0.25) ** 2)  # noqa: E731
                    + jnp.exp(-300 * (x - 0.75) ** 2))
    res = minimax(fh, 1)
    fv = np.asarray(fh(jnp.asarray(XS)))
    print(f"err = {res.err:.15f}")
    _pair_plot(fv, _p_eval(res)(XS), res.err,
               "f and its best approximation", "OddEven_repl_05.png")

    feh = lambda x: (fh(x) + fh(-x)) / 2  # noqa: E731
    res_e = minimax(feh, 1)
    print(f"erreven = {res_e.err:.15f}")
    _pair_plot(np.asarray(feh(jnp.asarray(XS))), _p_eval(res_e)(XS),
               res_e.err, "Approximation of the even part",
               "OddEven_repl_06.png")

    foh = lambda x: (fh(x) - fh(-x)) / 2  # noqa: E731
    res_o = minimax(foh, 1)
    print(f"errodd = {res_o.err:.15f}")
    _pair_plot(np.asarray(foh(jnp.asarray(XS))), _p_eval(res_o)(XS),
               res_o.err, "Approximation of the odd part",
               "OddEven_repl_07.png")

    psum = _p_eval(res_e)(XS) + _p_eval(res_o)(XS)
    errsum = float(np.max(np.abs(fv - psum)))
    print(f"errsum = {errsum:.15f}")
    _pair_plot(fv, psum, errsum, "combined", "OddEven_repl_08.png")


if __name__ == "__main__":
    run()
