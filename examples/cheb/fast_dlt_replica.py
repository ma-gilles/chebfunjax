"""The fast discrete Legendre transform.

Faithful replica of cheb/FastDLT.m by Nick Hale and Alex Townsend
(March 2015): converting between Legendre coefficients and values at
Legendre points, a frequency-domain look at a modified Legendre
polynomial, and the closeness of Legendre and Chebyshev points.

Original: https://www.chebfun.org/examples/cheb/FastDLT.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.polynomials import legpoly
from chebfunjax.utils.quadrature import chebpts, legpts
from chebfunjax.utils.transforms import legcoeffs2legvals, legvals2legcoeffs

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'cheb')


def run():
    os.makedirs(_IMG, exist_ok=True)

    # Timing of the DLT at N = 1e4
    c = jnp.asarray(np.random.RandomState(5489).standard_normal(10**4))
    t0 = time.time()
    legcoeffs2legvals(c).block_until_ready()
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")

    # Frequency content of sqrt(sin(theta)) * P_N(cos(theta)):
    # concentrated near the wavenumber N (here N = 2000; MATLAB uses
    # 1e4, whose degree-1e4 Clenshaw evaluation is slow in our setup)
    Ndeg = 2000
    P = cj.chebfun(legpoly(Ndeg), coeffs=True)
    theta = np.linspace(0, 2 * np.pi, 4 * Ndeg)
    with np.errstate(invalid="ignore"):
        sig = (np.sqrt(np.abs(np.sin(theta)))
               * np.asarray(P(jnp.asarray(np.cos(theta)))))
    modes = np.abs(np.fft.fft(sig))
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.plot(modes[:2 * Ndeg], lw=1.2)
    ax.set_xlabel("Frequency bins", fontsize=13)
    ax.set_ylabel("Magnitude", fontsize=13)
    ax.set_title("Frequency analysis of modified Legendre polynomial",
                 fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "FastDLT_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Legendre points are close to Chebyshev (1st-kind) points in angle
    NN = np.unique(np.floor(np.logspace(1, 3.5, 40)).astype(int))
    maxdiff = []
    for N in NN:
        t_leg = np.arccos(np.asarray(legpts(int(N))[0]))
        t_cheb = np.arccos(np.asarray(chebpts(int(N), kind=1)))
        maxdiff.append(np.max(np.abs(np.sort(t_leg)
                                     - np.sort(t_cheb))))
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.loglog(NN, maxdiff, '-', lw=2,
              label=r"$\|x^{leg} - x^{cheb}\|_\infty$")
    ax.loglog(NN, 0.83845 / NN, '--', lw=2, label="Theoretical bound")
    ax.legend()
    ax.set_xlabel("N", fontsize=13)
    ax.set_ylabel("Max abs diff", fontsize=13)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "FastDLT_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Roundtrip DLT/IDLT accuracy
    f = cj.chebfun(lambda x: 1.0 / (1 + 10000 * x**2))
    from chebfunjax.utils.transforms import cheb2leg
    c_leg = cheb2leg(f.coeffs)
    t0 = time.time()
    f_leg = legcoeffs2legvals(c_leg)
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")
    back = legvals2legcoeffs(f_leg)
    print("ans =")
    print(f"     {float(jnp.max(jnp.abs(back - c_leg))):.15e}")


if __name__ == "__main__":
    run()
