"""Random polynomials and their roots in [-1,1].

Translation of roots/RandomPolys.m by Nick Trefethen
(June 2014): degree-n polynomials with random normalized-Legendre
coefficients have, on average, a fraction 1/sqrt(3) = 0.5774 of
their roots in [-1,1] as n -> infinity.

This replay uses the captured MATLAB R2025b rng(default) input sequence.
It does not implement or claim general MATLAB random-stream parity.

Original: https://www.chebfun.org/examples/roots/RandomPolys.html
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
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.transforms import leg2cheb

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.environ.get("RANDOMPOLYS_IMAGE_DIR") or os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(
        _IMG, f"RandomPolys_{FIG[0]:02d}.png"), layout="matlab")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    capture = os.environ.get("RANDOMPOLYS_CAPTURE_DIR")
    if capture:
        os.makedirs(capture, exist_ok=True)
    fixture = os.environ.get("RANDOMPOLYS_INPUT_FIXTURE")
    if not fixture:
        fixture = os.path.join(os.path.dirname(__file__), "randompolys_inputs.npz")
    with np.load(fixture, allow_pickle=False) as inputs:
        cleg30 = inputs["cleg30"].copy()
        cleg1000 = inputs["cleg1000"].copy()
    assert cleg30.shape == (31,) and cleg1000.shape == (1001, 10)

    n = 30
    cleg = cleg30
    ccheb = np.asarray(leg2cheb(jnp.asarray(cleg), normalize=True))
    p = cj.Chebfun.from_coeffs(jnp.asarray(ccheb))

    rr = np.asarray(p.roots())
    if capture:
        np.savez(os.path.join(capture, "degree30.npz"), cleg=cleg, ccheb=ccheb, roots=rr)
    fig, ax = plt.subplots(figsize=(6.0, 2.7), dpi=100)
    p.plot(ax=ax, color="b", linewidth=1.2)
    ax.axis([-1.1, 1.1, -n, n])
    ax.grid(True)
    ax.plot(rr, np.asarray(p(rr)), '.r', ms=6)
    ratio = len(rr) / n
    ax.set_title(f"fraction of roots in [-1,1]: {ratio:.5g}",
                 fontsize=10)
    _save(fig)

    r = np.asarray(p.roots(all_roots=True))
    if capture:
        np.save(os.path.join(capture, "degree30_all_roots.npy"), r)
    fig, ax = plt.subplots(figsize=(6.0, 2.7), dpi=100)
    ax.plot([-1, 1], [0, 0], 'k')
    ax.grid(True)
    ax.plot(r.real, r.imag, '.r', ms=6)
    ax.set_xlim(-2.5, 2.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks(range(-2, 3))
    _save(fig)

    n = 1000
    data = []
    for _k in range(10):
        cleg = cleg1000[:, _k]
        ccheb = np.asarray(leg2cheb(jnp.asarray(cleg),
                                    normalize=True))
        p = cj.Chebfun.from_coeffs(jnp.asarray(ccheb))
        rr = np.asarray(p.roots())
        ratio = len(rr) / n
        data.append(ratio)
        if capture:
            np.savez(os.path.join(capture, f"degree1000_{_k + 1:02d}.npz"), cleg=cleg, ccheb=ccheb, roots=rr, ratio=ratio)
        print(f"fraction of roots in [-1,1]: {ratio:g}")
    if capture:
        np.save(os.path.join(capture, "data.npy"), np.asarray(data))
    print("ans =")
    print(f"   {np.mean(data):.15f}")


if __name__ == "__main__":
    run()
