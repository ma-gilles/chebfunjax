"""Nonsmoothness of the field of values boundary.

Faithful replica of linalg/NonsmoothFOV.m by Nick Trefethen
(March 2019): how smooth is the boundary curve of the field of
values?  Chebyshev/Fourier coefficient decay, AAA pole clustering,
and derivative jumps reveal near-corners, both with respect to the
Johnson angle and the true boundary angle.

The first matrix is a randn draw (not bit-reproducible vs MATLAB);
the 5x5 complex matrix of Caldwell-Greenbaum-Li is hardcoded in the
example and matches exactly.

Original: https://www.chebfun.org/examples/linalg/NonsmoothFOV.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, plotregion
from chebfunjax.utils.aaa import aaa

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"NonsmoothFOV_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fov_point(A, theta_arr):
    theta_arr = np.atleast_1d(np.asarray(theta_arr, dtype=float))
    out = np.empty(theta_arr.shape, dtype=complex)
    for i, th in enumerate(theta_arr.ravel()):
        r = np.exp(1j * th)
        H = (r * A + np.conj(r) * A.conj().T) / 2
        _, V = np.linalg.eigh(H)
        v = V[:, -1]
        out.ravel()[i] = (v.conj() @ A @ v) / (v.conj() @ v)
    return out.reshape(theta_arr.shape)


def _plotcoeffs(c, title, fourier=False):
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    cc = np.abs(np.asarray(c.funs[0].tech.coeffs))
    cc = np.maximum(cc, 1e-20)
    ax.semilogy(np.arange(len(cc)), cc, '.', ms=3)
    ax.grid(True)
    ax.set_title(title, fontsize=11)
    _save(fig)


def _curve_plot(A, c, xlim):
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    t = np.linspace(0, 2 * np.pi, 1200)
    v = np.asarray(c(t))
    ax.plot(v.real, v.imag, lw=2)
    w = np.linalg.eigvals(A)
    ax.plot(w.real, w.imag, '.k', ms=7)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.grid(True)
    ax.set_title("field of values and eigenvalues", fontsize=11)
    _save(fig)


def _analyze(A, xlim):
    op = lambda t: jnp.asarray(_fov_point(A, np.asarray(t)))  # noqa: E731
    c = cj.chebfun(op, domain=(0.0, 2 * np.pi))
    _curve_plot(A, c, xlim)
    print("ans =")
    print(f"   {len(c)}")
    _plotcoeffs(c, "Chebyshev coefficients wrt Johnson angle t")

    ct = cj.chebfun(lambda t: jnp.asarray(c(t)),
                    domain=(0.0, 2 * np.pi), trig=True)
    print("ans =")
    print(f"   {len(ct)}")
    _plotcoeffs(ct, "Fourier coefficients wrt Johnson angle t",
                fourier=True)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    plotregion(c, ax=ax, title="")
    ax.set_aspect("equal")
    ax.grid(True)
    xx = np.linspace(0, 2 * np.pi, 1000)
    _, poles, *_ = aaa(jnp.asarray(np.asarray(c(xx))),
                       jnp.asarray(xx), tol=1e-10)
    poles = np.asarray(poles)
    ax.plot(poles.real, poles.imag, '.r', ms=6)
    ax.set_ylim(-1.6, 1.6)
    _save(fig)

    ac = c.abs()
    dac = ac.diff()
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(xx, np.asarray(dac(xx)), lw=1.2)
    ax.grid(True)
    ax.set_xlabel("t")
    ax.set_title("derivative of abs(c) wrt Johnson angle t",
                 fontsize=11)
    _save(fig)

    # true boundary angle a(t) = 2*pi + unwrap(angle(c))
    tt = np.linspace(0, 2 * np.pi, 8000)
    av = 2 * np.pi + np.unwrap(np.angle(np.asarray(c(tt))))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(tt, av, lw=1.2)
    ax.grid(True)
    ax.set_xlabel("Johnson angle t")
    ax.set_ylabel("true angle a")
    _save(fig)

    # invert a(t) on a dense monotone grid and recompose: sample the
    # curve at equispaced TRUE angles and build the periodic chebfun
    # from those values (equivalent to MATLAB's c(inv(a)) composition)
    if av[-1] < av[0]:
        av, tt2 = av[::-1], tt[::-1]
    else:
        tt2 = tt
    a0, a1 = av[0], av[-1]
    m = 4096
    sgrid = a0 + (a1 - a0) * np.arange(m) / m
    t_of_a = np.interp(sgrid, av, tt2)
    dvals = np.asarray(c(t_of_a))
    ss = np.linspace(0, 2 * np.pi, 2000)
    from chebfunjax.chebfun1d.chebfun import Chebfun, Domain, _Piece
    from chebfunjax.tech.trigtech import Trigtech
    tech = Trigtech.from_values(jnp.asarray(dvals)).simplify()
    dfun = Chebfun(funs=[_Piece(tech=tech,
                                interval=(0.0, 2 * np.pi))],
                   domain=Domain((0.0, 2 * np.pi)))
    print("ans =")
    print(f"   {len(dfun)}")
    _plotcoeffs(dfun, "Fourier coefficients wrt true angle a",
                fourier=True)

    add = dfun.abs().diff()
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(ss, np.asarray(add(ss)), lw=1.2)
    ax.grid(True)
    ax.set_xlabel("a")
    ax.set_title("derivative of abs(d) wrt true angle a",
                 fontsize=11)
    _save(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    rs = np.random.RandomState(1)
    n = 60
    A = rs.randn(n, n) / np.sqrt(n)
    _analyze(A, (-3, 3))

    A5 = np.array([
        [0.2560 + 0.0573j, 0.0568 + 0.0800j, 0.1597 + 0.2204j,
         -0.1649 + 0.1315j, -0.3639 + 0.0091j],
        [0.4733 + 0.2805j, -0.3192 + 0.1267j, 0.0810 + 0.0687j,
         0.5213 + 0.1574j, -0.0596 + 0.2879j],
        [0.1447 + 0.3037j, 0.2942 + 0.1844j, -0.2918 + 0.0364j,
         -0.2714 + 0.0265j, -0.0849 + 0.2264j],
        [-0.0650 + 0.1360j, 0.0952 + 0.0813j, -0.0503 + 0.0920j,
         -0.1500 + 0.0814j, 0.4742 + 0.1514j],
        [0.1938 + 0.0344j, 0.0419 + 0.1868j, -0.0453 + 0.0988j,
         -0.2207 + 0.2483j, -0.0772 + 0.1793j]])
    _analyze(A5, (-2, 2))


if __name__ == "__main__":
    run()
