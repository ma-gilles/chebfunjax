"""The analytic SVD.

Translation of linalg/AnalyticSVD.m by Yuji Nakatsukasa and
Vanni Noferini (May 2016): the singular values of the matrix family
A t + B(1-t) as functions of t.  Sorted singular values have kinks
where branches cross; flipping signs across the crossings recovers
the analytic SVD, in which singular values may go negative but every
branch is smooth.

rng(10) randn is not bit-reproducible vs MATLAB; the crossing
structure of our draw differs while the phenomenon replicates.

Original: https://www.chebfun.org/examples/linalg/AnalyticSVD.html
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
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')

M = N = 4
FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"AnalyticSVD_{FIG[0]:02d}.png"))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()

    rs = np.random.RandomState(10)
    A = rs.randn(M, N)
    B = rs.randn(M, N)

    def AA(t):
        return A * t + B * (1 - t)

    def sigma_k(t_arr, k):
        t_arr = np.atleast_1d(np.asarray(t_arr, dtype=float))
        out = np.empty_like(t_arr)
        for i, t in enumerate(t_arr.ravel()):
            out.ravel()[i] = np.linalg.svd(AA(t),
                                           compute_uv=False)[k]
        return out.reshape(np.shape(t_arr))

    # sorted singular values: chebfuns with splitting; kinks marked
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for k in range(N):
        f = cj.chebfun(
            lambda t, _k=k: jnp.asarray(sigma_k(np.asarray(t), _k)),
            splitting=True)
        xs = np.linspace(-1, 1, 800)
        ax.plot(xs, np.asarray(f(xs)), lw=2)
        for bp in [float(v) for v in f.domain.breakpoints][1:-1]:
            ax.plot([bp, bp], [0, 4], 'k', lw=0.8)
            ax.plot(bp, float(f(bp)), 'ko', ms=5, mfc='none')
    ax.grid(True)
    ax.set_title("sorted singular values, kinks at crossings",
                 fontsize=12)
    _save(fig)

    def uvsvd(t, i, j, pos):
        """(i, j) element of U (pos=1), S (pos=2) or V (pos=3) of AA(t)."""
        tt = np.asarray(t, dtype=float)
        ta = np.atleast_1d(tt).ravel()
        Us, ss, Vts = np.linalg.svd(A[None] * ta[:, None, None]
                                    + B[None] * (1 - ta)[:, None, None])
        if pos == 1:
            y = Us[:, i, j]
        elif pos == 2:
            y = ss[:, i]
        else:
            y = Vts[:, j, i]
        return jnp.asarray(y.reshape(tt.shape))

    def split(fun):
        return cj.chebfun(fun, splitting=True)

    def breaks(f):
        return np.array([float(v) for v in f.domain.breakpoints][1:-1])

    def flip(f, b):
        return split(lambda t: f(t) * jnp.sign(b - t))

    def plot_usv(sspos, uupos, vvpos):
        fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.7))
        xs = np.linspace(-1, 1, 2001)
        for pos in range(N):
            for ax, f in zip(axes, (sspos[pos], uupos[pos], vvpos[pos])):
                ax.plot(xs, np.asarray(f(xs)), lw=2)
        for ax, ttl in zip(axes, ("singular values", "U", "V")):
            ax.set_title(ttl)
        axes[0].grid(True)
        return fig, axes

    # first entries of U and V and the singular values, as piecewise
    # chebfuns: LAPACK's sign choices introduce jumps
    uupos, sspos, vvpos = [], [], []
    for pos in range(N):
        uupos.append(split(lambda t, _p=pos: uvsvd(t, 0, _p, 1)))
        sspos.append(split(lambda t, _p=pos: uvsvd(t, _p, _p, 2)))
        vvpos.append(split(lambda t, _p=pos: uvsvd(t, 0, _p, 3)))
    fig, _ = plot_usv(sspos, uupos, vvpos)
    _save(fig)

    # flip the sign of sigma past each kink, and with it U or V
    for pos in range(N):
        uu, vv, ss = uupos[pos], vvpos[pos], sspos[pos]
        endsss = breaks(ss)
        ssp = ss.diff()
        sdisc = np.array([], dtype=int)
        if endsss.size:
            spleft = np.asarray(ssp(jnp.asarray(endsss), 'left'))
            spright = np.asarray(ssp(jnp.asarray(endsss), 'right'))
            sdisc = np.nonzero(np.abs(spleft - spright) > 1e-8)[0]
        if sdisc.size:
            uleft = np.asarray(uu(jnp.asarray(endsss[sdisc]), 'left'))
            uright = np.asarray(uu(jnp.asarray(endsss[sdisc]), 'right'))
            ujump = bool(np.all(np.abs(uleft - uright) > 1e-8))
        for ii in sdisc:
            ss = flip(ss, endsss[ii])
            if ujump:
                uu = flip(uu, endsss[ii])
            else:
                vv = flip(vv, endsss[ii])
        uupos[pos], vvpos[pos], sspos[pos] = uu, vv, ss
    fig, _ = plot_usv(sspos, uupos, vvpos)
    _save(fig)

    # remaining jumps in U and V are simultaneous sign flips of both
    for pos in range(N):
        uu, vv = uupos[pos], vvpos[pos]
        endsuu = breaks(uu)
        if endsuu.size:
            uleft = np.asarray(uu(jnp.asarray(endsuu), 'left'))
            uright = np.asarray(uu(jnp.asarray(endsuu), 'right'))
            for ii in np.nonzero(np.abs(uleft - uright) > 1e-8)[0]:
                uu = flip(uu, endsuu[ii])
                vv = flip(vv, endsuu[ii])
        uupos[pos], vvpos[pos] = uu, vv
    fig, axes = plot_usv(sspos, uupos, vvpos)
    for ax in axes:
        ax.grid(True)
    _save(fig)

    # the analytic SVD: every branch is a smooth global chebfun
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    eps = np.finfo(float).eps
    for pos in (0, N - 1):
        uu = cj.chebfun(lambda t, _f=uupos[pos]: _f(t))
        ss = cj.chebfun(lambda t, _f=sspos[pos]: _f(t))
        vv = cj.chebfun(lambda t, _f=vvpos[pos]: _f(t))
        for f, c in ((uu, 'b'), (ss, 'k'), (vv, 'r')):
            cf = np.abs(np.asarray(f.coeffs))
            ax.semilogy(np.arange(len(cf)), cf, '.', color=c, ms=4)
        ax.text(len(uu) + 5, eps * 10, f"$U_{{{pos + 1}1}}$", color='b',
                fontsize=11)
        ax.text(len(ss) + 5, eps / 10, rf"$\sigma_{pos + 1}$", color='k',
                fontsize=11)
        ax.text(len(vv) + 5, eps / 1e3, f"$V_{{{pos + 1}1}}$", color='r',
                fontsize=11)
    ax.set_xlabel("Degree of Chebyshev polynomial", fontsize=9)
    ax.set_ylabel("Magnitude of coefficient", fontsize=9)
    _save(fig)

    print("time_in_seconds =")
    print(f"     {time.time() - t0:.15e}")


if __name__ == "__main__":
    run()
