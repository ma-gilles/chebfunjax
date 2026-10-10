"""The analytic SVD.

Translation of linalg/AnalyticSVD.m by Yuji Nakatsukasa, Vanni Noferini,
and Nick Trefethen (July 2016). The source uses MATLAB rng(10), followed
by A=randn(4,4), B=randn(4,4). Supply those matrices using --matrix-input;
the default run uses captured MATLAB R2025b matrices from those source commands.
Historical July2016 LAPACK sign choices and cached figure identity remain
unverified. Other input matrices are explicitly unmatched diagnostics.

Original: https://www.chebfun.org/examples/linalg/AnalyticSVD.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, plotcoeffs
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MATRIX_INPUT = Path(_HERE) / "analytic_svd_matlab_rng10.json"
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')

M = N = 4
FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(
        _IMG, f"AnalyticSVD_{FIG[0]:02d}.png"), size=(608, 271))
    plt.close(fig)


def run(matrix_input=_DEFAULT_MATRIX_INPUT):
    """Run the source computation on explicitly supplied 4-by-4 matrices.

    The JSON input contains A, B, and provenance. Input provenance and hashes
    belong in the execution manifest; a non-MATLAB fixture cannot establish
    random-input or cached-figure parity.
    """
    os.makedirs(_IMG, exist_ok=True)
    FIG[0] = 0
    t0 = time.time()
    data = json.loads(Path(matrix_input).read_text())
    A = jnp.asarray(data["A"], dtype=jnp.float64)
    B = jnp.asarray(data["B"], dtype=jnp.float64)
    if A.shape != (M, N) or B.shape != (M, N):
        raise ValueError("AnalyticSVD requires two 4-by-4 matrices")
    if not data.get("provenance"):
        raise ValueError("Matrix input must state its provenance")
    if not bool(jnp.all(jnp.isfinite(A)) & jnp.all(jnp.isfinite(B))):
        raise ValueError("AnalyticSVD matrix input must be finite")

    def uvsvd(t, i, j=0, pos=2):
        # Native UVSVD always computes [U,S,V]=svd(A,0), including the
        # singular-value-only caller. Public vectorize supplies scalar t.
        U, singular, Vh = jnp.linalg.svd(A * t + B * (1 - t),
                                         full_matrices=False)
        if pos == 1:
            return U[i, j]
        if pos == 2:
            return singular[i]
        return jnp.conj(Vh[j, i])

    # Native sorted singular-value loop, including marked breakpoints.
    fig, ax = plt.subplots()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k in range(N):
        f = cj.chebfun(lambda t, _k=k: uvsvd(t, _k),
                       splitting=True, vectorize=True)
        f.plot(ax=ax, color=colors[k], linewidth=2)
        for bp in f.domain.breakpoints[1:-1]:
            x = float(bp)
            ax.plot([x, x], [0, 4], 'k')
            ax.plot(x, float(f(x)), 'ko', mfc='none')
    ax.grid(True)
    _save(fig)

    def split(fun):
        return cj.chebfun(fun, splitting=True, vectorize=True)

    def breaks(f):
        return jnp.asarray(f.domain.breakpoints[1:-1])

    def flip(f, b):
        return split(lambda t: f(t) * jnp.sign(b - t))

    def plot_usv(sspos, uupos, vvpos):
        fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.7))
        for pos in range(N):
            for ax, f in zip(axes, (sspos[pos], uupos[pos], vvpos[pos])):
                f.plot(ax=ax, color=colors[pos], linewidth=2)
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
        sdisc = jnp.array([], dtype=int)
        if endsss.size:
            spleft = jnp.asarray(ssp(jnp.asarray(endsss), 'left'))
            spright = jnp.asarray(ssp(jnp.asarray(endsss), 'right'))
            sdisc = jnp.nonzero(jnp.abs(spleft - spright) > 1e-8)[0]
        if sdisc.size:
            uleft = jnp.asarray(uu(jnp.asarray(endsss[sdisc]), 'left'))
            uright = jnp.asarray(uu(jnp.asarray(endsss[sdisc]), 'right'))
            # MATLAB if(vector) is true only when every element is true.
            ujump = bool(jnp.all(jnp.abs(uleft - uright) > 1e-8))
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
            uleft = jnp.asarray(uu(jnp.asarray(endsuu), 'left'))
            uright = jnp.asarray(uu(jnp.asarray(endsuu), 'right'))
            for ii in jnp.nonzero(jnp.abs(uleft - uright) > 1e-8)[0]:
                uu = flip(uu, endsuu[ii])
                vv = flip(vv, endsuu[ii])
        uupos[pos], vvpos[pos] = uu, vv
    fig, axes = plot_usv(sspos, uupos, vvpos)
    for ax in axes:
        ax.grid(True)
    _save(fig)

    # the analytic SVD: every branch is a smooth global chebfun
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    eps = jnp.finfo(jnp.float64).eps
    for pos in (0, N - 1):
        uu = cj.chebfun(lambda t, _f=uupos[pos]: _f(t))
        ss = cj.chebfun(lambda t, _f=sspos[pos]: _f(t))
        vv = cj.chebfun(lambda t, _f=vvpos[pos]: _f(t))
        for f, c in ((uu, 'b'), (ss, 'k'), (vv, 'r')):
            plotcoeffs(f, ax=ax, fmt=c + '.', source=True,
                       linewidth=2, hold=bool(ax.lines))
        ax.text(len(uu) + 5, eps * 10, f"$U_{{{pos + 1}1}}$", color='b',
                fontsize=14)
        ax.text(len(ss) + 5, eps / 10, rf"$\sigma_{pos + 1}$", color='k',
                fontsize=14)
        ax.text(len(vv) + 5, eps / 1e3, f"$V_{{{pos + 1}1}}$", color='r',
                fontsize=14)
    _save(fig)

    print("time_in_seconds =")
    print(f"     {time.time() - t0:.15e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-input", default=_DEFAULT_MATRIX_INPUT,
                        help="JSON containing A, B and explicit provenance")
    run(parser.parse_args().matrix_input)
