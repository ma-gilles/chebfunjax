"""Introducing breakpoints speeds up difficult calculations.

Translation of ode-linear/Breakpoints.m by Nick Trefethen
(November 2016): boundary- and interior-layer BVPs solved first on
plain domains and then with strategically placed breakpoints, showing
dramatic reductions in length; plus a nonlinear shock problem with
zero, one, and two breakpoints.

Original: https://www.chebfun.org/examples/ode-linear/Breakpoints.html
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

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')

FIG = [0]
HEAD = "        ep      pos(max(u))    length(u)    time (secs.) "
FS = "%12.1e %14.9f %9d %14.2f"


def _mnum(v):
    """MATLAB ``format long`` display of a real scalar (value line)."""
    v = float(v)
    if v == int(v) and abs(v) < 1e9:
        return f"{int(v):6d}"
    if 1e-3 <= abs(v) < 100:
        return f"{v:.15f}".rjust(20)
    return f"{v:.15e}".rjust(26)


def _mcol(vals):
    """MATLAB ``format long`` display lines of a real column vector."""
    vals = [float(v) for v in vals]
    if all(v == int(v) for v in vals):
        return [f"{int(v):6d}" for v in vals]
    m = max(abs(v) for v in vals)
    if 1e-3 <= m < 100:
        return [f"{v:.15f}".rjust(20) for v in vals]
    e = int(np.floor(np.log10(m))) + (1 if m < 1e-3 else 0)
    return [f"   1.0e{e:+03d} *"] + [f"{v / 10.0**e:.15f}".rjust(20)
                                    for v in vals]


def _mdisp(name, v):
    """Print ``name = v`` as MATLAB does (scalar or column vector)."""
    print(f"{name} =")
    if np.ndim(v) == 0:
        print(_mnum(v))
    else:
        print("\n".join(_mcol(np.ravel(v))))


def _num2str(v):
    """MATLAB num2str of a positive scalar."""
    return f"{v:.{max(1, int(np.floor(np.log10(abs(v)))) + 5)}g}"


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"Breakpoints_{FIG[0]:02d}.png"))
    plt.close(fig)


def _pos_max(u):
    mx = u.max()
    return float(mx[0]) if isinstance(mx, tuple) else float("nan")


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Problem A on plain [0,1], ep = 1e-1..1e-5
    def LA(ep, dom):
        N = Chebop(lambda x, u, _e=ep: -_e * u.diff(2) - u.diff(),
                   domain=dom)
        N.bc = "dirichlet"
        return N

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    print(HEAD, flush=True)
    tt = np.linspace(0, 1, 2000)
    for k in range(1, 6):
        ep = 10.0**(-k)
        t0 = time.time()
        u = LA(ep, (0, 1)).solve(1.0)
        el = time.time() - t0
        print(FS % (ep, _pos_max(u), len(u), el), flush=True)
        ax.plot(tt, np.asarray(u(tt)), 'b', lw=1.0)
    ax.grid(True)
    ax.axis([-0.03, 1, 0, 1.03])
    ax.set_title(r"Boundary layers for $\epsilon$ = 1e-1, ..., 1e-5")
    _save(fig)

    # Problem A with a moving breakpoint, ep down to 1e-8.  As in the
    # MATLAB code, fprintf runs before [val,pos] = max(u), so each row
    # shows the previous solution's pos.
    pos = _pos_max(u)
    print(HEAD, flush=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for k in range(1, 9):
        ep = 10.0**(-k)
        dom = (0.0, min(0.5, 40 * ep), 1.0)
        t0 = time.time()
        u = LA(ep, dom).solve(1.0)
        el = time.time() - t0
        print(FS % (ep, pos, len(u), el), flush=True)
        pos = _pos_max(u)
        if k == 3:
            ax.plot(tt, np.asarray(u(tt)), 'b', lw=1.2)
            bp = dom[1]
            ax.plot(bp, float(u(jnp.asarray(bp))), '.r', ms=12)
    ax.grid(True)
    ax.axis([-0.03, 1, 0, 1.03])
    ax.set_title(r"The same computed with a breakpoint, "
                 r"$\epsilon$ = 1e-3")
    _save(fig)
    print("u =")
    print(repr(u))

    # Problem B: interior layers, plain [-2,2]
    def LB(ep, dom):
        N = Chebop(lambda x, u, _e=ep: _e * u.diff(2)
                   + x * u.diff() + x * u, domain=dom)
        N.lbc = -4
        N.rbc = 2
        return N

    t2 = np.linspace(-2, 2, 2400)
    print(HEAD, flush=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for k in range(1, 5):
        ep = 10.0**(-k)
        t0 = time.time()
        u = LB(ep, (-2, 2)).solve(0.0)
        el = time.time() - t0
        print(FS % (ep, _pos_max(u), len(u), el), flush=True)
        ax.plot(t2, np.asarray(u(t2)), 'm', lw=1.0)
    ax.grid(True)
    ax.axis([-2, 2, -6, 17])
    ax.set_title(r"Interior layers for $\epsilon$ = 1e-1, ..., 1e-4")
    _save(fig)

    # Problem B with two breakpoints, ep down to 1e-8
    print(HEAD, flush=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for k in range(1, 9):
        ep = 10.0**(-k)
        d = min(0.5, 10 * np.sqrt(ep))
        dom = (-2.0, -d, d, 2.0)
        t0 = time.time()
        u = LB(ep, dom).solve(0.0)
        el = time.time() - t0
        print(FS % (ep, _pos_max(u), len(u), el), flush=True)
        if k == 4:
            ax.plot(t2, np.asarray(u(t2)), 'm', lw=1.2)
            for bp in (-d, d):
                ax.plot(bp, float(u(jnp.asarray(bp))), '.k', ms=12)
    ax.grid(True)
    ax.axis([-2, 2, -6, 17])
    ax.set_title(r"The same computed with two breakpoints, "
                 r"$\epsilon$ = 1e-4")
    _save(fig)
    print("u =")
    print(repr(u))

    # Nonlinear problem with 0, 1, 2 breakpoints
    for k, (doms, title) in enumerate((
        ((0.0, 1.0), "Nonlinear problem"),
        ((0.0, 1 / 3, 1.0), "Same but with one breakpoint"),
        ((0.0, 0.30, 0.36, 1.0), "Same but with two breakpoints"),
    )):
        N = Chebop(lambda u: 0.005 * u.diff(2) + u * u.diff() - u,
                   domain=doms)
        N.lbc = -7.0 / 6
        N.rbc = 3.0 / 2
        t0 = time.time()
        u = N.solve(0.0)
        el = time.time() - t0
        print("u =")
        print(repr(u))
        if k > 0:
            _mdisp("t", el)
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        ax.plot(tt, np.asarray(u(tt)), lw=1.4)
        for bp in doms[1:-1]:
            ax.plot(bp, float(u(jnp.asarray(bp))), '.r', ms=12)
        ax.grid(True)
        ax.set_title(f"{title}: time {_num2str(el)} secs")
        _save(fig)


if __name__ == "__main__":
    run()
