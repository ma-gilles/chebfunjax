"""Black-Scholes PDE using operator exponential.

Translation of pde/BSexponential.m by Toby Driscoll (June 2014):
the Black-Scholes equation

    v_t = -(sigma^2/2) s^2 v_ss - r s v_s + r v

on [0, 500] (sigma = 0.45, r = 0.03) with v(0) = 0 and v' -> 1 as
s -> infinity, solved by operator exponential.  The inhomogeneous
right condition is removed by the particular-solution trick
(u = A\\0 with Bu = q; propagate w = v - u homogeneously), and the
payoff v_T = max(0, s - 50) is advanced to t = 0.1..0.5 with expm.

Original: https://www.chebfun.org/examples/pde/BSExponential.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'pde')

D = (0.0, 500.0)
SIGMA, R = 0.45, 0.03


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


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    A = Chebop(lambda s, v: (-SIGMA**2 / 2 * s**2 * v.diff(2)
                             - R * s * v.diff() + R * v), domain=D)
    A.lbc = 0.0
    A.rbc = lambda v: v.diff() - 1     # replaces v -> s as s -> inf

    # Particular solution carrying the inhomogeneous BC.
    u = A.solve(0.0)
    A.rbc = 0.0                        # homogeneous BCs for w = v - u

    s = chebfun(lambda t: t, domain=D)
    vT = (s - 50).maximum(0.0)         # payoff at maturity
    wT = vT - u

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    xx = np.linspace(40, 60, 600)
    ax.plot(xx, np.asarray(vT(xx)), lw=2)
    ax.set_ylim(-0.5, 14)
    ax.set_xlim(40, 60)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "BSExponential_01.png"))
    v55 = None
    for t in np.arange(0.1, 0.51, 0.1):
        w = A.expm(-t, wT, n=700)
        v = w + u
        ax.plot(xx, np.asarray(v(xx)), 'k', lw=1.0)
        v55 = float(v(55.0))
    _savefig(fig, os.path.join(_IMG, "BSExponential_02.png"))    # hold on
    plt.close(fig)

    # Value of the option at s = 55, six months before maturity.
    _mdisp("ans", v55)
    print("w =")
    print(repr(w))

    # The second derivative is continuous across the strike s = 50.
    eps = float(np.finfo(float).eps)
    wss = w.diff(2)
    jump2 = float(wss(50 + 100 * eps)) - float(wss(50 - 100 * eps))
    _mdisp("jump2", jump2)


if __name__ == "__main__":
    run()
