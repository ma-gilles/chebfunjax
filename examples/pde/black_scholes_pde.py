"""Black-Scholes PDE via the operator exponential.

Faithful port of pde/BSExponential.m by Toby Driscoll (June 2014).  Prices a
European call by evolving the Black-Scholes operator with ``expm`` (the
operator exponential) rather than time stepping.  The far-field boundary
``v -> s`` is handled by a steady correction ``u = A\\0`` (with the
inhomogeneous Neumann condition ``v'(s_max) = 1``); the adjusted variable
``w = v - u`` then satisfies homogeneous boundary conditions and
``w_t = A w``, integrated by ``w = expm(A, -t, wT)``.

Original: https://www.chebfun.org/examples/pde/BSExponential.html
Copyright 2014 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): this example exercises two chebop features on
the Black-Scholes operator ``A = -sigma^2/2 s^2 D^2 - r s D + r`` over the
large, *singular* interval ``[0, 500]`` (the leading coefficient ``s^2``
vanishes at ``s = 0``).  Two scheme-dependent walls result:

1. ``u = A\\0`` (steady state).  On a non-singular interval the operator and
   the inhomogeneous Robin condition solve exactly -- verified on ``[1,2]``,
   where ``A\\0`` returns ``u = s`` to 1e-9.  On ``[0,500]`` the s=0
   endpoint makes the collocation matrix singular: our dense collocation
   returns a function that meets both boundary conditions but leaves an
   interior residual ``A(u) ~ 1.7`` near ``s=500`` (MATLAB's ultraspherical
   discretization handles the vanishing leading coefficient; ours does not).
   The exact steady solution is analytically ``u(s) = s`` (the ``s^{-0.293}``
   mode is killed by ``u(0)=0``, and ``u'(500)=1`` fixes the slope), so we
   substitute it -- this is the exact ``A\\0`` answer, not an approximation.

2. ``expm(A, -t, wT)`` with a corner.  ``wT = max(0,s-50) - u`` has a corner
   at the strike ``s=50``, so the evolved ``w`` is only piecewise smooth and
   the fixed-size operator exponential converges algebraically:
   v(55) = 9.8504 at n=256 vs the published 9.849887661936435 (relative
   error ~5e-5; n=192 gives 9.8456, n=384 gives 9.8514 -- it brackets the
   MATLAB value but no fixed n reproduces its 16 displayed digits).  The
   vertical scale of w, 49.25, matches to ~5 figures; the chebfun length
   (MATLAB's adaptive 97 vs our fixed 256) is scheme-dependent; and jump2,
   the second-derivative jump at the corner, is a numerically-zero residual
   (~4e-17 vs published -2.4e-17).  Adaptive ``expm`` (auto n) currently
   segfaults on this operator, so a fixed n is used.
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import CHEBFUN_BLUE, chebfun_style

chebfun_style()

_HERE = os.path.dirname(os.path.abspath(__file__))

_N = 256   # fixed collocation size (adaptive expm segfaults on this operator)


def run():
    d = (0.0, 500.0)
    sigma = 0.45
    r = 0.03
    s = chebfun(lambda x: x, domain=d)

    # A = -sigma^2/2 s^2 v'' - r s v' + r v
    op = lambda s, v: -sigma**2 / 2 * s**2 * v.diff(2) - r * s * v.diff() + r * v
    A = Chebop(op, domain=d)
    A.lbc = 0.0
    A.rbc = lambda v: v.diff() - 1     # replaces v -> s as s -> infinity

    # u = A\0.  The exact steady state is u = s (see the module docstring:
    # our dense collocation cannot solve the s=0-singular operator, so we
    # use the analytically exact solution rather than a failing discretization).
    u = s

    A.rbc = 0.0                        # homogeneous BCs for the evolution

    vT = chebfun(lambda x: np.maximum(0.0, x - 50.0), domain=d)
    wT = vT - u                        # w_t = A w, B w = 0

    v = None
    w = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in np.arange(0.1, 0.51, 0.1):
            w = A.expm(-float(t), wT, n=_N)
            v = w + u

    # v(55) after t = 0.5.
    print("ans =")
    print(f"   {float(v(55.0)):.15f}")

    # w display (MATLAB chebfun summary).
    coeffs = np.asarray(w.funs[0].coeffs)
    vscale = float(np.max(np.abs(np.asarray([w(x) for x in
                    np.linspace(0.0, 500.0, 401)]))))
    print("w =")
    print("   chebfun column (1 smooth piece)")
    print("       interval       length   endpoint values  ")
    print(f"[       0,   5e+02]      {len(coeffs)}    "
          f"{float(w(0.0)):.1e}  {float(w(500.0)):.1e} ")
    print(f"Vscale = {vscale:.6e}.")

    # jump2 = w''(50+) - w''(50-): second-derivative jump at the strike.
    wss = w.diff(2)
    eps = float(np.finfo(float).eps)
    jump2 = float(wss(50.0 + 100 * eps)) - float(wss(50.0 - 100 * eps))
    print("jump2 =")
    print(f"    {jump2:.15e}")

    # ------------------------------------------------------------------
    # Plot: the option value v(s) at the evolved times, near the strike.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sx = np.linspace(40, 60, 400)
    ax.plot(sx, np.maximum(0.0, sx - 50.0), color=CHEBFUN_BLUE, lw=1.5,
            label="payoff v_T")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in np.arange(0.1, 0.51, 0.1):
            w_t = A.expm(-float(t), wT, n=_N)
            v_t = w_t + u
            ax.plot(sx, np.asarray([float(v_t(x)) for x in sx]), "k", lw=0.9)
    ax.set_xlim([40, 60])
    ax.set_ylim([-0.5, 14])
    ax.set_xlabel("asset price s")
    ax.set_ylabel("option value v")
    ax.set_title("Black-Scholes via operator exponential")
    ax.legend(framealpha=0.9)

    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_HERE, "black_scholes_pde.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return True


if __name__ == "__main__":
    run()
