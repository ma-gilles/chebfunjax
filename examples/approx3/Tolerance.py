"""Loosening the Chebfun3 tolerance for faster construction.

Faithful port of approx3/Tolerance.m by Nick Trefethen (June 2016).  Shows
that machine precision is the Chebfun3 default but is often more than needed:
a triple integral of a smooth 3D function can be obtained far faster with a
loosened tolerance, at a controlled accuracy cost.

Original: https://www.chebfun.org/examples/approx3/Tolerance.html
Copyright 2016 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured):
- The three default tolerances (chebfuneps, chebfun2eps, chebfun3eps) are
  EXACT: 2.220446049250313e-16.
- Machine-precision integrals match to ~13 significant figures:
  sum3(exp(sin(xyz+exp(xyz))))       = 17.8856934116068  (published ...855)
  sum3(exp(sin(10 xyz+exp(xyz))))    = 13.5800209530689  (published ...932)
  Both constructions of f (functional constructor and the ``cheb.xyz``
  arithmetic path) agree, as in MATLAB.

Scheme-dependent walls (documented, not defects):
- Fiber lengths / core sizes (published 42,41,42 and 156,156,156 and
  86,86,86): our adaptive chebtech happiness test resolves the same
  functions to different Chebyshev degrees.
- The loosened-tolerance integrals are only accurate to O(tol) and depend on
  the exact adaptive construction:  at eps=1e-8 the integral differs from the
  machine-precision value at the ~1e-7 level, at eps=1e-4 at the ~3e-2 level.
  MATLAB furthermore reports *two different* eps=1e-8 values (13.5800207195
  via the ``'eps'`` constructor argument vs 13.5800234623 via the global
  ``chebfun3eps`` setting) -- our single tolerance path cannot reproduce that
  internal MATLAB distinction, so both collapse to one construction.  These
  loosened values are therefore reported honestly but do not (and cannot)
  match the published digits.
"""
import matplotlib

matplotlib.use("Agg")
import os
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.plotting import CHEBFUN_BLUE, CHEBFUN_RED, chebfun_style

chebfun_style()

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)), "docs", "images", "approx3"
)
os.makedirs(_IMG_DIR, exist_ok=True)

_DOM = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
_EPS = float(jnp.finfo(jnp.float64).eps)


def _lengths(f3):
    return (max(len(np.asarray(c.coeffs)) for c in f3.cols),
            max(len(np.asarray(r.coeffs)) for r in f3.rows),
            max(len(np.asarray(t.coeffs)) for t in f3.tubes))


def _vscale(raw):
    g = np.linspace(-1, 1, 80)
    XX, YY, ZZ = np.meshgrid(g, g, g, indexing="ij")
    return float(np.max(np.abs(np.asarray(
        raw(jnp.asarray(XX), jnp.asarray(YY), jnp.asarray(ZZ))))))


def _print_display(name, f3, raw):
    """Print the MATLAB-style chebfun3 object display and length(f)."""
    r1, r2, r3 = f3.rank
    print(f"{name} =")
    print("   chebfun3 object ")
    print(f"   cols: [Inf x {r1} chebfun]")
    print(f"   rows: [Inf x {r2} chebfun]")
    print(f"  tubes: [Inf x {r3} chebfun]")
    print(f"   core: [{r1} x {r2} x {r3} double]")
    print(" domain: [-1, 1] x [-1, 1] x [-1, 1]")
    print(f" vertical scale = {_vscale(raw):.2g} ")
    m, n, pp = _lengths(f3)
    print("m =")
    print(f"    {m}")
    print("n =")
    print(f"    {n}")
    print("p =")
    print(f"    {pp}")


def run():
    # ------------------------------------------------------------------
    # The default construction tolerances.
    # ------------------------------------------------------------------
    for _ in range(3):        # chebfuneps, chebfun2eps, chebfun3eps
        print("ans =")
        print(f"     {_EPS:.15e}")

    # ------------------------------------------------------------------
    # Section 1: a smooth function, machine precision.
    #   f = exp(sin(x*y*z + exp(x*y*z)));  I = sum3(f)
    # ------------------------------------------------------------------
    ff = lambda x, y, z: jnp.exp(jnp.sin(x * y * z + jnp.exp(x * y * z)))
    t0 = time.time()
    f = chebfun3(ff, domain=_DOM)
    I = float(f.sum3())
    print("I =")
    print(f"  {I:.15f}")
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")

    # Alternative syntax: cheb.xyz arithmetic build (same integral).
    x = chebfun3(lambda x, y, z: x, domain=_DOM)
    y = chebfun3(lambda x, y, z: y, domain=_DOM)
    z = chebfun3(lambda x, y, z: z, domain=_DOM)
    t0 = time.time()
    p = x * y * z
    f2 = (p + p.exp()).sin().exp()
    I = float(f2.sum3())
    print("I =")
    print(f"  {I:.15f}")
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")

    _print_display("f", f, ff)

    # ------------------------------------------------------------------
    # Section 2: a more complicated function, machine precision (slow).
    #   g = exp(sin(10*x*y*z + exp(x*y*z)))
    # ------------------------------------------------------------------
    gf = lambda x, y, z: jnp.exp(jnp.sin(10 * x * y * z + jnp.exp(x * y * z)))
    t0 = time.time()
    g = chebfun3(gf, domain=_DOM)
    I = float(g.sum3())
    print("I =")
    print(f"  {I:.15f}")
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")
    _print_display("g", g, gf)

    # ------------------------------------------------------------------
    # Section 3: loosened tolerance eps=1e-8 -- much faster.
    # ------------------------------------------------------------------
    t0 = time.time()
    g8 = chebfun3(gf, domain=_DOM, tol=1e-8)
    I8 = float(g8.sum3())
    print("I =")
    print(f"  {I8:.15f}")
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")
    _print_display("g", g8, gf)

    # Global chebfun3eps 1e-8 (in MATLAB a distinct construction; here our
    # single tolerance path reproduces the same eps=1e-8 build).
    print("I =")
    print(f"  {I8:.15f}")
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")

    # Further loosened, eps=1e-4.
    t0 = time.time()
    g4 = chebfun3(gf, domain=_DOM, tol=1e-4)
    I4 = float(g4.sum3())
    print("I =")
    print(f"  {I4:.15f}")
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")

    # ------------------------------------------------------------------
    # Plot: tolerance vs rank/error, and coefficient decay.
    # ------------------------------------------------------------------
    tol_vals = [1e-8, 1e-6, 1e-4]
    ranks, errs = [], []
    for tol in tol_vals:
        gt = chebfun3(gf, domain=_DOM, tol=tol)
        ranks.append(gt.rank[0])
        errs.append(abs(float(gt.sum3()) - I))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    ax1 = axes[0]
    ax1.loglog(tol_vals, ranks, "o-", color=CHEBFUN_BLUE, lw=1.5, ms=7)
    ax1.invert_xaxis()
    ax1.set_title("rank vs tolerance", fontsize=10)
    ax1.set_xlabel("tolerance", fontsize=9)
    ax1.set_ylabel("Tucker rank", fontsize=9)

    ax2 = axes[1]
    ax2.loglog(tol_vals, errs, "o-", color=CHEBFUN_RED, lw=1.5, ms=7)
    ax2.loglog(tol_vals, tol_vals, "--k", alpha=0.4, label="error = tol")
    ax2.invert_xaxis()
    ax2.set_title("error vs tolerance", fontsize=10)
    ax2.set_xlabel("tolerance", fontsize=9)
    ax2.set_ylabel("integral error", fontsize=9)
    ax2.legend(fontsize=8, framealpha=0.9)

    ax3 = axes[2]
    coeffs0 = np.abs(np.asarray(g.rows[0].coeffs))
    ax3.semilogy(range(len(coeffs0)), coeffs0, "o-", color=CHEBFUN_BLUE,
                 ms=3, lw=1.2)
    ax3.set_title("plotcoeffs(g.rows)", fontsize=10)
    ax3.set_xlabel("degree n", fontsize=9)
    ax3.set_ylabel("|a_n|", fontsize=9)

    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG_DIR, "Tolerance.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return True


if __name__ == "__main__":
    run()
