"""Low-rank Tucker representation of Wagon's function.

Faithful port of approx3/Wagon.m by Behnam Hashemi (July 2016).  Stan Wagon's
deliberately complicated 3D oscillatory function has, despite its complexity,
a surprisingly low Tucker rank (4, 3, 5).  We build it as a Chebfun3, report
its global minimum via ``min3``, and plot its Tucker factor fibers.

Original: https://www.chebfun.org/examples/approx3/Wagon.html
Copyright 2016 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): the headline results match MATLAB.
- Tucker rank (4, 3, 5): EXACT.
- min3(f) = -3.328338345663264 vs published -3.328338345663268 (4e-15,
  last-digit roundoff) -- SOFT match.  (This required a multi-start polish
  in minandmax3; single-start descent from a coarse seed grid found only a
  local min at -3.287.)

The following published numbers are scheme-dependent walls, not defects:
- Fiber lengths (published 666, 1054, 124): our adaptive chebtech
  construction resolves the same function to slightly different Chebyshev
  degrees (705, 1025, 129) -- the happiness/plateau test differs from
  MATLAB's, so the polynomial lengths differ while the function agrees to
  machine precision.
- Tube-endpoint values (published f.tubes(-1,end)=0.0216,
  f.tubes(-0.2,end)=-0.1133): these read individual Tucker *factor* fibers,
  whose normalization and sign are set by the ACA pivoting convention.  Our
  factors carry a different (unnormalized) scaling -- the scaling lives in
  the core -- so the raw factor values differ even though f itself is
  identical.  Reproducing them would require replicating MATLAB's exact ACA
  normalization, not the mathematics of the example.
"""
import matplotlib

matplotlib.use("Agg")
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.plotting import CHEBFUN_BLUE, chebfun_style

chebfun_style()

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)), "docs", "images", "approx3"
)
os.makedirs(_IMG_DIR, exist_ok=True)


def wagon_func(x, y, z):
    """Stan Wagon's 3D function from the SIAM 100-Digit Challenge."""
    return (
        jnp.exp(jnp.sin(50 * x))
        + jnp.sin(60 * jnp.exp(y)) * jnp.sin(60 * z)
        + jnp.sin(70 * jnp.sin(x)) * jnp.cos(10 * z)
        + jnp.sin(jnp.sin(80 * y))
        - jnp.sin(10 * (x + z))
        + (x**2 + y**2 + z**2) / 4
    )


def run():
    # ------------------------------------------------------------------
    # f = chebfun3(ff)
    # ------------------------------------------------------------------
    f = chebfun3(wagon_func)
    r1, r2, r3 = f.rank
    m = max(len(np.asarray(c.coeffs)) for c in f.cols)
    n = max(len(np.asarray(r.coeffs)) for r in f.rows)
    p = max(len(np.asarray(t.coeffs)) for t in f.tubes)

    # Global extrema (one pass) -> min3 and the vertical scale sup|f|.
    mm_vals, _ = f.minandmax3()
    fmin, fmax = float(mm_vals[0]), float(mm_vals[1])
    vscale = max(abs(fmin), abs(fmax))

    print("f =")
    print("   chebfun3 object ")
    print(f"   cols: Inf x {r1} chebfun")
    print(f"   rows: Inf x {r2} chebfun")
    print(f"  tubes: Inf x {r3} chebfun")
    print(f"   core: {r1} x {r2} x {r3}")
    print(f" length: {m}, {n}, {p}")
    print(" domain: [-1, 1] x [-1, 1] x [-1, 1]")
    print(f" vertical scale = {vscale:.2g}")

    # ------------------------------------------------------------------
    # min3_f = min3(f)
    # ------------------------------------------------------------------
    min3_f = fmin
    print("min3_f =")
    print(f"  {min3_f:.15f}")

    # ------------------------------------------------------------------
    # format short, f.tubes(-1, end), f.tubes(-0.2, end)
    # (last Tucker tube fiber -- scheme-dependent factor scaling, see note)
    # ------------------------------------------------------------------
    print("ans =")
    print(f"    {float(f.tubes[-1](-1.0)):.4f}")
    print("ans =")
    print(f"   {float(f.tubes[-1](-0.2)):.4f}")

    # ------------------------------------------------------------------
    # Plot: Tucker factor fibers and coefficient decay of tube[0].
    # ------------------------------------------------------------------
    t_ref = np.linspace(-1, 1, 400)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    ax1 = axes[0, 0]
    for i, col in enumerate(f.cols):
        ax1.plot(t_ref, np.asarray(col(jnp.asarray(t_ref))), lw=1.2,
                 label=f"col {i+1}")
    ax1.set_title(f"columns ({r1})", fontsize=10)
    ax1.legend(fontsize=8, framealpha=0.9, ncol=2)
    ax1.set_xlim([-1, 1])

    ax2 = axes[0, 1]
    for j, row in enumerate(f.rows):
        ax2.plot(t_ref, np.asarray(row(jnp.asarray(t_ref))), lw=1.2,
                 label=f"row {j+1}")
    ax2.set_title(f"rows ({r2})", fontsize=10)
    ax2.legend(fontsize=8, framealpha=0.9, ncol=2)
    ax2.set_xlim([-1, 1])

    ax3 = axes[1, 0]
    for k, tube in enumerate(f.tubes):
        ax3.plot(t_ref, np.asarray(tube(jnp.asarray(t_ref))), lw=1.2,
                 label=f"tube {k+1}")
    ax3.set_title(f"tubes ({r3})", fontsize=10)
    ax3.legend(fontsize=8, framealpha=0.9, ncol=2)
    ax3.set_xlim([-1, 1])

    ax4 = axes[1, 1]
    tube0_coeffs = np.abs(np.asarray(f.tubes[0].coeffs))
    ax4.semilogy(range(len(tube0_coeffs)), tube0_coeffs, "o-",
                 color=CHEBFUN_BLUE, ms=3, lw=1.2)
    ax4.set_title("plotcoeffs(f.tubes(:,1))", fontsize=10)
    ax4.set_xlabel("degree n", fontsize=9)
    ax4.set_ylabel("|a_n|", fontsize=9)

    fig.suptitle(f"Wagon's function: Tucker rank ({r1},{r2},{r3})",
                 fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG_DIR, "Wagon.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return True


if __name__ == "__main__":
    run()
