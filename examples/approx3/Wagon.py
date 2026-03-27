"""Low-rank Tucker representation of Wagon's function.

Stan Wagon's function is a deliberately complicated 3D oscillatory function
that despite its complexity has surprisingly low Tucker rank. This example
constructs it as a Chebfun3 and plots its Tucker factor fibers (cols, rows,
tubes).

Original MATLAB Chebfun: approx3/Wagon.m by Behnam Hashemi, July 2016.
See https://www.chebfun.org/examples/approx3/Wagon.html
Copyright 2016 by The University of Oxford and The Chebfun Developers.
"""

import matplotlib
matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt
from chebfunjax.plotting import chebfun_style
chebfun_style()

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3

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
    print("=" * 60)
    print("Wagon's function: low-rank Tucker representation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Construct Wagon's function as a Chebfun3
    # ------------------------------------------------------------------
    print("\nConstructing Wagon's function...")
    f = chebfun3(wagon_func)
    rx, ry, rz = f.rank
    print(f"  Tucker rank: ({rx}, {ry}, {rz})")
    print(f"  Representation: {f}")

    # MATLAB gives rank (4, 3, 5) — Python may differ slightly
    # but rank should be small (< 20)
    assert max(rx, ry, rz) <= 20, f"Rank unexpectedly large: ({rx},{ry},{rz})"

    # ------------------------------------------------------------------
    # Find minimum using grid search (global minimization)
    # ------------------------------------------------------------------
    print("\n--- Finding global minimum ---")
    # Evaluate on a coarse grid to locate minimum
    n_grid = 50
    xs = np.linspace(-1, 1, n_grid)
    XX, YY, ZZ = np.meshgrid(xs, xs, xs, indexing="ij")
    vals = np.array(wagon_func(jnp.array(XX), jnp.array(YY), jnp.array(ZZ)))
    idx_min = np.unravel_index(np.argmin(vals), vals.shape)
    x0 = xs[idx_min[0]]
    y0 = xs[idx_min[1]]
    z0 = xs[idx_min[2]]
    f_min_grid = float(vals[idx_min])
    print(f"  Grid search minimum: f({x0:.3f},{y0:.3f},{z0:.3f}) = {f_min_grid:.6f}")

    # Refine using Chebfun3 evaluation
    f_min_cheb = float(f(jnp.array(x0), jnp.array(y0), jnp.array(z0)))
    print(f"  Chebfun3 at grid min: {f_min_cheb:.6f}")

    # MATLAB reports minimum around -3.31 to -3.3
    assert f_min_grid < -2.0, f"Minimum should be well below 0, got {f_min_grid}"

    # ------------------------------------------------------------------
    # Tucker factor fibers: columns (x-direction), rows (y), tubes (z)
    # ------------------------------------------------------------------
    print(f"\n--- Tucker factors: cols={rx}, rows={ry}, tubes={rz} ---")
    # Each factor is a Chebtech2 object with .coeffs attribute
    # Evaluate on reference grid [-1, 1]
    t_ref = np.linspace(-1, 1, 200)

    cols_vals = []
    for i, col in enumerate(f.cols):
        vals_col = np.array(col(jnp.array(t_ref)))
        cols_vals.append(vals_col)
        n_c = len(np.array(col.coeffs))
        print(f"  col[{i}]: {n_c} Chebyshev coefficients")

    rows_vals = []
    for j, row in enumerate(f.rows):
        vals_row = np.array(row(jnp.array(t_ref)))
        rows_vals.append(vals_row)

    tubes_vals = []
    for k, tube in enumerate(f.tubes):
        vals_tube = np.array(tube(jnp.array(t_ref)))
        tubes_vals.append(vals_tube)

    # Chebyshev coefficients of first tube
    tube0_coeffs = np.abs(np.array(f.tubes[0].coeffs))
    print(f"\n  Tube[0] Chebyshev coefficients: {len(tube0_coeffs)} terms")
    print(f"    Max: {tube0_coeffs.max():.4e}, Min: {tube0_coeffs.min():.4e}")

    # ------------------------------------------------------------------
    # Plot: Tucker factor fibers and coefficient decay
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Plot columns
    ax1 = axes[0, 0]
    for i, col_v in enumerate(cols_vals):
        ax1.plot(t_ref, col_v, lw=1.8, label=f"col {i+1}")
    ax1.set_xlabel("x", fontsize=11)
    ax1.set_ylabel("value", fontsize=11)
    ax1.set_title(f"Columns ({rx} Tucker x-fibers)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.4)
    ax1.set_xlim([-1, 1])

    # Plot rows
    ax2 = axes[0, 1]
    for j, row_v in enumerate(rows_vals):
        ax2.plot(t_ref, row_v, lw=1.8, label=f"row {j+1}")
    ax2.set_xlabel("y", fontsize=11)
    ax2.set_ylabel("value", fontsize=11)
    ax2.set_title(f"Rows ({ry} Tucker y-fibers)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)
    ax2.set_xlim([-1, 1])

    # Plot tubes
    ax3 = axes[1, 0]
    for k, tube_v in enumerate(tubes_vals):
        ax3.plot(t_ref, tube_v, lw=1.8, label=f"tube {k+1}")
    ax3.set_xlabel("z", fontsize=11)
    ax3.set_ylabel("value", fontsize=11)
    ax3.set_title(f"Tubes ({rz} Tucker z-fibers)", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.4)
    ax3.set_xlim([-1, 1])

    # Chebyshev coefficients of first tube
    ax4 = axes[1, 1]
    ax4.semilogy(range(len(tube0_coeffs)), tube0_coeffs, "o-b", ms=4, lw=1.5)
    ax4.set_xlabel("Chebyshev degree", fontsize=11)
    ax4.set_ylabel("|coefficient|", fontsize=11)
    ax4.set_title("Chebyshev coefficients of tube[0]", fontsize=11)
    ax4.grid(True, which="both", alpha=0.4)

    fig.suptitle(
        f"Wagon's function: Tucker rank ({rx},{ry},{rz})\n"
        f"low rank despite high oscillation",
        fontsize=12
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(_IMG_DIR, "Wagon.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
