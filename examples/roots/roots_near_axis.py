"""Complex roots near the real axis.

A Chebfun approximation of a smooth function is accurate throughout a
Bernstein ellipse in the complex plane. The complex roots of the underlying
function that lie near the real axis can thus be found.

Credit: Inspired by Chebfun example roots/RootsNearAxis.m
(Nick Trefethen, October 2011).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj


def run():
    print("=" * 60)
    print("Complex roots near the real axis")
    print("=" * 60)

    pi = float(jnp.pi)

    # f(x) = 3 + sin(x) + sin(pi*x) on [0, 30]
    # This function has no real roots (minimum value > 0)
    dom = (0.0, 30.0)
    f = cj.chebfun(lambda x: 3.0 + jnp.sin(x) + jnp.sin(pi * x), domain=dom)

    print(f"\nf(x) = 3 + sin(x) + sin(pi*x) on [0, 30]")
    r_real = f.roots()
    print(f"  Real roots on [0,30]: {len(np.array(r_real))}  (should be 0)")
    assert len(np.array(r_real)) == 0, "Expected no real roots"

    # The Chebfun has polynomial degree n; it accurately represents the function
    # in the Bernstein ellipse of parameter rho ~ exp(1/n * ... ).
    # Complex roots near the real axis: f(z) = 0 where z = x + iy, |y| small.
    # For f = 3 + sin(z) + sin(pi*z), the imaginary part makes some roots complex.
    # We find them by examining where |f(z)| ≈ 0 in the complex plane.

    # Build a grid in the complex strip [0,30] x [-1, 1]i
    xs = np.linspace(0, 30, 500)
    ys = np.linspace(-1.0, 1.0, 200)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = XX + 1j * YY
    FF = 3.0 + np.sin(ZZ) + np.sin(pi * ZZ)
    absFF = np.abs(FF)

    # Find approximate complex roots (where |f| < threshold)
    thresh = 0.05
    mask = absFF < thresh
    if np.any(mask):
        root_xs = XX[mask]
        root_ys = YY[mask]
        print(f"\nApproximate complex roots near real axis (|f| < {thresh}):")
        # Cluster and show a few
        pts = list(zip(root_xs, root_ys))
        # Show at most 5 representative ones
        for i, (rx, ry) in enumerate(pts[:5]):
            val = abs(3.0 + np.sin(rx + 1j*ry) + np.sin(pi*(rx + 1j*ry)))
            print(f"  z ≈ {rx:.4f} + {ry:.4f}i,  |f(z)| = {val:.2e}")

    print(f"\nChebfun length (polynomial degree): {len(f)}")

    # Verify minimum value of f on real axis (should be > 0)
    x_min, f_min = f.min()
    print(f"  min f on [0,30] = {f_min:.6f} at x = {x_min:.4f}  (should be > 0)")
    assert float(f_min) > 0.0, f"f should have no real roots, but min = {f_min}"

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: f on the real axis
    xs_plot = np.linspace(0, 30, 500)
    ys_plot = np.array(f(jnp.array(xs_plot)))
    axes[0].plot(xs_plot, ys_plot, color="#1e77b4", linewidth=1.5)
    axes[0].axhline(0, color="k", linewidth=0.5)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].set_title("$f(x) = 3 + \\sin x + \\sin(\\pi x)$: no real roots")
    axes[0].grid(True, alpha=0.4)

    # Right: |f(z)| in the complex strip
    im = axes[1].contourf(XX, YY, np.log10(absFF + 1e-15),
                          levels=50, cmap='RdYlBu_r')
    axes[1].axhline(0, color="k", linewidth=0.8)
    axes[1].set_xlabel("Re(z)")
    axes[1].set_ylabel("Im(z)")
    axes[1].set_title("$\\log_{10}|f(z)|$ in complex strip")
    plt.colorbar(im, ax=axes[1], label="$\\log_{10}|f|$")
    axes[1].set_xlim(0, 30)
    axes[1].set_ylim(-1, 1)

    fig.suptitle("Complex roots near the real axis", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "roots_near_axis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
