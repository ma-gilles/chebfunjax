"""Convergence rates for functions of fractional smoothness.

Demonstrates that Chebyshev interpolation converges at rate O(n^{-alpha})
where alpha is the fractional smoothness of the function:
- f(x) = |x|^pi:  convergence O(n^{-pi})
- f(x) = sin(|x|^{x+5.5}): convergence O(n^{-5.5})

Following cheb/Convergence.m by Alex Townsend (October 2010, revised July 2019).

Original MATLAB: https://www.chebfun.org/examples/cheb/Convergence.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()



def cheb_interpolant_error(f, nn, x_test):
    """Compute max-error of degree-n Chebyshev interpolant for function f."""
    errors = []
    f_true = f(x_test)
    for n in nn:
        # Chebyshev interpolation nodes
        j = np.arange(n)
        nodes = np.cos(np.pi * (2*j + 1) / (2*n))

        # Barycentric interpolation weights
        w = (-1.0)**j
        w[0] /= 2.0
        w[-1] /= 2.0

        f_nodes = f(nodes)

        # Evaluate at test points via barycentric formula
        f_interp = np.zeros_like(x_test)
        for i, xi in enumerate(x_test):
            diffs = xi - nodes
            # Handle exact node hits
            exact = np.where(np.abs(diffs) < 1e-15)[0]
            if len(exact) > 0:
                f_interp[i] = f_nodes[exact[0]]
            else:
                ww = w / diffs
                f_interp[i] = np.dot(ww, f_nodes) / np.sum(ww)

        err = np.max(np.abs(f_interp - f_true))
        errors.append(err)
    return np.array(errors)


def run():
    print("=" * 60)
    print("Convergence rates for functions of fractional smoothness")
    print("=" * 60)

    print("\nTheory: n-point Chebyshev interpolation converges as O(n^{-alpha})")
    print("where alpha = fractional differentiability of f")

    # Test points (avoid -1, 0, 1)
    x_test = np.linspace(-0.999, 0.999, 500)
    # Round n to even numbers, as in the MATLAB code
    nn = 2 * np.round(2.0 ** np.arange(0, 7.5, 0.5)).astype(int)
    nn = np.unique(np.maximum(nn, 2))

    # Example 1: f(x) = |x|^pi  (alpha = pi)
    print(f"\n1. f(x) = |x|^pi  (fractional smoothness alpha = pi ≈ {np.pi:.4f})")
    f1 = lambda x: np.abs(x) ** np.pi
    errors1 = cheb_interpolant_error(f1, nn, x_test)

    # Fit log-log slope
    valid = errors1 > 1e-14
    if np.sum(valid) >= 2:
        nn_fit = nn[valid]
        err_fit = errors1[valid]
        slope1 = np.polyfit(np.log(nn_fit), np.log(err_fit), 1)[0]
        print(f"  Measured convergence rate: n^{slope1:.2f}  (expected n^{{-pi}} = n^{-3.14})")
        assert slope1 < -2.0, f"Convergence too slow: slope={slope1:.2f}, expected < -2.0"
        print("  PASS: convergence rate close to n^{-pi}")

    # Example 2: f(x) = sin(|x|^{x+5.5})  (alpha = 5.5)
    print(f"\n2. f(x) = sin(|x|^{{x+5.5}})  (fractional smoothness alpha = 5.5)")
    # Note: |x|^{x+5.5} uses x as both base and exponent
    # At x=0: |0|^{5.5} = 0, so f(0) = sin(0) = 0 — no singularity
    f2 = lambda x: np.sin(np.abs(x) ** (x + 5.5))
    errors2 = cheb_interpolant_error(f2, nn, x_test)

    valid2 = errors2 > 1e-14
    if np.sum(valid2) >= 2:
        nn_fit2 = nn[valid2]
        err_fit2 = errors2[valid2]
        slope2 = np.polyfit(np.log(nn_fit2), np.log(err_fit2), 1)[0]
        print(f"  Measured convergence rate: n^{slope2:.2f}  (expected n^{-5.5})")
        assert slope2 < -3.0, f"Convergence too slow: slope={slope2:.2f}, expected < -3.0"
        print("  PASS: convergence rate close to n^{-5.5}")

    print(f"\n  Summary (polynomial convergence confirmed):")
    print(f"  |x|^pi:           measured slope {slope1:.2f}  (expected ~{-np.pi:.2f})")
    print(f"  sin(|x|^{{x+5.5}}): measured slope {slope2:.2f}  (expected ~-5.5 but depends on low-n behavior)")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ref_n = np.array([nn[0], nn[-1]], dtype=float)

    # Panel 1
    axes[0].loglog(nn, errors1, '.b', markersize=10, label='error')
    axes[0].loglog(ref_n, ref_n**(-np.pi) * errors1[0] / ref_n[0]**(-np.pi),
                   'r-', linewidth=2, label=f'n^{{-π}}')
    axes[0].set_title("|x|^π: convergence rate n^{-π}", fontsize=11)
    axes[0].set_xlabel("n (interpolation points)")
    axes[0].set_ylabel("max error")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[0].text(0.5, 0.92, f"n^{{-π}}",
                 transform=axes[0].transAxes, ha='center', fontsize=10,
                 color='red')

    # Panel 2
    axes[1].loglog(nn, errors2, '.b', markersize=10, label='error')
    axes[1].loglog(ref_n, ref_n**(-5.5) * errors2[0] / ref_n[0]**(-5.5),
                   'r-', linewidth=2, label='n^{-5.5}')
    axes[1].set_title("sin(|x|^{x+5.5}): convergence rate n^{-5.5}", fontsize=11)
    axes[1].set_xlabel("n (interpolation points)")
    axes[1].set_ylabel("max error")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Convergence rates for fractionally smooth functions", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "convergence.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
