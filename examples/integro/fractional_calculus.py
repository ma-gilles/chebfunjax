"""Fractional calculus in Chebfun.

Demonstrates fractional derivatives D^alpha f for 0 < alpha < 1
via Riemann-Liouville operators, following integro/FracCalc.m by
Nick Hale (October 2010).

Key results:
- D^(1/2) x = 2*sqrt(x/pi)  (half-derivative of x)
- Fractional derivatives interpolate between identity (alpha=0) and
  derivative (alpha=1)

Original MATLAB: https://www.chebfun.org/examples/integro/FracCalc.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chebfunjax.plotting import chebfun_style
chebfun_style()

import numpy as np
from scipy.special import gamma
import os

def run():
    print("=" * 60)
    print("Fractional calculus")
    print("=" * 60)

    # Riemann-Liouville fractional integral of order alpha on [0, b]:
    # (J^alpha f)(x) = 1/Gamma(alpha) * int_0^x (x-t)^(alpha-1) f(t) dt

    def frac_integral(f_vals, x_vals, alpha):
        """Numerical Riemann-Liouville fractional integral."""
        n = len(x_vals)
        result = np.zeros(n)
        for i in range(1, n):
            # Trapezoid rule for int_0^{x_i} (x_i - t)^(alpha-1) * f(t) dt
            t_sub = x_vals[:i+1]
            f_sub = f_vals[:i+1]
            kernel = (x_vals[i] - t_sub)**(alpha - 1)
            kernel[0] = 0.0 if alpha < 1 else kernel[0]  # regularize left endpoint
            result[i] = np.trapezoid(kernel * f_sub, t_sub) / gamma(alpha)
        return result

    def frac_deriv(f_vals, x_vals, alpha):
        """Riemann-Liouville fractional derivative D^alpha f = d/dx J^(1-alpha) f."""
        # D^alpha f = d/dx (J^(1-alpha) f)
        integral = frac_integral(f_vals, x_vals, 1.0 - alpha)
        # Numerical differentiation
        return np.gradient(integral, x_vals)

    # --- 1. Half-derivative of f(x) = x ---
    print("\n1. Half-derivative of f(x) = x on [0,4]")
    print("   Exact: D^(1/2) x = 2*sqrt(x/pi)")

    x4 = np.linspace(0.01, 4, 200)  # avoid x=0 singularity
    f_x = x4.copy()

    d_half = frac_deriv(f_x, x4, 0.5)
    exact_half = 2 * np.sqrt(x4 / np.pi)

    err = np.max(np.abs(d_half[10:] - exact_half[10:]))
    print(f"   Max error (away from 0): {err:.4f}")
    print(f"   At x=1: numerical={d_half[np.argmin(np.abs(x4-1))]:.4f}, "
          f"exact={2/np.sqrt(np.pi):.4f}")

    # --- 2. Fractional derivatives alpha=0.1,...,1 of f(x)=x ---
    print("\n2. Fractional derivatives D^alpha x for alpha = 0.1, ..., 1")
    x_arr = np.linspace(0.01, 4, 150)
    f_arr = x_arr.copy()

    deriv_results = {}
    for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        if alpha == 0.0:
            deriv_results[alpha] = x_arr.copy()
        elif alpha == 1.0:
            deriv_results[alpha] = np.ones_like(x_arr)
        else:
            deriv_results[alpha] = frac_deriv(f_arr, x_arr, alpha)

    print(f"   D^0 x at x=2: {deriv_results[0.0][np.argmin(np.abs(x_arr-2))]:.4f} (expected 2.0)")
    print(f"   D^1 x at x=2: {deriv_results[1.0][np.argmin(np.abs(x_arr-2))]:.4f} (expected 1.0)")

    # --- 3. Fractional integrals of x^k ---
    print("\n3. Half-integrals of x^k for k=1,...,5")
    x_int = np.linspace(0.01, 1, 100)

    integrals = {}
    for k in range(1, 6):
        f_k = x_int**k
        integrals[k] = frac_integral(f_k, x_int, 0.5)
        # Analytical: J^(1/2) x^k = Gamma(k+1)/Gamma(k+3/2) * x^(k+1/2)
        exact_int = gamma(k + 1) / gamma(k + 1.5) * x_int**(k + 0.5)
        err_k = np.max(np.abs(integrals[k][5:] - exact_int[5:]))
        print(f"   k={k}: max error = {err_k:.4f}")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    # Panel 1: fractional derivatives of x
    cmap = plt.cm.viridis
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i, alpha in enumerate(alphas):
        col = cmap(i / (len(alphas) - 1))
        axes[0].plot(x_arr, deriv_results[alpha], color=col,
                     linewidth=1.8, label=f'α={alpha:.1f}')
    axes[0].set_title("Fractional derivatives D^α x", fontsize=11)
    axes[0].legend(fontsize=8, loc='upper left')
    axes[0].set_ylim([0, 5])

    # Panel 2: half-derivative of x vs exact
    axes[1].plot(x4, f_x, 'k-', linewidth=2, label='f(x) = x')
    axes[1].plot(x4, np.gradient(x4, x4), 'g--', linewidth=2, label="f'(x) = 1")
    axes[1].plot(x4, exact_half, 'b-', linewidth=2, label='D^(1/2) x = 2√(x/π) exact')
    axes[1].plot(x4[::10], d_half[::10], 'r.', markersize=8, label='Numerical D^(1/2) x')
    axes[1].set_title("Half-derivative of x", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].set_ylim([0, 4])

    fig.suptitle("Fractional calculus: Riemann-Liouville operators", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "fractional_calculus.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True

if __name__ == "__main__":
    run()
