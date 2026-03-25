"""Fractional calculus in Chebfun.

Demonstrates the Riemann-Liouville fractional derivative d^a/dx^a for
various values of a in (0,1), including the half-derivative of x.
Translated from temp/FracCalc.m (original: integro/FracCalc.m).

Original: https://www.chebfun.org/examples/integro/FracCalc.html
Author: Nick Hale, October 2010
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj


def riemann_liouville(f, x, alpha, a=0.0):
    """Compute Riemann-Liouville fractional integral of order alpha.

    I^alpha f(x) = 1/Gamma(alpha) * int_a^x (x-t)^(alpha-1) f(t) dt
    """
    dx = x[1] - x[0]
    result = np.zeros_like(x)
    for i in range(1, len(x)):
        ts = x[:i]
        kernel = (x[i] - ts)**(alpha - 1)
        result[i] = np.trapezoid(kernel * f[:i], ts)
    return result / gamma(alpha)


def fractional_derivative(f, x, alpha, a=0.0):
    """Riemann-Liouville fractional derivative D^alpha = D^1 * I^(1-alpha).

    For 0 < alpha < 1: D^alpha f = d/dx [I^(1-alpha) f]
    """
    if alpha == 0:
        return f.copy()
    if alpha == 1:
        return np.gradient(f, x)
    # I^(1-alpha)
    integral = riemann_liouville(f, x, 1 - alpha, a)
    # d/dx
    return np.gradient(integral, x)


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/temp')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: f(x) = x with half-derivative ---
    x = np.linspace(0, 4, 400)
    f = x.copy()  # f(x) = x

    axes[0].plot(x, f, '-', linewidth=2, label='x')
    axes[0].plot(x, np.ones_like(x), '-', linewidth=2, label="x'=1")
    axes[0].plot(x, x**2 / 2, '-', linewidth=2, label='∫x = x²/2')

    # Half-derivative: D^{1/2}[x] = 2√(x/π)
    half_deriv_exact = 2 * np.sqrt(x / np.pi)
    half_deriv_num = fractional_derivative(f, x, 0.5)

    axes[0].plot(x, half_deriv_exact, 'k--', linewidth=2, label='2√(x/π)')
    axes[0].plot(x[1:], half_deriv_num[1:], 'r:', linewidth=2.5,
                 label='D^{1/2}x (numerical)')

    err = np.max(np.abs(half_deriv_exact[10:] - half_deriv_num[10:]))
    print(f"Half-derivative of x: max error = {err:.4f}")
    axes[0].set_xlim(0, 4); axes[0].set_ylim(0, 4)
    axes[0].set_title("f(x)=x and its half-derivative\nD^{1/2}x = 2√(x/π)", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('x')

    # --- Panel 2: Fractional derivatives of x for alpha=0.1,...,1.0 ---
    alphas = np.arange(0.1, 1.01, 0.1)
    colors2 = plt.cm.viridis(np.linspace(0, 1, len(alphas)))

    for alpha, col in zip(alphas, colors2):
        da = fractional_derivative(f, x, alpha)
        axes[1].plot(x[2:], da[2:], '-', color=col, linewidth=1.5,
                     label=f'α={alpha:.1f}' if alpha in [0.1, 0.5, 1.0] else '')

    axes[1].set_xlim(0, 4); axes[1].set_ylim(0, 4)
    axes[1].set_title('Fractional derivatives of x\nα from 0.1 to 1.0', fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('x'); axes[1].set_ylabel('D^α x')
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                 norm=plt.Normalize(0.1, 1.0))
    plt.colorbar(sm, ax=axes[1], label='α')

    # --- Panel 3: Fractional derivatives of sin(x) ---
    x3 = np.linspace(0, 20, 1000)
    f3 = np.sin(x3)
    alphas3 = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors3 = ['b', 'c', 'g', 'orange', 'r']

    for alpha, col in zip(alphas3, colors3):
        if alpha == 0:
            da = f3.copy()
        elif alpha == 1:
            da = np.cos(x3)
        else:
            da = fractional_derivative(f3, x3, alpha)
        skip = 20 if alpha > 0 else 0
        axes[2].plot(x3[skip:], da[skip:], '-', color=col,
                     linewidth=1.5, label=f'α={alpha:.2f}')

    axes[2].set_xlim(0, 20); axes[2].set_ylim(-1.5, 1.5)
    axes[2].set_title('Fractional derivatives of sin(x)\nfor α=0,0.25,0.5,0.75,1', fontsize=10)
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel('x')

    fig.suptitle('Fractional Calculus: Riemann-Liouville Derivatives', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'frac_calc.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("frac_calc: done")
    return True


if __name__ == "__main__":
    run()
