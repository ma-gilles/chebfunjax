"""Fractional calculus in Chebfun (cont.).

Explains the implementation of fractional integrals using the fact that
fractional integrals of Legendre and Jacobi polynomials are known in closed form.
Translated from temp/FracCalc2.m (original: integro/FracCalc2.m).

Original: https://www.chebfun.org/examples/integro/FracCalc2.html
Author: Nick Hale, February 2015
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma, legendre, eval_legendre, eval_chebyt
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

def half_integral_legendre(n, x):
    """Half-integral of P_n(x) via the explicit formula.

    J^{1/2} P_n(x) = (T_n(x) + T_{n+1}(x)) / (Gamma(1/2) * (n+1/2) * sqrt(1+x))
    """
    Tn = eval_chebyt(n, x)
    Tn1 = eval_chebyt(n+1, x)
    denom = gamma(0.5) * (n + 0.5) * np.sqrt(np.abs(1 + x) + 1e-15)
    return (Tn + Tn1) / denom

def half_integral_numerical(f, x):
    """Numerical half-integral via quadrature: J^{1/2}f(x) = int_{-1}^x f(t)/sqrt(x-t) dt."""
    result = np.zeros_like(x)
    for i in range(1, len(x)):
        ts = x[:i]
        kernel = 1.0 / np.sqrt(x[i] - ts + 1e-14)
        result[i] = np.trapezoid(f[:i] * kernel, ts)
    return result

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/temp')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3)

    x = np.linspace(-0.999, 1.0, 300)

    # --- Panel 1: Half-integral of P_n (explicit vs numerical) ---
    n = 4
    Pn = eval_legendre(n, x)
    J_exact = half_integral_legendre(n, x)
    J_num = half_integral_numerical(Pn, x)

    axes[0].plot(x, Pn, 'k-', linewidth=2, label=f'$P_{n}(x)$')
    axes[0].plot(x, J_exact, 'b--', linewidth=2, label='Explicit formula')
    axes[0].plot(x[5:], J_num[5:], 'r:', linewidth=2.5, label='Numerical')
    axes[0].set_title(f'Half-integral of $P_{n}(x)$\n'
                       f'$(T_n + T_{{n+1}}) / (\\Gamma(1/2)(n+1/2)\\sqrt{{1+x}})$',
                       fontsize=9)
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(-1, 1)

    err = np.max(np.abs(J_exact[20:] - J_num[20:]))
    print(f"Half-integral P_4: max error explicit vs numerical = {err:.4f}")

    # --- Panel 2: Fractional integrals of exp(x) ---
    f_exp = np.exp(x)
    alphas = [0.25, 0.5, 0.75, 1.0]
    colors2 = ['b', 'g', 'r', 'm']

    axes[1].plot(x, f_exp, 'k-', linewidth=2, label='exp(x)')
    for alpha, col in zip(alphas, colors2):
        if abs(alpha - 1.0) < 0.01:
            # Integral of exp(x) = exp(x) - exp(-1)
            J_a = np.exp(x) - np.exp(-1)
        else:
            # Numerical approximation
            result = np.zeros_like(x)
            for i in range(1, len(x)):
                ts = x[:i]
                kernel = (x[i] - ts)**(alpha - 1)
                result[i] = np.trapezoid(f_exp[:i] * kernel, ts)
            J_a = result / gamma(alpha)
        axes[1].plot(x[5:], J_a[5:], '-', color=col, linewidth=1.5,
                     label=f'J^{{{alpha:.2f}}}')

    axes[1].set_title('Fractional integrals of exp(x)\nfor α=0.25, 0.5, 0.75, 1', fontsize=10)
    axes[1].legend(fontsize=9)

    # --- Panel 3: Beta function and Jacobi polynomial connection ---
    # B(z,w) = Gamma(z)*Gamma(w)/Gamma(z+w)
    z_vals = np.linspace(0.1, 3, 100)
    B_half = gamma(z_vals) * gamma(0.5) / gamma(z_vals + 0.5)

    axes[2].plot(z_vals, B_half, 'b-', linewidth=2, label='B(z, 1/2)')
    axes[2].plot(z_vals, np.pi / (2*np.sqrt(z_vals)), 'r--', linewidth=2,
                 alpha=0.7, label='π/(2√z) (large z)')

    # Show Gamma function ratio
    axes[2].set_title('Beta function B(z, 1/2)\nΓ(z)Γ(1/2)/Γ(z+1/2)', fontsize=10)
    axes[2].legend(fontsize=9)
    axes[2].set_ylim(0, 10)

    print("Fractional calculus implementation via Legendre/Jacobi polynomials")
    print(f"  Key formula: J^{{1/2}}P_n = (T_n + T_{{n+1}}) / (Γ(1/2)(n+1/2)√(1+x))")

    fig.suptitle('Fractional Calculus via Orthogonal Polynomials', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'frac_calc2.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("frac_calc2: done")
    return True

if __name__ == "__main__":
    run()
