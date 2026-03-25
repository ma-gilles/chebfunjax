"""Halphen's constant for approximation of exp(x).

Illustrates Halphen's constant C ≈ 9.2890... governing the exponential
rate of decay of best rational approximation errors to exp(x) on (-inf,0].

Credit: Nick Trefethen, May 2011.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/Halphen.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')

HALPHEN = 9.289025491920818918755449435951


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Known errors for best type-(n,n) rational approximation to exp(x) on (-inf,0]
    errors_known = [
        0.500, 0.0668, 0.00736, 0.000799, 0.0000865,
        0.00000934, 0.000001008, 0.0000001087, 0.00000001172
    ]
    ns = np.arange(len(errors_known))

    # Halphen asymptotic: error ~ 2 * C^{-n-1/2}
    ns_fine = np.linspace(0, 8, 200)
    halphen_bound = 2.0 * HALPHEN**(-ns_fine - 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.semilogy(ns, errors_known, 'b.', ms=12, label='known errors')
    ax.semilogy(ns_fine, halphen_bound, 'r--', lw=1.5,
                label=f'2·C^(−n−1/2), C={HALPHEN:.3f}')
    ax.set_title("Best type-(n,n) rational approx error for e^x on (-∞,0]", fontsize=10)
    ax.set_xlabel('n')
    ax.set_ylabel('error')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Demonstration: L2 polyfit errors for exp(x) on [-1,1]
    f = cj.chebfun(jnp.exp)
    ns_poly = [1, 2, 4, 6, 8, 10, 14]
    poly_errs = []
    for n in ns_poly:
        pn = f.polyfit(n)
        err = float(jnp.max(jnp.abs(jnp.array(
            [float((f - pn)(jnp.array(x))) for x in np.linspace(-1, 1, 200)]))))
        poly_errs.append(err)

    ax2 = axes[1]
    ax2.semilogy(ns_poly, poly_errs, 'b.-', lw=1.5, ms=8,
                 label='L2 poly approx error')
    ax2.set_title('Polynomial approx errors for exp(x) on [-1,1]', fontsize=10)
    ax2.set_xlabel('degree n')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Halphen's constant and rational approximation of exp(x)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'Halphen.png'), dpi=150)
    plt.close(fig)

    print(f"Halphen: C = {HALPHEN:.6f}")
    return True


if __name__ == '__main__':
    run()
