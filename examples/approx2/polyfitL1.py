"""Best polynomial approximation in the L1 norm.

Compares L∞ (minimax), L2 (least squares), and L1 best polynomial approximants.
The L1 approximant has error concentrated near singularities.

Credit: Yuji Nakatsukasa and Alex Townsend, July 2019.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/polyfitL1.html
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


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Test function: |x - 1/4| on [-1, 1]
    f = cj.chebfun(lambda x: jnp.abs(x - 0.25))
    deg = 20

    # L2 approximation
    p_L2 = f.polyfit(deg)

    xx = np.linspace(-1.0, 1.0, 500)
    f_vals = np.abs(xx - 0.25)
    p_L2_vals = np.array([float(p_L2(jnp.array(x))) for x in xx])
    err_L2 = p_L2_vals - f_vals

    # L1-like approximation: minimize sum |f(xi) - p(xi)| via scipy IRLS
    from scipy.optimize import linprog
    # Sample points for L1 fitting
    xi = np.linspace(-1.0, 1.0, 200)
    yi = np.abs(xi - 0.25)

    # Construct Vandermonde in Chebyshev basis
    V = np.zeros((len(xi), deg + 1))
    for k in range(deg + 1):
        V[:, k] = np.cos(k * np.arccos(xi))

    # IRLS for L1 fit
    w = np.ones(len(xi))
    for _ in range(10):
        WV = (V.T * w).T
        Wy = w * yi
        coeffs_L1, _, _, _ = np.linalg.lstsq(WV, Wy, rcond=None)
        residuals = np.abs(yi - V @ coeffs_L1)
        w = 1.0 / np.maximum(residuals, 1e-10)

    # Evaluate L1 fit
    V_eval = np.zeros((len(xx), deg + 1))
    for k in range(deg + 1):
        V_eval[:, k] = np.cos(k * np.arccos(np.clip(xx, -1, 1)))
    p_L1_vals = V_eval @ coeffs_L1
    err_L1 = p_L1_vals - f_vals

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(xx, f_vals, 'k', lw=1.5, label='f = |x−1/4|')
    ax.plot(xx, p_L2_vals, 'b--', lw=1.3, label=f'L2 deg {deg}')
    ax.plot(xx, p_L1_vals, 'r--', lw=1.3, label=f'L1 deg {deg}')
    ax.set_title(f'|x−1/4| and polynomial approximants (deg {deg})', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(xx, err_L2, 'b', lw=1.5, label='L2 error')
    ax2.plot(xx, err_L1, 'r', lw=1.5, label='L1 error')
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_title('Error curves: L2 vs. L1 approximation', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('L1 vs. L2 polynomial approximation', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'polyfitL1.png'), dpi=150)
    plt.close(fig)

    print(f"polyfitL1: L2 max err={np.max(np.abs(err_L2)):.3e}, "
          f"L1 max err={np.max(np.abs(err_L1)):.3e}")
    return True


if __name__ == '__main__':
    run()
