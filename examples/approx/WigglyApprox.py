"""A wiggly function and its best approximations.

Explores approximation of the oscillatory function sin²(x) + sin(x²)
using Chebyshev polynomial interpolation.

Credit: Ricardo Pachon and Nick Trefethen, November 2010.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/WigglyApprox.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')

def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    dom = (0.0, 14.0)
    def wiggly(x): return jnp.sin(x)**2 + jnp.sin(x**2)
    f = cj.chebfun(wiggly, domain=dom)

    xx = np.linspace(0.0, 14.0, 1000)
    f_vals = np.array([float(f(jnp.array(x))) for x in xx])
    true_vals = np.sin(xx)**2 + np.sin(xx**2)

    # Show low-degree approximant
    p50 = f.polyfit(50)
    p50_vals = np.array([float(p50(jnp.array(x))) for x in xx])

    fig, axes = plt.subplots(2, 1)

    ax = axes[0]
    ax.plot(xx, f_vals, 'b', lw=1.5, label=f'adaptive (len={len(f)})')
    ax.plot(xx, p50_vals, color='#D95319', linestyle='--', lw=1.3, label='deg-50 approx')
    ax.set_title('Wiggly function: sin²(x) + sin(x²)', fontsize=11)
    ax.legend(fontsize=9)
    ax2 = axes[1]
    coeffs = np.abs(np.array(f.coeffs)) + 1e-18
    ax2.semilogy(np.arange(len(coeffs)), coeffs, color='#0072BD', marker='.', linestyle='none', ms=4)
    ax2.set_title('Chebyshev coefficients', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'WigglyApprox.png'), dpi=150)
    plt.close(fig)

    err = np.max(np.abs(f_vals - true_vals))
    print(f"WigglyApprox: len(f)={len(f)}, max err vs. true = {err:.2e}")
    return True

if __name__ == '__main__':
    run()
