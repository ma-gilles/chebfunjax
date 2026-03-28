"""Approximating Pushnitski's reciprocal log function.

The function 1/|log|x|| on [-1,1] is continuous but has a logarithmic
singularity at 0 that requires many terms for polynomial approximation.

Credit: Nick Trefethen, November 2016.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/Pushnitski.html
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

from chebfunjax.utils.aaa import aaa

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')

def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Pushnitski function: 1/|log|x||, approximated away from origin
    def pushnitski(x):
        return 1.0 / jnp.abs(jnp.log(jnp.abs(x) + 1e-15))

    # Piecewise to avoid singularity at 0
    dom = (-1.0, -0.001, 0.001, 1.0)
    f = cj.chebfun(pushnitski, domain=dom)

    xx = np.linspace(-1.0, 1.0, 800)
    # Avoid exact 0
    xx_safe = xx[np.abs(xx) > 0.001]
    f_vals = np.array([float(pushnitski(jnp.array(x))) for x in xx_safe])

    # Polynomial approximation lengths vs. degree
    degrees = [10, 50, 100]
    poly_errs = []
    for n in degrees:
        pn = f.polyfit(n)
        pn_safe = np.array([float(pn(jnp.array(x))) for x in xx_safe])
        poly_errs.append(np.max(np.abs(pn_safe - f_vals)))

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    ax.plot(xx_safe, f_vals, 'b', lw=1.8)
    ax.set_title('Pushnitski function 1/|log|x||', fontsize=11)
    ax2 = axes[1]
    ax2.semilogy(degrees, poly_errs, color='#0072BD', linestyle='.-', lw=1.5, ms=10)
    ax2.set_title('Polynomial approx errors for 1/|log|x||', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'Pushnitski.png'), dpi=150)
    plt.close(fig)

    print(f"Pushnitski: len(f)={len(f)}")
    return True

if __name__ == '__main__':
    run()
