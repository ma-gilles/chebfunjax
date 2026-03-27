"""Least-squares approximation in Chebfun.

Demonstrates L2 best polynomial approximation using polyfit, including
convergence rates for |x| (the algebraic O(n^{-3/2}) case).

Credit: Alex Townsend, October 2013.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/BestL2Approximation.html
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

    x = cj.chebfun(lambda t: t)

    # 1. Best L2 of degree 5 for |x|
    f = cj.chebfun(jnp.abs)
    p5 = f.polyfit(5)
    xx = np.linspace(-1.0, 1.0, 400)
    f_vals = np.abs(xx)
    p5_vals = np.array([float(p5(jnp.array(xi))) for xi in xx])

    fig, axes = plt.subplots(1, 3)

    ax = axes[0]
    ax.plot(xx, f_vals, 'b', lw=1.8, label='|x|')
    ax.plot(xx, p5_vals, 'r', lw=1.5, label='p5 L2')
    ax.set_title('Best L2 approximation of |x|, degree 5', fontsize=10)
    ax.legend(fontsize=9)
    # 2. L2 of degree 10 for Runge function
    f_runge = cj.chebfun(lambda x: 1.0 / (1.0 + 25.0 * x**2))
    p10 = f_runge.polyfit(10)
    runge_vals = np.array([float(f_runge(jnp.array(xi))) for xi in xx])
    p10_vals = np.array([float(p10(jnp.array(xi))) for xi in xx])

    ax2 = axes[1]
    ax2.plot(xx, runge_vals, 'b', lw=1.8, label='Runge')
    ax2.plot(xx, p10_vals, 'r', lw=1.5, label='p10 L2')
    ax2.set_title('Best L2 of Runge, degree 10', fontsize=10)
    ax2.legend(fontsize=9)
    # 3. Convergence of ||f - p_n||_2 for |x|: expect O(n^{-3/2})
    f_abs = cj.chebfun(jnp.abs)
    ns = [1, 2, 5, 10, 20, 50, 100]
    errs = []
    for n in ns:
        pn = f_abs.polyfit(n)
        err = float(jnp.abs((f_abs - pn).norm(2)))
        errs.append(err)

    ns_arr = np.array(ns)
    errs_arr = np.array(errs)
    ax3 = axes[2]
    ax3.loglog(ns_arr, errs_arr, 'k.-', lw=1.8, ms=10)
    ax3.loglog(ns_arr, ns_arr**(-1.5), 'k--', lw=1.2, label='n^{-3/2}')
    ax3.set_title('Convergence for |x|', fontsize=10)
    ax3.set_xlabel('n')
    ax3.set_ylabel('‖|x| − p_n‖₂')
    ax3.legend(fontsize=9)
    fig.suptitle('Least-squares (L2) approximation in Chebfun', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'BestL2Approximation.png'), dpi=150)
    plt.close(fig)

    print(f"BestL2Approximation: done. deg-10 Runge max err = "
          f"{np.max(np.abs(p10_vals - runge_vals)):.3e}")
    return True


if __name__ == '__main__':
    run()
