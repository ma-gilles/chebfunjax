"""Eight Shades of Rational Approximation.

Overview of rational approximation methods available in Chebfun: AAA,
Padé, CF, minimax, and polynomial approximation for comparison.

Credit: Mohsin Javed and Nick Trefethen, January 2016.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/EightShades.html
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

    # Test function: exp(x) on [-1, 1]
    f = cj.chebfun(jnp.exp)
    xx = np.linspace(-1.0, 1.0, 400)
    f_true = np.exp(xx)

    fig, axes = plt.subplots(2, 3)
    axes = axes.flatten()

    # 1. Polynomial degree 4 (Taylor truncation)
    p4 = f.polyfit(4)
    p4_vals = np.array([float(p4(jnp.array(x))) for x in xx])
    err4 = np.abs(p4_vals - f_true)
    axes[0].semilogy(xx, err4 + 1e-18, 'b', lw=1.5)
    axes[0].set_title('1. Polynomial L2, deg 4', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 2. Polynomial degree 8
    p8 = f.polyfit(8)
    p8_vals = np.array([float(p8(jnp.array(x))) for x in xx])
    err8 = np.abs(p8_vals - f_true)
    axes[1].semilogy(xx, err8 + 1e-18, 'r', lw=1.5)
    axes[1].set_title('2. Polynomial L2, deg 8', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # 3. AAA rational approximation (near-minimax)
    xs_aaa = jnp.linspace(-1.0, 1.0, 300)
    ys_aaa = jnp.exp(xs_aaa)
    r_aaa, pol_aaa, *_ = aaa(ys_aaa, xs_aaa)
    err_aaa = np.array([abs(float((jnp.exp(jnp.array(x)) - r_aaa(jnp.array(x))).real))
                        for x in xx])
    axes[2].semilogy(xx, err_aaa + 1e-18, 'g', lw=1.5)
    axes[2].set_title(f'3. AAA rational ({len(pol_aaa)} poles)', fontsize=10)
    axes[2].grid(True, alpha=0.3)

    # 4. Spline approximation (Chebfun.spline is a class method taking x, y arrays)
    xk = np.linspace(-1.0, 1.0, 10)
    yk = np.exp(xk)
    f_spline = cj.Chebfun.spline(xk, yk)
    sp_vals = np.array([float(f_spline(jnp.array(x))) for x in xx])
    err_sp = np.abs(sp_vals - f_true)
    axes[3].semilogy(xx, err_sp + 1e-18, 'm', lw=1.5)
    axes[3].set_title('4. Spline (10 knots)', fontsize=10)
    axes[3].grid(True, alpha=0.3)

    # 5. Pchip (Chebfun.pchip is a class method taking x, y arrays)
    f_pchip = cj.Chebfun.pchip(xk, yk)
    pchip_vals = np.array([float(f_pchip(jnp.array(x))) for x in xx])
    err_pchip = np.abs(pchip_vals - f_true)
    axes[4].semilogy(xx, err_pchip + 1e-18, 'c', lw=1.5)
    axes[4].set_title('5. Pchip (10 knots)', fontsize=10)
    axes[4].grid(True, alpha=0.3)

    # 6. Summary comparison
    axes[5].semilogy(xx, err4 + 1e-18, 'b', lw=1.2, label='poly deg 4')
    axes[5].semilogy(xx, err8 + 1e-18, 'r', lw=1.2, label='poly deg 8')
    axes[5].semilogy(xx, err_aaa + 1e-18, 'g', lw=1.2, label='AAA')
    axes[5].semilogy(xx, err_sp + 1e-18, 'm', lw=1.2, label='spline')
    axes[5].set_title('6. Comparison of methods', fontsize=10)
    axes[5].legend(fontsize=8)
    axes[5].grid(True, alpha=0.3)

    for ax in axes:
        pass
    fig.suptitle('Eight Shades of Rational Approximation (for exp(x))', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'EightShades.png'), dpi=150)
    plt.close(fig)

    print("EightShades: done.")
    return True


if __name__ == '__main__':
    run()