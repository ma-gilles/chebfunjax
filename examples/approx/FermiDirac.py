"""Rational approximation of the Fermi-Dirac function.

The Fermi-Dirac function f(x) = 1/(1 + exp(x)) has a rapid transition
near x=0 that makes it ideal for rational approximation.

Credit: Nick Trefethen, July 2019.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/FermiDirac.html
"""

import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()

from chebfunjax.utils.aaa import aaa

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')

def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # MATLAB: L = 20; f = @(x) 1./(1+exp(x-L));
    # The transplant s <-> x maps the semi-infinite decay onto [-1, 1] so the
    # smooth transition can be resolved; g is the transplanted Fermi-Dirac.
    #   x = @(s) (s*L+L)./(1-s);  s = @(x) (x-L)./(x+L);  g = @(s) f(x(s));
    # This is exactly symmetric in appearance but NOT symmetric about s = 0,
    # which the two printed values demonstrate.
    L = 20.0
    f_fd = lambda x: 1.0 / (1.0 + jnp.exp(x - L))
    x_of_s = lambda s: (s * L + L) / (1.0 - s)
    g = lambda s: f_fd(x_of_s(s))
    g_pos = float(g(jnp.array(0.1)))
    one_minus_g_neg = float(1.0 - g(jnp.array(-0.1)))
    # MATLAB: disp([g(.1) 1-g(-.1)])
    print(f"{g_pos:.15f}   {one_minus_g_neg:.15f}")

    def fermi_dirac(x): return jnp.array(1.0 / (1.0 + jnp.exp(x)))

    # The function on [-10, 10]
    f = cj.chebfun(fermi_dirac, domain=(-10.0, 10.0))

    xx = np.linspace(-10.0, 10.0, 600)
    f_vals = np.array([float(f(jnp.array(x))) for x in xx])

    # AAA approximation
    xs_aaa = jnp.linspace(-10.0, 10.0, 500)
    ys_aaa = jnp.array([fermi_dirac(x) for x in xs_aaa])
    r, pol, res, zer, *_ = aaa(ys_aaa, xs_aaa)

    r_vals = np.array([float(r(jnp.array(x)).real) for x in xx])
    err_vals = np.abs(r_vals - f_vals)

    fig, axes = plt.subplots(1, 3)

    ax = axes[0]
    ax.plot(xx, f_vals, 'b', lw=1.8)
    ax.set_title('Fermi-Dirac function 1/(1+exp(x))', fontsize=10)
    ax2 = axes[1]
    ax2.semilogy(xx, err_vals + 1e-18, 'b', lw=1.5)
    ax2.set_title(f'AAA approximation error ({len(pol)} poles)', fontsize=10)
    ax3 = axes[2]
    pol_arr = np.array([complex(p) for p in pol])
    ax3.plot(pol_arr.real, pol_arr.imag, '.r', ms=10)
    ax3.set_title('Poles of AAA approximant', fontsize=10)
    ax3.axhline(0, color='k', lw=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'FermiDirac.png'), dpi=150)
    plt.close(fig)

    print(f"FermiDirac: {len(pol)} AAA poles, max err = {np.max(err_vals):.2e}")
    return True

if __name__ == '__main__':
    run()
