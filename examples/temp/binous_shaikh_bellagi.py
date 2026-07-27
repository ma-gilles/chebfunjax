"""Problems from Binous, Shaikh and Bellagi.

Two boundary-value / evolution problems solved with chebop: a nonlinear
BVP  u*u' - u'' = 1  on [-1, 1], and the heat equation u_t = u_yy on [0, 1]
advanced with the operator exponential.  Faithful port of
temp/BinousShaikhBellagi.m.

Original: https://www.chebfun.org/examples/temp/BinousShaikhBellagi.html
Author: Nick Trefethen, September 2014
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
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/temp')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2)

    # --- Problem 1: nonlinear BVP  u*u' - u'' = 1  -----------------------
    # MATLAB:
    #   N = chebop(-1,1);
    #   N.op = @(x,u) u.*diff(u) - diff(u,2);
    #   N.lbc = 0; N.rbc = 2;
    #   u = N\1;
    #   u([0 1/sqrt(2)])
    N = Chebop(domain=(-1.0, 1.0))
    N.op = lambda x, u: u * u.diff() - u.diff(2)
    N.lbc = 0.0
    N.rbc = 2.0
    u = N.solve(1.0)
    u0 = float(u(jnp.array(0.0)))
    us = float(u(jnp.array(1.0 / np.sqrt(2.0))))
    print(f"u([0 1/sqrt(2)]) = {u0:.15f}   {us:.15f}")

    xx = np.linspace(-1.0, 1.0, 400)
    axes[0].plot(xx, np.asarray(u(jnp.array(xx))), color='#0072BD', lw=2.5)
    axes[0].plot([0.0, 1.0 / np.sqrt(2.0)], [u0, us], 'o', color='#D95319')
    axes[0].set_title("Nonlinear BVP: u*u' - u'' = 1", fontsize=10)

    # --- Problem 2: heat equation via the operator exponential -----------
    # MATLAB:
    #   L = chebop(0,1); L.op = @(y,u) diff(u,2); L.bc = 'dirichlet';
    #   u0 = chebfun('1',[0 1]); t = 0.0126; u1 = expm(L,t,u0);
    #   u1(0.5)
    L = Chebop(domain=(0.0, 1.0))
    L.op = lambda y, u: u.diff(2)
    L.bc = 'dirichlet'
    u0f = cj.chebfun(1.0, domain=(0.0, 1.0))
    u1 = L.expm(0.0126, u0f)
    print(f"u1(0.5) = {float(u1(jnp.array(0.5))):.15f}")

    yy = np.linspace(0.0, 1.0, 400)
    axes[1].plot(yy, np.asarray(u1(jnp.array(yy))), color='#D95319', lw=2.5,
                 label='t = 0.0126')
    axes[1].axhline(1.0, color='k', ls='--', lw=1.2, label='t = 0 (IC)')
    axes[1].set_title('Heat equation: u_t = u_yy', fontsize=10)
    axes[1].legend(fontsize=9)

    fig.suptitle('Problems from Binous, Shaikh and Bellagi', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'binous_shaikh_bellagi.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("binous_shaikh_bellagi: done")
    return True


if __name__ == "__main__":
    run()
