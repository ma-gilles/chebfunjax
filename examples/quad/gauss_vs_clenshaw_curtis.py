"""Gauss and Clenshaw-Curtis quadrature.

Compares Gauss-Legendre and Clenshaw-Curtis quadrature convergence.
Based on Chebfun example quad/GaussClenCurt.m by Nick Trefethen (September 2010).

Original: https://www.chebfun.org/examples/quad/GaussClenCurt.html
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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
from chebfunjax.utils.quadrature import chebpts, chebweights, legpts

chebfun_style()


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/quad')
    os.makedirs(outdir, exist_ok=True)

    # MATLAB: x = chebfun('x'); f = @(x) x.*sin(2*exp(2*sin(2*exp(2*x))));
    #         fc = chebfun(f); Ichebfun = sum(fc); Npts = length(fc);
    f_fn = lambda x: x * np.sin(2 * np.exp(2 * np.sin(2 * np.exp(2 * x))))
    f_jax = lambda x: x * jnp.sin(2 * jnp.exp(2 * jnp.sin(2 * jnp.exp(2 * x))))

    fc = cj.chebfun(f_jax)
    Ichebfun = float(fc.sum())
    Npts = len(np.asarray(fc.coeffs))
    print(f"Ichebfun = {Ichebfun:.15f}")
    print(f"Npts = {Npts}")

    # Clenshaw-Curtis quadrature at Npts points: [s,w]=chebpts(Npts); w*f(s)
    s = np.asarray(chebpts(Npts, 2))
    w = np.asarray(chebweights(Npts, 2))
    Iclenshawcurtis = float(w @ f_fn(s))
    print(f"Iclenshawcurtis = {Iclenshawcurtis:.15f}")

    # Gauss quadrature at Npts points: [s,w]=legpts(Npts); w*f(s)
    sg, wg = legpts(Npts)
    Igauss = float(np.asarray(wg) @ f_fn(np.asarray(sg)))
    print(f"Igauss = {Igauss:.15f}")

    # --- Plot -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2)
    xx = np.linspace(-1, 1, 800)
    axes[0].plot(xx, f_fn(xx), color='#0072BD', lw=1.4)
    axes[0].set_title(r'$f(x) = x\sin(2e^{2\sin(2e^{2x})})$', fontsize=11)
    NN = list(range(10, 700, 20))
    err_g, err_c = [], []
    for n in NN:
        xg, wgn = np.asarray(legpts(n)[0]), np.asarray(legpts(n)[1])
        err_g.append(abs(float(wgn @ f_fn(xg)) - Ichebfun))
        sc, wc = np.asarray(chebpts(n, 2)), np.asarray(chebweights(n, 2))
        err_c.append(abs(float(wc @ f_fn(sc)) - Ichebfun))
    axes[1].semilogy(NN, err_g, color='#0072BD', marker='.', lw=1.4, label='Gauss')
    axes[1].semilogy(NN, err_c, color='#D95319', marker='.', lw=1.4, label='Clenshaw-Curtis')
    axes[1].set_title('Convergence', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].set_ylim(bottom=1e-17)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'gauss_vs_clenshaw_curtis.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("gauss_vs_clenshaw_curtis: done")
    return True

if __name__ == "__main__":
    run()
