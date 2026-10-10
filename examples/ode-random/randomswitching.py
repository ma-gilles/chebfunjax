"""Linear ODEs with random switching, source example f4b9ea4.

Original: https://www.chebfun.org/examples/ode-random/RandomSwitching.html
Nick Trefethen, May 2017. Copyright by The University of Oxford and
the Chebfun Developers.

All four original computations use source-order draws and native solver
defaults. JAX key 1 is an explicit RNG adapter, not MATLAB rng(1).
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import jax

from chebfunjax import randnfun
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-random')
DOM = (0.0, 40.0)


def _plot_scalar(y):
    fig, ax = plt.subplots(figsize=(6, 2.69), dpi=100)
    matlab_plot(y, ax=ax, linewidth=4)
    ax.grid(True)
    ax.set_xlim(*DOM)
    ax.set_xticks(np.arange(0, 41, 5))
    _savefig(fig, os.path.join(_IMG, 'RandomSwitching_01.png'),
             size=(600, 269), layout='matlab')
    plt.close(fig)


def _plot_pair(u, v, index, width, linear_limits, log_limits, exponents):
    fig, axes = plt.subplots(2, 1, figsize=(6, 2.69), dpi=100)
    matlab_plot(u, ax=axes[0], linewidth=width)
    matlab_plot(v, ax=axes[0], linewidth=width)
    axes[0].set_title('u and v on linear scale')
    axes[0].set_ylim(*linear_limits)
    # Form the continuous function as in the source before plotting it.
    matlab_plot(u**2 + v**2, 'k', ax=axes[1], linewidth=width)
    axes[1].set_yscale('log')
    axes[1].set_title('norm of (u,v) on log scale')
    axes[1].set_ylim(*log_limits)
    ticks = 10.0**np.asarray(exponents)
    axes[1].set_yticks(ticks[(ticks >= log_limits[0]) & (ticks <= log_limits[1])])
    for ax in axes:
        ax.grid(True)
        ax.set_xlim(*DOM)
        ax.set_xticks(np.arange(0, 41, 5))
    # Renderer mapping of the two axes boxes in the 600 x 269 references.
    fig.subplots_adjust(left=.13, right=.905, bottom=.115, top=.889, hspace=.57)
    _savefig(fig, os.path.join(_IMG, f'RandomSwitching_{index:02d}.png'),
             size=(600, 269))
    plt.close(fig)


def _print_matrix(name, matrix):
    print(f'{name} =')
    for row in matrix:
        print(''.join(f'{value:6.0f}' for value in row))
    print()


def run():
    os.makedirs(_IMG, exist_ok=True)
    started = time.perf_counter()
    key = jax.random.PRNGKey(1)
    scalar = Chebop(domain=DOM)
    scalar.lbc = 1.
    c = randnfun(1., DOM, key=key).sign()
    key, _ = jax.random.split(key)
    scalar.op = lambda t, y: y.diff()-c*y
    y = scalar.solve(0.)
    _plot_scalar(y)

    A = np.asarray([[-1., 5.], [0., -1.]])
    B = np.asarray([[-1., 0.], [-5., -1.]])
    _print_matrix('A', A)
    _print_matrix('B', B)
    operator = Chebop(domain=DOM)
    operator.lbc = lambda u, v: [u-1, v-1]
    operator.maxnorm = [200., 200.]
    panels = [
        (3., 4., (-3., 3.), (1e-5, 1e2), [-4, -2, 0, 2, 4]),
        (1., 3., (-300., 300.), (1e-1, 1e6), [-4, -2, 0, 2, 4]),
        (1./3., 2.5, (-3., 3.), (1e-8, 1e2), [-8, -4, 0, 4]),
    ]
    for index, (lam, width, linear_limits, log_limits, exponents) in enumerate(panels, 2):
        f = 5*(1+randnfun(lam, DOM, key=key).sign())/2
        key, _ = jax.random.split(key)
        operator.op = lambda t, u, v: [u.diff()+u-f*v, v.diff()+v+(5-f)*u]
        u, v = operator.solve(0.)
        _plot_pair(u, v, index, width, linear_limits, log_limits, exponents)
    print('total_time_in_seconds =')
    print(f'  {time.perf_counter()-started:.6f}')


if __name__ == '__main__':
    run()
