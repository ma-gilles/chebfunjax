"""Conformal mapping in Chebfun — native continuous-boundary computation.

Translation of complex/ConformalMapping.m, Nick Trefethen, October 2019.
The deterministic JAX random stream is not MATLAB rng(0); boundary geometry,
poles and numerical outputs therefore remain unmatched to the website draw.

Original: https://www.chebfun.org/examples/complex/ConformalMapping.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use('Agg')
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.plotting import chebfun_style, matlab_axis_equal, matlab_plot, save_chebfun_figure
from chebfunjax.utils.conformal import conformal
from chebfunjax.utils.randnfun import randnfun

chebfun_style()
_IMG = Path(__file__).resolve().parents[2] / 'docs' / 'images' / 'complex'
_SIZE = (600, 253)
_DPI = 100


def _boundary_input():
    """One explicit JAX stream; not a MATLAB normal/uniform stream oracle.

    The current periodic randnfun source consumes exactly one split normal
    draw. Preserve the earlier key0 boundary realization and continue that
    key state for the two source uniform vectors; no independent reseeding.
    """
    key = jax.random.PRNGKey(0)
    C = chebfun(lambda t: jnp.exp(1j * jnp.pi * t), trig=True)
    C = C * (1 + .15 * randnfun(.2, key=key))
    key, _ = jax.random.split(key)
    return C, key


def _disk_points(key):
    """Two successive uniform vectors, drawn after computing the map."""
    key, first = jax.random.split(key)
    real_draw = jax.random.uniform(first, (20000,), dtype=jnp.float64)
    key, second = jax.random.split(key)
    imag_draw = jax.random.uniform(second, (20000,), dtype=jnp.float64)
    W = 2 * real_draw - 1 + 2j * imag_draw - 1j
    W = W[jnp.abs(W) < 1]
    W = W[:10000]
    if W.size != 10000:
        raise ValueError('Source sample contains fewer than 10000 disk points')
    return W


def _plot_images(C, Z):
    """Native second figure: default axes, fixed ylim and equal data units."""
    fig = plt.figure(figsize=(_SIZE[0]/_DPI, _SIZE[1]/_DPI), dpi=_DPI)
    ax = fig.add_axes([.13, .11, .775, .815])
    matlab_plot(C, 'b', ax=ax)
    # Native Line point diameter is MarkerSize/3, as in matlab_spy.
    ax.plot(jnp.real(Z), jnp.imag(Z), color='k', linestyle='none',
            marker=MarkerStyle('.').scaled(2/3), markersize=3, markeredgewidth=0)
    ax.set_ylim(-1.3, 1.3)
    matlab_axis_equal(ax)
    return fig, ax


def run():
    os.makedirs(_IMG, exist_ok=True)
    C, key = _boundary_input()
    fig = plt.figure(figsize=(_SIZE[0]/_DPI, _SIZE[1]/_DPI), dpi=_DPI)
    t0 = time.perf_counter()
    f, finv, pol, polinv = conformal(C, plots=True)
    print(f'Elapsed time is {time.perf_counter()-t0:.6f} seconds.')
    save_chebfun_figure(fig, _IMG / 'ConformalMapping_01.png', size=_SIZE, dpi=_DPI)
    plt.close(fig)

    W = _disk_points(key)
    t0 = time.perf_counter()
    Z = finv(W)
    Z.block_until_ready()
    print(f'Elapsed time is {time.perf_counter()-t0:.6f} seconds.')
    fig, _ = _plot_images(C, Z)
    save_chebfun_figure(fig, _IMG / 'ConformalMapping_02.png', size=_SIZE, dpi=_DPI)
    plt.close(fig)

    Wb = C(jnp.linspace(-1, 1, 1001))
    Zb = f(Wb)
    deviation = jnp.linalg.norm(jnp.abs(Zb)-1, ord=jnp.inf)
    print('max_deviation_from_circle =')
    print(f'     {float(deviation):.15e}')
    W2 = finv(Zb)
    error = jnp.linalg.norm(Wb-W2, ord=jnp.inf)
    print('max_back_and_forth_error =')
    print(f'     {float(error):.15e}')
    return {'boundary': C, 'disk_points': W, 'mapped_points': Z,
            'boundary_points': Wb, 'mapped_boundary': Zb, 'returned_boundary': W2,
            'poles': pol, 'inverse_poles': polinv,
            'deviation': deviation, 'back_and_forth_error': error}


if __name__ == '__main__':
    run()
