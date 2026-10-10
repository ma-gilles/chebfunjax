"""Random level hopping.

Translation of ode-random/LevelHopping.m, Nick Trefethen, May 2017.
Both source nonperiodic random forcings and [0,100] solves are retained.
The explicit sequential JAX stream is unmatched to MATLAB rng(0).

Original: https://www.chebfun.org/examples/ode-random/LevelHopping.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use('Agg')
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from chebfunjax import randnfun
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style, matlab_plot, save_chebfun_figure

chebfun_style()
_IMG = Path(__file__).resolve().parents[2] / 'docs/images/ode-random'
DOM = (0.0, 100.0)
_SIZE = (600, 269)
_DPI = 72.0


def _draw_forcing(lam, key):
    forcing = randnfun(lam, DOM, 'big', key=key)
    # The real nonperiodic source branch consumes one normal matrix draw.
    key, _ = jax.random.split(key)
    return forcing, key


def _operator(y):
    return y.diff() + 2 * (2 * jnp.pi * y).sin()


def _plot(solution, linewidth, filename):
    fig, ax = plt.subplots(figsize=(_SIZE[0]/_DPI, _SIZE[1]/_DPI), dpi=_DPI)
    matlab_plot(solution, ax=ax, linewidth=linewidth)
    ax.grid(True)
    ax.set_xlabel('t', fontsize=32)
    ax.set_ylabel('y', fontsize=32)
    save_chebfun_figure(fig, _IMG/filename, size=_SIZE, dpi=_DPI, layout='matlab')
    plt.close(fig)


def run():
    _IMG.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(0)
    started = time.perf_counter()
    N = Chebop(_operator, domain=DOM)
    lam = 0.4
    f1, key = _draw_forcing(lam, key)
    N.lbc = 0.0
    y1 = N.solve(f1)
    _plot(y1, 2, 'LevelHopping_01.png')
    lam = lam/2
    f2, key = _draw_forcing(lam, key)
    y2 = N.solve(f2)
    _plot(y2, 1, 'LevelHopping_02.png')
    elapsed = time.perf_counter() - started
    print('total_time_in_seconds =')
    print(f'  {elapsed:.6f}')
    return {'forcing': (f1, f2), 'solutions': (y1, y2),
            'elapsed_seconds': elapsed}


if __name__ == '__main__':
    run()
