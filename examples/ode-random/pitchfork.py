"""Pitchfork bifurcation triggered by noise.

Translation of ode-random/Pitchfork.m, Nick Trefethen, May 2017.
All five solves retain [0,600] and the source forcing signs. The explicit
sequential JAX stream is unmatched to MATLAB rng(0).

Original: https://www.chebfun.org/examples/ode-random/Pitchfork.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use('Agg')
import sys
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from chebfunjax import randnfun
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style, matlab_plot, save_chebfun_figure

chebfun_style()
_IMG = Path(__file__).resolve().parents[2] / 'docs/images/ode-random'
DOM = (0.0, 600.0)
LAMBDA = 2.0
_SIZE = (600, 269)
_DPI = 72.0


def _draw_forcing(key):
    forcing = 0.003 * randnfun(DOM, LAMBDA, 'big', key=key)
    # This nonperiodic source branch consumes one real normal matrix draw.
    key, _ = jax.random.split(key)
    return forcing, key


def _operator(damped=False):
    if not damped:
        return lambda t, y: y.diff(2) - 2*(-1+t/300)*y + 4*y**3
    return lambda t, y: y.diff(2) - 2*(-1+t/300)*y + 4*y**3 + 0.2*y.diff()


def _plot(solutions, title, filename):
    fig, ax = plt.subplots(figsize=(_SIZE[0]/_DPI, _SIZE[1]/_DPI), dpi=_DPI)
    for solution, style in zip(solutions, ('--k', 'b', 'r')):
        matlab_plot(solution, style, ax=ax, linewidth=2.5)
    ax.set_xlabel('t', fontsize=32)
    ax.set_ylabel('y', fontsize=32)
    ax.set_title(title, fontsize=32)
    ax.set_xlim(*DOM)
    ax.set_ylim(-0.8, 0.8)
    ax.grid(True)
    save_chebfun_figure(fig, _IMG/filename, size=_SIZE, dpi=_DPI, layout='matlab')
    plt.close(fig)


def run():
    _IMG.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    key = jax.random.PRNGKey(0)
    N = Chebop(_operator(), domain=DOM)
    N.lbc = [0.0, 0.0]
    y1 = N.solve(0.0)
    f1, key = _draw_forcing(key)
    y2 = N.solve(f1)
    f2, key = _draw_forcing(key)
    y3 = N.solve(f2)
    undamped = (y1, y2, y3)
    _plot(undamped, 'Pitchfork', 'Pitchfork_01.png')
    N.op = _operator(damped=True)
    y2 = N.solve(-f1)
    y3 = N.solve(f2)
    damped = (y1, y2, y3)
    _plot(damped, 'Pitchfork with damping', 'Pitchfork_02.png')
    elapsed = time.perf_counter() - started
    print('total_time_in_seconds =')
    print(f'  {elapsed:.6f}')
    return {'forcing': (f1, f2), 'undamped': undamped, 'damped': damped,
            'elapsed_seconds': elapsed}


if __name__ == '__main__':
    run()
