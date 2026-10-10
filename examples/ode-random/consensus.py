"""Collective dynamics and consensus.

Translation of ode-random/Consensus.m, Nick Trefethen, May 2017.
The physical IVP domain is [0,40]. Source nonperiodic randnfun embeds its
forcing on [0,48] before restriction; this does not extend the IVP interval.
An explicit sequential JAX stream is unmatched to MATLAB rng(3).

Original: https://www.chebfun.org/examples/ode-random/Consensus.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use('Agg')
import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from chebfunjax import randnfun
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style, matlab_plot, save_chebfun_figure

chebfun_style()
_IMG = Path(__file__).resolve().parents[2] / 'docs/images/ode-random'
DOM = (0.0, 40.0)
_SIZE = (600, 269)
_SOURCE_FONT_SIZE = 32
_SOURCE_LINE_WIDTH = 2.5
_EXPORT_DPI = 72.0


def _forcing():
    """Native nonperiodic big construction; explicit unmatched JAX stream."""
    key = jax.random.PRNGKey(3)
    f = randnfun(.2, DOM, 'big', key=key)
    # This source nonperiodic branch consumes exactly one normal matrix draw.
    key, _ = jax.random.split(key)
    g = randnfun(.2, DOM, 'big', key=key)
    return f, g


def _operator(strength, f, g):
    if strength == 0:
        return lambda t, u, v: [u.diff() + f, v.diff() + g]
    return lambda t, u, v: [
        u.diff() + f + strength * (u-v) * (-(u-v)**2).exp(),
        v.diff() + g + strength * (v-u) * (-(v-u)**2).exp()]


def _plot_solution(solution, title):
    fig, ax = plt.subplots(figsize=(_SIZE[0]/_EXPORT_DPI,
                                    _SIZE[1]/_EXPORT_DPI), dpi=_EXPORT_DPI)
    matlab_plot(solution[0], ax=ax, linewidth=_SOURCE_LINE_WIDTH)
    ax.grid(True)
    matlab_plot(solution[1], ax=ax, linewidth=_SOURCE_LINE_WIDTH)
    ax.set_xlabel('t', fontsize=_SOURCE_FONT_SIZE)
    ax.set_ylabel('u,v', fontsize=_SOURCE_FONT_SIZE)
    ax.set_title(title, fontsize=_SOURCE_FONT_SIZE)
    return fig, ax


def run():
    _IMG.mkdir(parents=True, exist_ok=True)
    f, g = _forcing()
    N = Chebop(_operator(0, f, g), domain=DOM)
    N.lbc = lambda u, v: [u-1, v+1]
    cases = [(0, 'Two independent random walks'),
             (3, 'Walks strongly attracted together'),
             (1.0, 'Walks weakly attracted together')]
    solutions = []
    for index, (strength, title) in enumerate(cases, start=1):
        N.op = _operator(strength, f, g)
        solution = N.solve(0.0)
        solutions.append({'strength': strength, 'solution': solution,
                          'backend': getattr(N, '_ivp_backend_used', None)})
        fig, _ = _plot_solution(solution, title)
        save_chebfun_figure(fig, _IMG/f'Consensus_{index:02d}.png',
                           size=_SIZE, dpi=_EXPORT_DPI, layout='matlab')
        plt.close(fig)
    return {'forcing': (f, g), 'solutions': solutions}


if __name__ == '__main__':
    run()
