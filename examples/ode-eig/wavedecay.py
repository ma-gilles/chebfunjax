"""WaveDecay.m source computations and two600x480 publication figures.

Original: Nick Trefethen, November2010, Chebfun examples f4b9ea46.
https://www.chebfun.org/examples/ode-eig/WaveDecay.html
Copyright by The University of Oxford and The Chebfun Developers.
Public eigs retains current library implementation; historical eigenfunction
signs and graphics defaults are not inferred from image pixels.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import jax.numpy as jnp
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT/'src') not in sys.path:
    sys.path.insert(0, str(_ROOT/'src'))

from chebfunjax.chebfun1d.chebfun import chebfun  # noqa: E402
from chebfunjax.operators.chebop import Chebop  # noqa: E402
from chebfunjax.plotting import (  # noqa: E402
    chebfun_style,
    matlab_plot,
    save_chebfun_figure,
)


def _panel_plot(path, modes, band=None):
    # Source figure position600x480. ReferencePNG physical resolution72.009dpi
    # is the PNG integer-pixels/meter representation of72dpi, not a font fit.
    fig, axes = plt.subplots(4, 1, figsize=(600/72, 480/72), dpi=72)
    for index, (ax, (number, eigenvalue, vector)) in enumerate(zip(axes, modes)):
        vector = vector/vector.norm(float('inf'))
        if band is not None:
            # Native patch default black edges; no source alpha override.
            ax.fill(band*jnp.array([-1., 1., 1., -1.]),
                    [-1.6, -1.6, 2.2, 2.2], facecolor=(1., .8, .8),
                    edgecolor='black', linewidth=.5)
        matlab_plot(vector, ax=ax)
        ax.set_xlim(-jnp.pi/2, jnp.pi/2)
        ax.set_ylim(-1.6, 2.2)
        if index < 3:
            ax.set_xticks([])
        ax.text(.3, 1.6, f'mode {number}         lam = {float(eigenvalue):6.3f}', fontsize=12)
    fig.set_facecolor('white')
    save_chebfun_figure(fig, path, size=(600, 480), dpi=72)
    plt.close(fig)


def _modes(operator):
    numbers = [1, 2, 20, 40]
    eigenvalues, vectors = operator.eigs(k=max(numbers), return_eigenfunctions=True)
    # The source real spectrum is sorted descending, then the same columns
    # are selected. Preserve solver-provided signs instead of fitting images.
    order = jnp.argsort(-eigenvalues)
    return [(number, eigenvalues[order[number-1]], vectors[int(order[number-1])])
            for number in numbers]


def run(output_dir=None):
    output = Path(output_dir) if output_dir is not None else _ROOT/'docs/images/ode-eig'
    output.mkdir(parents=True, exist_ok=True)
    chebfun_style()
    operator = Chebop(lambda u: u.diff(2), domain=(-jnp.pi/2, jnp.pi/2))
    operator.bc = 'dirichlet'
    _panel_plot(output/'WaveDecay_01.png', _modes(operator))

    a = .2
    x = chebfun('x', domain=(-jnp.pi/2, jnp.pi/2))
    middle = abs(x) <= a
    operator.op = lambda x_, u: u.diff(2)+(2/a)*middle*u.diff()
    _panel_plot(output/'WaveDecay_02.png', _modes(operator), band=a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path)
    run(parser.parse_args().output_dir)
