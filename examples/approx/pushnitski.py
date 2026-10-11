"""Approximating Pushnitski's reciprocal log function.

Provenance
----------
MATLAB source: approx/Pushnitski.m, Nick Trefethen, November 2016.
Chebfun commit: 7574c77 (reference library).
Original: https://www.chebfun.org/examples/approx/Pushnitski.html
"""
import json
import os
import time
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, plotcoeffs, save_chebfun_figure
from chebfunjax.utils.cfpade import cf
from chebfunjax.utils.minimax import minimax

_HERE = Path(__file__).resolve().parent
_IMG = _HERE / '../../docs/images/approx'
_CAPTURE = os.environ.get('CHEBFUNJAX_CAPTURE_DIR')


def _capture(name, f):
    if not _CAPTURE:
        return
    arrays = {'domain': np.asarray(f.domain.breakpoints),
              'point_values': np.asarray(f._breakpoint_values())}
    for k, piece in enumerate(f.funs):
        arrays[f'coeffs_{k}'] = np.asarray(piece.coeffs)
    np.savez(Path(_CAPTURE) / f'{name}.npz', **arrays)


def _save(fig, number):
    fig.tight_layout()
    save_chebfun_figure(fig, _IMG / f'Pushnitski_{number:02d}.png',
                       size=(600, 270))
    plt.close(fig)


def run():
    chebfun_style()
    _IMG.mkdir(parents=True, exist_ok=True)
    if _CAPTURE:
        Path(_CAPTURE).mkdir(parents=True, exist_ok=True)
    f = cj.chebfun('-heaviside(x)./log(x)', domain=[-.1, .1], splitting=True)
    _capture('initial_f', f)
    fig, ax = plt.subplots(figsize=(6, 2.7))
    f.plot(ax=ax, color='k', linewidth=2)
    ax.set_ylim(-.2, .5)
    ax.grid(True)
    _save(fig, 1)

    f1000 = cj.chebfun('-heaviside(x)./log(x)', domain=[-.1, .1], n=1000)
    _capture('f1000', f1000)
    fig, ax = plt.subplots(figsize=(6, 2.7))
    plotcoeffs(f1000, ax=ax, loglog=True, source=True)
    ax.set_xlim(0, 500)
    ax.grid(True)
    _save(fig, 2)

    records = []
    t0 = time.perf_counter()
    f = cj.chebfun([0, lambda x: -1 / jnp.log(x)],
                  domain=[-.1, 0, .1], n=1000)
    _capture('piecewise_f', f)
    for cell, number in [('polynomial', 3), ('rational', 4), ('cf', 5)]:
        if cell != 'polynomial':
            t0 = time.perf_counter()
        fig, axes = plt.subplots(2, 2, figsize=(6, 2.7))
        for m in range(1, 5):
            if cell == 'polynomial':
                result = minimax(f, 4*m, domain=(-.1, .1))
                p = cj.chebfun(result.coeffs, coeffs=True, domain=(-.1, .1))
                residual = f - p
                title = f'degree {2*m}'  # Literal native title, degree is 4*m.
            elif cell == 'rational':
                result = minimax(f, m-1, rational=True, denom=m-1,
                                 domain=(-.1, .1))
                p, q = result.as_chebfuns()
                residual = f - p/q
                title = f'type ({m-1},{m-1})'
            else:
                p, q, _, error = cf(f, m-1, m-1, 4000)
                residual = f - p/q
                title = f'type ({m-1},{m-1})'
            _capture(f'{cell}_{m}_p', p)
            if cell != 'polynomial':
                _capture(f'{cell}_{m}_q', q)
            _capture(f'{cell}_{m}_residual', residual)
            record = {'cell': cell, 'm': m}
            if cell == 'cf':
                record['error'] = float(error)
            else:
                record.update(error=float(result.err), iterations=int(result.iter),
                              delta=float(result.delta))
            records.append(record)
            if _CAPTURE:
                (Path(_CAPTURE) / 'fits.json').write_text(json.dumps(records, indent=2)+'\n')
            ax = axes.flat[m-1]
            residual.plot(ax=ax)
            ax.grid(True)
            ax.set_title(title, fontsize=10)
        _save(fig, number)
        print(f'Elapsed time is {time.perf_counter()-t0:.6f} seconds.', flush=True)


if __name__ == '__main__':
    run()
