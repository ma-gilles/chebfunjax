"""Rational approximation of the Fermi–Dirac function.

Nick Trefethen, July 2019. Original:
https://www.chebfun.org/examples/approx/FermiDirac.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import os
import sys
import time

import matplotlib

matplotlib.use('Agg')
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from chebfunjax.utils.minimax import minimax
from chebfunjax.utils.quadrature import chebpts_ab

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def make_g(L):
    return lambda s: 1 / (1 + jnp.exp((s*L+L)/(1-s)-L))


def _style(ax, title):
    if title.startswith('Fermi-Dirac'):
        # Explicit y avoids Matplotlib raising the title above the canvas
        # to make room for the scientific-notation offset.
        ax.set_title(title, fontsize=10.5, y=1.0, pad=3)
    else:
        ax.set_title(title, fontsize=11, pad=5)
    ax.grid(True, color='.8', linewidth=.45)
    ax.tick_params(direction='in', top=True, right=True, labelsize=10,
                   width=.5, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(.5)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_fontsize(9)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    if not title.startswith('Fermi-Dirac'):
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))


def _save(fig, name):
    fig.savefig(os.path.join(_IMG, name), dpi=100)


def fermi(L, n, fname):
    g = make_g(L)
    result = minimax(g, n, rational=True, denom=n)
    err, poles = result.err, jnp.asarray(result.poles)
    ss = jnp.concatenate((chebpts_ab(1000, -1., 0.),
                          chebpts_ab(1000, 0., 1.)))
    # Historical six 600x270 canvases and visible axes rectangles. These
    # positions adapt MATLAB subplot/axis equal rendering to Matplotlib.
    fig = plt.figure(figsize=(6, 2.7), dpi=100, facecolor='white')
    ax = fig.add_axes([.13, .5889, .775, .337])
    ax.plot(ss, result.r(ss)-g(ss), color='#0072BD', lw=1.4)
    ax.plot([-1, 1], [-err, -err], '--r', lw=1.4)
    ax.plot([-1, 1], [err, err], '--r', lw=1.4)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-2*err, 2*err)
    if L == 1000 and n == 20:
        ax.set_yticks(jnp.asarray([-4, -2, 0, 2, 4])*1e-6)
    ax.set_xticks([-1, -.5, 0, .5, 1])
    _style(ax, f'Fermi-Dirac transplanted to [-1,1], L = {L}, n = {n}')
    for left, limits, title in ((.16, [-20, 20, -10, 10], 'poles'),
                                (.60, [-1, 1, -.5, .5], 'closeup')):
        ax = fig.add_axes([left, .115, .275, .3])
        line, = ax.plot(jnp.real(poles), jnp.imag(poles), '.r', ms=4.0)
        # Match the historical window's visible markers while retaining all
        # pole coordinates; Matplotlib otherwise paints partial markers
        # whose centers are outside the requested source axes limits.
        visible = ((jnp.real(poles) >= limits[0]) & (jnp.real(poles) <= limits[1])
                   & (jnp.imag(poles) >= limits[2]) & (jnp.imag(poles) <= limits[3]))
        line.set_markevery(jnp.flatnonzero(visible).tolist())
        ax.set_aspect('equal')
        ax.axis(limits)
        ax.set_xticks([limits[0], 0, limits[1]])
        ax.set_yticks([limits[2], 0, limits[3]])
        _style(ax, title)
    _save(fig, fname)
    return result, fig


def run():
    os.makedirs(_IMG, exist_ok=True)
    L = 20
    f = lambda x: 1/(1+jnp.exp(x-L))  # noqa: E731
    g = make_g(L)
    # Explicit rendering adapter for the two MATLAB builtin fplot calls:
    # 2000 uniform points over the complete requested interval. No exponent
    # clipping or endpoint truncation; native fplot adaptive sampling is open.
    figures = []
    for index, (callback, domain, title) in enumerate((
        (f, (0, 80), 'physical domain'),
        (g, (-1, 1), 'transplantation to [-1,1]'),
    ), 1):
        xs = jnp.linspace(*domain, 2000)
        fig = plt.figure(figsize=(6, 2.7), dpi=100, facecolor='white')
        ax = fig.add_axes([.13, .115, .775, .8])
        ax.plot(xs, callback(xs), color='#0072BD', lw=.75)
        ax.set_xlim(*domain)
        ax.set_ylim(-1, 2)
        if index == 2:
            ax.set_xticks([-1, -.5, 0, .5, 1])
        _style(ax, title)
        _save(fig, f'FermiDirac_{index:02d}.png')
        figures.append(fig)
    print(f'   {float(g(jnp.asarray(.1))):.15f}   {float(1-g(jnp.asarray(-.1))):.15f}', flush=True)
    results = []
    for index, (L, n) in enumerate(((10, 10), (100, 15), (1000, 20), (1000, 30)), 3):
        start = time.perf_counter()
        result, fig = fermi(L, n, f'FermiDirac_{index:02d}.png')
        print(f'Elapsed time is {time.perf_counter()-start:.6f} seconds.', flush=True)
        figures.append(fig)
        results.append(result)
    return {'figures': figures, 'results': results}


if __name__ == '__main__':
    run()
