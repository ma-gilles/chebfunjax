"""Source computations and plots for approx/BestApprox.m.

Nick Trefethen, September 2010. Original:
https://www.chebfun.org/examples/approx/BestApprox.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import curve_plot_data
from chebfunjax.utils.minimax import minimax

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def _errplot(xs, values, err, dom, ylim, title, number):
    # Historical publication canvas/axes/ticks. This is a rendering adapter;
    # the MATLAB source sets line widths, font size and data limits.
    fig = plt.figure(figsize=(6, 2.69), dpi=100, facecolor='white')
    ax = fig.add_axes([.13, .11, .775, .815])
    ax.plot(xs, values, color='#0072BD', lw=.6)
    ax.plot(dom, [err, err], '--', color='.55', lw=.6)
    ax.plot(dom, [-err, -err], '--', color='.55', lw=.6)
    ax.set_xlim(*dom)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=4.5, pad=2)
    ax.tick_params(direction='in', top=True, right=True, labelsize=10,
                   colors='.25', width=.4, length=3)
    for spine in ax.spines.values():
        spine.set_color('.55')
        spine.set_linewidth(.5)
    if number <= 3:
        ax.set_xticks(jnp.linspace(-1, 1, 5))
    elif number == 4:
        ax.set_xticks([.45, .5, .55])
    else:
        ax.set_xticks([.498, .499, .5, .501, .502])
    if number >= 3:
        ax.set_yticks(jnp.linspace(-4e-5, 4e-5, 5))
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    if number == 1:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    fig.savefig(os.path.join(_IMG, f'BestApprox_{number:02d}.png'), dpi=100)
    return fig


def run():
    os.makedirs(_IMG, exist_ok=True)
    x = cj.chebfun('x')
    f = abs(x - .5)
    result = minimax(f, 16)
    p = cj.chebfun(jnp.asarray(result.coeffs), coeffs=True)
    # Public source Chebyshev plotData sampling, piece by piece. Encoding the
    # graph as x+i*y preserves each error piece's oversampling count.
    data = curve_plot_data(x + 1j * (f - p))
    figures = [_errplot(data['xLine'], data['yLine'], result.err,
                        (-1, 1), (-.03, .03),
                        'Degree 16 polynomial error curve', 1)]
    r88 = minimax(f, 8, rational=True, denom=8)
    p, q = r88.as_chebfuns()
    data = curve_plot_data(x + 1j * (f - p / q))
    figures.append(_errplot(data['xLine'], data['yLine'], r88.err,
                           (-1, 1), (-.003, .003),
                           'Type (8,8) rational error curve', 2))
    r16 = minimax(f, 16, rational=True, denom=16)
    for number, domain, title in (
        (3, (-1, 1), 'Type (16,16) rational error curve'),
        (4, (.45, .55), 'Zoom near singularity'),
        (5, (.4975, .5025), 'Closer zoom'),
    ):
        xx = jnp.linspace(*domain, 3000)
        figures.append(_errplot(xx, f(xx) - r16.r(xx), r16.err, domain,
                               (-4e-5, 4e-5), title, number))
    return {'figures': figures, 'polynomial': result, 'r88': r88, 'r16': r16}


if __name__ == '__main__':
    run()
