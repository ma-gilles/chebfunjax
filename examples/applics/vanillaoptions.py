"""Exploring Vanilla Options (Pachon, December 2014).

Translation of applics/VanillaOptions.m, using public Chebfun operations.
Original: https://www.chebfun.org/examples/applics/VanillaOptions.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import json
import os
import sys
import time

import jax.numpy as jnp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np  # uses-numpy: plotting coordinates and JSON observations only
from jax.scipy.special import ndtr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from chebfunjax import chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.plotting import chebfun_style, matlab_plot, surf
from chebfunjax.plotting import save_chebfun_figure as _savefig

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'applics')
_PAGE_REPORT = None
_TRACE = None
K, VOL = 100, 0.45


def vanilla(s, strike, maturity, vol, rate, sign):
    """Source Black–Scholes callback, including its s=0 limit."""
    d1 = (jnp.log(s / strike) + (rate + 0.5 * vol**2) * maturity) / (vol * jnp.sqrt(maturity))
    d2 = (jnp.log(s / strike) + (rate - 0.5 * vol**2) * maturity) / (vol * jnp.sqrt(maturity))
    return sign * (s * ndtr(sign * d1) - strike * ndtr(sign * d2) * jnp.exp(-rate * maturity))


def payoff(s, strike, sign):
    return jnp.maximum(0, sign * (s - strike))


def _trace(label):
    if _TRACE:
        with open(_TRACE, 'a') as f:
            f.write(json.dumps({'operation': label, 'time': time.monotonic()}) + '\n')


def _save(fig, number):
    # Historical page raster and explicit normalized MATLAB axes positions.
    _savefig(fig, os.path.join(_IMG, f'VanillaOptions_{number:02d}.png'), size=(600, 268))
    plt.close(fig)
    _trace(f'figure_{number}_complete')


def _pair(xmax, ylim, first=False):
    fig, axes = plt.subplots(1, 2)
    for index, (ax, title) in enumerate(zip(axes, ('put', 'call'), strict=True)):
        ax.set_position([.05 + .5 * index, .13, .42, .75])
        ax.set_title(title, fontsize=6.5 if first else 5.5)
        ax.set_xlabel('S', fontsize=5.5)
        ax.set_xticks(np.arange(0, 351, 50))
        ax.set_xlim(0, xmax)
        ax.set_ylim(*ylim)
        ax.grid(True)
    return fig, axes


def _option(strike, maturity, rate, sign, domain):
    return chebfun(lambda s: vanilla(s, strike, maturity, VOL, rate, sign), domain=domain)


def _hedge(rate, strike3, number, report):
    _trace(f'hedge_{rate}_start')
    asset = chebfun(lambda s: s, domain=(0, 300))
    call = _option(100, .5, rate, +1, (0, 300))
    loan = K * jnp.exp(-rate * .5)
    puts = [_option(100, .5, rate, -1, (0, 300)),
            _option(105.5, .25, rate, -1, (0, 300)),
            1.03 * _option(strike3, .75, rate, -1, (0, 300))]
    errors = [(put + asset - loan) - call for put in puts]
    fig, ax = plt.subplots()
    ax.set_position([.13, .11, .775, .815])
    labels = ('1.00 put/same K/same T', '1.00 put/higher K/ shorter T', '1.03 put/lower K/ longer T')
    for error, color, label in zip(errors, ('b', 'r', 'k'), labels, strict=True):
        matlab_plot(asset, error, color, ax=ax, linewidth=.6, label=label)
    ax.set_xlim(0, 300)
    ax.set_ylim(-6, 8)
    ax.set_yticks(np.arange(-6, 9, 2))
    ax.set_title('Imperfect put-call parity relation, r=0' if rate == 0 else 'Imperfect put-call parity relation, r = 5%', fontsize=6)
    ax.set_xlabel('S', fontsize=6)
    ax.set_ylabel('profit/loss (instant)', fontsize=6)
    ax.legend(loc='upper right', fontsize=5.5)
    ax.grid(True)
    _save(fig, number)
    minima = []
    for index, error in enumerate(errors[1:], start=2):
        location, value = error.min()  # Python API returns location before value.
        location_text = f'{location:.4f}'.rstrip('0').rstrip('.')
        print(f'Max loss stgy {index}: {value:.4f} at {location_text}')
        minima.append({'location': float(location), 'value': float(value)})
    report[f'hedge_{rate}'] = {'minima': minima, 'exact_hedge_norm': float(errors[0].norm(jnp.inf))}
    _trace(f'hedge_{rate}_complete')


def run():
    os.makedirs(_IMG, exist_ok=True)
    chebfun_style()
    plt.rcParams.update({'font.size': 5.5, 'axes.linewidth': .35, 'xtick.labelsize': 5.5,
                         'ytick.labelsize': 5.5, 'grid.linewidth': .3, 'grid.alpha': .35,
                         'lines.linewidth': .4, 'figure.facecolor': 'white'})
    report = {}
    domain = (0, K, 350)
    payoffs = [chebfun(lambda s, sign=sign: payoff(s, K, sign), domain=domain) for sign in (-1, +1)]
    _trace('profiles_start')
    fig, axes = _pair(250, (-10, 250), first=True)
    for ax, terminal, sign in zip(axes, payoffs, (-1, +1), strict=True):
        matlab_plot(terminal, ax=ax, color='#0072bd', linewidth=.6)
        for exponent in range(-1, 9):
            matlab_plot(_option(K, 2.**exponent, 0, sign, domain), 'k', ax=ax)
        matlab_plot(_option(K, 1000, 0, sign, domain), 'r', ax=ax)
    for ax in axes:
        ax.set_xlim(0, 250)
        ax.set_ylim(-10, 250)
        ax.set_xticks(np.arange(0, 251, 50))
    _save(fig, 1)
    _hedge(0, 94.5, 2, report)
    for number, rate, exponents in [(3, 0, range(-1, 9)), (4, .015, range(-1, 5))]:
        _trace(f'time_values_{rate}_start')
        fig, axes = _pair(350, (-30, 110))
        for ax, terminal, sign in zip(axes, payoffs, (-1, +1), strict=True):
            for exponent in exponents:
                matlab_plot(_option(K, 2.**exponent, rate, sign, domain) - terminal, 'k', ax=ax)
            if rate == 0:
                matlab_plot(_option(K, 1000, rate, sign, domain) - terminal, 'r', ax=ax)
        for ax in axes:
            ax.set_xlim(0, 350)
            ax.set_ylim(-30, 110)
        _save(fig, number)
    if _PAGE_REPORT:
        with open(_PAGE_REPORT, 'w') as f:
            json.dump(report, f, indent=2)
    fig = plt.figure()
    for index, rate in enumerate((0, .015)):
        _trace(f'surface_{rate}_construct_start')
        put = Chebfun2.from_function(lambda s, t: vanilla(s, K, t, VOL, rate, -1), domain=(0, 200, .001, 200))
        report[f'surface_{rate}'] = {'domain': list(put.domain), 'rank': put.rank, 'length': list(put.length())}
        _trace(f'surface_{rate}_construct_complete')
        ax = fig.add_subplot(1, 2, index + 1, projection='3d')
        surf(put, ax=ax)
        for surface in ax.collections:
            surface.set_antialiased(False)
            surface.set_edgecolor('face')
        ax.set_proj_type('ortho')
        ax.set_box_aspect((1, 1, .75), zoom=1.3)
        ax.zaxis._axinfo['juggled'] = (1, 2, 0)
        ax.set_position([.05 + .5 * index, .13, .42, .75])
        # Fixed page coordinates keep projected labels inside the historic raster.
        ax.set_xlabel('')
        ax.set_ylabel('')
        fig.text(.095 + .5 * index, .11, 'S', fontsize=5.5, ha='center')
        fig.text(.32 + .5 * index, .035, 'T', fontsize=5.5, ha='center')
        ax.set_zlim(0, 100)
        ax.set_xticks(np.arange(0, 201, 50))
        ax.set_yticks(np.arange(0, 201, 20))
        ax.set_zticks(np.arange(0, 101, 10))
        ax.set_title('put, r=0' if rate == 0 else 'put, r=1.5%', fontsize=5.5)
        # Source camera is [1347 -347 346], with target near box center.
        ax.view_init(elev=24.1, azim=-19.7)
    for index, ax in enumerate(fig.axes):
        ax.set_position([.05 + .5 * index, .13, .42, .75])
        ax.tick_params(labelsize=4.3, pad=0)
    _save(fig, 5)
    fig, ax = plt.subplots()
    ax.set_position([.13, .11, .775, .815])
    for index in range(6):
        rate = .0005 + .010 * index
        _trace(f'boundary_{rate}_construct_start')
        put = Chebfun2.from_function(lambda t, s: vanilla(s, K, t, VOL, rate, -1) - payoff(s, K, -1), domain=(.001, 25, 0, 100))
        _trace(f'boundary_{rate}_roots_start')
        curves = put.roots()
        report[f'boundary_{rate}'] = {'domain': list(put.domain), 'rank': put.rank, 'length': list(put.length()), 'curve_count': len(curves)}
        for curve in curves:
            matlab_plot(curve, ax=ax, linewidth=.6)
        _trace(f'boundary_{rate}_complete')
    ax.tick_params(labelsize=8.5)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 110)
    ax.set_xlabel('Time to maturity', fontsize=5.5, labelpad=-2)
    ax.set_ylabel('Asset level', fontsize=5.5)
    ax.set_title('Asset level at which time value of a put becomes negative', fontsize=5.5)
    ax.grid(True)
    _save(fig, 6)
    _hedge(.05, 95, 7, report)
    if _PAGE_REPORT:
        with open(_PAGE_REPORT, 'w') as f:
            json.dump(report, f, indent=2)


if __name__ == '__main__':
    run()
