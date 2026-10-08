"""Pricing other European Options: Puts, Digitals, Powers.

Translation of applics/EuropeanOptions.m (Pachon, 2014).
Original: https://www.chebfun.org/examples/applics/EuropeanOptions.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import os
import sys

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.special import ndtr
from matplotlib.ticker import FormatStrFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'applics')
FIG = [0]
S0, VOL, R, T, MAXS = 100, 0.45, 0.01, 0.5, 10000


def logn(s):
    # The source expression has the removable 0/0 endpoint at s=0.
    # Supply its analytic limit while retaining the source's [0,maxS] domain.
    positive = jnp.where(s > 0, s, 1.0)
    value = (jnp.exp(-(jnp.log(positive / S0) - (R - 0.5 * VOL**2) * T)**2
                     / (2 * VOL**2 * T))
             / (VOL * positive * jnp.sqrt(2 * jnp.pi * T)))
    return jnp.where(s > 0, value, 0.0)


def _axes():
    fig, ax = plt.subplots()
    ax.set_position([0.13, 0.11, 0.775, 0.815])
    ax.xaxis.set_label_coords(.5, -.075)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.tick_params(direction='in', top=True, right=True, labelsize=6,
                   width=0.5, length=3, colors='0.25')
    for spine in ax.spines.values():
        spine.set_color('0.5')
        spine.set_linewidth(0.5)
    return fig, ax


def _save(fig):
    FIG[0] += 1
    # Historical published PNGs are 600x268; tick sizes vary by source cell.
    _savefig(fig, os.path.join(_IMG, f"EuropeanOptions_{FIG[0]:02d}.png"),
             size=(600, 268))
    plt.close(fig)


def _pdf_plot(pdf, xlim, approx, markerheight, ylim=None):
    fig, ax = _axes()
    xx = np.linspace(float(pdf.domain.a), float(pdf.domain.b), 2001)
    ax.fill_between(xx, np.asarray(pdf.remove_deltas()(xx)), color=(0.3, 0.9, 0.4))
    matlab_plot(pdf, 'k', ax=ax, linewidth=0.8, deltaline='r', markersize=2)
    ax.plot([approx, approx], [0, markerheight], 'b--', linewidth=0.8)
    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 50, 100, 150])
    else:
        # The historical power-density PNG labels only zero on the y axis.
        ax.set_yticks(np.arange(0, .61, .1), ['0']+['']*6)
    ax.set_xlabel(r'$e^{-rT}V(S_T)$', fontsize=6)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.2)
    _save(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    lognPDF = chebfun(logn, domain=(0, MAXS))
    lognCDF = lognPDF.cumsum()
    disc = np.exp(-R*T)
    fig, ax = _axes()
    matlab_plot(lognPDF, 'k', ax=ax, interval=(0, 200), linewidth=0.8)
    ax.set_ylim(0, 0.015)
    ax.set_yticks(np.arange(0, 0.0151, 0.003))
    ax.set_xticks(np.arange(0, 201, 20))
    ax.set_xlabel(r'$S_T$', fontsize=6)
    ax.grid(True, alpha=0.2)
    _save(fig)

    K = 150.0
    put = chebfun(lambda s: jnp.maximum(0, K-s), domain=(0, K, 250))
    fig, ax = _axes()
    matlab_plot(put, 'k', ax=ax, interval=(0, 250), linewidth=0.8)
    ax.set_xlabel('S', fontsize=6)
    ax.set_ylabel('V(S)', fontsize=6)
    ax.set_ylim(-10, K)
    _save(fig)
    probOOM = 1-float(lognCDF(K))
    print(f"probOOM =\n   {probOOM:.15f}")
    x = chebfun('x', domain=(0, K))
    OOM = 2*probOOM*x.dirac()
    ITM = chebfun(lambda s: logn(K-s), domain=(0, K))
    payoffPDF = disc*(OOM+ITM)
    approx = float((x*payoffPDF).sum())
    _pdf_plot(payoffPDF, (-10, K), approx, .025, (0, .08))
    print(f"approx = {approx:.15f}")
    d1 = (np.log(S0/K)+(R+0.5*VOL**2)*T)/(VOL*np.sqrt(T))
    d2 = d1-VOL*np.sqrt(T)
    exact = -float(S0*ndtr(-d1)-K*ndtr(-d2)*disc)
    print(f"exact  = {exact:.15f}\napprox = {approx:.15f}")
    results = {'put': {'approx': approx, 'exact': exact, 'oom_mass': float(OOM.sum())}}

    K = 100.0
    s = chebfun('x', domain=(0, 200))
    digital = (s-K).heaviside()
    fig, ax = _axes()
    matlab_plot(digital, 'k', ax=ax, linewidth=0.8, jumpline='k:')
    ax.set_xlabel('S', fontsize=6)
    ax.set_ylabel('V(S)', fontsize=6)
    ax.set_ylim(-.1, 1.1)
    ax.set_xticks(np.arange(0, 201, 20))
    _save(fig)
    probOOM = float(lognCDF(K))
    probITM = 1-probOOM
    x = chebfun('x', domain=(0, 1))
    OOM = 2*probOOM*x.dirac()
    ITM = 2*probITM*(x-1).dirac()
    payoffPDF = disc*(OOM+ITM)
    approx = float((x*payoffPDF).sum())
    fig, ax = _axes()
    matlab_plot(OOM, ax=ax, color='r', linewidth=0.8, markersize=2)
    matlab_plot(ITM, ax=ax, color='lime', linewidth=0.8, markersize=2)
    ax.plot([approx, approx], [0, .3], 'b--', linewidth=0.8)
    ax.set_xlim(-.5, 1.5)
    ax.set_ylim(0, .6)
    ax.set_xticks([-.5, 0, .5, 1, 1.5])
    ax.set_yticks(np.arange(0, .61, .1))
    ax.set_xlabel(r'$e^{-rT}V(S_T)$', fontsize=6)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.2)
    _save(fig)
    print(f"approx = {approx:.15f}")
    d1 = (np.log(S0/K)+(R+0.5*VOL**2)*T)/(VOL*np.sqrt(T))
    d2 = d1-VOL*np.sqrt(T)
    exact = float(ndtr(d2))*disc
    print(f"exact  = {exact:.15f}\napprox = {approx:.15f}")
    results['digital'] = {'approx': approx, 'exact': exact,
                          'oom_mass': float(OOM.sum()), 'itm_mass': float(ITM.sum()),
                          'values': np.asarray(digital(jnp.array([99., 100., 101.]))).tolist()}

    K, alpha, maxV = 9.1, 0.5, 50.0
    alphainv = 1/alpha
    power = chebfun(lambda s: jnp.maximum(0, s**alpha-K), domain=(0, K**alphainv, 1000))
    fig, ax = _axes()
    matlab_plot(power, 'k', ax=ax, linewidth=0.8)
    ax.set_xlabel('S', fontsize=6)
    ax.set_ylabel('V(S)', fontsize=6)
    _save(fig)
    probOOM = float(lognCDF(K**alphainv))
    x = chebfun('x', domain=(0, maxV))
    OOM = 2*probOOM*x.dirac()
    ITM = chebfun(lambda s: logn((s+K)**alphainv)
                  * jnp.abs(alphainv*(s+K)**(alphainv-1)), domain=(0, maxV))
    payoffPDF = disc*(OOM+ITM)
    approx = float((x*payoffPDF).sum())
    _pdf_plot(payoffPDF, (-.5, 10), approx, .3)
    print(f"approx = {approx:.15f}")
    d1 = (np.log(S0/K**alphainv)+(R+(alpha-.5)*VOL**2)*T)/(VOL*np.sqrt(T))
    d2 = d1-alpha*VOL*np.sqrt(T)
    m = (R+.5*alpha*VOL**2)*(alpha-1)
    exact = float(S0**alpha*np.exp(m*T)*ndtr(d1)-disc*K*ndtr(d2))
    print(f"exact  = {exact:.15f}\napprox = {approx:.15f}")
    results['power'] = {'approx': approx, 'exact': exact, 'oom_mass': float(OOM.sum())}
    if '_PAGE_REPORT' in globals():
        import json
        from pathlib import Path
        Path(globals()["_PAGE_REPORT"]).write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    run()
