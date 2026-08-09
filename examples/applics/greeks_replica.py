"""Accurate Greeks.

Faithful replica of applics/Greeks.m (Pachon, 2014): risk
sensitivities of European calls and puts computed by differentiating
2D chebfun "slices" of the lognormal payoff density with respect to
the bumped parameter, then integrating -- compared against the
closed-form Black-Scholes Greeks.

Original: https://www.chebfun.org/examples/applics/Greeks.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'applics')
FIG = [0]

ST0 = 100.0
VOL = 0.45
TAU = 0.5
MU = 0.07
R = 0.01
K = 100.0


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Greeks_repl_{FIG[0]:02d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def f_pdf(ST, St, vol, tau, mu):
    ST = np.maximum(np.asarray(ST, dtype=float), 1e-300)
    return (np.exp(-(np.log(ST / St) - (mu - 0.5 * vol**2) * tau)**2
                   / (2 * vol**2 * tau))
            / (vol * ST * np.sqrt(2 * np.pi * tau)))


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    ss = np.linspace(1e-6, 200, 1200)

    # --- bumping the PDF (forward finite differences) ---
    PDF = f_pdf(ss, ST0, VOL, TAU, MU)
    dSt, dvol, dtau, dmu = 0.001, 0.0001, 0.0001, 0.0001
    bumps = [
        ((f_pdf(ss, ST0 + dSt, VOL, TAU, MU) - PDF) / dSt,
         r'$[f(S+\delta S)-f(S)]/\delta S$'),
        ((f_pdf(ss, ST0, VOL + dvol, TAU, MU) - PDF) / dvol,
         r'$[f(\sigma+\delta\sigma)-f(\sigma)]/\delta\sigma$'),
        ((f_pdf(ss, ST0, VOL, TAU + dtau, MU) - PDF) / dtau,
         r'$[f(t+\delta t)-f(t)]/\delta t$'),
        ((f_pdf(ss, ST0, VOL, TAU, MU + dmu) - PDF) / dmu,
         r'$[f(\mu+\delta\mu)-f(\mu)]/\delta\mu$'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    for ax, (v, lab) in zip(axes.ravel(), bumps):
        ax.plot(ss, v, lw=1.6)
        ax.set_xlabel(r'$S_T$')
        ax.set_title(lab, fontsize=12)
        ax.grid(True)
    _save(fig)

    # --- 2D slices and their partial derivatives ---
    minS = 0.01
    slices = [
        (Chebfun2.from_function(
            lambda s, SSt: f_pdf(s, SSt, VOL, TAU, MU),
            domain=(minS, 200, 90, 110)), ST0, r'$\partial f/\partial S$'),
        (Chebfun2.from_function(
            lambda s, vvol: f_pdf(s, ST0, vvol, TAU, MU),
            domain=(minS, 200, 0.40, 0.5)), VOL,
         r'$\partial f/\partial\sigma$'),
        (Chebfun2.from_function(
            lambda s, ttau: f_pdf(s, ST0, VOL, ttau, MU),
            domain=(minS, 200, 0.45, 0.55)), TAU,
         r'$\partial f/\partial\tau$'),
        (Chebfun2.from_function(
            lambda s, mmu: f_pdf(s, ST0, VOL, TAU, mmu),
            domain=(minS, 200, 0.06, 0.08)), MU,
         r'$\partial f/\partial\mu$'),
    ]
    partials = [(sl.diff(dim=1), lev, lab) for (sl, lev, lab) in slices]

    fig = plt.figure(figsize=(11.6, 8.2))
    for j, (p, lev, lab) in enumerate(partials, start=1):
        ax = fig.add_subplot(2, 2, j, projection="3d")
        xa, xb, ya, yb = p.domain
        gx = np.linspace(xa, xb, 140)
        gy = np.linspace(ya, yb, 60)
        GX, GY = np.meshgrid(gx, gy)
        ax.plot_surface(GX, GY, np.asarray(p(GX, GY)), cmap="viridis",
                        rstride=1, cstride=2, linewidth=0)
        ax.set_xlabel(r'$S_T$')
        ax.set_title(lab, fontsize=13)
    _save(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    for ax, (p, lev, lab) in zip(axes.ravel(), partials):
        vals = np.asarray(p(ss, np.full_like(ss, lev)))
        ax.plot(ss, vals, lw=1.6)
        ax.set_xlabel(r'$S_T$')
        ax.set_title(lab + " at the prescribed level", fontsize=12)
        ax.grid(True)
    _save(fig)

    # --- Greeks from payoff-density slices ---
    def greeks(W, domx):
        sl_St = Chebfun2.from_function(
            lambda S, SSt: np.exp(-R * TAU)
            * f_pdf(W * S + K, SSt, VOL, TAU, R),
            domain=(0, domx, 80, 120))
        sl_vol = Chebfun2.from_function(
            lambda S, vvol: np.exp(-R * TAU)
            * f_pdf(W * S + K, ST0, vvol, TAU, R),
            domain=(0, domx, 0.41, .5))
        sl_tau = Chebfun2.from_function(
            lambda S, ttau: -np.exp(-R * ttau)
            * f_pdf(W * S + K, ST0, VOL, ttau, R),
            domain=(0, domx, 0.49, 0.51))
        sl_r = Chebfun2.from_function(
            lambda S, rr: np.exp(-rr * TAU)
            * f_pdf(W * S + K, ST0, VOL, TAU, rr),
            domain=(0, domx, 0.006, 0.013))
        parts = [(sl_St.diff(dim=1), ST0), (sl_vol.diff(dim=1), VOL),
                 (sl_tau.diff(dim=1), TAU), (sl_r.diff(dim=1), R)]
        out = []
        for p, lev in parts:
            g1 = chebfun(
                lambda s: np.asarray(p(s, np.full_like(np.asarray(s),
                                                       lev))),
                domain=(0, domx))
            x1 = chebfun(lambda s: np.asarray(s), domain=(0, domx))
            out.append(float((x1 * g1).sum()))
        return out, parts

    (cd, cv, ct, cr), parts = greeks(+1, 5000)

    fig = plt.figure(figsize=(11.6, 8.2))
    labs = [r'$\partial g/\partial S$', r'$\partial g/\partial\sigma$',
            r'$\partial g/\partial\tau$', r'$\partial g/\partial r$']
    for j, ((p, lev), lab) in enumerate(zip(parts, labs), start=1):
        ax = fig.add_subplot(2, 2, j, projection="3d")
        xa, xb, ya, yb = p.domain
        gx = np.linspace(xa, min(xb, 200), 140)
        gy = np.linspace(ya, yb, 60)
        GX, GY = np.meshgrid(gx, gy)
        ax.plot_surface(GX, GY, np.asarray(p(GX, GY)), cmap="viridis",
                        rstride=1, cstride=2, linewidth=0)
        ax.set_xlabel(r'$S_T$')
        ax.set_title(lab, fontsize=13)
    _save(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    ssp = np.linspace(0, 200, 1000)
    for ax, ((p, lev), lab) in zip(axes.ravel(), zip(parts, labs)):
        ax.plot(ssp, np.asarray(p(ssp, np.full_like(ssp, lev))), lw=1.6)
        ax.set_xlabel(r'$S_T$')
        ax.set_title(lab + " at the prescribed level", fontsize=12)
        ax.grid(True)
    _save(fig)

    print(f"delta approx [call] = {cd:.15f}")
    print(f"vega approx  [call] = {cv:.15f}")
    print(f"theta approx [call] = {ct:.15f}")
    print(f"rho approx   [call] = {cr:.15f}")

    (pd_, pv, pt, pr), _ = greeks(-1, 99)
    print(f"delta approx [put] = {pd_:.15f}")
    print(f"vega approx  [put] = {pv:.15f}")
    print(f"theta approx [put] = {pt:.15f}")
    print(f"rho approx   [put] = {pr:.15f}")

    # --- closed-form comparison ---
    d1 = (np.log(ST0 / K) + (R + 0.5 * VOL**2) * TAU) / (VOL * np.sqrt(TAU))
    d2 = d1 - VOL * np.sqrt(TAU)
    phi = norm.pdf(d1)
    delta_c, delta_p = norm.cdf(d1), norm.cdf(d1) - 1
    vega = ST0 * phi * np.sqrt(TAU)
    theta_c = (-ST0 * phi * VOL / (2 * np.sqrt(TAU))
               - R * K * np.exp(-R * TAU) * norm.cdf(d2))
    theta_p = (-ST0 * phi * VOL / (2 * np.sqrt(TAU))
               + R * K * np.exp(-R * TAU) * norm.cdf(-d2))
    rho_c = K * TAU * np.exp(-R * TAU) * norm.cdf(d2)
    rho_p = -K * TAU * np.exp(-R * TAU) * norm.cdf(-d2)

    rows = [
        ("delta", delta_c, cd, delta_p, pd_),
        ("vega ", vega, cv, vega, pv),
        ("theta", theta_c, ct, theta_p, pt),
        ("rho  ", rho_c, cr, rho_p, pr),
    ]
    print("                 call                 put")
    for name, ec, ac, ep, ap in rows:
        print(f"{name} exact  : {ec:.15f}    {ep:.15f}")
        print(f"{name} approx : {ac:.15f}    {ap:.15f}")
        print(f"{name} error  : {abs(ec - ac):.4e}            "
              f"{abs(ep - ap):.4e}")
        print("-" * 55)


if __name__ == "__main__":
    run()
