"""Fractional calculus: closed-form formulas.

Faithful replica of integro/FracCalc2.m by Nick Hale (June 2014,
based on a Chebfun workshop talk): closed-form identities for
Riemann-Liouville fractional integrals and derivatives of Legendre,
Chebyshev, and Jacobi polynomials and of exp(x), each verified
against ``cumsum(f, alpha)`` / ``diff(f, alpha)``.

Original: https://www.chebfun.org/examples/integro/FracCalc2.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import beta as scipy_beta
from scipy.special import gamma

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.polynomials import jacpoly, legpoly
from chebfunjax.utils.transforms import jac2cheb, leg2cheb

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'integro')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"FracCalc2_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _cmp_plot(J1, J2, t, title):
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(t, np.asarray(J1(jnp.asarray(t))).ravel(), 'b',
            lw=1.6, label="J1")
    ax.plot(t, np.asarray(J2(jnp.asarray(t))).ravel(), '--r',
            lw=1.6, label="J2")
    ax.set_title(title, fontsize=11)
    ax.legend()
    ax.grid(True)
    _save(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    x = cj.chebfun(lambda t: t, domain=(-1, 1))
    t = np.linspace(-1 + 1e-8, 1, 900)

    # Half-integral of P_4: (T_n + T_{n+1}) / (gamma(1/2)(n+1/2)sqrt(1+x))
    n = 4
    P4 = Chebfun.from_coeffs(legpoly(n))
    J1 = P4.cumsum(0.5)
    T4 = Chebfun.from_coeffs(jnp.zeros(n + 1).at[n].set(1.0))
    T5 = Chebfun.from_coeffs(jnp.zeros(n + 2).at[n + 1].set(1.0))
    J2 = (T4 + T5) / ((1 + x).sqrt() * float(gamma(0.5) * (n + 0.5)))
    _cmp_plot(J1, J2, t, "Half-integral of $P_4(x)$")

    # Half-integral of exp(x) via Legendre coefficients
    f = cj.chebfun(jnp.exp, domain=(-1, 1))
    nf = len(f)
    c = np.asarray(f.legcoeffs(nf))
    tmp = c / ((np.arange(nf) + 0.5) * gamma(0.5))
    b = np.concatenate([tmp, [0.0]]) + np.concatenate([[0.0], tmp])
    J1 = Chebfun.from_coeffs(leg2cheb(jnp.asarray(b))) / (1 + x).sqrt()
    J2 = f.cumsum(0.5)
    _cmp_plot(J1, J2, t, "Half-integral of $\\exp(x)$")

    # Quarter-integral of (1+x)^beta * P_n^{(0,beta)}
    n = 4
    bet = 0.3
    Jp = Chebfun.from_coeffs(jacpoly(n, 0.0, bet))
    fj = ((1 + x) ** bet) * Jp
    mu = 0.25
    J1 = fj.cumsum(mu)
    Jp2 = Chebfun.from_coeffs(jacpoly(n, -mu, bet + mu))
    J2 = (((1 + x) ** (bet + mu)) * Jp2
          * float(scipy_beta(bet + n + 1, mu) / gamma(mu)))
    _cmp_plot(J1, J2, t,
              "Quarter-integral of $(1+x)^\\beta P_n^{(0,\\beta)}(x)$")

    # Quarter-integral of exp(x) via Jacobi coefficients
    nf = len(f)
    cj_ = np.asarray(f.jaccoeffs(nf, 0.0, 0.0))
    tmp = (scipy_beta(np.arange(1, nf + 1), mu) / gamma(mu)) * cj_
    bcoef = jac2cheb(jnp.asarray(tmp), -mu, mu)
    J1 = Chebfun.from_coeffs(bcoef) * ((1 + x) ** mu)
    J2 = f.cumsum(mu)
    _cmp_plot(J1, J2, t, "Quarter-integral of $\\exp(x)$")

    # Caputo vs Riemann-Liouville quarter-derivative of exp(x)
    nceil = int(np.ceil(mu))
    Df_Caputo = f.diff(nceil).fracInt(nceil - mu)
    Df_RL = f.fracInt(nceil - mu).diff(nceil)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(t, np.asarray(Df_Caputo(jnp.asarray(t))).ravel(), 'b',
            lw=1.6, label="Caputo")
    ax.plot(t, np.asarray(Df_RL(jnp.asarray(t))).ravel(), '--r',
            lw=1.6, label="Riemann-Liouville")
    ax.set_title("Quarter-derivative of $\\exp(x)$", fontsize=11)
    ax.legend()
    ax.grid(True)
    _save(fig)

    # diff(diff(f, a), 1-a) = diff(f)
    d1 = Df_Caputo.fracDiff(1 - mu, kind="Caputo")
    d2 = Df_RL.fracDiff(1 - mu, kind="RL")
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(t, np.asarray(d1(jnp.asarray(t))).ravel(), 'b',
            lw=1.6, label="Caputo")
    ax.plot(t, np.asarray(d2(jnp.asarray(t))).ravel(), '--r',
            lw=1.6, label="Riemann-Liouville")
    ax.plot(t, np.asarray(f.diff()(jnp.asarray(t))).ravel(), ':g',
            lw=1.6, label="f'")
    ax.set_title("diff(diff(f, a), 1-a) = diff(f)", fontsize=11)
    ax.legend()
    ax.grid(True)
    _save(fig)


if __name__ == "__main__":
    run()
