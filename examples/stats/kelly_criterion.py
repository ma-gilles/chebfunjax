"""Optimal bet-sizing and the Kelly Criterion.

Demonstrates how to use calculus to find the optimal betting fraction
that maximizes long-run capital growth. Translated from
stats/KellyCriterion.m.

Original: https://www.chebfun.org/examples/stats/KellyCriterion.html
Author: Mark Richardson, October 2012
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()



def growth_rate(f, p, a):
    """Log capital growth rate: G(f) = p*log(1+a*f) + (1-p)*log(1-f)."""
    if f <= 0 or f >= 1:
        return -np.inf
    v1 = 1 + a * f
    v2 = 1 - f
    if v1 <= 0 or v2 <= 0:
        return -np.inf
    return p * np.log(v1) + (1 - p) * np.log(v2)


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- 1. Simple Kelly problem: a=2, p=0.5 ---
    p_simple, a_simple = 0.5, 2.0

    # Kelly optimal: f* = (a*p - (1-p)) / a
    f_kelly = (a_simple * p_simple - (1 - p_simple)) / a_simple
    print(f"Kelly criterion: f* = {f_kelly:.6f}  (exact: 0.25)")
    assert abs(f_kelly - 0.25) < 1e-10

    G_max = growth_rate(f_kelly, p_simple, a_simple)
    print(f"G(f*) = {G_max:.6f}")
    print(f"exp(G(f*)) = {np.exp(G_max):.6f} (growth factor per bet)")

    fs = np.linspace(0, 0.999, 1000)
    G_vals = np.array([growth_rate(f, p_simple, a_simple) for f in fs])
    valid = np.isfinite(G_vals)
    expG = np.where(valid, np.exp(G_vals), np.nan)

    axes[0].plot(fs[valid], expG[valid], 'b-', linewidth=2)
    axes[0].plot(f_kelly, np.exp(G_max), '.r', markersize=15, label=f'f*={f_kelly:.2f}')
    axes[0].axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Breakeven')
    axes[0].set_title('Capital growth: simple bet (a=2, p=0.5)', fontsize=10)
    axes[0].set_xlabel('Fraction bet f')
    axes[0].set_ylabel('exp(G(f))')
    axes[0].set_ylim(0, 1.2)
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    # Find breakeven (where G(f) = 0)
    from scipy.optimize import brentq
    try:
        f_break = brentq(lambda f: growth_rate(f, p_simple, a_simple), 0.5, 0.999)
        print(f"Breakeven at f = {f_break:.6f}")
    except Exception:
        f_break = None

    # --- 2. Multi-outcome Kelly ---
    # 6 possible outcomes
    probs = [0.4, 0.21, 0.26, 0.1, 0.02, 0.01]
    payoffs = [-1, 0, 1.2, 1.3, 1.4, 10]

    def G_multi(f):
        if f <= 0 or f >= 1:
            return -np.inf
        total = 0
        for pi, ai in zip(probs, payoffs):
            v = 1 + ai * f
            if v <= 0:
                return -np.inf
            total += pi * np.log(v)
        return total

    fs2 = np.linspace(0.001, 0.999, 1000)
    G2_vals = np.array([G_multi(f) for f in fs2])
    valid2 = np.isfinite(G2_vals)
    expG2 = np.where(valid2, np.exp(G2_vals), np.nan)

    # Find optimal numerically
    idx_max = np.nanargmax(expG2)
    f_opt = fs2[idx_max]
    G_opt = G2_vals[idx_max]
    print(f"\nMulti-outcome Kelly: f* = {f_opt:.6f}")
    print(f"exp(G(f*)) = {np.exp(G_opt):.6f}")

    axes[1].plot(fs2[valid2], expG2[valid2], 'b-', linewidth=2)
    axes[1].plot(f_opt, np.exp(G_opt), '.r', markersize=15, label=f'f*={f_opt:.3f}')
    axes[1].axhline(1.0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_title('Capital growth: multi-outcome bet', fontsize=10)
    axes[1].set_xlabel('Fraction bet f')
    axes[1].set_ylabel('exp(G(f))')
    axes[1].set_ylim(0, 1.2)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    fig.suptitle('Kelly Criterion: Optimal Bet Sizing', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'kelly_criterion.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("kelly_criterion: done")
    return True


if __name__ == "__main__":
    run()
