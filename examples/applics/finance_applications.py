"""Finance applications: Black-Scholes and option pricing.

Demonstrates using chebfunjax for financial mathematics: Black-Scholes
option pricing, Greeks computation, and European option valuation,
following applics/BlackScholes2D.m, applics/EuropeanOptions.m,
and applics/Greeks.m.

Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()



def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes European call option price."""
    S, K, T, r, sigma = [np.asarray(x, dtype=float) for x in [S, K, T, r, sigma]]
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)


def black_scholes_delta(S, K, T, r, sigma):
    """Delta: dC/dS"""
    S, K, T, r, sigma = [np.asarray(x, dtype=float) for x in [S, K, T, r, sigma]]
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)


def run():
    print("=" * 60)
    print("Finance: Black-Scholes option pricing")
    print("=" * 60)

    # Parameters
    K = 100.0   # strike
    T = 1.0     # time to expiry
    r = 0.05    # risk-free rate
    sigma = 0.2 # volatility

    # --- European call as function of stock price ---
    print("\nEuropean call: K=100, T=1, r=5%, σ=20%")
    S_domain = [50.0, 200.0]

    call_fn = cj.chebfun(
        lambda S: jnp.array(black_scholes_call(np.array(S), K, T, r, sigma)),
        domain=S_domain
    )
    print(f"  Degree: {len(call_fn.funs[0].tech.coeffs)}")

    # Test at S = K = 100 (at-the-money)
    val_atm = float(call_fn(jnp.array(100.0)))
    exact_atm = float(black_scholes_call(100.0, K, T, r, sigma))
    print(f"  C(S=100) = {val_atm:.6f}  (exact: {exact_atm:.6f})")
    assert abs(val_atm - exact_atm) < 1e-6

    # Test deep in-the-money
    val_itm = float(call_fn(jnp.array(180.0)))
    exact_itm = float(black_scholes_call(180.0, K, T, r, sigma))
    print(f"  C(S=180) = {val_itm:.6f}  (exact: {exact_itm:.6f})")
    assert abs(val_itm - exact_itm) < 1e-5

    # --- Delta (derivative of call w.r.t. S) ---
    delta_fn = call_fn.diff()
    delta_atm = float(delta_fn(jnp.array(100.0)))
    exact_delta = float(black_scholes_delta(100.0, K, T, r, sigma))
    print(f"\n  Delta(S=100) = {delta_atm:.6f}  (exact: {exact_delta:.6f})")
    assert abs(delta_atm - exact_delta) < 0.001

    # --- Put-call parity ---
    # C - P = S*exp(0) - K*exp(-r*T) = S - K*exp(-r*T)
    # P = C - S + K*exp(-r*T)
    put_fn = cj.chebfun(
        lambda S: jnp.array(
            black_scholes_call(np.array(S), K, T, r, sigma)
        ) - S + K * jnp.exp(jnp.array(-r*T)),
        domain=S_domain
    )
    val_put = float(put_fn(jnp.array(100.0)))
    print(f"\n  Put(S=100) via parity = {val_put:.6f}")
    assert val_put > 0

    # --- Volatility surface (sigma as function of S) ---
    print("\nImplied volatility-like surface:")
    sigmas = np.linspace(0.1, 0.5, 5)
    for sig in sigmas:
        c = black_scholes_call(100.0, K, T, r, sig)
        print(f"  σ={sig:.2f}: C(100) = {c:.4f}")

    # --- 2D: call as function of (S, sigma) ---
    f_2d = cj.chebfun2(
        lambda S, sig: jnp.array(
            black_scholes_call(np.array(50 + 150*((S+1)/2)), K, T, r,
                               np.array(0.1 + 0.4*((sig+1)/2)))
        )
    )
    print(f"\n2D Black-Scholes (S in [50,200], σ in [0.1,0.5]): rank {f_2d.rank}")

    # --- Plot ---
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/applics')
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    S_plot = np.linspace(50, 200, 200)
    C_plot = black_scholes_call(S_plot, K, T, r, sigma)
    D_plot = black_scholes_delta(S_plot, K, T, r, sigma)
    intrinsic = np.maximum(S_plot - K, 0)

    axes[0].plot(S_plot, C_plot, 'b-', linewidth=2, label='Call price')
    axes[0].plot(S_plot, intrinsic, 'r--', linewidth=1.5, label='Intrinsic value')
    axes[0].axvline(K, color='k', linestyle=':', alpha=0.5, label=f'Strike K={K}')
    axes[0].set_title("Black-Scholes call option", fontsize=12)
    axes[0].set_xlabel("Stock price S"); axes[0].set_ylabel("Option value")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    axes[1].plot(S_plot, D_plot, 'g-', linewidth=2)
    axes[1].axhline(0.5, color='k', linestyle='--', alpha=0.5, label='Δ=0.5 (ATM)')
    axes[1].set_title("Delta: dC/dS", fontsize=12)
    axes[1].set_xlabel("Stock price S"); axes[1].set_ylabel("Delta")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    # Volatility surface
    S_grid = np.linspace(60, 160, 40)
    sig_grid = np.linspace(0.1, 0.5, 30)
    SG, SIG = np.meshgrid(S_grid, sig_grid)
    C_vol = black_scholes_call(SG, K, T, r, SIG)
    im = axes[2].contourf(SG, SIG, C_vol, levels=20, cmap="viridis")
    axes[2].set_title("Volatility surface", fontsize=12)
    axes[2].set_xlabel("Stock price S"); axes[2].set_ylabel("σ")
    fig.colorbar(im, ax=axes[2], label='Call price')

    fig.suptitle("Black-Scholes option pricing", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "finance_applications.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
