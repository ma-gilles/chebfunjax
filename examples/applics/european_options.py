"""Pricing European Options: Puts, Digitals, Powers.

Extends the European call pricing to put options, digital options,
and power options, following applics/EuropeanOptions.m by Ricardo Pachon
(December 2014).

Original MATLAB: https://www.chebfun.org/examples/applics/EuropeanOptions.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chebfunjax.plotting import chebfun_style
chebfun_style()

import numpy as np
from scipy.stats import norm
import os

def run():
    print("=" * 60)
    print("European Options: Puts, Digitals, Powers")
    print("=" * 60)

    # Common parameters
    S0 = 100.0
    vol = 0.45
    r = 0.01
    T = 0.5
    maxS = 2000.0
    N = 3000
    S_grid = np.linspace(1e-3, maxS, N)
    dS = S_grid[1] - S_grid[0]

    # Lognormal PDF under risk-neutral measure
    def lognorm_pdf(S):
        mu_ln = np.log(S0) + (r - 0.5 * vol**2) * T
        sig_ln = vol * np.sqrt(T)
        return (np.exp(-(np.log(S) - mu_ln)**2 / (2 * sig_ln**2))
                / (S * sig_ln * np.sqrt(2 * np.pi)))

    pdf_vals = lognorm_pdf(S_grid)
    cdf_vals = np.cumsum(pdf_vals * dS)

    def price_option(payoff_fn):
        """Price option: e^(-rT) * E[payoff(S_T)]."""
        payoff = payoff_fn(S_grid)
        return np.exp(-r * T) * np.trapezoid(pdf_vals * payoff, S_grid)

    print(f"\nParameters: S0={S0}, sigma={vol}, r={r}, T={T}")

    # --- 1. European Put ---
    K_put = 150.0
    put_payoff = lambda S: np.maximum(K_put - S, 0.0)
    price_put_num = price_option(put_payoff)

    d1 = (np.log(S0 / K_put) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    price_put_bs = K_put * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    print(f"\nEuropean Put (K={K_put}):")
    print(f"  Numerical: {price_put_num:.6f}")
    print(f"  Black-Scholes: {price_put_bs:.6f}")
    print(f"  Error: {abs(price_put_num - price_put_bs):.6f}")
    assert abs(price_put_num - price_put_bs) < 0.05, "Put price error too large"
    print("  PASS")

    # Probability OOM for put (S_T > K)
    prob_OOM_put = 1 - np.interp(K_put, S_grid, cdf_vals)
    print(f"  Prob(S_T > {K_put}): {prob_OOM_put:.6f}")

    # --- 2. Digital (binary) option ---
    K_dig = 100.0
    # Digital call: pays 1 if S_T > K
    digital_call_payoff = lambda S: (S > K_dig).astype(float)
    price_dig_num = price_option(digital_call_payoff)

    d2_dig = ((np.log(S0 / K_dig) + (r - 0.5 * vol**2) * T)
              / (vol * np.sqrt(T)))
    price_dig_bs = np.exp(-r * T) * norm.cdf(d2_dig)

    print(f"\nDigital Call (K={K_dig}):")
    print(f"  Numerical: {price_dig_num:.6f}")
    print(f"  Black-Scholes: {price_dig_bs:.6f}")
    print(f"  Error: {abs(price_dig_num - price_dig_bs):.6f}")
    assert abs(price_dig_num - price_dig_bs) < 0.01, "Digital price error too large"
    print("  PASS")

    # --- 3. Power option ---
    K_pow = 100.0
    p = 2.0  # power
    # Power call: pays (S-K)^p if S > K
    power_payoff = lambda S: np.where(S > K_pow, (S - K_pow)**p, 0.0)
    price_pow_num = price_option(power_payoff)
    print(f"\nPower Call (K={K_pow}, p={p:.0f}):")
    print(f"  Numerical: {price_pow_num:.4f}")
    assert price_pow_num > 0, "Power call price should be positive"
    print("  PASS (positive price)")

    # Put-call parity check for standard call/put
    K_par = 100.0
    call_pay = lambda S: np.maximum(S - K_par, 0.0)
    put_pay = lambda S: np.maximum(K_par - S, 0.0)
    price_call_par = price_option(call_pay)
    price_put_par = price_option(put_pay)
    pcp_lhs = price_call_par - price_put_par
    pcp_rhs = S0 - K_par * np.exp(-r * T)
    print(f"\nPut-Call Parity check (K={K_par}):")
    print(f"  C - P = {pcp_lhs:.6f}")
    print(f"  S0 - K*e^(-rT) = {pcp_rhs:.6f}")
    print(f"  Error: {abs(pcp_lhs - pcp_rhs):.6f}")
    assert abs(pcp_lhs - pcp_rhs) < 0.05, "Put-call parity violation"
    print("  PASS: put-call parity satisfied")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 3)

    S_plot_max = 300.0
    mask = S_grid <= S_plot_max
    S_p = S_grid[mask]
    pdf_p = pdf_vals[mask]

    axes[0].plot(S_p, put_payoff(S_p), 'k-', linewidth=2, label=f'Put payoff K={K_put}')
    axes[0].fill_between(S_p, pdf_p / pdf_p.max() * 50, 0, alpha=0.3, label='PDF (scaled)')
    axes[0].axvline(K_put, color='r', linestyle='--', label=f'K={K_put}')
    axes[0].set_title("European Put option", fontsize=11)
    axes[0].legend(fontsize=9)

    axes[1].plot(S_p, digital_call_payoff(S_p), 'k-', linewidth=2,
                 label=f'Digital call K={K_dig}')
    axes[1].fill_between(S_p, pdf_p / pdf_p.max(), 0, alpha=0.3, label='PDF (scaled)')
    axes[1].axvline(K_dig, color='r', linestyle='--')
    axes[1].set_title("Digital (binary) option", fontsize=11)
    axes[1].legend(fontsize=9)

    axes[2].plot(S_p, np.where(S_p > K_pow, (S_p - K_pow)**2, 0), 'k-', linewidth=2,
                 label=f'Power call (p=2)')
    axes[2].fill_between(S_p, pdf_p / pdf_p.max() * 200, 0, alpha=0.3, label='PDF (scaled)')
    axes[2].axvline(K_pow, color='r', linestyle='--')
    axes[2].set_title(f"Power call (p={p:.0f})", fontsize=11)
    axes[2].legend(fontsize=9)

    fig.suptitle("European option types: put, digital, power", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "european_options.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True

if __name__ == "__main__":
    run()
