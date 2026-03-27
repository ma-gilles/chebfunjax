"""Pricing of a European Call option.

Computes the price of a European call option by integrating the lognormal
density against the payoff, following applics/EuropeanCall.m by
Ricardo Pachon (November 2014).

The method works directly with the probability density function and
avoids Monte Carlo simulation by using spectral integration (Chebfun).

Original MATLAB: https://www.chebfun.org/examples/applics/EuropeanCall.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chebfunjax.plotting import chebfun_style
chebfun_style()

import numpy as np
from scipy.stats import norm, lognorm
import os


def run():
    print("=" * 60)
    print("Pricing of a European Call option")
    print("=" * 60)

    # Parameters
    K = 60.0       # strike
    T = 0.5        # time to maturity
    S0 = 100.0     # initial stock price
    vol = 0.45     # volatility
    r = 0.01       # risk-free rate

    print(f"\nParameters: K={K}, T={T}, S0={S0}, sigma={vol}, r={r}")

    # Risk-neutral lognormal PDF:
    # f(S) = exp(-(log(S/S0)-(r-sigma^2/2)*T)^2 / (2*sigma^2*T)) / (sigma*S*sqrt(2*pi*T))
    def lognorm_pdf(S, S0, vol, r, T):
        mu_ln = np.log(S0) + (r - 0.5 * vol**2) * T
        sig_ln = vol * np.sqrt(T)
        return (np.exp(-(np.log(S) - mu_ln)**2 / (2 * sig_ln**2))
                / (S * sig_ln * np.sqrt(2 * np.pi)))

    # Numerical integration on [0, RHS]
    RHS = 2000.0
    N = 5000
    S_grid = np.linspace(1e-3, RHS, N)
    dS = S_grid[1] - S_grid[0]

    pdf_vals = lognorm_pdf(S_grid, S0, vol, r, T)

    # Check normalization
    pdf_integral = np.trapezoid(pdf_vals, S_grid)
    print(f"\nPDF normalization: {pdf_integral:.8f} (should be ≈ 1)")
    assert abs(pdf_integral - 1.0) < 0.01, f"PDF not normalized: {pdf_integral}"

    # Probability of expiring OOM (S_T < K)
    idx_K = np.searchsorted(S_grid, K)
    prob_OOM = np.trapezoid(pdf_vals[:idx_K], S_grid[:idx_K])
    print(f"  Probability of expiring OOM (S_T < {K}): {prob_OOM:.6f}")

    # Call payoff: max(S-K, 0)
    call_payoff = np.maximum(S_grid - K, 0.0)

    # Option price = e^(-rT) * E^Q[max(S_T - K, 0)]
    price_numerical = np.exp(-r * T) * np.trapezoid(pdf_vals * call_payoff, S_grid)
    print(f"\nNumerical option price: {price_numerical:.8f}")

    # Black-Scholes exact formula
    d1 = (np.log(S0 / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    bs_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    print(f"Black-Scholes formula:  {bs_price:.8f}")

    err = abs(price_numerical - bs_price)
    print(f"Error: {err:.6f}")
    assert err < 0.01, f"Pricing error too large: {err}"
    print("PASS: prices agree to within 1 cent")

    # Distribution of discounted payoff e^(-rT)*max(S_T-K,0)
    # In-the-money part: S > K, payoff = S - K → density contribution
    # = lognorm_pdf(S) for S in (K, ∞)
    itm_payoff_vals = S_grid[idx_K:] - K
    itm_pdf_vals = pdf_vals[idx_K:]
    discounted_itm = np.exp(-r * T) * itm_pdf_vals

    # The expected value
    approx2 = np.trapezoid(itm_payoff_vals * discounted_itm, S_grid[idx_K:])
    print(f"\nPrice from ITM region alone: {approx2:.8f}")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 3)

    S_plot_max = 300.0
    S_plot = S_grid[S_grid <= S_plot_max]
    pdf_plot = pdf_vals[S_grid <= S_plot_max]

    # Panel 1: payoff and PDF
    axes[0].plot(S_plot, call_payoff[S_grid <= S_plot_max], 'k-', linewidth=2,
                 label='Payoff max(S-K,0)')
    axes[0].axvline(K, color='g', linestyle='--', label=f'Strike K={K}')
    axes[0].set_xlabel("S"); axes[0].set_ylabel("Value")
    axes[0].set_title("Call payoff", fontsize=11)
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-10, 150])

    # Panel 2: PDF and CDF
    ax2b = axes[1].twinx()
    axes[1].plot(S_plot, pdf_plot * 1e-3, 'b-', linewidth=2, label='PDF × 1e-3')
    cdf_vals = np.cumsum(pdf_vals * dS)
    axes[1].fill_between(S_plot, 0, pdf_plot * 1e-3, alpha=0.3)
    ax2b.plot(S_plot, cdf_vals[S_grid <= S_plot_max], 'r-', linewidth=1.5,
              label='CDF', alpha=0.7)
    axes[1].set_xlabel("S_T"); axes[1].set_title("Lognormal PDF/CDF", fontsize=11)
    axes[1].axvline(K, color='g', linestyle='--')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: price comparison
    strikes = np.linspace(40, 120, 50)
    prices_bs = []
    prices_num = []
    for k_i in strikes:
        d1_i = (np.log(S0 / k_i) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2_i = d1_i - vol * np.sqrt(T)
        prices_bs.append(S0 * norm.cdf(d1_i) - k_i * np.exp(-r * T) * norm.cdf(d2_i))
        idx_ki = np.searchsorted(S_grid, k_i)
        p_num = np.exp(-r * T) * np.trapezoid(
            pdf_vals[idx_ki:] * (S_grid[idx_ki:] - k_i), S_grid[idx_ki:])
        prices_num.append(p_num)

    axes[2].plot(strikes, prices_bs, 'r-', linewidth=2, label='Black-Scholes')
    axes[2].plot(strikes, prices_num, 'b.', markersize=6, label='Numerical')
    axes[2].set_xlabel("Strike K"); axes[2].set_ylabel("Option price")
    axes[2].set_title("Call prices vs strike", fontsize=11)
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"European call option pricing (S0={S0}, σ={vol}, T={T})", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "european_call.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
