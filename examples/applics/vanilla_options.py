"""Exploring Vanilla Options.

Uses the Black-Scholes formula to explore how European call and put
prices depend on parameters (S, K, T, sigma, r), following
applics/VanillaOptions.m by Ricardo Pachon (December 2014).

Original MATLAB: https://www.chebfun.org/examples/applics/VanillaOptions.html
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
    print("Exploring Vanilla Options")
    print("=" * 60)

    # Black-Scholes formula for call (W=+1) and put (W=-1)
    def vanilla(S, K, T, vol, r, W):
        """Black-Scholes vanilla option price."""
        if np.isscalar(T) and T <= 0:
            return W * np.maximum(W * (S - K), 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
            d2 = d1 - vol * np.sqrt(T)
            return W * (S * norm.cdf(W * d1) - K * np.exp(-r * T) * norm.cdf(W * d2))

    # Payoff at T=0
    def payoff(S, K, W):
        return W * np.maximum(W * (S - K), 0.0)

    # Fixed parameters
    K = 100.0
    r = 0.01
    T_base = 0.5
    vol_base = 0.4
    S_base = 100.0

    S_range = np.linspace(40, 200, 200)

    print(f"\nBase parameters: K={K}, T={T_base}, sigma={vol_base}, r={r}")

    # --- 1. Impact of maturity T ---
    print("\n1. Impact of varying maturity T")
    T_vals = [0.1, 0.3, 0.5, 1.0, 2.0]
    for T in T_vals:
        C = vanilla(S_base, K, T, vol_base, r, 1)
        print(f"  T={T:.1f}: Call(S={S_base}) = {C:.4f}")

    # --- 2. Impact of volatility ---
    print("\n2. Impact of varying volatility sigma")
    vol_vals = [0.1, 0.2, 0.4, 0.6, 0.8]
    for v in vol_vals:
        C = vanilla(S_base, K, T_base, v, r, 1)
        print(f"  sigma={v:.1f}: Call(S={S_base}) = {C:.4f}")

    # Higher vol always increases option price
    C_low = vanilla(S_base, K, T_base, 0.2, r, 1)
    C_high = vanilla(S_base, K, T_base, 0.6, r, 1)
    assert C_high > C_low, "Higher vol should give higher call price"
    print(f"  PASS: C(sigma=0.6) > C(sigma=0.2)")

    # --- 3. Put-call parity ---
    print("\n3. Put-call parity: C - P = S*e^(-q*T) - K*e^(-r*T)")
    S_test_arr = [70, 100, 130]
    for S_test in S_test_arr:
        C = vanilla(S_test, K, T_base, vol_base, r, 1)
        P = vanilla(S_test, K, T_base, vol_base, r, -1)
        lhs = C - P
        rhs = S_test - K * np.exp(-r * T_base)
        err = abs(lhs - rhs)
        print(f"  S={S_test}: C-P={lhs:.4f}, S-K*e^(-rT)={rhs:.4f}, err={err:.6f}")
        assert err < 1e-8, f"Put-call parity violated: {err}"

    print("  PASS: Put-call parity holds")

    # --- 4. Moneyness: ITM, ATM, OTM ---
    print("\n4. Moneyness analysis")
    T_fixed = 0.5; vol_fixed = 0.4
    for S_test in [70, 100, 130]:
        C = vanilla(S_test, K, T_fixed, vol_fixed, r, 1)
        P = vanilla(S_test, K, T_fixed, vol_fixed, r, -1)
        intrinsic_call = max(S_test - K, 0)
        time_value = C - intrinsic_call
        print(f"  S={S_test}: Call={C:.3f}, Put={P:.3f}, "
              f"intrinsic={intrinsic_call:.1f}, time_val={time_value:.3f}")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(2, 2)

    # Panel 1: Call prices for different maturities
    colors = plt.cm.Blues(np.linspace(0.4, 0.95, len(T_vals)))
    for T_v, col in zip(T_vals, colors):
        C_arr = vanilla(S_range, K, T_v, vol_base, r, 1)
        axes[0, 0].plot(S_range, C_arr, color=col, linewidth=1.8, label=f'T={T_v}')
    axes[0, 0].plot(S_range, payoff(S_range, K, 1), 'k--', linewidth=1.5, label='Payoff (T=0)')
    axes[0, 0].axvline(K, color='gray', linestyle=':', alpha=0.5)
    axes[0, 0].set_title("Call price vs S (varying T)", fontsize=11)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylim([-2, 80])

    # Panel 2: Put prices for different maturities
    for T_v, col in zip(T_vals, colors):
        P_arr = vanilla(S_range, K, T_v, vol_base, r, -1)
        axes[0, 1].plot(S_range, P_arr, color=col, linewidth=1.8, label=f'T={T_v}')
    axes[0, 1].plot(S_range, payoff(S_range, K, -1), 'k--', linewidth=1.5, label='Payoff')
    axes[0, 1].axvline(K, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_title("Put price vs S (varying T)", fontsize=11)
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_ylim([-2, 80])

    # Panel 3: Sensitivity to volatility
    colors_v = plt.cm.Reds(np.linspace(0.4, 0.95, len(vol_vals)))
    for v, col in zip(vol_vals, colors_v):
        C_arr = vanilla(S_range, K, T_base, v, r, 1)
        axes[1, 0].plot(S_range, C_arr, color=col, linewidth=1.8, label=f'σ={v:.1f}')
    axes[1, 0].set_title(f"Call price vs S (varying σ, T={T_base})", fontsize=11)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_ylim([-2, 80])

    # Panel 4: Call vs Put comparison ATM
    S_atm = np.linspace(0.5, 3.0, 100) * K
    C_atm = vanilla(S_atm, K, T_base, vol_base, r, 1)
    P_atm = vanilla(S_atm, K, T_base, vol_base, r, -1)
    axes[1, 1].plot(S_atm, C_atm, 'b-', linewidth=2, label='Call')
    axes[1, 1].plot(S_atm, P_atm, 'r-', linewidth=2, label='Put')
    axes[1, 1].plot(S_atm, payoff(S_atm, K, 1), 'b--', linewidth=1, alpha=0.5)
    axes[1, 1].plot(S_atm, payoff(S_atm, K, -1), 'r--', linewidth=1, alpha=0.5)
    axes[1, 1].axvline(K, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_title("Call & Put vs payoff", fontsize=11)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].set_ylim([-5, 80])

    fig.suptitle(f"Vanilla options: K={K}, base σ={vol_base}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "vanilla_options.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True

if __name__ == "__main__":
    run()
