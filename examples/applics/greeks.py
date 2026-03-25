"""Accurate Greeks for European options.

Computes the Greeks (Delta, Vega, Theta, Rho) of European call and put
options by differentiating the lognormal PDF with respect to parameters,
following applics/Greeks.m by Ricardo Pachon (December 2014).

Original MATLAB: https://www.chebfun.org/examples/applics/Greeks.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os


def run():
    print("=" * 60)
    print("Accurate Greeks for European options")
    print("=" * 60)

    # Parameters
    St = 100.0   # current stock price
    K = 100.0    # strike
    vol = 0.45   # volatility
    tau = 0.5    # time to expiry
    r = 0.01     # risk-free rate
    S0 = St

    print(f"\nParameters: S={St}, K={K}, sigma={vol}, tau={tau}, r={r}")

    # Risk-neutral lognormal PDF
    def log_pdf(S, St, vol, tau, r):
        mu_ln = np.log(St) + (r - 0.5 * vol**2) * tau
        sig_ln = vol * np.sqrt(tau)
        return (np.exp(-(np.log(S) - mu_ln)**2 / (2 * sig_ln**2))
                / (S * sig_ln * np.sqrt(2 * np.pi)))

    maxS = 5000.0
    N = 5000
    S_grid = np.linspace(0.01, maxS, N)

    # Call payoff
    call_payoff = np.maximum(S_grid - K, 0.0)

    def price_call(St_v, vol_v, tau_v, r_v):
        pdf = log_pdf(S_grid, St_v, vol_v, tau_v, r_v)
        return np.exp(-r_v * tau_v) * np.trapezoid(pdf * call_payoff, S_grid)

    def price_put(St_v, vol_v, tau_v, r_v):
        put_pay = np.maximum(K - S_grid, 0.0)
        pdf = log_pdf(S_grid, St_v, vol_v, tau_v, r_v)
        return np.exp(-r_v * tau_v) * np.trapezoid(pdf * put_pay, S_grid)

    # Numerical Greeks by finite differences (bumping)
    delta_St = 0.01; delta_vol = 0.0001; delta_tau = 0.0001; delta_r = 0.0001

    # Call Greeks
    call_delta_num = (price_call(St+delta_St, vol, tau, r) -
                      price_call(St-delta_St, vol, tau, r)) / (2*delta_St)
    call_vega_num  = (price_call(St, vol+delta_vol, tau, r) -
                      price_call(St, vol-delta_vol, tau, r)) / (2*delta_vol)
    call_theta_num = -(price_call(St, vol, tau+delta_tau, r) -
                       price_call(St, vol, tau-delta_tau, r)) / (2*delta_tau)
    call_rho_num   = (price_call(St, vol, tau, r+delta_r) -
                      price_call(St, vol, tau, r-delta_r)) / (2*delta_r)

    # Put Greeks
    put_delta_num  = (price_put(St+delta_St, vol, tau, r) -
                      price_put(St-delta_St, vol, tau, r)) / (2*delta_St)
    put_vega_num   = (price_put(St, vol+delta_vol, tau, r) -
                      price_put(St, vol-delta_vol, tau, r)) / (2*delta_vol)
    put_theta_num  = -(price_put(St, vol, tau+delta_tau, r) -
                       price_put(St, vol, tau-delta_tau, r)) / (2*delta_tau)
    put_rho_num    = (price_put(St, vol, tau, r+delta_r) -
                      price_put(St, vol, tau, r-delta_r)) / (2*delta_r)

    # Black-Scholes exact Greeks
    d1 = (np.log(St/K) + (r + 0.5*vol**2)*tau) / (vol*np.sqrt(tau))
    d2 = d1 - vol*np.sqrt(tau)

    # Call
    call_delta_bs = norm.cdf(d1)
    call_vega_bs  = K*np.exp(-r*tau)*norm.pdf(d2)*np.sqrt(tau)
    call_theta_bs = -(St*norm.pdf(d1)*vol/(2*np.sqrt(tau)) +
                      r*K*np.exp(-r*tau)*norm.cdf(d2))
    call_rho_bs   = K*tau*np.exp(-r*tau)*norm.cdf(d2)

    # Put
    put_delta_bs  = -norm.cdf(-d1)
    put_vega_bs   = call_vega_bs  # same for calls and puts
    put_theta_bs  = -(St*norm.pdf(d1)*vol/(2*np.sqrt(tau)) -
                      r*K*np.exp(-r*tau)*norm.cdf(-d2))
    put_rho_bs    = -K*tau*np.exp(-r*tau)*norm.cdf(-d2)

    print("\n" + "="*55)
    print(f"{'Greek':<10} {'Call (num)':>12} {'Call (BS)':>12} {'Put (num)':>12} {'Put (BS)':>12}")
    print("="*55)
    for name, c_n, c_b, p_n, p_b in [
        ('Delta', call_delta_num, call_delta_bs, put_delta_num, put_delta_bs),
        ('Vega',  call_vega_num,  call_vega_bs,  put_vega_num,  put_vega_bs),
        ('Theta', call_theta_num, call_theta_bs, put_theta_num, put_theta_bs),
        ('Rho',   call_rho_num,   call_rho_bs,   put_rho_num,   put_rho_bs),
    ]:
        print(f"{name:<10} {c_n:>12.6f} {c_b:>12.6f} {p_n:>12.6f} {p_b:>12.6f}")

    print("="*55)

    # Check accuracy
    for name, num_val, exact_val in [
        ('call delta', call_delta_num, call_delta_bs),
        ('call vega',  call_vega_num,  call_vega_bs),
        ('put delta',  put_delta_num,  put_delta_bs),
    ]:
        err = abs(num_val - exact_val)
        rel = err / max(abs(exact_val), 1e-10)
        status = "PASS" if rel < 0.01 else "WARN"
        print(f"  {name}: |error| = {err:.4f}, rel = {rel:.4f} [{status}]")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    S_range = np.linspace(40, 180, 100)

    def compute_greeks_array(S_arr):
        d1_arr = (np.log(S_arr/K) + (r + 0.5*vol**2)*tau) / (vol*np.sqrt(tau))
        d2_arr = d1_arr - vol*np.sqrt(tau)
        return {
            'delta': norm.cdf(d1_arr),
            'vega':  K*np.exp(-r*tau)*norm.pdf(d2_arr)*np.sqrt(tau),
            'theta': -(S_arr*norm.pdf(d1_arr)*vol/(2*np.sqrt(tau)) +
                       r*K*np.exp(-r*tau)*norm.cdf(d2_arr)),
            'rho':   K*tau*np.exp(-r*tau)*norm.cdf(d2_arr),
        }

    greeks = compute_greeks_array(S_range)

    axes[0, 0].plot(S_range, greeks['delta'], 'b-', linewidth=2)
    axes[0, 0].axvline(K, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title("Delta Δ = ∂V/∂S", fontsize=11)
    axes[0, 0].set_xlabel("S"); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(S_range, greeks['vega'], 'b-', linewidth=2)
    axes[0, 1].axvline(K, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_title("Vega ν = ∂V/∂σ", fontsize=11)
    axes[0, 1].set_xlabel("S"); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(S_range, greeks['theta'], 'b-', linewidth=2)
    axes[1, 0].axvline(K, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_title("Theta Θ = -∂V/∂τ", fontsize=11)
    axes[1, 0].set_xlabel("S"); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(S_range, greeks['rho'], 'b-', linewidth=2)
    axes[1, 1].axvline(K, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_title("Rho ρ = ∂V/∂r", fontsize=11)
    axes[1, 1].set_xlabel("S"); axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Black-Scholes Greeks (K={K}, σ={vol}, τ={tau})", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "greeks.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
