"""Exponential, logistic, and Gompertz growth.

Compares three population growth models, following applics/Gompertz.m
by Toby Driscoll (June 2015).

Models:
- Exponential: P' = r*P (unbounded)
- Logistic: P' = r*P*(K-P)/K (bounded by carrying capacity K)
- Gompertz: P' = r*P*log(K/P)/log(K/P0) (bounded, slower approach to K)

Original MATLAB: https://www.chebfun.org/examples/applics/Gompertz.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

def run():
    print("=" * 60)
    print("Exponential, logistic, and Gompertz growth")
    print("=" * 60)

    # Parameters
    P0 = 0.2     # initial population
    r = 0.5      # growth rate
    K = 6.0      # carrying capacity
    T = 25.0     # end time

    t_eval = np.linspace(0, T, 500)

    print(f"\nParameters: P0={P0}, r={r}, K={K}, T={T}")

    # --- Exponential growth: P' = r*P ---
    print("\n1. Exponential growth: P' = r*P")
    sol_exp = solve_ivp(lambda t, P: [r * P[0]], [0, T], [P0],
                        t_eval=t_eval, rtol=1e-9, atol=1e-11)
    P_exp = sol_exp.y[0]

    # Exact: P(t) = P0 * exp(r*t)
    t_check = np.array([5.0, 10.0, 15.0])
    for t in t_check:
        idx = np.argmin(np.abs(t_eval - t))
        t_actual = t_eval[idx]          # actual t_eval grid point (may not be exactly t)
        exact = P0 * np.exp(r * t_actual)
        err = abs(P_exp[idx] - exact) / exact
        print(f"  P({t:.0f}) = {P_exp[idx]:.4f}, exact = {exact:.4f}, rel_err = {err:.2e}")
        assert err < 1e-5, f"Exponential error too large at t={t}: {err}"

    print("  PASS: exponential growth matches exact solution")

    # --- Logistic growth: P' = r*P*(K-P)/K ---
    print("\n2. Logistic growth: P' = r*P*(K-P)/K")
    sol_log = solve_ivp(lambda t, P: [r * P[0] * (K - P[0]) / K], [0, T], [P0],
                        t_eval=t_eval, rtol=1e-9, atol=1e-11)
    P_log = sol_log.y[0]

    P_log_final = P_log[-1]
    print(f"  P({T}) = {P_log_final:.4f} (should approach K={K:.4f})")
    assert abs(P_log_final - K) < 0.05, f"Logistic didn't converge to K: {P_log_final}"
    print("  PASS: logistic growth converges to K")

    # --- Gompertz growth: P' = r*P*log(K/P)/log(K/P0) ---
    print("\n3. Gompertz growth: P' = r*P*log(K/P)/log(K/P0)")
    log_factor = np.log(K / P0)
    sol_gom = solve_ivp(
        lambda t, P: [r * P[0] * np.log(K / max(P[0], 1e-15)) / log_factor],
        [0, T], [P0], t_eval=t_eval, rtol=1e-9, atol=1e-11
    )
    P_gom = sol_gom.y[0]

    P_gom_final = P_gom[-1]
    print(f"  P({T}) = {P_gom_final:.4f} (should approach K={K:.4f})")
    # Gompertz converges more slowly than logistic — only check that P is close to K
    # and monotonically increasing (hasn't overshot)
    assert P_gom_final > 0.8 * K, f"Gompertz didn't grow towards K: {P_gom_final}"
    assert P_gom_final < K * 1.01, f"Gompertz overshot K: {P_gom_final}"
    print("  PASS: Gompertz growth approaches K from below")

    # --- Compare the three models ---
    print("\nComparison at t=10:")
    idx10 = np.argmin(np.abs(t_eval - 10.0))
    print(f"  Exponential: {P_exp[idx10]:.3f}")
    print(f"  Logistic:    {P_log[idx10]:.3f}")
    print(f"  Gompertz:    {P_gom[idx10]:.3f}")
    print(f"  Carrying capacity K = {K}")

    # Gompertz < Logistic < K at t=10 (before saturation)
    assert P_gom[idx10] < P_log[idx10], "Gompertz should grow slower than logistic"
    print("  PASS: Gompertz grows slower than logistic (as expected)")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    # Cap exponential for display
    P_exp_display = np.minimum(P_exp, 12.0)

    axes[0].plot(t_eval, P_exp_display, color='#0072BD', linestyle='-', linewidth=2, label='Exponential')
    axes[0].plot(t_eval, P_log, color='#D95319', linestyle='-', linewidth=2, label='Logistic')
    axes[0].plot(t_eval, P_gom, color='#77AC30', linestyle='-', linewidth=2, label='Gompertz')
    axes[0].axhline(K, color='k', linestyle='--', alpha=0.5, label=f'K={K}')
    axes[0].set_title("Population growth models", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim([0, 10])

    # Per-capita growth rates
    P_range = np.linspace(P0, K, 100)
    exp_rate = r * np.ones_like(P_range)
    log_rate = r * (K - P_range) / K
    gom_rate = r * np.log(K / P_range) / log_factor

    axes[1].plot(P_range, exp_rate, color='#0072BD', linestyle='-', linewidth=2, label='Exponential: r')
    axes[1].plot(P_range, log_rate, color='#D95319', linestyle='-', linewidth=2, label='Logistic: r(K-P)/K')
    axes[1].plot(P_range, gom_rate, color='#77AC30', linestyle='-', linewidth=2, label='Gompertz: r·log(K/P)/log(K/P₀)')
    axes[1].set_title("Per-capita growth rates", fontsize=11)
    axes[1].legend(fontsize=9)

    fig.suptitle("Growth models: exponential, logistic, Gompertz", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "gompertz.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True

if __name__ == "__main__":
    run()
