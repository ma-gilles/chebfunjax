"""Exponential distribution exercises.

Demonstrates computations with the exponential distribution including
conditional probabilities, memorylessness, and reliability.
Translated from stats/ExponentialExercises.m.

Original: https://www.chebfun.org/examples/stats/ExponentialExercises.html
Authors: Jie Gao and Nick Trefethen, May 2013
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()



def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- 1. P[X<1 | X<2] for exponential with mean=2 (lambda=1/2) ---
    lam = 0.5
    # CDF: F(x) = 1 - exp(-lam*x)
    F = lambda x: 1 - np.exp(-lam * x)

    prob_cond = F(1) / F(2)
    print(f"P[X<1|X<2] = {prob_cond:.10f}")
    # Exact: (1 - e^{-1/2}) / (1 - e^{-1})
    exact = (1 - np.exp(-0.5)) / (1 - np.exp(-1.0))
    print(f"Exact:       {exact:.10f}")
    assert abs(prob_cond - exact) < 1e-12

    xs = np.linspace(0, 5, 300)
    pdf = lam * np.exp(-lam * xs)
    axes[0].plot(xs, pdf, 'k-', linewidth=2)
    mask = xs <= 1
    axes[0].fill_between(xs[mask], pdf[mask], alpha=0.5, color='blue',
                         label=f'P[X<1|X<2] = {prob_cond:.4f}')
    mask2 = (xs > 1) & (xs <= 2)
    axes[0].fill_between(xs[mask2], pdf[mask2], alpha=0.3, color='cyan',
                         label='X in [1,2]')
    axes[0].set_title('Exponential(λ=0.5): P[X<1|X<2]', fontsize=10)
    axes[0].set_xlabel('x'); axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Memorylessness: P[X>8|X>5] = P[X>3]
    p_memoryless = (1 - F(8)) / (1 - F(5))
    p_direct = 1 - F(3)
    print(f"\nMemorylessness:")
    print(f"P[X>8|X>5] = {p_memoryless:.10f}")
    print(f"P[X>3]     = {p_direct:.10f}")
    assert abs(p_memoryless - p_direct) < 1e-12

    # --- 2. Reliability of light bulb: Exp(lambda=0.2) ---
    lam2 = 0.2
    xs2 = np.linspace(0, 20, 500)
    pdf2 = lam2 * np.exp(-lam2 * xs2)
    Rel = np.exp(-lam2 * xs2)  # P[T > t]

    p_700 = np.exp(-lam2 * 7)  # 700 hours = 7 units
    p_900 = np.exp(-lam2 * 9)
    d_10pct = -np.log(0.1) / lam2  # 10% reliability threshold
    print(f"\nLight bulb reliability (lambda=0.2):")
    print(f"P[T>700h] = P[T>7] = {p_700:.6f}")
    print(f"P[T>900h] = P[T>9] = {p_900:.6f}")
    print(f"10% reliability at t = {d_10pct:.2f} (hundreds of hours)")

    axes[1].plot(xs2, Rel, 'k-', linewidth=2)
    axes[1].axhline(0.1, color='r', linestyle='--', label='10% reliability')
    axes[1].axvline(d_10pct, color='r', linestyle='--')
    axes[1].set_title('Light bulb reliability Rel(t) = P[T>t]', fontsize=10)
    axes[1].set_xlabel('t (hundreds of hours)')
    axes[1].set_ylabel('Reliability')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    # --- 3. Find lambda such that median = 1 (P[X<=1] = 0.5) ---
    # For exponential: CDF = 1 - exp(-lam*x), so P[X<=1] = 1 - exp(-lam)
    # P[X<=1] = 0.5 => lambda = log(2)
    lam_median1 = np.log(2)
    check = 1 - np.exp(-lam_median1 * 1)
    print(f"\nLambda with median=1: lambda = ln(2) = {lam_median1:.10f}")
    print(f"P[X<=1] at lambda=ln(2): {check:.10f}  (should be 0.5)")
    assert abs(check - 0.5) < 1e-12

    # Variance = 1/lambda^2
    var_val = 1 / lam_median1**2
    std_val = 1 / lam_median1
    print(f"Variance = 1/ln(2)^2 = {var_val:.6f}")
    print(f"Std dev  = 1/ln(2)   = {std_val:.6f}")

    xs3 = np.linspace(0, 8, 400)
    pdf3 = lam_median1 * np.exp(-lam_median1 * xs3)
    axes[2].plot(xs3, pdf3, 'k-', linewidth=2)
    axes[2].axvline(std_val, color='r', linewidth=2, linestyle='--',
                    label=f'std = {std_val:.3f}')
    axes[2].axvline(1/lam_median1, color='b', linewidth=2, linestyle=':',
                    label=f'mean = {1/lam_median1:.3f}')
    axes[2].set_title('Exp with median=1 (λ=ln2)', fontsize=10)
    axes[2].set_xlabel('x'); axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Exponential Distribution Exercises', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'exponential_exercises.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("exponential_exercises: done")
    return True


if __name__ == "__main__":
    run()
