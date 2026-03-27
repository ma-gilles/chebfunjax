"""Uniform distribution exercises.

Demonstrates computations with the uniform distribution including
conditional probabilities and lottery wheel applications.
Translated from stats/UniformExercises.m.

Original: https://www.chebfun.org/examples/stats/UniformExercises.html
Author: Jie Gao, July 2013
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

    fig, axes = plt.subplots(1, 3)

    # --- 1. X ~ Uniform(1,2): P[X > z + mu_x] = 1/4 ---
    a_u, b_u = 1.0, 2.0
    f_pdf = 1.0 / (b_u - a_u)   # = 1.0
    mu_x = (a_u + b_u) / 2      # = 1.5
    # CDF: F(x) = (x - a)/(b - a)
    F = lambda x: (x - a_u) / (b_u - a_u)
    # P[X > a_thresh] = 1/4  =>  a_thresh = F^{-1}(3/4) = a_u + 3/4*(b_u-a_u) = 1.75
    a_thresh = a_u + 0.75 * (b_u - a_u)
    z_val = a_thresh - mu_x
    print(f"Uniform(1,2): mu_x = {mu_x}")
    print(f"a = z + mu_x = {a_thresh:.4f}  (exact: 1.75)")
    print(f"z = {z_val:.4f}  (exact: 0.25)")
    assert abs(a_thresh - 1.75) < 1e-12

    xs_u = np.linspace(0.8, 2.2, 400)
    f_u = np.where((xs_u >= a_u) & (xs_u <= b_u), f_pdf, 0)
    axes[0].plot(xs_u, f_u, 'k-', linewidth=2)
    mask = xs_u >= a_thresh
    axes[0].fill_between(xs_u, f_u, where=mask & (xs_u <= b_u),
                         alpha=0.5, color='purple',
                         label=f'P[X>z+μ]=1/4, z={z_val:.2f}')
    axes[0].set_title('Uniform(1,2): find z', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 2)

    # --- 2. Uniform with mean=1, variance=4/3 -> find a, b ---
    # mean = (a+b)/2 = 1, variance = (b-a)^2/12 = 4/3
    # => b - a = sqrt(12*4/3) = 4, and a + b = 2
    # => a = -1, b = 3
    mean_target = 1.0
    var_target = 4.0 / 3.0
    b_minus_a = np.sqrt(12 * var_target)
    a_new = mean_target - b_minus_a / 2
    b_new = mean_target + b_minus_a / 2
    print(f"\nUniform with mean=1, var=4/3: a={a_new}, b={b_new}")
    assert abs(a_new - (-1)) < 1e-10
    assert abs(b_new - 3) < 1e-10

    F2 = lambda x: (x - a_new) / (b_new - a_new)
    p_lt_0 = F2(0)
    print(f"P[X<0] = {p_lt_0:.4f}  (exact: 1/4)")
    assert abs(p_lt_0 - 0.25) < 1e-12

    xs_u2 = np.linspace(a_new - 0.2, b_new + 0.2, 400)
    f_u2 = np.where((xs_u2 >= a_new) & (xs_u2 <= b_new), 1.0/(b_new-a_new), 0)
    axes[1].plot(xs_u2, f_u2, 'k-', linewidth=2)
    mask2 = (xs_u2 >= a_new) & (xs_u2 <= 0)
    axes[1].fill_between(xs_u2, f_u2, where=mask2,
                         alpha=0.5, color='darkorange',
                         label=f'P[X<0] = {p_lt_0:.3f}')
    axes[1].set_title('Uniform(-1,3): P[X<0]', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0, 0.4)

    # --- 3. Lottery wheel ---
    # Sectors (degrees): red=5, cyan=15, yellow=35, green=50, white=65, blue=80, black=110
    sectors = [
        ('Red', 0, 5, [1, 0, 0]),
        ('Cyan', 5, 20, [0, 0.8, 0.8]),
        ('Yellow', 20, 55, [1, 1, 0]),
        ('Green', 55, 105, [0, 0.7, 0]),
        ('White', 105, 170, [0.9, 0.9, 0.9]),
        ('Blue', 170, 250, [0, 0, 0.8]),
        ('Black', 250, 360, [0.2, 0.2, 0.2]),
    ]

    # Q1: P[red or cyan] = 20/360
    p_rc = 20 / 360
    print(f"\nLottery wheel:")
    print(f"P[red or cyan] = {p_rc:.6f}  (= {20}/360 = {p_rc:.4f})")

    # Q2: P[neither black nor yellow | not blue]
    p_not_blue = 1 - 80/360
    p_neither_given_notblue = (1 - (35 + 110 + 80)/360) / p_not_blue
    print(f"P[not black, not yellow | not blue] = {p_neither_given_notblue:.6f}")

    # Pie chart of lottery wheel
    angles = [s[2] - s[1] for s in sectors]
    colors = [s[3] for s in sectors]
    labels = [s[0] for s in sectors]
    axes[2].pie(angles, labels=labels, colors=colors,
                autopct='%1.0f°', startangle=90)
    axes[2].set_title(f'Lottery wheel\nP[red|cyan]={p_rc:.3f}', fontsize=10)

    fig.suptitle('Uniform Distribution Exercises', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'uniform_exercises.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("uniform_exercises: done")
    return True

if __name__ == "__main__":
    run()
