"""Generate all plots for Guide Chapter 4: Chebfun and Approximation Theory.

Faithful translation of the MATLAB Chebfun Guide Chapter 4 plots.
"""

import matplotlib
matplotlib.use('Agg')

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.polynomials import chebpoly
from chebfunjax.utils.quadrature import chebpts

chebfun_style()

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT_DIR, exist_ok=True)

plot_idx = 0

# --------------------------------------------------------------------------
# Plot 1: T_2, T_3, T_15, T_50 -- Section 4.1
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for ax, N in zip(axes.flat, [2, 3, 15, 50]):
        c = chebpoly(N)
        T_N = cj.Chebfun.from_coeffs(jnp.array(c))
        xx = jnp.linspace(-1, 1, 500)
        ax.plot(xx, T_N(xx), linewidth=1.5)
        ax.set_ylim([-1.5, 1.5])
        ax.set_title(f'$T_{{{N}}}$', fontsize=12)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide04_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide04_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide04_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 2: sign(x) interpolated in 10 and 20 Chebyshev points -- Section 4.3
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, N in zip(axes, [10, 20]):
        f = cj.chebfun(lambda x: jnp.sign(x), n=N)
        pts = chebpts(N)
        vals = jnp.sign(pts)
        xx = jnp.linspace(-1, 1, 1000)
        ax.plot(xx, f(xx), '.-', markersize=4, linewidth=1.2)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'sign(x), N={N}')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide04_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide04_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide04_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 3: sign(x) zoomed -- Section 4.3
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, N, xlim_right in zip(axes, [10, 20], [0.8, 0.4]):
        f = cj.chebfun(lambda x: jnp.sign(x), n=N)
        xx = jnp.linspace(0, xlim_right, 500)
        ax.plot(xx, f(xx), '.-', markersize=4, linewidth=1.2)
        ax.set_xlim([0, xlim_right])
        ax.set_ylim([0.5, 1.5])
        ax.grid(True, alpha=0.3)
        ax.set_title(f'sign(x) zoomed, N={N}')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide04_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide04_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide04_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 4: sign(x) N=100 and N=1000 zoomed -- Section 4.3
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, N, xlim_right in zip(axes, [100, 1000], [0.08, 0.008]):
        f = cj.chebfun(lambda x: jnp.sign(x), n=N)
        xx = jnp.linspace(0, xlim_right, 500)
        ax.plot(xx, f(xx), '.-', markersize=4, linewidth=1.2)
        ax.set_xlim([0, xlim_right])
        ax.set_ylim([0.5, 1.5])
        ax.grid(True, alpha=0.3)
        ax.set_title(f'sign(x) zoomed, N={N}')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide04_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide04_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide04_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 5: |x| interpolated in 10 and 20 Chebyshev points -- Section 4.4
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, N in zip(axes, [10, 20]):
        f = cj.chebfun(lambda x: jnp.abs(x), n=N)
        pts = chebpts(N)
        xx = jnp.linspace(-1, 1, 1000)
        ax.plot(xx, f(xx), '.-', markersize=4, linewidth=1.2)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.set_title(f'|x|, N={N}')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide04_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide04_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide04_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 6: |x| interpolated in 100 and 1000 Chebyshev points -- Section 4.4
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, N in zip(axes, [100, 1000]):
        f = cj.chebfun(lambda x: jnp.abs(x), n=N)
        xx = jnp.linspace(-1, 1, 1000)
        ax.plot(xx, f(xx), linewidth=1.2)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.set_title(f'|x|, N={N}')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide04_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide04_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide04_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 7: |x|^5 convergence -- loglog and semilogy -- Section 4.4
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    s = '|x|^5'
    exact = cj.chebfun(lambda x: jnp.abs(x)**5)
    NN = np.arange(1, 101)
    e = []
    for N in NN:
        fN = cj.chebfun(lambda x: jnp.abs(x)**5, n=int(N))
        err = float((fN - exact).norm(jnp.inf))
        e.append(err)
    e = np.array(e)
    e[e == 0] = 1e-16  # avoid log(0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].loglog(NN, e, linewidth=1.2)
    axes[0].loglog(NN, NN.astype(float)**(-5), '--r', linewidth=1.2)
    axes[0].set_ylim([1e-10, 10])
    axes[0].set_title('loglog scale')
    axes[0].grid(True, alpha=0.3)
    axes[0].text(6, 4e-7, r'$N^{-5}$', color='r', fontsize=14)

    axes[1].semilogy(NN, e, linewidth=1.2)
    axes[1].semilogy(NN, NN.astype(float)**(-5), '--r', linewidth=1.2)
    axes[1].set_ylim([1e-10, 10])
    axes[1].set_title('semilog scale')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide04_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide04_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide04_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 8: 1/(1+25x^2) convergence -- loglog and semilogy -- Section 4.4
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    exact = cj.chebfun(lambda x: 1.0 / (1 + 25 * x**2))
    NN = np.arange(1, 101)
    e = []
    for N in NN:
        fN = cj.chebfun(lambda x: 1.0 / (1 + 25 * x**2), n=int(N))
        err = float((fN - exact).norm(jnp.inf))
        e.append(err)
    e = np.array(e)
    e[e == 0] = 1e-16

    c = 1.0/5 + np.sqrt(1 + 1.0/25)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].loglog(NN, e, linewidth=1.2)
    axes[0].loglog(NN, c**(-NN.astype(float)), '--r', linewidth=1.2)
    axes[0].set_ylim([1e-10, 10])
    axes[0].set_title('loglog scale')
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(NN, e, linewidth=1.2)
    axes[1].semilogy(NN, c**(-NN.astype(float)), '--r', linewidth=1.2)
    axes[1].set_ylim([1e-10, 10])
    axes[1].set_title('semilog scale')
    axes[1].grid(True, alpha=0.3)
    axes[1].text(45, 1e-3, r'$C^{-N}$', color='r', fontsize=14)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide04_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide04_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide04_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 9: sinefun1 and sinefun2 gallery -- Section 4.5
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # sinefun1: sin(6*pi*sin(2*pi*x))
    f1 = cj.chebfun(lambda x: jnp.sin(6 * jnp.pi * jnp.sin(2 * jnp.pi * x)))
    xx = jnp.linspace(-1, 1, 1000)
    axes[0].plot(xx, f1(xx), linewidth=1.2)
    axes[0].set_ylim([0, 3.5])
    axes[0].set_title(f'sinefun1, length = {len(f1)}')
    axes[0].grid(True, alpha=0.3)

    # sinefun2: sin(6*pi*sin(2*pi*sin(2*pi*x)))
    f2 = cj.chebfun(lambda x: jnp.sin(6 * jnp.pi * jnp.sin(2 * jnp.pi * jnp.sin(2 * jnp.pi * x))))
    axes[1].plot(xx, f2(xx), linewidth=1.2)
    axes[1].set_ylim([0, 3.5])
    axes[1].set_title(f'sinefun2, length = {len(f2)}')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide04_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide04_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide04_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 10: best approximation of sqrt(|x-3|) on [0,4] -- Section 4.6
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    from chebfunjax.utils.minimax import minimax

    f = cj.chebfun(lambda x: jnp.sqrt(jnp.abs(x - 3.0)), domain=[0, 4])
    result = minimax(f, 20)
    p = cj.Chebfun.from_coeffs(jnp.array(result.coeffs), domain=[0.0, 4.0])

    xx = jnp.linspace(0, 4, 1000)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xx, f(xx), 'b', linewidth=1.5, label='f(x)')
    ax.plot(xx, p(xx), 'r', linewidth=1.5, label='minimax p(x)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(r'$\sqrt{|x-3|}$ and its degree 20 best approximation')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide04_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide04_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide04_{plot_idx:02d}.png FAILED: {e}")
    plot_idx += 1  # skip 11 and 12 too since they depend on minimax
    plot_idx += 1

# --------------------------------------------------------------------------
# Plot 11: equioscillation of best approx error -- Section 4.6
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    err_val = float(result.err)
    xx = jnp.linspace(0, 4, 1000)
    err_curve = f(xx) - p(xx)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xx, err_curve, 'm', linewidth=1.2)
    ax.axhline(err_val, color='k', linestyle='--', linewidth=0.8)
    ax.axhline(-err_val, color='k', linestyle='--', linewidth=0.8)
    ax.set_ylim([-3 * abs(err_val), 3 * abs(err_val)])
    ax.set_title('Error curve f - p (minimax)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide04_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide04_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide04_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 12: minimax vs Chebyshev interpolant error -- Section 4.6
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    pinterp = cj.chebfun(lambda x: jnp.sqrt(jnp.abs(x - 3.0)), domain=[0, 4], n=21)
    xx = jnp.linspace(0, 4, 1000)
    err_interp = f(xx) - pinterp(xx)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xx, f(xx) - p(xx), 'm', linewidth=1.2, label='f - p (minimax)')
    ax.axhline(err_val, color='k', linestyle='--', linewidth=0.8)
    ax.axhline(-err_val, color='k', linestyle='--', linewidth=0.8)
    ax.plot(xx, err_interp, 'b', linewidth=1.2, label='f - p (Chebyshev interp)')
    ax.set_ylim([-3 * abs(err_val), 3 * abs(err_val)])
    ax.set_title('Minimax vs Chebyshev interpolant error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide04_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide04_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide04_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 13: CF approximation of exp(x) -- Section 4.6
# --------------------------------------------------------------------------
# CF approximation not available in chebfunjax, skip
plot_idx += 1
print(f"guide04_{plot_idx:02d}.png SKIPPED (CF approximation not available)")

# --------------------------------------------------------------------------
# Plot 14: CF approximation of |x-0.3| -- Section 4.6
# --------------------------------------------------------------------------
plot_idx += 1
print(f"guide04_{plot_idx:02d}.png SKIPPED (CF approximation not available)")

# --------------------------------------------------------------------------
# Plot 15: equispaced interpolation of tanh(10x) with 10 points -- Section 4.7
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    x = cj.chebfun(lambda x: x)
    f = cj.chebfun(lambda x: jnp.tanh(10 * x))
    s = jnp.linspace(-1, 1, 10)
    p = cj.Chebfun.interp1(s, jnp.tanh(10 * s))

    xx = jnp.linspace(-1, 1, 500)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xx, f(xx), 'b', linewidth=1.5, label='tanh(10x)')
    ax.plot(xx, p(xx), 'r', linewidth=1.2, label='equispaced interp')
    ax.plot(s, p(s), '.r', markersize=8)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Equispaced interpolation, N=10')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide04_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide04_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide04_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 16: equispaced interpolation of tanh(10x) with 20 points -- Section 4.7
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    s = jnp.linspace(-1, 1, 20)
    p = cj.Chebfun.interp1(s, jnp.tanh(10 * s))

    xx = jnp.linspace(-1, 1, 500)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xx, f(xx), 'b', linewidth=1.5, label='tanh(10x)')
    ax.plot(xx, p(xx), 'r', linewidth=1.2, label='equispaced interp')
    ax.plot(s, p(s), '.r', markersize=8)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Equispaced interpolation, N=20')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide04_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide04_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide04_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 17: Lebesgue function for 20 equispaced points -- Section 4.7
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    from chebfunjax.utils.lebesgue import lebesgue_function
    s = jnp.linspace(-1, 1, 20)
    L_fun, L_const = lebesgue_function(s)

    xx = jnp.linspace(-1, 1, 1000)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(xx, L_fun(xx), linewidth=1.2)
    ax.set_title(f'Lebesgue function, 20 equispaced points')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide04_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide04_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide04_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 18: Lebesgue function for 40 equispaced points -- Section 4.7
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    s = jnp.linspace(-1, 1, 40)
    L_fun, L_const = lebesgue_function(s)

    xx = jnp.linspace(-1, 1, 1000)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(xx, L_fun(xx), linewidth=1.2)
    ax.set_title(f'Lebesgue function, 40 equispaced points')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide04_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide04_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide04_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 19: tanh(pi*x/2) + x/20 on [-10,10] -- Section 4.8
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    f = cj.chebfun(lambda x: jnp.tanh(jnp.pi * x / 2) + x / 20, domain=[-10, 10])

    fig, ax = plt.subplots(figsize=(8, 5))
    xx = jnp.linspace(-10, 10, 1000)
    ax.plot(xx, f(xx), linewidth=1.5)
    ax.set_title(f'tanh(pi*x/2) + x/20, length = {len(f)}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide04_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide04_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide04_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 20-22: rational approximation errors -- Section 4.8
# Chebpade, ratinterp, CF comparisons
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    from chebfunjax.utils.ratapprox import chebpade, ratinterp

    f = cj.chebfun(lambda x: jnp.tanh(jnp.pi * x / 2) + x / 20, domain=[-10, 10])
    xx = jnp.linspace(-10, 10, 2000)
    f_vals = f(xx)

    # Try chebpade
    try:
        cp_result = chebpade(f, 40, 4)
        r_handle = cp_result[0]
        r_vals = jnp.array([r_handle(float(xi)) for xi in xx])
        err_cp = f_vals - r_vals

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xx, err_cp, 'r', linewidth=1.0)
        ax.set_title('Chebyshev-Pade approximation error')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'guide04_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"guide04_{plot_idx:02d}.png saved")
    except Exception as e2:
        print(f"guide04_{plot_idx:02d}.png FAILED (chebpade): {e2}")

    # ratinterp
    plot_idx += 1
    try:
        ri_result = ratinterp(f, 40, 4)
        r_handle = ri_result[0]
        r_vals = jnp.array([r_handle(float(xi)) for xi in xx])
        err_ri = f_vals - r_vals

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xx, err_ri, 'm', linewidth=1.0)
        ax.set_title('Rational interpolation error (ratinterp)')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'guide04_{plot_idx:02d}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"guide04_{plot_idx:02d}.png saved")
    except Exception as e2:
        print(f"guide04_{plot_idx:02d}.png FAILED (ratinterp): {e2}")

    # CF not available
    plot_idx += 1
    print(f"guide04_{plot_idx:02d}.png SKIPPED (CF not available)")

except Exception as e:
    print(f"guide04_{plot_idx:02d}.png FAILED: {e}")

print(f"\nGuide 04 plot generation complete. Generated {plot_idx} plots.")
