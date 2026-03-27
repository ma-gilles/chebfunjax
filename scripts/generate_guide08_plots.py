"""Generate all plots for Guide Chapter 8: Chebfun Preferences."""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax.numpy as jnp
import numpy as np
import warnings
import chebfunjax as cj
from chebfunjax.pref import pref
from chebfunjax.plotting import chebfun_style, CHEBFUN_BLUE, CHEBFUN_RED, CHEBFUN_GREEN, CHEBFUN_ORANGE

chebfun_style()

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUTDIR, exist_ok=True)

plot_index = 0

def save(fig):
    global plot_index
    plot_index += 1
    path = os.path.join(OUTDIR, f"guide08_{plot_index:02d}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  guide08_{plot_index:02d}.png saved")

PI = float(jnp.pi)
tt = jnp.linspace(-1, 1, 600)

# ==========================================================================
# Plot 1: domain preference -- sin(19t), cos(20t) Lissajous on [0,2pi] -- Sec 8.2
# ==========================================================================
try:
    with pref.context(domain=(0.0, 2*PI)):
        f_sin = cj.chebfun(lambda t: jnp.sin(19*t))
        g_cos = cj.chebfun(lambda t: jnp.cos(20*t))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    tt_2pi = jnp.linspace(0, 2*PI, 1000)
    ax.plot(np.array(f_sin(tt_2pi)), np.array(g_cos(tt_2pi)),
            color=CHEBFUN_BLUE, linewidth=0.8)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide08_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 2: min(|x|, exp(x)/6) -- splitting example -- Section 8.3
# ==========================================================================
try:
    x = cj.chebfun(lambda t: t)
    f_abs = cj.abs(x)
    f_exp6 = cj.exp(x) * (1.0/6.0)
    # min(|x|, exp(x)/6) -- compute pointwise
    f_min = cj.chebfun(lambda t: jnp.minimum(jnp.abs(t), jnp.exp(t)/6.0))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tt, np.array(f_min(tt)), color=CHEBFUN_BLUE, linewidth=1.8)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide08_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 3: sqrt(x) on [0,1] -- Section 8.3
# (Would require splitting; show the resolved version)
# ==========================================================================
try:
    # Approximate sqrt on [eps, 1] to avoid singularity
    f_sqrt = cj.chebfun(lambda t: jnp.sqrt(jnp.maximum(t, 1e-15)), domain=(0.0, 1.0))

    fig, ax = plt.subplots(figsize=(6, 4))
    tt01 = jnp.linspace(0, 1, 600)
    ax.plot(tt01, np.array(f_sqrt(tt01)), color=CHEBFUN_BLUE, linewidth=1.8)
    ax.plot(tt01, np.sqrt(np.array(tt01)), color=CHEBFUN_RED, linewidth=1.0,
            linestyle='--', alpha=0.5, label='exact')
    ax.legend()
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide08_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 4: sin(x)*tanh(3*exp(x)*sin(15x)) -- globally complicated -- Sec 8.3
# ==========================================================================
try:
    ff = lambda t: jnp.sin(t) * jnp.tanh(3*jnp.exp(t)*jnp.sin(15*t))
    f_comp = cj.chebfun(ff)
    print(f"  Complicated function length: {len(f_comp)}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tt, np.array(f_comp(tt)), color=CHEBFUN_BLUE, linewidth=1.0)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide08_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 5: Same on [-3,3] -- very high degree -- Section 8.3
# ==========================================================================
try:
    ff3 = lambda t: jnp.sin(t) * jnp.tanh(3*jnp.exp(t)*jnp.sin(15*t))
    f3_comp = cj.chebfun(ff3, domain=(-3.0, 3.0))
    print(f"  Complicated function on [-3,3] length: {len(f3_comp)}")

    fig, ax = plt.subplots(figsize=(6, 4))
    tt3 = jnp.linspace(-3, 3, 1200)
    ax.plot(tt3, np.array(f3_comp(tt3)), color=CHEBFUN_BLUE, linewidth=0.8)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide08_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 6: sign(x) interpolant through 65 points -- Section 8.5
# ==========================================================================
try:
    f_sign65 = cj.chebfun(lambda x: jnp.sign(x), n=65)
    print(f"  sign(x), n=65: length = {len(f_sign65)}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tt, np.array(f_sign65(tt)), color=CHEBFUN_BLUE, linewidth=1.5)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide08_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 7: 1/(1+1e8*x^2) at high maxLength -- Section 8.5
# ==========================================================================
try:
    with pref.context(max_length=1000000):
        f_spike = cj.chebfun(lambda x: 1.0 / (1.0 + 1e8 * x**2))
    print(f"  1/(1+1e8*x^2) length: {len(f_spike)}, sum: {float(f_spike.sum()):.15e}")

    fig, ax = plt.subplots(figsize=(6, 4))
    # Zoom near origin to show the spike
    tt_zoom = jnp.linspace(-0.001, 0.001, 600)
    ax.plot(tt_zoom, np.array(f_spike(tt_zoom)), color=CHEBFUN_BLUE, linewidth=1.8)
    ax.set_title(f'$1/(1+10^8 x^2)$, length = {len(f_spike)}')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide08_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 8: Bump with exponent 2 (found) -- Section 8.6
# ==========================================================================
try:
    f_bump2 = cj.chebfun(lambda x: -x - x**2 + jnp.exp(-(30*(x - 0.47))**2))
    print(f"  Bump (exp 2) length: {len(f_bump2)}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tt, np.array(f_bump2(tt)), color=CHEBFUN_BLUE, linewidth=1.5)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide08_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 9: Bump with exponent 4 (missed) -- Section 8.6
# ==========================================================================
try:
    f_bump4 = cj.chebfun(lambda x: -x - x**2 + jnp.exp(-(30*(x - 0.47))**4))
    print(f"  Bump (exp 4) length: {len(f_bump4)}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tt, np.array(f_bump4(tt)), color=CHEBFUN_BLUE, linewidth=1.5)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide08_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 10: max(.85, sin(x+x^2)) - x/20 on [0,10] (missing spikes) -- Sec 8.6
# ==========================================================================
try:
    ff_spike = lambda t: jnp.maximum(0.85, jnp.sin(t + t**2)) - t/20.0
    f_spikes = cj.chebfun(ff_spike, domain=(0.0, 10.0))
    print(f"  Spike function length: {len(f_spikes)}")

    fig, ax = plt.subplots(figsize=(6, 4))
    tt010 = jnp.linspace(0, 10, 1200)
    ax.plot(tt010, np.array(f_spikes(tt010)), color=CHEBFUN_BLUE, linewidth=1.0)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide08_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 11: Effect of eps on length -- Section 8.8
# ==========================================================================
try:
    eps_values = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 2.22e-16]
    lengths = []
    for eps_val in eps_values:
        with pref.context(eps=eps_val):
            f_e = cj.chebfun(lambda x: jnp.exp(jnp.sin(10*x)))
            lengths.append(len(f_e))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(eps_values, lengths, 'o-', color=CHEBFUN_BLUE, linewidth=1.8, markersize=6)
    ax.set_xlabel('eps')
    ax.set_ylabel('Length')
    ax.set_title('Effect of eps on polynomial length')
    ax.invert_xaxis()
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide08_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 12: Fixed-length sign(x) at various n -- Section 8.5/8.9
# ==========================================================================
try:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for idx, n_pts in enumerate([17, 65, 257]):
        f_sgn = cj.chebfun(lambda x: jnp.sign(x), n=n_pts)
        xs = np.linspace(-1, 1, 1000)
        ys = np.array(f_sgn(jnp.array(xs)))
        axes[idx].plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.5)
        axes[idx].plot(xs, np.sign(xs), color=CHEBFUN_RED, linewidth=1.0,
                       linestyle='--', alpha=0.5)
        axes[idx].set_title(f'sign(x), n={n_pts}')
        axes[idx].set_ylim(-1.5, 1.5)
        axes[idx].grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
        axes[idx].spines['top'].set_visible(False); axes[idx].spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide08_{plot_index:02d}.png FAILED: {e}")

print(f"\nGuide 08: Generated {plot_index} plots.")
