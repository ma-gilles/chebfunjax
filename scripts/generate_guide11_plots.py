"""Generate plots for Guide Chapter 11: Periodic Chebfuns."""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax.numpy as jnp
import numpy as np
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, CHEBFUN_BLUE, CHEBFUN_RED, CHEBFUN_GREEN
chebfun_style()

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUTDIR, exist_ok=True)

plot_index = 0

def save(fig, name_hint=""):
    global plot_index
    plot_index += 1
    path = os.path.join(OUTDIR, f"guide11_{plot_index:02d}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# --- Plot 1: Trigtech function: tanh(3*sin(pi*t)) - sin(pi*(t+0.5)) ---
try:
    from chebfunjax.tech.trigtech import Trigtech

    f = Trigtech.from_function(
        lambda t: jnp.tanh(3 * jnp.sin(jnp.pi * t)) - jnp.sin(jnp.pi * (t + 0.5)),
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.linspace(-1, 1, 500)
    ys = np.array(f(jnp.array(xs)))
    ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.set_title(f"$\\tanh(3\\sin(\\pi t)) - \\sin(\\pi(t+0.5))$  (n={f.n})")
    ax.set_xlabel("t")
    fig.tight_layout()
    save(fig)
except Exception as e:
    print(f"  Skipping plot 1 (Trigtech basic): {e}")

# --- Plot 2: Trigtech differentiation: sin(pi*x) and its derivative pi*cos(pi*x) ---
try:
    from chebfunjax.tech.trigtech import Trigtech

    f = Trigtech.from_function(lambda t: jnp.sin(jnp.pi * t))
    df = f.diff()

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.linspace(-1, 1, 500)
    ys_f = np.array(f(jnp.array(xs)))
    ys_df = np.array(df(jnp.array(xs)))
    ax.plot(xs, ys_f, color=CHEBFUN_BLUE, linewidth=1.8, label="$\\sin(\\pi t)$")
    ax.plot(xs, ys_df, color=CHEBFUN_RED, linewidth=1.8, label="$\\pi\\cos(\\pi t)$ (derivative)")
    ax.set_title("Trigonometric differentiation is exact")
    ax.set_xlabel("t")
    ax.legend()
    fig.tight_layout()
    save(fig)
except Exception as e:
    print(f"  Skipping plot 2 (Trigtech diff): {e}")

# --- Plot 3: Chebyshev vs trigonometric length comparison ---
try:
    from chebfunjax.tech.trigtech import Trigtech

    ff = lambda t: jnp.cos(11 * jnp.sin(3 * (t - 1.0 / jnp.pi)))

    f_cheb = cj.chebfun(ff, domain=(-float(jnp.pi), float(jnp.pi)))
    f_trig = Trigtech.from_function(
        lambda t: ff(jnp.pi * t),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    xs_phys = np.linspace(-float(jnp.pi), float(jnp.pi), 500)
    ys_cheb = np.array(f_cheb(jnp.array(xs_phys)))
    axes[0].plot(xs_phys, ys_cheb, color=CHEBFUN_BLUE, linewidth=1.5)
    axes[0].set_title(f"Chebyshev (length {len(f_cheb)})")
    axes[0].set_xlabel("t")

    xs_ref = np.linspace(-1, 1, 500)
    ys_trig = np.array(f_trig(jnp.array(xs_ref)))
    axes[1].plot(xs_ref, ys_trig, color=CHEBFUN_RED, linewidth=1.5)
    axes[1].set_title(f"Trigtech (length {f_trig.n})")
    axes[1].set_xlabel("s (reference)")

    ratio = len(f_cheb) / f_trig.n if f_trig.n > 0 else float('inf')
    fig.suptitle(f"Chebyshev vs Trig: ratio = {ratio:.2f}", fontsize=12)
    fig.tight_layout()
    save(fig)
except Exception as e:
    print(f"  Skipping plot 3 (Cheb vs Trig): {e}")

# --- Plot 4: Fourier coefficient decay for different smoothness classes ---
try:
    from chebfunjax.tech.trigtech import Trigtech, trig_vals2coeffs

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) Entire function: exp(sin(pi*t))
    f1 = Trigtech.from_function(lambda t: jnp.exp(jnp.sin(jnp.pi * t)))
    c1 = np.abs(np.array(f1.coeffs))
    c1 = np.maximum(c1, 1e-18)
    axes[0].semilogy(np.arange(len(c1)) - len(c1) // 2, c1, '.', color=CHEBFUN_BLUE, markersize=4)
    axes[0].set_title(f"$e^{{\\sin(\\pi t)}}$ (entire, n={f1.n})")
    axes[0].set_xlabel("Fourier mode k")
    axes[0].set_ylabel("|$c_k$|")

    # (b) Analytic: 1/(2 - cos(pi*t))
    f2 = Trigtech.from_function(lambda t: 1.0 / (2.0 - jnp.cos(jnp.pi * t)))
    c2 = np.abs(np.array(f2.coeffs))
    c2 = np.maximum(c2, 1e-18)
    axes[1].semilogy(np.arange(len(c2)) - len(c2) // 2, c2, '.', color=CHEBFUN_BLUE, markersize=4)
    axes[1].set_title(f"$1/(2-\\cos(\\pi t))$ (analytic, n={f2.n})")
    axes[1].set_xlabel("Fourier mode k")

    # (c) Finite smoothness: |sin(pi*t)|^5
    f3 = Trigtech.from_function(lambda t: jnp.abs(jnp.sin(jnp.pi * t))**5)
    c3 = np.abs(np.array(f3.coeffs))
    c3 = np.maximum(c3, 1e-18)
    axes[2].semilogy(np.arange(len(c3)) - len(c3) // 2, c3, '.', color=CHEBFUN_BLUE, markersize=4)
    axes[2].set_title(f"$|\\sin(\\pi t)|^5$ ($C^4$, n={f3.n})")
    axes[2].set_xlabel("Fourier mode k")

    fig.suptitle("Fourier coefficient decay by smoothness class", fontsize=12)
    fig.tight_layout()
    save(fig)
except Exception as e:
    print(f"  Skipping plot 4 (Fourier decay): {e}")

# --- Plot 5: Gibbs phenomenon — truncated Fourier series of square wave ---
try:
    from chebfunjax.tech.trigtech import trig_vals2coeffs, trig_coeffs2vals

    N = 201
    t = np.linspace(-1, 1, N, endpoint=False)
    vals = np.sign(np.sin(np.pi * t))
    coeffs_np = np.array(trig_vals2coeffs(jnp.array(vals)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    M = N // 2

    for idx, degree in enumerate([5, 15, 50]):
        trunc = np.zeros(2 * degree + 1, dtype=np.complex128)
        for k in range(-degree, degree + 1):
            if abs(k + M) < len(coeffs_np):
                trunc[k + degree] = coeffs_np[k + M]
        vals_trunc = np.array(trig_coeffs2vals(jnp.array(trunc)))
        t_trunc = np.linspace(-1, 1, len(vals_trunc), endpoint=False)

        axes[idx].plot(t_trunc, np.real(vals_trunc), color=CHEBFUN_BLUE, linewidth=1.5)
        axes[idx].plot(t, vals, color=CHEBFUN_RED, linewidth=0.8, linestyle='--', alpha=0.5)
        axes[idx].set_title(f"Degree {degree}")
        axes[idx].set_xlabel("t")
        axes[idx].set_ylim(-1.5, 1.5)

    fig.suptitle("Gibbs phenomenon: truncated Fourier series of a square wave", fontsize=12)
    fig.tight_layout()
    save(fig)
except Exception as e:
    print(f"  Skipping plot 5 (Gibbs): {e}")

# --- Plot 6: Non-periodic function warning: t^2 on [-1,1] ---
try:
    from chebfunjax.tech.trigtech import Trigtech
    import warnings

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        f_bad = Trigtech.from_function(lambda t: t**2)

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.linspace(-1, 1, 500)
    ys = np.array(f_bad(jnp.array(xs)))
    ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.5, label=f"Trigtech (n={f_bad.n})")
    ax.plot(xs, xs**2, color=CHEBFUN_RED, linewidth=1.5, linestyle='--',
            alpha=0.6, label="exact $t^2$")
    ax.set_title("Non-periodic function in Trigtech: $t^2$")
    ax.set_xlabel("t")
    ax.legend()
    fig.tight_layout()
    save(fig)
except Exception as e:
    print(f"  Skipping plot 6 (non-periodic): {e}")

print(f"\nGuide 11: Generated {plot_index} plots.")
