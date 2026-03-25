"""Fourier series and convergence.

Demonstrates construction of Fourier series approximations using chebfunjax,
including the Gibbs phenomenon for discontinuous functions.

Original: https://www.chebfun.org/examples/fourier/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj


def fourier_partial_sum(f_coeffs, x, N):
    """Compute N-term Fourier partial sum at points x in [0, 2*pi]."""
    result = f_coeffs[0] * np.ones_like(x)
    for k in range(1, N + 1):
        result += f_coeffs[2*k-1] * np.cos(k * x) + f_coeffs[2*k] * np.sin(k * x)
    return result


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/fourier')
    os.makedirs(outdir, exist_ok=True)

    T = 2 * float(jnp.pi)

    # --- Fourier series of a smooth function: f = sin(x)^3 + cos(2x)^2 ---
    f = cj.chebfun(lambda x: jnp.sin(x)**3 + jnp.cos(2*x)**2,
                   domain=(0.0, T))
    integral_f = float(f.sum())
    print(f"∫₀^2π f(x)dx = {integral_f:.8f}  (exact: π)")
    assert abs(integral_f - float(jnp.pi)) < 1e-10

    # --- Gibbs phenomenon: square wave -----------------------------------
    # Step function: f(x) = sign(sin(x)) on [0, 2*pi]
    # Approximate by Fourier partial sums of odd order
    xx = np.linspace(0.01, T - 0.01, 1000)
    sq_wave = np.sign(np.sin(xx))

    # Manually compute Fourier partial sums
    def fourier_squarewave(x, N):
        """N-term Fourier series of square wave."""
        result = np.zeros_like(x)
        for k in range(1, N + 1, 2):   # odd harmonics only
            result += (4 / (np.pi * k)) * np.sin(k * x)
        return result

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Smooth function
    fv = np.array(f(jnp.array(xx)))
    axes[0].plot(xx, fv, 'b-', linewidth=1.8, label='$f(x) = \\sin^3(x) + \\cos^2(2x)$')
    axes[0].set_title('A smooth periodic function on $[0, 2\\pi]$', fontsize=11)
    axes[0].set_xlabel('$x$')
    axes[0].set_xlim(0, T)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='k', linewidth=0.5)

    # Gibbs phenomenon
    axes[1].plot(xx, sq_wave, 'k-', linewidth=1.5, alpha=0.3, label='Square wave')
    colors2 = ['blue', 'red', 'green']
    for N, color in zip([5, 15, 51], colors2):
        Fs = fourier_squarewave(xx, N)
        axes[1].plot(xx, Fs, '-', color=color, linewidth=1.2,
                     label=f'$N={N}$ terms')
    axes[1].set_title('Gibbs phenomenon: square wave Fourier series', fontsize=11)
    axes[1].set_xlabel('$x$')
    axes[1].set_xlim(0, T)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-1.5, 1.5)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fourier_series.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Verify: integral of sin(nx)*sin(mx) over [0,2pi]
    print("\nTrig orthogonality check:")
    for m, n in [(1,1), (2,2), (1,2)]:
        sm = cj.chebfun(lambda x, m=m: jnp.sin(m*x), domain=(0.0, T))
        sn = cj.chebfun(lambda x, n=n: jnp.sin(n*x), domain=(0.0, T))
        inn = float(sm.inner(sn))
        expected = float(jnp.pi) if m == n else 0.0
        print(f"  <sin({m}x), sin({n}x)> = {inn:.8f}  (expected {expected:.8f})")
        assert abs(inn - expected) < 1e-10

    print("fourier_series: done")
    return True


if __name__ == "__main__":
    run()
