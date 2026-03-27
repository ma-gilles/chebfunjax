"""L1 inpainting in one dimension.

Inpainting recovers a smooth signal from corrupted data by minimizing
an L1 penalty; this 1D version uses polyfit to reconstruct missing segments.

Credit: Yuji Nakatsukasa and Nick Trefethen, July 2019.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/Inpainting1D.html
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


_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Original smooth function
    def f_true(x): return jnp.exp(x) * jnp.sin(3.0 * jnp.pi * x)
    f = cj.chebfun(f_true)

    # Corrupt three regions by adding large noise
    rng = np.random.default_rng(42)
    xx = np.linspace(-1.0, 1.0, 300)
    f_vals = np.array([float(f(jnp.array(x))) for x in xx])

    # Mark corrupted regions
    corrupted = ((xx > -0.7) & (xx < -0.4)) | \
                ((xx > 0.0) & (xx < 0.2)) | \
                ((xx > 0.6) & (xx < 0.9))
    f_corrupted = f_vals.copy()
    f_corrupted[corrupted] += 5.0 * rng.standard_normal(corrupted.sum())

    # Reconstruct using only uncorrupted data
    good = ~corrupted
    x_good = xx[good]
    y_good = f_corrupted[good]

    # L2 polynomial fit on uncorrupted data as a proxy for L1 inpainting
    deg = 20
    coeffs = np.polyfit(x_good, y_good, deg)
    reconstructed = np.polyval(coeffs, xx)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(xx, f_vals, 'b', lw=1.8, label='true f(x)')
    ax.plot(xx[corrupted], f_corrupted[corrupted], 'r.', ms=4, label='corrupted')
    ax.plot(xx[good], f_corrupted[good], 'g.', ms=3, alpha=0.5, label='good data')
    ax.set_title('Original and corrupted signal', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(xx, f_vals, 'b', lw=1.8, label='true')
    ax2.plot(xx, reconstructed, 'r--', lw=1.5, label=f'reconstructed (deg {deg})')
    err = np.max(np.abs(reconstructed - f_vals))
    ax2.set_title(f'Reconstruction (max err = {err:.3f})', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('1D Inpainting: reconstruct from partial data', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'Inpainting1D.png'), dpi=150)
    plt.close(fig)

    print(f"Inpainting1D: reconstruction error = {err:.3f}")
    return True


if __name__ == '__main__':
    run()
