"""A pathological function of Weierstrass.

Constructs a partial sum of the Weierstrass nowhere-differentiable function
F(x) = sum_{k=0}^{inf} 2^{-k} cos(pi/2 * 4^k * x) on [-1,1].

Credit: Hrothgar, October 2013.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/WeierstrassFunction.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Build partial sums F_k
    def make_fk(k):
        return lambda x: 2.0**(-k) * jnp.cos(jnp.pi / 2 * x * 4.0**k)

    F = cj.chebfun(make_fk(0))
    for k in range(1, 8):
        F = F + cj.chebfun(make_fk(k))

    xx = np.linspace(-1.0, 1.0, 2000)
    F_vals = np.array([float(F(jnp.array(x))) for x in xx])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(xx, F_vals, 'k', lw=1.0)
    ax.set_title('Weierstrass-type function F(x), 8 terms', fontsize=11)
    ax.set_xlabel('x')
    ax.grid(True, alpha=0.3)

    # Zoom in
    ax2 = axes[1]
    mask = (xx >= 0) & (xx <= 0.005)
    ax2.plot(xx[mask], F_vals[mask], 'k', lw=1.5)
    ax2.set_title('Close-up near x=0 (zoomed 200×)', fontsize=11)
    ax2.set_xlabel('x')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Weierstrass nowhere-differentiable function', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'WeierstrassFunction.png'), dpi=150)
    plt.close(fig)

    # Integral should equal 4/pi
    integral = float(F.sum())
    print(f"WeierstrassFunction: integral = {integral:.6f}, "
          f"4/pi = {4/np.pi:.6f}, error = {abs(integral - 4/np.pi):.2e}")
    return True


if __name__ == '__main__':
    run()
