"""Interactive interpolation.

Demonstrates polynomial interpolation at user-specified points, comparing
Chebyshev nodes vs. equispaced nodes and showing Lebesgue functions.

Credit: Nick Hale, November 2012.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/InteractiveInterp.html
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

    # Demonstrate interpolation at different node distributions
    def f_func(x): return jnp.sin(2.0 * jnp.pi * x)
    n_nodes = 12

    # Chebyshev nodes
    k = np.arange(n_nodes)
    cheb_nodes = np.cos(np.pi * k / (n_nodes - 1))

    # Equispaced nodes
    eq_nodes = np.linspace(-1.0, 1.0, n_nodes)

    xx = np.linspace(-1.0, 1.0, 400)
    f_true = np.sin(2.0 * np.pi * xx)

    def poly_interp(nodes, f_fn, x_eval):
        y_nodes = np.array([float(f_fn(jnp.array(n))) for n in nodes])
        coeffs = np.polyfit(nodes, y_nodes, len(nodes) - 1)
        return np.polyval(coeffs, x_eval)

    p_cheb = poly_interp(cheb_nodes, f_func, xx)
    p_eq = poly_interp(eq_nodes, f_func, xx)

    err_cheb = np.abs(p_cheb - f_true)
    err_eq = np.abs(p_eq - f_true)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.plot(xx, f_true, 'b', lw=1.8, label='f(x)')
    ax.plot(xx, p_cheb, 'r--', lw=1.5, label='Chebyshev interp')
    ax.plot(cheb_nodes, np.sin(2 * np.pi * cheb_nodes), '.r', ms=8)
    ax.set_title(f'Chebyshev nodes (n={n_nodes})', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(xx, f_true, 'b', lw=1.8, label='f(x)')
    ax2.plot(xx, p_eq, 'r--', lw=1.5, label='equispaced interp')
    ax2.plot(eq_nodes, np.sin(2 * np.pi * eq_nodes), '.r', ms=8)
    ax2.set_title(f'Equispaced nodes (n={n_nodes})', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.semilogy(xx, err_cheb + 1e-18, 'r', lw=1.5, label='Chebyshev')
    ax3.set_title('Error: Chebyshev', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.semilogy(xx, err_eq + 1e-18, 'g', lw=1.5, label='equispaced')
    ax4.set_title('Error: equispaced (Runge phenomenon)', fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('x')

    fig.suptitle('Polynomial interpolation: Chebyshev vs. equispaced', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'InteractiveInterp.png'), dpi=150)
    plt.close(fig)

    print(f"InteractiveInterp: cheb max err={np.max(err_cheb):.2e}, "
          f"eq max err={np.max(err_eq):.2e}")
    return True


if __name__ == '__main__':
    run()
