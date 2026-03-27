"""Gallery and Gallerytrig.

Demonstrates chebfunjax's gallery functions for generating interesting
test functions, analogous to MATLAB's gallery command.

Credit: Hrothgar and Nick Trefethen, December 2014.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/Galleries.html
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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

    # Try chebfunjax gallery if available
    try:
        from chebfunjax.utils.gallery import gallery
        gallery_names = ['gamma', 'abs', 'step', 'tower', 'runge']
        funcs = []
        for name in gallery_names:
            try:
                funcs.append((name, gallery(name)))
            except Exception:
                pass
    except ImportError:
        funcs = []

    # Fallback: define our own interesting functions
    interesting = [
        ('abs(x)', cj.chebfun(jnp.abs)),
        ('sign(x)', cj.chebfun(lambda x: jnp.sign(x), domain=(-1.0, 0.0, 1.0))),
        ('tanh(10x)', cj.chebfun(lambda x: jnp.tanh(10.0 * x))),
        ('Runge 1/(1+25x²)', cj.chebfun(lambda x: 1.0 / (1.0 + 25.0 * x**2))),
        ('exp(-1/x²)', cj.chebfun(lambda x: jnp.exp(-1.0 / (x**2 + 0.01)))),
        ('sin(20x)·exp(-x²)', cj.chebfun(lambda x: jnp.sin(20.0 * x) * jnp.exp(-x**2))),
    ]

    if funcs:
        # Use gallery functions
        to_plot = funcs[:min(6, len(funcs))]
    else:
        to_plot = interesting

    n = len(to_plot)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4 * rows))
    axes_flat = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for i, (name, f) in enumerate(to_plot):
        ax = axes_flat[i]
        xx = np.linspace(float(f.domain.breakpoints[0]),
                         float(f.domain.breakpoints[-1]), 500)
        vals = np.array([float(f(jnp.array(x))) for x in xx])
        ax.plot(xx, vals, 'b', lw=1.5)
        ax.set_title(name, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')

    for i in range(len(to_plot), len(axes_flat)):
        axes_flat[i].axis('off')

    fig.suptitle('Gallery of interesting Chebfun functions', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'Galleries.png'), dpi=150)
    plt.close(fig)

    print(f"Galleries: {len(to_plot)} functions plotted.")
    return True


if __name__ == '__main__':
    run()
