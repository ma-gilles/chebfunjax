"""Smooth functions of compact support.

Constructs infinitely smooth functions with compact support by convolving
a characteristic function with itself multiple times.

Credit: Nick Trefethen, July 2014.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/SmoothCompact.html
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

    h = 0.5
    # B0 = characteristic function on [-h, h]
    B0 = cj.chebfun(lambda x: jnp.ones_like(x), domain=(-h, h))

    # Build B1, B2, B3 by convolution
    B1 = B0.conv(B0)
    B2 = B1.conv(B0)
    B3 = B2.conv(B0)

    ax_range = [-3, 3, -0.1, 1.2]
    xx = np.linspace(ax_range[0], ax_range[1], 600)

    fig, axes = plt.subplots(2, 2)
    axes_flat = axes.flatten()
    splines = [(B0, 'B0 (box)', 0), (B1, 'B1 (hat)', 1),
               (B2, 'B2 (C¹)', 2), (B3, 'B3 (C²)', 3)]

    for ax, (Bk, title, k) in zip(axes_flat, splines):
        dom_a = float(Bk.domain.breakpoints[0])
        dom_b = float(Bk.domain.breakpoints[-1])
        in_dom = (xx >= dom_a) & (xx <= dom_b)
        vals = np.zeros_like(xx)
        vals[in_dom] = np.array([float(Bk(jnp.array(x))) for x in xx[in_dom]])
        ax.plot(xx, vals, 'b', lw=1.8)
        ax.set_xlim(ax_range[0], ax_range[1])
        ax.set_ylim(ax_range[2], ax_range[3])
        ax.set_title(f'{title} (C^{k-1} smooth)', fontsize=10)
    fig.suptitle('Smooth compactly supported functions via convolution', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'SmoothCompact.png'), dpi=150)
    plt.close(fig)

    print("SmoothCompact: done.")
    return True


if __name__ == '__main__':
    run()
