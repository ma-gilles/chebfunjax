"""MathJax introduction.

Demonstrates key mathematical formulae in Chebfun examples using
LaTeX rendering, illustrated with Chebyshev's key theorems.
Translated from fun/MathJax.m.

Original: https://www.chebfun.org/examples/fun/MathJax.html
Author: Nick Hale, March 2012
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()


# Enable LaTeX rendering if available
plt.rcParams['text.usetex'] = False  # Use mathtext instead


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/fun')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3)

    x = np.linspace(-1, 1, 500)

    # --- Panel 1: Chebyshev polynomials with LaTeX labels ---
    colors = ['b', 'r', 'g', 'm', 'c']
    for n, col in enumerate(colors):
        Tn = np.cos(n * np.arccos(x))
        axes[0].plot(x, Tn, '-', color=col, linewidth=1.5,
                     label=f'$T_{n}(x)$')
    axes[0].axhline(0, color='k', linewidth=0.5)
    axes[0].set_title('Chebyshev polynomials\n$T_n(x) = \\cos(n\\arccos x)$',
                       fontsize=10)
    axes[0].legend(fontsize=10, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$T_n(x)$')

    # --- Panel 2: Orthogonality integral ---
    ns = np.arange(0, 8)
    ms = np.arange(0, 8)
    gram = np.zeros((len(ns), len(ms)))
    for i, n in enumerate(ns):
        for j, m in enumerate(ms):
            Tn = np.cos(n * np.arccos(x))
            Tm = np.cos(m * np.arccos(x))
            w = 1 / np.sqrt(1 - x**2 + 1e-14)
            # Trapezoidal quadrature
            gram[i, j] = np.trapezoid(Tn * Tm * w, x)

    im = axes[1].imshow(np.abs(gram), cmap='Blues', aspect='auto')
    axes[1].set_title('Orthogonality matrix\n$\\int_{-1}^{1} T_m T_n w\\, dx$',
                       fontsize=10)
    axes[1].set_xlabel('$m$'); axes[1].set_ylabel('$n$')
    plt.colorbar(im, ax=axes[1])

    print("MathJax/Chebyshev example:")
    print("  Orthogonality: max off-diagonal element = "
          f"{np.max(np.abs(gram - np.diag(np.diag(gram)))):.4f}")

    # --- Panel 3: Famous formulae ---
    ax3 = axes[2]
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.axis('off')

    formulae = [
        (0.5, 0.90, r'Chebyshev nodes:', 12, 'bold'),
        (0.5, 0.80, r'$x_k = \cos\!\left(\frac{(2k-1)\pi}{2n}\right)$', 14, 'normal'),
        (0.5, 0.65, r'Chebyshev series:', 12, 'bold'),
        (0.5, 0.55, r'$f(x) = \sum_{k=0}^\infty a_k T_k(x)$', 14, 'normal'),
        (0.5, 0.42, r'Barycentric formula:', 12, 'bold'),
        (0.5, 0.32, r'$p(x) = \frac{\sum_k w_k f_k/(x-x_k)}{\sum_k w_k/(x-x_k)}$',
         14, 'normal'),
        (0.5, 0.15, r'$w_k = (-1)^k \sin\!\left(\frac{k\pi}{n}\right)$', 12, 'normal'),
    ]

    for x_pos, y_pos, text, fontsize, weight in formulae:
        ax3.text(x_pos, y_pos, text, ha='center', va='center',
                 fontsize=fontsize, fontweight=weight, transform=ax3.transAxes)

    ax3.set_title('Key Chebyshev formulae', fontsize=11)
    ax3.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False,
                                   edgecolor='gray', linewidth=2,
                                   transform=ax3.transAxes))

    fig.suptitle('MathJax: Beautiful Mathematics in Chebfun', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'mathjax.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("mathjax: done")
    return True


if __name__ == "__main__":
    run()
