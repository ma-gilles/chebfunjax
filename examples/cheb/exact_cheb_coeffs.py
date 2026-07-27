"""Exact Chebyshev expansion coefficients of a function.

The Chebyshev coefficients of f(x) = 1/(5+x) have a closed form (Elliott's
residue formula); the example tabulates the exact coefficients next to the
ones Chebfun computes and their difference.  Faithful port of
cheb/ExactChebCoeffs.m by Mark Richardson (June 2012).

Original MATLAB: https://www.chebfun.org/examples/cheb/ExactChebCoeffs.html
"""

import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()


def run():
    # MATLAB: f = @(x) 1/(5+x); fc = chebfun(f); k = 1:length(fc);
    fc = cj.chebfun(lambda x: 1.0 / (5.0 + x))
    cheb_coeffs = np.asarray(fc.coeffs, dtype=float)
    n = len(cheb_coeffs)
    k = np.arange(1, n + 1)

    # Elliott's exact coefficients:
    #   exact = (1/sqrt(6)) .* (-1).^(k-1) ./ (5+sqrt(24)).^(k-1)
    exact_coeffs = (1.0 / np.sqrt(6.0)) * (-1.0) ** (k - 1) / (5.0 + np.sqrt(24.0)) ** (k - 1)

    # MATLAB: display([exact_coeffs cheb_coeffs exact_coeffs-cheb_coeffs])
    diff = exact_coeffs - cheb_coeffs
    for i in range(n):
        print(f"{exact_coeffs[i]:>19.15f} {cheb_coeffs[i]:>19.15f} {diff[i]:>19.15f}")

    # --- Plot: coefficient decay ----------------------------------------
    fig, ax = plt.subplots()
    ax.semilogy(np.arange(n), np.abs(cheb_coeffs) + 1e-300, '.-',
                color='#0072BD')
    ax.set_title('Chebyshev coefficients of 1/(5+x)', fontsize=12)
    ax.set_xlabel('n')
    ax.set_ylabel('|a_n|')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "exact_cheb_coeffs.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print("exact_cheb_coeffs: done")
    return True


if __name__ == "__main__":
    run()
