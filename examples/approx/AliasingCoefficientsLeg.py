"""Accuracy of Legendre coefficients via aliasing.

Follow-up to AliasingCoefficients exploring aliasing errors in Legendre
rather than Chebyshev coefficients.

Credit: Yuji Nakatsukasa, April 2016.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/AliasingCoefficientsLeg.html
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

    def fori(x): return jnp.log(jnp.sin(10.0 * x) + 2.0)

    f = cj.chebfun(fori)
    n_low = max(5, len(f) // 3)
    p = cj.chebfun(fori, n=n_low)

    # Convert Chebyshev coefficients to Legendre via cheb2leg
    try:
        from chebfunjax.utils.transforms import cheb2leg, leg2cheb
        fc_cheb = np.array(f.coeffs)
        pc_cheb = np.array(p.coeffs)
        fc_leg = np.array(cheb2leg(jnp.array(fc_cheb)))
        pc_leg = np.array(cheb2leg(jnp.array(pc_cheb)))
        n_p = len(pc_leg)
        err_leg = np.abs(pc_leg - fc_leg[:n_p]) + 1e-18

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.semilogy(np.arange(len(fc_leg)), np.abs(fc_leg) + 1e-18,
                    '.g', ms=7, label='f Legendre coeffs')
        ax.semilogy(np.arange(n_p), np.abs(pc_leg) + 1e-18,
                    '.b', ms=7, label='p Legendre coeffs')
        ax.semilogy(np.arange(n_p), err_leg, '.r', ms=7,
                    label='|f−p| (aliasing error)')
        ax.set_title('Aliasing of Legendre coefficients', fontsize=11)
        ax.set_xlabel('degree')
        ax.set_ylabel('|coefficient|')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(_OUTDIR, 'AliasingCoefficientsLeg.png'), dpi=150)
        plt.close(fig)
        print("AliasingCoefficientsLeg: done with cheb2leg transform.")
    except ImportError:
        # Fallback: use Chebyshev coefficients with a note
        fc = np.array(f.coeffs)
        pc = np.array(p.coeffs)
        n_p = len(pc)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.semilogy(np.arange(len(fc)), np.abs(fc) + 1e-18, '.g', ms=7,
                    label='f Chebyshev coeffs (proxy for Legendre)')
        ax.semilogy(np.arange(n_p), np.abs(pc) + 1e-18, '.b', ms=7,
                    label='p coeffs')
        ax.semilogy(np.arange(n_p), np.abs(pc - fc[:n_p]) + 1e-18, '.r',
                    ms=7, label='aliasing error')
        ax.set_title('Aliasing of coefficients (Chebyshev basis)', fontsize=11)
        ax.set_xlabel('degree')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(_OUTDIR, 'AliasingCoefficientsLeg.png'), dpi=150)
        plt.close(fig)
        print("AliasingCoefficientsLeg: cheb2leg not available, used Chebyshev basis.")

    return True


if __name__ == '__main__':
    run()
