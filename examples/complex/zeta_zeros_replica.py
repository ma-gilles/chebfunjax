"""Zeros of the Riemann zeta function.

Faithful replica of complex/ZetaZeros.m by Nick Trefethen (October
2011): computing zeros of zeta(s) in the critical strip by analytic
continuation of a chebfun on the line Re(s) = 4 and complex
rootfinding.

Original: https://www.chebfun.org/examples/complex/ZetaZeros.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'complex')

KK = np.arange(1e5, 0, -1)


def zeta(s):
    s = np.asarray(s, dtype=complex)
    return np.sum(KK[None, :] ** (-np.atleast_1d(s)[:, None]),
                  axis=1).reshape(np.shape(s))


def run():
    os.makedirs(_IMG, exist_ok=True)
    t_start = time.time()

    v = complex(zeta(4.0))
    print("ans =")
    print(f"   {v.real:.15f}")
    print("exact =")
    print(f"   {np.pi**4/90:.15f}")

    # chebfun of zeta(4 + it) on t in [5, 50]
    def fop(t):
        arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
        vals = zeta(4.0 + 1j * arr.ravel())
        return jnp.asarray(vals.reshape(arr.shape))

    f = cj.chebfun(fop, domain=(5.0, 50.0))
    print("f =")
    print(repr(f))

    # complex roots of the chebfun = zeros of zeta continued into the
    # strip (t complex; s = 4 + i t maps them to the critical line)
    zt = np.asarray(f.roots(complex_roots=True))
    # zeta zeros at s = 1/2 + i*gamma map to t = gamma + 3.5i
    zt = zt[(np.real(zt) > 5) & (np.real(zt) < 50)
            & (np.imag(zt) > 2) & (np.imag(zt) < 5)]
    zt = zt[np.argsort(np.real(zt))]
    zeros_s = 4.0 + 1j * zt

    ts = np.linspace(5, 50, 2000)
    fv = np.asarray(f(jnp.asarray(ts)))
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.plot(ts, np.zeros_like(ts), 'k:', lw=0.8)
    ax.plot(zt.real, zt.imag, '.r', ms=10)
    ax.plot([0], [3], 'xk', ms=12)
    ax.set_xlim(-5, 60)
    ax.set_yticks(range(-12, 13, 4))
    ax.grid(True)
    ax.set_title("complex roots of the chebfun (red)", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ZetaZeros_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    zeros_exact = 0.5 + 1j * np.array([
        14.1347251417, 21.0220396388, 25.0108575801, 30.4248761259,
        32.9350615877, 37.5861781588, 40.9187190121, 43.3270732809])
    print("            Chebfun                          Exact")
    for zc, ze in zip(zeros_s[:len(zeros_exact)], zeros_exact):
        print(f"{zc.real:13.10f} + {zc.imag:13.10f}i   "
              f"{ze.real:13.10f} + {ze.imag:13.10f}i")

    # real/imag parts along the critical line (t shifted by 3.5i)
    ftv = np.asarray([complex(zeta(4 + 1j * (tt + 3.5j)))
                      for tt in ts])
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.plot(ts, ftv.imag, lw=1.2)
    ax.plot(ts, ftv.real, lw=1.2)
    ax.plot(zt.real, np.zeros(len(zt)), '.k', ms=10)
    ax.grid(True)
    ax.set_title("Real and imaginary parts of zeta(s) along critical "
                 "line", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ZetaZeros_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Elapsed time is {time.time()-t_start:.6f} seconds.")


if __name__ == "__main__":
    run()
