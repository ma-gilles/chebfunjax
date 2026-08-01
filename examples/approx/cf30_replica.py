"""CF approximation, 30 years ago.

Faithful replica of approx/CF30.m by Nick Trefethen (October 2010):
type (1,1) Caratheodory-Fejer approximation of sqrt(1.2-x) via the
modern cf command, minimax comparison, and the 'historical' 1980s
RCF code (FFT + Toeplitz + SVD).

Original: https://www.chebfun.org/examples/approx/CF30.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.cfpade import cf
from chebfunjax.utils.minimax import minimax

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

XS = np.linspace(-1, 1, 3000)


def _poly_coeffs(pcheb):
    c = np.asarray(pcheb.coeffs)
    mono = np.polynomial.chebyshev.cheb2poly(c)
    return mono[::-1]     # MATLAB poly(): descending powers


def _errplot(fv, rv, err, title, fname, color='C0'):
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(XS, fv - rv, color, lw=1.6)
    ax.grid(True)
    ax.set_ylim(-0.02, 0.02)
    ax.plot([-1, 1], [err, err], '--k', lw=1.6)
    ax.plot([-1, 1], [-err, -err], '--k', lw=1.6)
    ax.set_title(title, fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def historical_rcf(Fx, m, n, nfft, K):
    np_ = n + 1
    dim = K + n - m
    nfft2 = nfft // 2
    z = np.exp(2j * np.pi * np.arange(nfft) / nfft)
    x = z.real
    F = Fx(x)
    Fc = np.real(np.fft.fft(F)) / nfft2
    idx = np.remainder(np.arange(dim, 0, -1) + nfft + m - n, nfft)
    from scipy.linalg import svd, toeplitz
    H = toeplitz(Fc[idx])
    H = np.triu(H)
    H = H[:, ::-1]
    u_, s_, vt = svd(H)
    s = s_[np_ - 1]
    u = u_[::-1, np_ - 1]
    v = vt[np_ - 1, :]
    zr = np.roots(v)
    qout = [r for r in zr if abs(r) > 1]
    qc = np.real(np.poly(qout))
    qc = qc / qc[np_ - 1]
    q = np.polyval(qc, z)
    Q = q * np.conj(q)
    Qc = np.real(np.fft.fft(Q)) / nfft2
    Qc[0] = Qc[0] / 2
    Q = np.real(Q / Qc[0])
    Qc = Qc[:np_] / Qc[0]
    b = (np.fft.fft(np.concatenate([u, np.zeros(nfft - dim)]))
         / np.fft.fft(np.concatenate([v, np.zeros(nfft - dim)])))
    Rt = F - np.real(s * z**K * b)
    Rtc = np.real(np.fft.fft(Rt)) / nfft2
    gam = np.real(np.fft.fft(1.0 / Q)) / nfft2
    gam = toeplitz(gam[:2 * m + 1])
    if m == 0:
        Pc = 2 * Rtc[:1] @ np.linalg.inv(gam)
    else:
        rhs = np.concatenate([Rtc[m:0:-1], Rtc[:m + 1]])
        Pc = 2 * np.linalg.solve(gam.T, rhs)
    Pc = Pc[m:2 * m + 1]
    Pc[0] = Pc[0] / 2
    P = np.real(np.polyval(Pc[::-1], z))
    R = P / Q
    err = np.max(np.abs(F - R))
    print("s =")
    print(f"   {s:.15f}")
    print("err =")
    print(f"   {err:.15f}")
    print("Pc =")
    print("   " + "  ".join(f"{v:.15f}" for v in Pc))
    print("Qc =")
    print("   " + "  ".join(f"{v:.15f}" for v in Qc))


def run():
    os.makedirs(_IMG, exist_ok=True)

    f = cj.chebfun(lambda x: jnp.sqrt(1.2 - x))
    t0 = time.time()
    p, q, rh, s = cf(f, 1, 1)
    t_cf = time.time() - t0
    print("ans =")
    print("  " + "   ".join(f"{v:.15f}" for v in _poly_coeffs(p)))
    print("ans =")
    print("  " + "   ".join(f"{v:.15f}" for v in _poly_coeffs(q)))
    print(f"Elapsed time is {t_cf:.6f} seconds.")

    fv = np.asarray(f(jnp.asarray(XS)))
    rv = np.asarray(rh(jnp.asarray(XS)))
    err = np.max(np.abs(fv - rv))
    _errplot(fv, rv, err,
             f"type (1,1) CF approximation:  error = {err:.6g}",
             "CF30_repl_01.png")

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimax(lambda x: jnp.sqrt(1.2 - x), 1, rational=True,
                      denom=1)
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")
    rvb = np.asarray(res.r(XS))
    _errplot(fv, rvb, res.err,
             f"type (1,1) best approximation:  error = {res.err:.6g}",
             "CF30_repl_02.png", color='m')

    # The historical RCF code
    print("Fx = ")
    print("    @(x)sqrt(1.2-x)")
    t0 = time.time()
    historical_rcf(lambda x: np.sqrt(1.2 - x), 1, 1, 128, 20)
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")


if __name__ == "__main__":
    run()
