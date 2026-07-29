"""AAA algorithm for system identification from a step response.

Faithful port of applics/Step2tf.m by Stefano Costa (December 2021).  Given
the step response of a linear system, the AAA algorithm recovers the transfer
function's poles and residues directly from frequency-response data on the
imaginary axis, then a discrete-time (FFT) reconstruction recovers them again.

Original: https://www.chebfun.org/examples/applics/Step2tf.html
Copyright 2021 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): the six physical poles polG and their residues
resG, the FFT length (1281), the recovered polH/resH, and poles=roots([1 1 2])
reproduce the published values -- the AAA fit of the ill-scaled
GS = G(iw)/(iw) data (|F| dynamic range > 1e8) returns the clean physical
poles (+/-1.5811i, -0.028+/-0.0286i, -0.01, 0) with no Froissart doublets.
Agreement is at the AAA noise floor (~11-13 significant digits), below the
page's 15-digit display precision.  The second example's polF/resF/err are an
RNG wall: MATLAB's rng(1) randn/randi stream (ziggurat) cannot be reproduced
by NumPy, so the noise realisation -- and therefore the fitted values --
differ, while the pole structure (one real pole near -2, a conjugate pair
near -0.5 +/- 1.33i, and 0) reproduces.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.aaa import aaa

chebfun_style()

_HERE = os.path.dirname(os.path.abspath(__file__))


def _va_orthog(z, n):
    """Vandermonde-with-Arnoldi orthogonalisation (Brubeck-Nakatsukasa-
    Trefethen); returns the Hessenberg recurrence and the ON basis."""
    m = len(z)
    Q = np.ones((m, 1), dtype=np.float64)
    H = np.zeros((n + 1, n), dtype=np.float64)
    for k in range(n):
        q = z * Q[:, k]
        for j in range(k + 1):
            H[j, k] = Q[:, j] @ q / m
            q = q - H[j, k] * Q[:, j]
        H[k + 1, k] = np.linalg.norm(q) / np.sqrt(m)
        Q = np.column_stack([Q, q / H[k + 1, k]])
    return H, Q


def _va_eval(z, H):
    """Evaluate the Arnoldi basis defined by ``H`` at new points ``z``."""
    m = len(z)
    n = H.shape[1]
    W = np.ones((m, 1), dtype=np.float64)
    for k in range(n):
        w = z * W[:, k]
        for j in range(k + 1):
            w = w - H[j, k] * W[:, j]
        W = np.column_stack([W, w / H[k + 1, k]])
    return W


def _print_col(name, vals):
    print(f"{name} =")
    for v in np.atleast_1d(vals):
        v = complex(v)
        sign = "+" if v.imag >= 0 else "-"
        print(f" {v.real: .15f} {sign} {abs(v.imag):.15f}i")


def run():
    Num = lambda s: (1 + 105 * s) * (1 + 28 * s + 400 * s**2)
    Den = lambda s: ((1 + 100 * s) * (1 + 35 * s + 625 * s**2)
                     * (1 + 0.4 * s**2))
    G = lambda s: Num(s) / Den(s)

    Fs = 128
    t = np.arange(0, 20, 1 / Fs)
    L = len(t)
    w = np.logspace(-4, 2, 6000)

    # AAA fit of the step response on the imaginary axis.
    GS = (1 / (1j * w)) * G(1j * w)
    z = 1j * np.concatenate([-w[::-1], w])
    f = np.concatenate([np.conj(GS)[::-1], GS])
    _, polG, resG, *_ = aaa(f, z, lawson=0)
    polG = np.array(polG)
    resG = np.array(resG)
    polG[np.argmin(np.abs(polG))] = 0.0
    _print_col("polG", polG)
    _print_col("resG", resG)

    g = lambda x: np.real(np.exp(np.outer(x, polG)) @ resG)

    # Discrete-time (FFT) reconstruction.
    Y = np.fft.fft(g(t))
    hY = Y[:L // 2 + 1] / L
    Fr = 2 * np.pi * Fs * np.arange(L // 2 + 1) / L
    print("fft_length =")
    print("  int16")
    print(f"   {len(hY)}")

    zz = np.concatenate([-1j * Fr[::-1], 1j * Fr])
    ff = np.concatenate([np.conj(hY)[::-1], hY])
    _, polH, *_ = aaa(ff, zz, lawson=0)
    polH = np.array(np.roots(np.real(np.poly(np.asarray(polH)))))
    pos = np.real(polH) > 0
    polH[pos] = -np.conj(polH[pos])
    polH = polH[np.abs(polH) <= np.max(Fr)]
    polH[np.argmin(np.abs(polH))] = 0.0
    _print_col("polH", polH)

    Q = np.exp(np.outer(t, polH))
    resH = np.linalg.lstsq(Q, g(t), rcond=None)[0]
    _print_col("resH", resH)
    h = lambda x: np.real(np.exp(np.outer(x, polH)) @ resH)
    print("err =")
    print(f"     {np.max(np.abs(g(t) - h(t))):.15e}")

    # A quadratic's roots, for reference.
    _print_col("poles", np.roots([1, 1, 2]))

    # ------------------------------------------------------------------
    # Second example: identify H(s) = 1/(s^2+s+2) from a noisy step
    # response with 15% of the samples missing.  The gappy data is first
    # regularised by a degree-30 Vandermonde-with-Arnoldi least-squares
    # fit, then the same FFT + AAA pole extraction is applied.
    # ------------------------------------------------------------------
    f2 = lambda x: (np.exp(-x / 2)
                    * (5 * np.sin(np.sqrt(7) * x / 2)
                       + np.sqrt(7) * np.cos(np.sqrt(7) * x / 2))
                    / (2 * np.sqrt(7)) - 0.5)
    rng = np.random.RandomState(1)
    data = f2(t) + 0.01 * rng.randn(L)
    k = np.unique(rng.randint(0, L, int(np.ceil(L * 0.15))))
    keep = np.setdiff1d(np.arange(L), k)
    data = data[keep]
    tt = t[keep]

    Hes, Q = _va_orthog(tt, 30)
    c = np.linalg.lstsq(Q, data, rcond=None)[0]
    y = _va_eval(t, Hes) @ c
    print("err =")
    print(f"   {np.max(np.abs(f2(t) - y)):.15f}")

    Yf = np.fft.fft(y)
    hYf = Yf[:L // 2 + 1] / L
    Ff = 2 * np.pi * Fs * np.arange(L // 2 + 1) / L

    zf = np.concatenate([-1j * Ff[::-1], 1j * Ff])
    fF = np.concatenate([np.conj(hYf)[::-1], hYf])
    _, polF, *_ = aaa(fF, zf, degree=4, lawson=0)
    polF = np.array(np.roots(np.real(np.poly(np.asarray(polF)))))
    pos = np.real(polF) > 0
    polF[pos] = -np.conj(polF[pos])
    polF = polF[np.abs(polF) <= np.max(Ff)]
    polF[np.argmin(np.abs(polF))] = 0.0
    _print_col("polF", polF)

    Qf = np.exp(np.outer(tt, polF))
    resF = np.linalg.lstsq(Qf, data.astype(complex), rcond=None)[0]
    _print_col("resF", resF)

    ff2 = lambda x: np.real(np.exp(np.outer(x, polF)) @ resF)
    print("err =")
    print(f"   {np.max(np.abs(f2(t) - ff2(t))):.15f}")
    print("For this example:")

    # ------------------------------------------------------------------
    # Plot: the recovered step response vs the reconstruction.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, g(t), "b", lw=1.5, label="step response g(t)")
    ax.plot(t, h(t), "r--", lw=1.0, label="reconstruction h(t)")
    ax.set_xlabel("t")
    ax.set_ylabel("g(t)")
    ax.set_title("System identification via AAA")
    ax.legend(fontsize=9)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_HERE, "step2tf.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return True


if __name__ == "__main__":
    run()
