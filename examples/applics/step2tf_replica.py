"""The AAA algorithm for system identification (2).

Faithful replica of applics/Step2tf.m (Costa, 2021): identifying LTI
systems from step-response data with AAA + FFT + Vandermonde with
Arnoldi -- pole extraction from the Laplace-domain step response,
re-identification from the FFT of the time signal, and recovery from
noisy data with 15% of samples missing.

The noise/missing-sample draws use a seeded numpy stream (MATLAB's
rng(1) stream is not reproducible outside MATLAB); the noise-free
sections are deterministic and match the published poles/residues.

Original: https://www.chebfun.org/examples/applics/Step2tf.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.aaa import aaa
from chebfunjax.utils.va import va_eval, va_orthog

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'applics')


def _save(fig, k):
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Step2tf_repl_{k:02d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def _print_cvec(name, v):
    print(f"{name} =")
    for z in np.asarray(v).ravel():
        sgn = '+' if z.imag >= 0 else '-'
        print(f" {z.real:18.15f} {sgn} {abs(z.imag):.15f}i")


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()

    def Num(s):
        return (1 + 105 * s) * (1 + 28 * s + 400 * s**2)

    def Den(s):
        return (1 + 100 * s) * (1 + 35 * s + 625 * s**2) * (1 + 0.4 * s**2)

    def G(s):
        return Num(s) / Den(s)

    Fs = 128
    t = np.arange(0, 20, 1 / Fs)
    L = t.shape[0]
    w = np.logspace(-4, 2, 6000)

    # Step response in the Laplace domain; AAA with mirrored samples.
    GS = (1 / (1j * w)) * G(1j * w)
    Zs = np.concatenate([-1j * w[::-1], 1j * w])
    Fv = np.concatenate([np.conj(GS[::-1]), GS])
    _, polG, resG, *_ = aaa(Fv, Zs, lawson=0)
    k = int(np.argmin(np.abs(polG)))
    polG = np.array(polG)
    polG[k] = 0.0
    order = np.argsort(-np.abs(np.asarray(resG)))
    _print_cvec("polG", polG[order])
    _print_cvec("resG", np.asarray(resG)[order])

    def g(x):
        return np.real(np.exp(np.outer(x, polG)) @ np.asarray(resG))

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(t, g(t), 'b-', lw=1.5, label='step response of G(s)')
    ax.grid(True)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('amplitude')
    ax.legend(loc='lower right')
    _save(fig, 1)

    # FFT of the time signal -> single-sided spectrum -> AAA.
    Y = np.fft.fft(g(t))
    hY = Y[:L // 2 + 1] / L
    F = 2 * np.pi * Fs * np.arange(L // 2 + 1) / L
    print("fft_length =")
    print("  int16")
    print(f"   {hY.shape[0]}")

    Zf = np.concatenate([-1j * F[::-1], 1j * F])
    Ff = np.concatenate([np.conj(hY[::-1]), hY])
    _, polH, *_ = aaa(Ff, Zf, lawson=0)
    polH = np.roots(np.real(np.poly(np.asarray(polH))))
    polH = np.where(np.real(polH) > 0, -np.conj(polH), polH)
    polH = polH[np.abs(polH) <= F.max()]
    k = int(np.argmin(np.abs(polH)))
    polH[k] = 0.0
    _print_cvec("polH", polH[np.argsort(-np.abs(polH))][:6])

    # LS directly on the original signal for the residues.
    Q = np.exp(np.outer(t, polH))
    resH = np.linalg.lstsq(Q, g(t), rcond=None)[0]
    _print_cvec("resH", resH[np.argsort(-np.abs(resH))][:6])

    def h(x):
        return np.real(np.exp(np.outer(x, polH)) @ resH)

    err = float(np.max(np.abs(g(t) - h(t))))
    print("err =")
    print(f"     {err:.15e}")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(t, g(t), 'b-', lw=1.5, label='step response of G(s)')
    ax.plot(t, h(t), 'k--', lw=1.5, label='step response of H(s)')
    ax.grid(True)
    ax.set_title(f"Error in step response data = {err:.6e}")
    ax.set_xlabel('time [s]')
    ax.set_ylabel('amplitude')
    ax.legend(loc='lower right')
    _save(fig, 2)

    def H(s):
        return (1 / (np.asarray(s)[:, None] - polH[None, :])) @ resH

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.loglog(w, np.abs(GS), 'b-', lw=1.5, label='G(s)')
    ax.loglog(w, np.abs(H(1j * w)), 'k--', lw=1.5, label='H(s)')
    ax.grid(True)
    ax.set_title("Frequency responses")
    ax.set_xlabel('frequency [rad/s]')
    ax.set_ylabel('magnitude')
    ax.legend(loc='lower left')
    _save(fig, 3)

    # --- noisy data with missing samples ---
    print("poles =")
    for z in np.roots([1, 1, 2]):
        sgn = '+' if z.imag >= 0 else '-'
        print(f" {z.real:18.15f} {sgn} {abs(z.imag):.15f}i")

    def f(x):
        return (np.exp(-x / 2) * (5 * np.sin(np.sqrt(7) * x / 2)
                                  + np.sqrt(7) * np.cos(np.sqrt(7) * x / 2))
                / (2 * np.sqrt(7)) - 0.5)

    rng = np.random.default_rng(1)
    data = f(t) + 0.01 * rng.standard_normal(L)
    drop = np.unique(rng.integers(0, L, int(np.ceil(L * 0.15))))
    keep = np.setdiff1d(np.arange(L), drop)
    tt, data = t[keep], data[keep]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(tt, data, 'r.', ms=3, label='corrupted data')
    ax.plot(t, f(t), 'b-', lw=1.5, label='original signal')
    ax.grid(True)
    ax.set_title("Signal with noise and missing samples")
    ax.set_xlabel('time [s]')
    ax.set_ylabel('amplitude')
    ax.legend(loc='upper right')
    _save(fig, 4)

    # Vandermonde with Arnoldi smooths the noise.
    Hes, R = va_orthog(tt, 30)
    c = np.linalg.lstsq(R, data, rcond=None)[0]
    y = np.real(va_eval(t, Hes) @ c)
    err = float(np.max(np.abs(f(t) - y)))
    print("err =")
    print(f"   {err:.15f}")

    # Identify a 4th-degree model from the FFT of the smoothed signal.
    Yf = np.fft.fft(y)
    hYf = Yf[:L // 2 + 1] / L
    Ffreq = 2 * np.pi * Fs * np.arange(L // 2 + 1) / L
    Zf = np.concatenate([-1j * Ffreq[::-1], 1j * Ffreq])
    Fv = np.concatenate([np.conj(hYf[::-1]), hYf])
    _, polF, *_ = aaa(Fv, Zf, degree=4, lawson=0)
    polF = np.roots(np.real(np.poly(np.asarray(polF))))
    polF = np.where(np.real(polF) > 0, -np.conj(polF), polF)
    polF = polF[np.abs(polF) <= Ffreq.max()]
    k = int(np.argmin(np.abs(polF)))
    polF[k] = 0.0
    _print_cvec("polF", polF)
    Qf = np.exp(np.outer(tt, polF))
    resF = np.linalg.lstsq(Qf, data, rcond=None)[0]
    _print_cvec("resF", resF)

    def ffit(x):
        return np.real(np.exp(np.outer(x, polF)) @ resF)

    err = float(np.max(np.abs(f(t) - ffit(t))))
    print("err =")
    print(f"   {err:.15f}")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(tt, data, 'r.', ms=3, label='corrupted data')
    ax.plot(t, f(t), 'b-', lw=1.5, label='original signal')
    ax.plot(t, ffit(t), 'k--', lw=1.5,
            label='step response of LTI model')
    ax.grid(True)
    ax.set_title("Signal with noise and missing samples\n"
                 f"error in original signal = {err:.6e}")
    ax.set_xlabel('time [s]')
    ax.set_ylabel('amplitude')
    ax.legend(loc='upper right')
    _save(fig, 5)

    print("For this example:")
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")


if __name__ == "__main__":
    run()
