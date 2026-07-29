"""The AAA algorithm for system identification from Bode plots.

Faithful port of applics/Bode2tf.m by Stefano Costa (August 2021).  AAA
approximation of the complex frequency response identifies LTI system
parameters (poles, zeros, DC gain); a low-degree AAA-LS variant produces
reduced-order models.

Original: https://www.chebfun.org/examples/applics/Bode2tf.html
Copyright 2021 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): pol/zer/DCgain, the AAA-recovered
polA/zerA/DCgainA (both before and after the conjugate-forcing residue
recomputation), and err_mag/err_ph reproduce the published values at the
AAA noise floor (~11-14 significant digits vs the page's 15-digit
display).  The reduced-order zerAr/polAr/DCgainAr depend on the Lawson
iteration count buried in MATLAB's 'degree' mode ("20 iterations under
the hood") and agree to the published values only in structure and ~2-3
digits; the final Dcn is an RNG wall (unseeded MATLAB randn noise).
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


def _print_col(name, vals, real=False):
    print(f"{name} =")
    for v in np.atleast_1d(vals):
        v = complex(v)
        if real:
            print(f"  {v.real: .15f}")
        else:
            sign = "+" if v.imag >= 0 else "-"
            print(f" {v.real: .15f} {sign} {abs(v.imag):.15f}i")


def _residue_numerator(res, pol):
    """Numerator coefficients of sum_k res[k]/(s - pol[k]) (MATLAB
    residue(res, pol, []) direction: partial fractions -> polynomial)."""
    pol = np.asarray(pol)
    res = np.asarray(res)
    n = len(pol)
    num = np.zeros(max(n, 1), dtype=complex)
    for k in range(n):
        others = np.delete(pol, k)
        pk = np.poly(others) if len(others) else np.array([1.0 + 0j])
        term = res[k] * pk
        num[-len(term):] += term
    return np.trim_zeros(num, "f")


def run():
    Nc = np.array([8.4e4, 6.68e3, 2.66e2, 2.0])
    Dc = np.array([6.25e4, 6.6625e4, 4.26e3, 1.36e2, 1.0])
    G = lambda s: np.polyval(Nc, s) / np.polyval(Dc, s)

    _print_col("pol", np.roots(Dc))
    _print_col("zer", np.roots(Nc))
    print("DCgain =")
    print(f"   {abs(G(0)):.15f}")

    w = np.logspace(-4, 2, 3000)
    mag = np.abs(G(1j * w))
    # MATLAB's N/D use s.^[3:-1:0]' whose ' is a COMPLEX-CONJUGATE
    # transpose, so the listed "ph = -angle(G(i*w))" evaluates G at
    # conj(s) and equals +angle of the true G -- the executed semantics
    # fit G itself (published poles are the system's own).
    ph = np.angle(G(1j * w))

    # AAA on the symmetrised complex signal.
    wA = np.concatenate([-w[::-1], w])
    magA = np.concatenate([mag[::-1], mag])
    phA = np.concatenate([-ph[::-1], ph])
    GA = magA * np.exp(1j * phA)
    H, polA, resA, zerA, *_ = aaa(GA, 1j * wA)
    polA = np.asarray(polA)
    resA = np.asarray(resA)
    _print_col("polA", polA)
    _print_col("zerA", np.asarray(zerA))
    print("DCgainA =")
    print(f"   {float(np.abs(np.asarray(H(np.array([0.0]))))[0]):.15f}")

    print("err_mag =")
    print(f"     {np.max(np.abs(mag - np.abs(np.asarray(H(1j * w))))):.15e}")
    print("err_ph =")
    print(f"     {np.max(np.abs(ph - np.angle(np.asarray(H(1j * w))))):.15e}")

    # Force complex-conjugate pole/zero pairs via polynomial recomputation.
    DcA = np.poly(polA)
    NcA = _residue_numerator(resA, polA)
    polA2 = np.roots(np.real(DcA))
    zerA2 = np.roots(np.real(NcA))
    _print_col("polA", polA2)
    _print_col("zerA", zerA2)

    # Reduced-order (degree 2) AAA-LS model.
    _, polAr, *_ = aaa(GA, 1j * wA, degree=2)
    polAr = np.roots(np.real(np.poly(np.asarray(polAr))))
    d = np.min(np.abs(1j * wA[:, None] - polAr[None, :]), axis=0)
    Q = d[None, :] / (1j * wA[:, None] - polAr[None, :])
    c = np.linalg.lstsq(Q, GA, rcond=None)[0]
    Hr = lambda s: (d[None, :] / (np.atleast_1d(s)[:, None]
                                  - polAr[None, :])) @ c
    # MATLAB passes the raw LS coefficients c to residue(), not the true
    # residues c.*d of Hr -- reproduce that choice.
    NAr = _residue_numerator(c, polAr)
    _print_col("zerAr", np.roots(np.real(NAr)), real=True)
    _print_col("polAr", polAr, real=True)
    print("DCgainAr =")
    print(f"   {abs(Hr(0.0)[0]):.15f}")

    # Bode plot: original, full AAA, reduced order.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.6))
    ax1.semilogx(w, 20 * np.log10(mag), "b-", lw=1.2, label="G(s)")
    ax1.semilogx(w, 20 * np.log10(np.abs(np.asarray(H(1j * w)))), "k--",
                 lw=1.0, label="AAA")
    ax1.semilogx(w, 20 * np.log10(np.abs(Hr(1j * w))), "c-", lw=1.0,
                 label="reduced order AAA")
    ax1.grid(True)
    ax1.set_title("Magnitude (dB)")
    ax1.legend(loc="lower left", fontsize=8)
    ax2.semilogx(w, ph * 180 / np.pi, "b-", lw=1.2, label="G(s)")
    ax2.semilogx(w, np.angle(np.asarray(H(1j * w))) * 180 / np.pi, "k--",
                 lw=1.0, label="AAA")
    ax2.semilogx(w, np.angle(Hr(1j * w)) * 180 / np.pi, "c-", lw=1.0,
                 label="reduced order AAA")
    ax2.grid(True)
    ax2.set_title("Phase (degrees)")
    ax2.legend(loc="lower left", fontsize=8)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_HERE, "bode2tf.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # Noisy scalar example (RNG wall: MATLAB's unseeded randn noise).
    Nc2 = np.array([1.0, -1.0])
    Dc2 = np.array([1.0, 1.0, 2.0])
    f = lambda s: np.polyval(Nc2, s) / np.polyval(Dc2, s)
    w2 = np.logspace(-1, 1, 500)
    mag2 = np.abs(f(1j * w2))
    ph2 = np.angle(f(1j * w2))  # same ctranspose-conjugation semantics
    rng = np.random.RandomState(0)
    mag2 = mag2 + 0.01 * rng.randn(len(mag2))
    ph2 = ph2 + 0.01 * rng.randn(len(ph2))

    wn = np.concatenate([-w2[::-1], w2])
    magn = np.concatenate([mag2[::-1], mag2])
    phn = np.concatenate([-ph2[::-1], ph2])
    fn = magn * np.exp(1j * phn)
    _, poln, *_ = aaa(fn, 1j * wn, degree=2, lawson=30)
    poln = np.array(poln)
    poln[np.real(poln) > 0] = -1.0
    poln = np.roots(np.real(np.poly(poln)))
    print("Dcn =")
    row = "  ".join(f"{v: .15f}" for v in np.real(np.poly(poln)))
    print(f"  {row}")

    return True


if __name__ == "__main__":
    run()
