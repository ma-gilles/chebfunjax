"""The AAA algorithm for system identification.

Faithful replica of applics/Bode2tf.m (Costa, 2021): identifying LTI
system poles, zeros, and DC gain from Bode plots via AAA on the
mirrored complex frequency response; reduced-order AAA-LS models; and
degree-2 identification from noisy Bode data (seeded numpy noise;
MATLAB's randn stream is not reproducible outside MATLAB).

Original: https://www.chebfun.org/examples/applics/Bode2tf.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.aaa import aaa

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'applics')
FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Bode2tf_repl_{FIG[0]:02d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def _print_cvec(name, v):
    print(f"{name} =")
    for z in np.asarray(v).ravel():
        sgn = '+' if z.imag >= 0 else '-'
        print(f" {z.real:18.15f} {sgn} {abs(z.imag):.15f}i")


def _residue_polys(res, pol):
    """Numerator/denominator polynomials from a partial-fraction
    expansion (MATLAB residue(res, pol, []) inverse direction)."""
    pol = np.asarray(pol)
    res = np.asarray(res)
    D = np.poly(pol)
    N = np.zeros(pol.shape[0], dtype=complex)
    for i in range(pol.shape[0]):
        others = np.delete(pol, i)
        N = N + res[i] * np.concatenate(
            [[0] * (1 + others.shape[0] - np.poly(others).shape[0]),
             np.poly(others)])
    return N, D


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    Nc = [8.4e4, 6.68e3, 2.66e2, 2]
    Dc = [6.25e4, 6.6625e4, 4.26e3, 1.36e2, 1]

    def G(s):
        s = np.asarray(s, dtype=complex)
        return np.polyval(Nc, s) / np.polyval(Dc, s)

    _print_cvec("pol", np.roots(Dc))
    _print_cvec("zer", np.roots(Nc))
    print("DCgain =")
    print(f"     {abs(G(0)):.0f}")

    w = np.logspace(-4, 2, 3000)
    mag = np.abs(G(1j * w))
    ph = -np.angle(G(1j * w))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 6.6))
    ax1.semilogx(w, 20 * np.log10(mag), 'b-')
    ax1.grid(True)
    ax1.set_title("Magnitude (dB)")
    ax2.semilogx(w, ph * 180 / np.pi, 'b-')
    ax2.grid(True)
    ax2.set_title("Phase (degrees)")
    _save(fig)

    # AAA on the mirrored complex signal.
    wA = np.concatenate([-w[::-1], w])
    magA = np.concatenate([mag[::-1], mag])
    phA = np.concatenate([-ph[::-1], ph])
    GA = magA * np.exp(1j * phA)
    # The mirrored data with negated phase represents s -> G(-s); the
    # continued fit H therefore has its poles/zeros at the NEGATIVES of
    # the system's (verified in R2025b: MATLAB's H likewise blows up at
    # +1 while its aaa PRINTS -1).  Negate to report system parameters.
    H, polA, resA, zerA, *_ = aaa(GA, 1j * wA)
    polA = -np.asarray(polA)
    zerA = -np.asarray(zerA)
    _print_cvec("polA", polA)
    _print_cvec("zerA", zerA)
    H0 = abs(complex(np.asarray(H(np.array([0.0 + 0j])))[0]))
    print("DCgainA =")
    print(f"   {H0:.15f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 6.6))
    Hw = np.asarray(H(1j * w))
    ax1.semilogx(w, 20 * np.log10(mag), 'b-', label='G(s)')
    ax1.semilogx(w, 20 * np.log10(np.abs(Hw)), 'k--', label='AAA')
    ax1.grid(True)
    ax1.set_title("Magnitude (dB)")
    ax1.legend(loc='lower left')
    ax2.semilogx(w, ph * 180 / np.pi, 'b-', label='G(s)')
    ax2.semilogx(w, np.angle(Hw) * 180 / np.pi, 'k--', label='AAA')
    ax2.grid(True)
    ax2.set_title("Phase (degrees)")
    ax2.legend(loc='lower left')
    _save(fig)

    print("err_mag =")
    print(f"     {np.max(np.abs(mag - np.abs(Hw))):.15e}")
    print("err_ph =")
    print(f"     {np.max(np.abs(ph - np.angle(Hw))):.15e}")

    # Pole recomputation via real polynomial coefficients.
    NcA, DcA = _residue_polys(np.asarray(resA), np.asarray(polA))
    _print_cvec("polA", np.roots(np.real(DcA)))
    _print_cvec("zerA", np.roots(np.real(NcA)))

    # Reduced-order (degree 2) AAA-LS.  The LS runs in the raw data
    # frame (continued fit = G(-s)); reported poles/zeros are negated
    # to the system frame, like the full-order case above.
    _, polAr, *_ = aaa(GA, 1j * wA, degree=2)
    polAr = np.roots(np.real(np.poly(np.asarray(polAr))))
    d = np.min(np.abs(1j * wA[:, None] - polAr[None, :]), axis=0)
    Q = d[None, :] / (1j * wA[:, None] - polAr[None, :])
    c = np.linalg.lstsq(Q, GA, rcond=None)[0]

    def Hr(s):
        return (d[None, :] / (np.asarray(s)[:, None]
                              - polAr[None, :])) @ c

    # MATLAB's example computes zerAr from residue(c, polAr) -- the
    # unweighted residues, without the d scaling its Hr carries.
    NAr, _ = _residue_polys(c, polAr)
    _print_cvec("zerAr", -np.roots(np.real(NAr)))
    _print_cvec("polAr", -polAr)
    print("DCgainAr =")
    print(f"   {abs(complex(Hr(np.array([0.0 + 0j]))[0])):.15f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 6.6))
    Hrw = Hr(1j * w)
    ax1.semilogx(w, 20 * np.log10(mag), 'b-', label='G(s)')
    ax1.semilogx(w, 20 * np.log10(np.abs(Hw)), 'k--', label='AAA')
    ax1.semilogx(w, 20 * np.log10(np.abs(Hrw)), 'c-',
                 label='reduced order AAA')
    ax1.grid(True)
    ax1.set_title("Magnitude (dB)")
    ax1.legend(loc='lower left')
    ax2.semilogx(w, ph * 180 / np.pi, 'b-', label='G(s)')
    ax2.semilogx(w, np.angle(Hw) * 180 / np.pi, 'k--', label='AAA')
    ax2.semilogx(w, np.angle(Hrw) * 180 / np.pi, 'c-',
                 label='reduced order AAA')
    ax2.grid(True)
    ax2.set_title("Phase (degrees)")
    ax2.legend(loc='lower left')
    _save(fig)

    # --- noisy Bode data, degree-2 AAA-LS with Lawson filtering ---
    Nc2 = [1, -1]
    Dc2 = [1, 1, 2]

    def f(s):
        s = np.asarray(s, dtype=complex)
        return np.polyval(Nc2, s) / np.polyval(Dc2, s)

    w = np.logspace(-1, 1, 500)
    mag = np.abs(f(1j * w))
    ph = -np.angle(f(1j * w))
    rng = np.random.default_rng(0)
    mag = mag + 0.01 * rng.standard_normal(mag.shape[0])
    ph = ph + 0.01 * rng.standard_normal(ph.shape[0])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 6.6))
    ax1.semilogx(w, 20 * np.log10(mag), 'r-')
    ax1.grid(True)
    ax1.set_title("Magnitude (dB)")
    ax2.semilogx(w, ph * 180 / np.pi, 'r-')
    ax2.grid(True)
    ax2.set_title("Phase (degrees)")
    _save(fig)

    wn = np.concatenate([-w[::-1], w])
    magn = np.concatenate([mag[::-1], mag])
    phn = np.concatenate([-ph[::-1], ph])
    fn = magn * np.exp(1j * phn)
    _, poln, *_ = aaa(fn, 1j * wn, degree=2, lawson=30)
    poln = np.roots(np.real(np.poly(np.asarray(poln))))
    dn = np.min(np.abs(1j * wn[:, None] - poln[None, :]), axis=0)
    Qn = dn[None, :] / (1j * wn[:, None] - poln[None, :])
    cn = np.linalg.lstsq(Qn, fn, rcond=None)[0]

    def Hn(s):
        return (dn[None, :] / (np.asarray(s)[:, None]
                               - poln[None, :])) @ cn

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 6.6))
    Hnw = Hn(1j * w)
    ax1.semilogx(w, 20 * np.log10(mag), 'r-', label='Noisy data')
    ax1.semilogx(w, 20 * np.log10(np.abs(Hnw)), 'b-',
                 label='AAA approximant')
    ax1.grid(True)
    ax1.set_title("Magnitude (dB)")
    ax1.legend(loc='lower left')
    ax2.semilogx(w, ph * 180 / np.pi, 'r-', label='Noisy data')
    ax2.semilogx(w, np.angle(Hnw) * 180 / np.pi, 'b-',
                 label='AAA approximant')
    ax2.grid(True)
    ax2.set_title("Phase (degrees)")
    ax2.legend(loc='lower left')
    _save(fig)

    Dcn = np.real(np.poly(-poln))       # system frame
    print("Dcn =")
    print("   " + "   ".join(f"{v:.15f}" for v in Dcn))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 6.6))
    ax1.loglog(w, np.abs(mag - np.abs(Hnw)), 'r-', lw=.5)
    ax1.grid(True)
    ax1.set_title("Estimated noise in magnitude")
    ax1.set_xlim(w.min(), w.max())
    ax1.set_ylim(1e-5, 1e-1)
    ax2.loglog(w, np.abs(ph - np.angle(Hnw)), 'r-', lw=.5)
    ax2.grid(True)
    ax2.set_title("Estimated noise in phase")
    ax2.set_xlim(w.min(), w.max())
    ax2.set_ylim(1e-5, 1e-1)
    _save(fig)


if __name__ == "__main__":
    run()
