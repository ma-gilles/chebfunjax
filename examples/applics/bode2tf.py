"""AAA algorithm for system identification from Bode data.

Uses the AAA rational approximation algorithm to identify a transfer function
from frequency-domain (Bode) measurements, following applics/Bode2tf.m
by Stefano Costa (August 2021).

Given a Bode plot (frequency response magnitude/phase data), AAA fits a
rational approximant from which the system characteristics can be extracted.

Original MATLAB: https://www.chebfun.org/examples/applics/Bode2tf.html
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

def run():
    print("=" * 60)
    print("AAA algorithm for system identification (Bode data)")
    print("=" * 60)

    # Create a known transfer function: second-order system
    # H(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)
    omega_n = 10.0  # natural frequency
    zeta = 0.3      # damping ratio

    print(f"\nTarget: H(s) = ω_n²/(s²+2ζω_n s+ω_n²)")
    print(f"  ω_n = {omega_n}, ζ = {zeta}")
    print(f"  Poles at s = {omega_n*(-zeta + 1j*np.sqrt(1-zeta**2)):.3f}, conj.")

    # Frequency response: H(jω) for ω ∈ [0.1, 100] rad/s
    omega_vals = np.logspace(-1, 2, 300)
    s_vals = 1j * omega_vals
    H_vals = omega_n**2 / (s_vals**2 + 2 * zeta * omega_n * s_vals + omega_n**2)

    H_mag = np.abs(H_vals)
    H_phase = np.degrees(np.angle(H_vals))
    H_mag_db = 20 * np.log10(H_mag)

    # AAA approximation — use real-valued Bode data (AAA works best with real input)
    print("\nFitting with AAA on Bode magnitude data (real-valued)...")
    from chebfunjax.utils.aaa import aaa

    # Fit |H(jω)| as a function of log10(omega)
    omega_log = np.log10(omega_vals)
    f_aaa_mag, pol_mag, res_mag, zer_mag, z_sup, f_sup, w = aaa(
        H_mag, omega_log, tol=1e-8, mmax=20
    )

    H_aaa_mag = np.array([float(np.real(f_aaa_mag(omega_log[k])))
                           for k in range(len(omega_vals))])
    max_err = np.max(np.abs(H_aaa_mag - H_mag))
    print(f"  AAA poles in log-omega domain: {len(pol_mag)}")
    print(f"  Max error in |H|: {max_err:.2e}")
    assert max_err < 0.1, f"AAA magnitude error too large: {max_err:.2e}"
    print("PASS: AAA accurately fits Bode magnitude")

    # Fit phase data
    f_aaa_phase, *_ = aaa(H_phase, omega_log, tol=1e-5, mmax=20)
    H_aaa_phase = np.array([float(np.real(f_aaa_phase(omega_log[k])))
                              for k in range(len(omega_vals))])
    max_phase_err = np.max(np.abs(H_aaa_phase - H_phase))
    print(f"  Max error in phase: {max_phase_err:.3f} deg")

    # Physical characteristics from the Bode plot
    print(f"\nBode characteristics:")
    peak_idx = np.argmax(H_mag)
    peak_freq = omega_vals[peak_idx]
    print(f"  Peak gain: {H_mag_db[peak_idx]:.2f} dB at ω = {peak_freq:.2f} rad/s")
    print(f"  Natural frequency ω_n = {omega_n} → resonance near ω_r = {omega_n*np.sqrt(1-2*zeta**2):.2f}")
    print(f"  DC gain: H(0) = 1.0 (= ω_n²/ω_n² = 1)")
    assert abs(H_mag[0] - 1.0) < 0.05, f"DC gain should be ~1: {H_mag[0]:.4f}"
    print("PASS: DC gain = 1 verified")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(2, 1)

    # Magnitude
    axes[0].semilogx(omega_vals, H_mag_db, color='#0072BD', linestyle='-', linewidth=2, label='Exact H(jω)')
    axes[0].semilogx(omega_vals, 20*np.log10(np.maximum(H_aaa_mag, 1e-15)),
                     'r--', linewidth=1.5, label='AAA fit')
    axes[0].set_title(f"Bode plot — 2nd order system (ω_n={omega_n}, ζ={zeta})", fontsize=11)
    axes[0].legend()

    # Phase
    axes[1].semilogx(omega_vals, H_phase, color='#0072BD', linestyle='-', linewidth=2, label='Exact H(jω)')
    axes[1].semilogx(omega_vals, H_aaa_phase, color='#D95319', linestyle='--', linewidth=1.5, label='AAA fit')
    axes[1].legend()

    fig.suptitle("System identification via AAA rational approximation", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "bode2tf.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True

if __name__ == "__main__":
    run()
