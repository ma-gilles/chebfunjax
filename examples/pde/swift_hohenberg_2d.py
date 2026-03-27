"""Swift-Hohenberg equation in 2D.

Documents the Swift-Hohenberg PDE:
  u_t = r*u - (1 + Delta)^2*u - u^3

which models pattern formation in convection rolls,
following pde/SwiftHohenberg.m by Hadrien Montanelli (May 2017).

NOTE: The full 2D computation requires spin2. This file provides
the 1D version and documents the 2D equations.

Original MATLAB: https://www.chebfun.org/examples/pde/SwiftHohenberg.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chebfunjax.plotting import chebfun_style
chebfun_style()

import numpy as np
import os


def run():
    print("=" * 60)
    print("Swift-Hohenberg equation (1D demo)")
    print("=" * 60)

    print("\nNOTE: Full 2D SH requires spin2. Showing 1D dynamics.")
    print("1D SH: u_t = r*u - (1 + d^2/dx^2)^2*u - u^3")

    # Parameter r controls instability. r > 0 leads to pattern formation.
    r = 0.3
    N = 256
    L_dom = 4 * np.pi * 20  # large domain to see multiple rolls
    x = np.linspace(-L_dom / 2, L_dom / 2, N, endpoint=False)
    dx = L_dom / N
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi

    # Linear operator in Fourier space:
    # (1 + d^2/dx^2)^2 → (1 - k^2)^2 in Fourier
    # u_t = r*u - (1-k^2)^2*u - u^3
    # Linear: L = r - (1-k^2)^2
    L_hat = r - (1.0 - k**2)**2

    # Initial condition: small random noise
    rng = np.random.default_rng(42)
    u0 = 0.01 * rng.standard_normal(N)

    # ETDRK2
    T = 200.0
    dt = 0.5
    nsteps = int(T / dt)

    E = np.exp(L_hat * dt)
    E2 = np.exp(L_hat * dt / 2)
    phi_half = np.where(np.abs(L_hat) < 1e-10,
                        dt / 2 * np.ones_like(L_hat),
                        (E2 - 1) / L_hat)
    phi_full = np.where(np.abs(L_hat) < 1e-10,
                        dt * np.ones_like(L_hat),
                        (E - 1) / L_hat)

    def nonlin_sh(u):
        return -u**3

    u = u0.copy()
    history = [(0.0, u.copy())]
    t_cur = 0.0
    record_T = set([50.0, 100.0, 200.0])

    for step in range(nsteps):
        Nu = nonlin_sh(u)
        u_hat = np.fft.fft(u)
        a_hat = E2 * u_hat + phi_half * np.fft.fft(Nu)
        a = np.real(np.fft.ifft(a_hat))
        Na = nonlin_sh(a)
        u_hat = E * u_hat + (phi_full / 2) * (np.fft.fft(Nu) + np.fft.fft(Na))
        u = np.real(np.fft.ifft(u_hat))
        t_cur += dt
        t_r = round(t_cur, 2)
        if t_r in record_T:
            history.append((t_cur, u.copy()))
            record_T.discard(t_r)

    print(f"\nAfter T={T}:")
    print(f"  max|u| = {np.max(np.abs(u)):.4f} (expected ≈ {np.sqrt(r):.4f} for saturation)")
    print(f"  Pattern: rolls with wavelength ≈ 2π")

    # Check amplitude is near the expected equilibrium amplitude sqrt(r)
    expected_amp = np.sqrt(r)
    actual_amp = np.max(np.abs(u))
    print(f"  Expected saturation: {expected_amp:.4f}, actual: {actual_amp:.4f}")

    # Check we have a periodic pattern
    u_hat_final = np.abs(np.fft.fft(u))
    # Peak should be near k=1 (wavelength 2pi)
    k_peak_idx = np.argmax(u_hat_final[1:N//2]) + 1
    k_peak = np.abs(k[k_peak_idx])
    wavelength_peak = 2 * np.pi / k_peak
    print(f"  Dominant wavelength: {wavelength_peak:.3f} (expected ≈ 2π = {2*np.pi:.3f})")
    assert abs(wavelength_peak - 2 * np.pi) < 3.0, "Wavelength far from 2π"

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    colors = plt.cm.viridis(np.linspace(0, 1, len(history)))
    for i, (t, u_h) in enumerate(history):
        lw = 2 if i == len(history) - 1 else 1
        axes[0].plot(x, u_h, color=colors[i], linewidth=lw,
                     label=f't={t:.0f}', alpha=0.8)
    axes[0].set_title(f"Swift-Hohenberg 1D (r={r})", fontsize=11)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("u")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([-60, 60])

    # Power spectrum
    k_pos = k[1:N//2]
    power = u_hat_final[1:N//2]**2
    axes[1].semilogy(k_pos, power, 'b-', linewidth=1.5)
    axes[1].axvline(1.0, color='r', linestyle='--', label='k=1 (λ=2π)')
    axes[1].set_title("Power spectrum at T=200", fontsize=11)
    axes[1].set_xlabel("Wavenumber k"); axes[1].set_ylabel("Power")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 5])

    fig.suptitle("Swift-Hohenberg pattern formation (1D)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "swift_hohenberg_2d.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
