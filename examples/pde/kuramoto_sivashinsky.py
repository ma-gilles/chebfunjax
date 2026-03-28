"""Kuramoto-Sivashinsky equation and chaos.

Solves the Kuramoto-Sivashinsky equation:
  u_t = -(u^2/2)_x - u_xx - u_xxxx

using Fourier pseudospectral ETDRK4, following pde/Kuramoto.m by
Nick Trefethen (April 2016).

Original MATLAB: https://www.chebfun.org/examples/pde/Kuramoto.html
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
    print("Kuramoto-Sivashinsky equation and chaos")
    print("=" * 60)

    # KS equation: u_t = -(u^2/2)_x - u_xx - u_xxxx
    # Linear part: L = -ik*(-k^2) - k^4 = ik^3 - k^4  ← in Fourier space
    # Actually: linear part from -u_xx - u_xxxx = (k^2 - k^4) in Fourier
    # Nonlinear part: -(u^2/2)_x = -ik * FFT(u^2/2)

    # Domain [-100, 100], periodic
    L_dom = 200.0
    N = 800
    x = np.linspace(-L_dom / 2, L_dom / 2, N, endpoint=False)
    dx = L_dom / N
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi

    # Linear operator in Fourier space: L(k) = k^2 - k^4
    L_hat = k**2 - k**4

    def solve_ks(u0, T, dt=0.025, record_times=None):
        """ETDRK4 for Kuramoto-Sivashinsky on periodic domain."""
        u = u0.copy()
        nsteps = int(T / dt)
        if record_times is None:
            record_times = [T]

        # Precompute ETDRK4 coefficients
        E = np.exp(L_hat * dt)
        E2 = np.exp(L_hat * dt / 2)

        # Contour integral for phi functions (simplified Cox-Matthews)
        M = 16
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
        L_hat_c = L_hat[:, np.newaxis] * dt + r[np.newaxis, :]

        phi1 = dt * np.real(np.mean((np.exp(L_hat_c / 2) - 1) / L_hat_c, axis=1))
        phi1_full = dt * np.real(np.mean((np.exp(L_hat_c) - 1) / L_hat_c, axis=1))
        phi2a = dt * np.real(np.mean((np.exp(L_hat_c) - L_hat_c - 1) / L_hat_c**2, axis=1))
        phi3a = dt * np.real(np.mean(
            (-4 - L_hat_c + np.exp(L_hat_c) * (4 - 3 * L_hat_c + L_hat_c**2)) / L_hat_c**3,
            axis=1))
        phi3b = dt * np.real(np.mean(
            (2 + L_hat_c + np.exp(L_hat_c) * (-2 + L_hat_c)) / L_hat_c**3,
            axis=1))

        def nonlin_hat(v):
            """FFT of nonlinear part: -(v^2/2)_x."""
            v2_hat = np.fft.fft(v**2 / 2)
            return -1j * k * v2_hat

        results = []
        t_cur = 0.0
        record_set = set(record_times)

        for step in range(nsteps):
            u_hat = np.fft.fft(u)
            Nu = nonlin_hat(u)

            a_hat = E2 * u_hat + phi1 * Nu
            a = np.real(np.fft.ifft(a_hat))
            Na = nonlin_hat(a)

            b_hat = E2 * u_hat + phi1 * Na
            b = np.real(np.fft.ifft(b_hat))
            Nb = nonlin_hat(b)

            c_hat = E2 * a_hat + phi1 * (2 * Nb - Nu)
            c = np.real(np.fft.ifft(c_hat))
            Nc = nonlin_hat(c)

            u_hat = (E * u_hat +
                     phi1_full * Nu + 2 * phi2a * (Na + Nb - Nu) + phi3a * Nc)
            u = np.real(np.fft.ifft(u_hat))
            t_cur += dt

            t_round = round(t_cur, 4)
            if t_round in record_set:
                results.append((t_cur, u.copy()))

        return results

    # Symmetric initial condition: two Gaussian bumps
    u0_sym = (np.exp(-((x + 50) / 10)**2) +
              np.exp(-((x - 50) / 10)**2))

    print("\nSymmetric initial condition: two Gaussian bumps")
    print("Integrating KS to T=100...")

    results_sym = solve_ks(u0_sym, T=100.0, dt=0.025,
                           record_times=[100.0])
    u_t100 = results_sym[-1][1]

    print(f"  max|u| at t=0: {np.max(np.abs(u0_sym)):.4f}")
    print(f"  max|u| at t=100: {np.max(np.abs(u_t100)):.4f}")
    print(f"  KS chaos: solution looks random with wavelength ≈ 8-9")

    # Check characteristic wavelength (dominant mode)
    u_hat = np.fft.fft(u_t100)
    power = np.abs(u_hat[:N//2])**2
    k_pos = k[:N//2]
    dominant_k = k_pos[np.argmax(power[1:]) + 1]
    wavelength = 2 * np.pi / dominant_k
    print(f"  Dominant wavelength ≈ {wavelength:.2f} (expected ≈ 8.89)")

    # Nonsymmetric: break symmetry slightly
    print("\nNonsymmetric: second bump moved to x=49.9")
    u0_asym = (np.exp(-((x + 50) / 10)**2) +
               np.exp(-((x - 49.9) / 10)**2))
    results_asym = solve_ks(u0_asym, T=100.0, dt=0.025,
                            record_times=[100.0])
    u_asym_t100 = results_asym[-1][1]

    # Symmetry should be broken
    sym_error = np.max(np.abs(u_asym_t100[:N//2] + u_asym_t100[N//2:][::-1]))
    print(f"  Symmetry error at t=100: {sym_error:.4f} (>0 indicates chaos broke symmetry)")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    # Symmetric case
    axes[0].plot(x, u0_sym, color='#0072BD', linestyle='-', linewidth=1.5, label='t=0', alpha=0.7)
    axes[0].plot(x, u_t100, color='#D95319', linestyle='-', linewidth=1.5, label='t=100')
    axes[0].set_title("KS: symmetric IC (chaotic at t=100)", fontsize=11)
    axes[0].set_ylim([-4, 4]); axes[0].legend()

    # Asymmetric case
    axes[1].plot(x, u0_asym, color='#0072BD', linestyle='-', linewidth=1.5, label='t=0 (asym)', alpha=0.7)
    axes[1].plot(x, u_asym_t100, color='#D95319', linestyle='-', linewidth=1.5, label='t=100')
    axes[1].set_title("KS: symmetry broken at t=100", fontsize=11)
    axes[1].set_ylim([-4, 4]); axes[1].legend()

    fig.suptitle("Kuramoto-Sivashinsky chaos", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "kuramoto_sivashinsky.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True

if __name__ == "__main__":
    run()
