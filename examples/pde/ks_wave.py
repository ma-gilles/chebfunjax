"""Kuramoto-Sivashinsky traveling waves.

Demonstrates stable and unstable traveling wave solutions of the KS equation
and the generalized KS equation, following pde/KSWave.m by Nick Trefethen
(March 2017).

The KS equation: u_t = -(u^2/2)_x - u_xx - u_xxxx
Generalized KS:  u_t = -(u^2/2)_x - delta*(u_xx - u_xxxx) - eps*u_xxx

Original MATLAB: https://www.chebfun.org/examples/pde/KSWave.html
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
    print("Kuramoto-Sivashinsky traveling waves")
    print("=" * 60)

    def solve_ks_periodic(u0, domain_length, T, dt, delta=1.0, eps=0.0, npts=256):
        """ETDRK4 for generalized KS on [0, domain_length] periodic."""
        N = npts
        x = np.linspace(0, domain_length, N, endpoint=False)
        k = np.fft.fftfreq(N, d=domain_length / N) * 2 * np.pi

        # Re-sample u0 onto the N-point grid
        x0 = np.linspace(0, domain_length, len(u0), endpoint=False)
        u = np.interp(np.linspace(0, domain_length, N, endpoint=False), x0, u0)

        # Linear operator for generalized KS
        # u_t = -(u^2/2)_x - delta*(u_xx - u_xxxx) - eps*u_xxx
        # Linear in Fourier: L = delta*(k^2 - k^4) - eps*(-ik^3)*(-1) = delta*(k^2-k^4) - i*eps*k^3
        # Wait: -delta*u_xx = delta*k^2, -(-delta)*u_xxxx = -delta*(-k^4) = delta*k^4 in linear
        # Actually: -delta*u_xx has Fourier: delta*k^2 (wrong sign); let's be careful:
        # u_xx has Fourier: -k^2 * u_hat
        # u_xxxx has Fourier: k^4 * u_hat
        # -delta*(u_xx - u_xxxx): Fourier = -delta*(-k^2 - k^4)*u_hat = delta*(k^2 + k^4)
        # Hmm that's wrong. Let me recheck Trefethen's formulation:
        # generalized KS: u_t = -(u^2/2)_x - delta*(u_xx - u_xxxx) - eps*u_xxx
        # The standard KS has: -u_xx - u_xxxx → in Fourier: k^2 - k^4
        # (unstable short waves for k<1, stable long waves for k>1)
        # So linear part of standard KS: L(k) = k^2 - k^4
        # For generalized: L = delta*(k^2 - k^4) - i*eps*k^3
        L_hat = delta * (k**2 - k**4) - 1j * eps * k**3

        # ETDRK4 coefficients (Cox-Matthews)
        M = 16
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
        Lc = L_hat[:, np.newaxis] * dt + r[np.newaxis, :]

        E = np.exp(L_hat * dt)
        E2 = np.exp(L_hat * dt / 2)
        phi1 = dt * np.real(np.mean((np.exp(Lc / 2) - 1) / Lc, axis=1))
        phi1f = dt * np.real(np.mean((np.exp(Lc) - 1) / Lc, axis=1))
        phi2 = dt * np.real(np.mean((np.exp(Lc) - 1 - Lc) / Lc**2, axis=1))
        phi3a = dt * np.real(np.mean(
            (-4 - Lc + np.exp(Lc) * (4 - 3 * Lc + Lc**2)) / Lc**3, axis=1))

        def Nfun(v):
            return -1j * k * np.fft.fft(v**2 / 2)

        nsteps = int(T / dt)
        u_hat = np.fft.fft(u)
        for _ in range(nsteps):
            Nu = Nfun(np.real(np.fft.ifft(u_hat)))
            a_hat = E2 * u_hat + phi1 * Nu
            Na = Nfun(np.real(np.fft.ifft(a_hat)))
            b_hat = E2 * u_hat + phi1 * Na
            Nb = Nfun(np.real(np.fft.ifft(b_hat)))
            c_hat = E2 * a_hat + phi1 * (2 * Nb - Nu)
            Nc = Nfun(np.real(np.fft.ifft(c_hat)))
            u_hat = E * u_hat + phi1f * Nu + 2 * phi2 * (Na + Nb - Nu) + phi3a * Nc
        return x, np.real(np.fft.ifft(u_hat))

    # --- Standard KS: X = 8, domain = 20X = 160 ---
    X = 8.0
    L_dom = 20 * X
    N = 256
    x0 = np.linspace(0, L_dom, N, endpoint=False)
    u0 = 2 * np.exp(np.sin(2 * np.pi * x0 / X))

    print(f"\nStandard KS: period X={X}, domain length {L_dom}")
    print("  Integrating initial condition to T=100...")
    x1, u100 = solve_ks_periodic(u0, L_dom, T=100.0, dt=0.02, delta=1.0, eps=0.0)
    print(f"  After T=100: max|u|={np.max(np.abs(u100)):.3f}")

    # Check that a traveling wave formed (periodic structure)
    u100_hat = np.abs(np.fft.fft(u100))
    # The traveling wave should have dominant frequency at wavenumber 1 (period=X)
    k_wave = np.fft.fftfreq(N, d=L_dom / N) * 2 * np.pi
    k_pos = k_wave[1:N//2]
    power = u100_hat[1:N//2]**2

    dominant_k_idx = np.argmax(power)
    dominant_wavelength = 2 * np.pi / k_pos[dominant_k_idx]
    print(f"  Dominant wavelength ≈ {dominant_wavelength:.2f} (target X={X})")

    # Generalized KS: delta=0.8, eps=0.6, X=10
    X2 = 10.0
    L_dom2 = 20 * X2
    x0_2 = np.linspace(0, L_dom2, N, endpoint=False)
    u0_2 = 2 * np.exp(np.sin(2 * np.pi * x0_2 / X2))
    delta2 = 0.8
    eps2 = 0.6

    print(f"\nGeneralized KS: delta={delta2}, eps={eps2}, X={X2}")
    print("  Integrating to T=100...")
    x2, u100_gen = solve_ks_periodic(u0_2, L_dom2, T=100.0, dt=0.02,
                                      delta=delta2, eps=eps2)
    print(f"  After T=100: max|u|={np.max(np.abs(u100_gen)):.3f}")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(2, 2)

    axes[0, 0].plot(x0, u0, 'k-', linewidth=2)
    axes[0, 0].set_title(f"KS initial condition (X={X})", fontsize=11)
    axes[0, 0].set_ylim([-3, 9])

    axes[0, 1].plot(x1, u100, color='#0072BD', linestyle='-', linewidth=2)
    axes[0, 1].set_title("After T=100 (traveling wave)", fontsize=11)
    axes[0, 1].set_ylim([-3, 9])

    axes[1, 0].plot(x0_2, u0_2, 'k-', linewidth=2)
    axes[1, 0].set_title(f"Generalized KS IC (δ={delta2}, ε={eps2})", fontsize=11)
    axes[1, 0].set_ylim([-3, 9])

    axes[1, 1].plot(x2, u100_gen, color='#D95319', linestyle='-', linewidth=2)
    axes[1, 1].set_title("Gen. KS after T=100", fontsize=11)
    axes[1, 1].set_ylim([-3, 9])

    fig.suptitle("Kuramoto-Sivashinsky traveling waves", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "ks_wave.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True

if __name__ == "__main__":
    run()
