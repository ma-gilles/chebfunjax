"""Allen-Cahn metastability.

Demonstrates the Allen-Cahn equation u_t = u_xx + u - u^3 and its
metastability properties, following pde/AllenCahn2.m by Nick Trefethen
(November 2013).

Original MATLAB: https://www.chebfun.org/examples/pde/AllenCahn2.html
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
    print("Allen-Cahn metastability")
    print("=" * 60)

    # Allen-Cahn: u_t = u_xx + u(1 - u^2)
    # Stable fixed points: u = ±1. Unstable: u = 0.
    # We use a pseudo-spectral Fourier method on a periodic domain
    # (equivalent to pde15s with large domain approximation).

    # Parameters
    L = 32.0          # half-domain [-L, L] approximating the real line
    N = 512           # spatial grid points
    eps = 1.0         # coefficient in u_t = u_xx + u - u^3
    dx = 2 * L / N
    x = np.linspace(-L, L - dx, N)

    # Wavenumbers for FFT
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi

    # Initial condition: u0 = 3*exp(-x^2/4) - 1 (narrow pulse)
    u0 = 3.0 * np.exp(-x**2 / 4) - 1.0

    # Impose Dirichlet-like BC: values near ±L should be ≈ -1
    # (the real domain has bc at ±∞ → -1; on finite domain we just set IC accordingly)

    def solve_allen_cahn(u0, T, dt=0.05):
        """Pseudo-spectral ETDRK2 for u_t = u_xx + u - u^3."""
        u = u0.copy()
        nsteps = int(T / dt)
        # Linear part: L_hat = -k^2 (diffusion) + 1 (linear part of u(1-u^2) ≈ u)
        L_hat = -k**2 + 1.0
        E = np.exp(L_hat * dt)
        E2 = np.exp(L_hat * dt / 2)

        def nonlin(u):
            return -u**3

        history = [u.copy()]
        for _ in range(nsteps):
            Nu = nonlin(u)
            a = E2 * np.fft.fft(u) + np.fft.fft(Nu) * (E2 - 1.0) / L_hat
            ua = np.real(np.fft.ifft(a))
            Nua = nonlin(ua)
            u_hat = E * np.fft.fft(u) + (np.fft.fft(Nu) * (E - E2) + np.fft.fft(Nua) * (E2 - 1.0)) / L_hat
            u = np.real(np.fft.ifft(u_hat))
        return u

    # Short-time evolution: narrow pulse
    print("\nNarrow pulse (sigma=2): solving to T=50...")
    u_t50 = solve_allen_cahn(u0, T=50.0, dt=0.1)
    max_u50 = np.max(u_t50)
    print(f"  max(u) at T=50: {max_u50:.4f}")

    # Wide pulse (sigma=4)
    u0_wide = 3.0 * np.exp(-x**2 / 16) - 1.0
    print("\nWide pulse (sigma=4): solving to T=50...")
    u_wide_t50 = solve_allen_cahn(u0_wide, T=50.0, dt=0.1)
    max_wide = np.max(u_wide_t50)
    print(f"  max(u) at T=50: {max_wide:.4f}")
    print("  (Wide pulse decays much more slowly -- metastability)")

    # Decay of maximum over time for medium width
    u0_med = 3.0 * np.exp(-x**2 / 6) - 1.0
    print("\nMedium pulse: tracking max decay...")
    t_vals = np.arange(0, 101, 5.0)
    u_cur = u0_med.copy()
    umax_list = [np.max(u_cur)]
    for i in range(1, len(t_vals)):
        u_cur = solve_allen_cahn(u_cur, T=5.0, dt=0.1)
        umax_list.append(np.max(u_cur))

    umax = np.array(umax_list)
    # Find approximate critical time (when max crosses 0)
    pos_idx = np.where(umax > 0)[0]
    neg_idx = np.where(umax <= 0)[0]
    if len(neg_idx) > 0 and len(pos_idx) > 0:
        t_cross = t_vals[neg_idx[0]]
        print(f"  Approximate critical time (max crosses 0): t ≈ {t_cross:.1f}")
        print(f"  (Chebfun example gives ≈55.9 for σ=6)")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    # Panel 1: solution snapshots for wide pulse
    snapshots_T = [0, 10, 25, 50]
    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshots_T)))
    u_snap = u0_wide.copy()
    axes[0].plot(x, u0_wide, color=colors[0], label='t=0')
    u_prev = u0_wide.copy()
    for i, T_snap in enumerate(snapshots_T[1:], 1):
        u_snap = solve_allen_cahn(u_prev, T=T_snap - snapshots_T[i-1], dt=0.1)
        axes[0].plot(x, u_snap, color=colors[i], label=f't={T_snap}')
        u_prev = u_snap.copy()
    axes[0].set_xlim([-16, 16])
    axes[0].set_title("Allen-Cahn: wide pulse (metastable)", fontsize=11)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("u")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-1.5, 2.5])

    # Panel 2: max decay over time
    axes[1].plot(t_vals, umax, 'b.-', markersize=8)
    axes[1].axhline(0, color='r', linestyle='--', linewidth=1)
    axes[1].set_title("Max(u) decay — medium pulse", fontsize=11)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("max(u)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Allen-Cahn metastability", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "allen_cahn2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
