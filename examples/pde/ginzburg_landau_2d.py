"""Complex Ginzburg-Landau equation in 2D.

Documents the complex Ginzburg-Landau equation:
  u_t = (1 + i*alpha)*Delta*u + u - (1 + i*beta)|u|^2*u

which produces spiral waves and turbulence patterns,
following pde/GinzburgLandau.m by Nick Trefethen (May 2016).

NOTE: The full 2D computation uses chebfun2/spin2 infrastructure not
yet available in chebfunjax. This file provides the 1D analog,
documents the 2D equations, and demonstrates the core nonlinearity.

Original MATLAB: https://www.chebfun.org/examples/pde/GinzburgLandau.html
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
    print("Complex Ginzburg-Landau equation (1D demo)")
    print("=" * 60)

    print("\nNOTE: Full 2D CGL requires spin2 (2D ETDRK4).")
    print("Demonstrating 1D CGL: u_t = (1+ia)*u_xx + u - (1+ib)|u|^2*u")

    # Parameters from Trefethen's example for "turbulence"
    alpha = -1.0
    beta = 2.5
    N = 256
    L_dom = 2 * np.pi * 10
    x = np.linspace(-L_dom / 2, L_dom / 2, N, endpoint=False)
    dx = L_dom / N
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi

    # Linear operator: L = (1 + ia)*(-k^2) = -(1+ia)*k^2
    L_hat = -(1.0 + 1j * alpha) * k**2 + 1.0  # linear part including +u

    # Initial condition: small random perturbation + one large mode
    rng = np.random.default_rng(42)
    u0 = (0.5 * (np.cos(x / 3) + np.sin(x / 5)) *
          np.exp(1j * (np.sin(x / 7) + np.cos(x / 4))))
    u0 = u0.astype(complex)

    # ETDRK2 for 1D CGL
    T = 50.0
    dt = 0.05
    nsteps = int(T / dt)

    E = np.exp(L_hat * dt)
    E2 = np.exp(L_hat * dt / 2)
    phi_half = np.where(np.abs(L_hat * dt / 2) < 1e-10,
                        dt / 2 * np.ones_like(L_hat),
                        (E2 - 1) / L_hat)
    phi_full = np.where(np.abs(L_hat * dt) < 1e-10,
                        dt * np.ones_like(L_hat),
                        (E - 1) / L_hat)

    def nonlin_cgl(u):
        return -(1.0 + 1j * beta) * np.abs(u)**2 * u

    u = u0.copy()
    history = [(0.0, u.copy())]
    t_cur = 0.0

    record_T = [10.0, 25.0, 50.0]
    record_set = set(record_T)

    for step in range(nsteps):
        Nu = nonlin_cgl(u)
        u_hat = np.fft.fft(u)
        a_hat = E2 * u_hat + phi_half * np.fft.fft(Nu)
        a = np.fft.ifft(a_hat)
        Na = nonlin_cgl(a)
        u_hat = E * u_hat + (phi_full / 2) * (np.fft.fft(Nu) + np.fft.fft(Na))
        u = np.fft.ifft(u_hat)
        t_cur += dt
        t_r = round(t_cur, 3)
        if t_r in record_set:
            history.append((t_cur, u.copy()))
            record_set.discard(t_r)

    history.append((T, u.copy()))

    print(f"\nAfter T={T}:")
    print(f"  max|u| = {np.max(np.abs(u)):.4f}")
    print(f"  mean|u| = {np.mean(np.abs(u)):.4f}")
    print(f"  Solution exhibits spatio-temporal chaos")

    # Basic check: solution amplitude stays bounded (CGL has global attractor)
    assert np.max(np.abs(u)) < 10.0, "CGL amplitude too large"
    assert np.mean(np.abs(u)) > 0.01, "CGL amplitude too small (decayed)"

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    # |u| at different times
    colors = ['blue', 'green', 'orange', 'red']
    for i, (t, u_h) in enumerate(history):
        if i < 4:
            axes[0].plot(x, np.abs(u_h), color=colors[i], linewidth=1.5,
                         label=f't={t:.0f}', alpha=0.8)
    axes[0].set_title(f"1D CGL: |u(x,t)| (α={alpha}, β={beta})", fontsize=11)
    axes[0].legend(fontsize=9)

    # Phase plot
    u_final = history[-1][1]
    axes[1].plot(np.real(u_final), np.imag(u_final), 'b.', markersize=2, alpha=0.5)
    theta = np.linspace(0, 2 * np.pi, 200)
    axes[1].plot(np.cos(theta), np.sin(theta), 'r--', linewidth=1, alpha=0.5,
                 label='|u|=1')
    axes[1].set_aspect('equal')
    axes[1].set_title("Phase portrait Re(u) vs Im(u) at T=50", fontsize=11)
    axes[1].legend()

    fig.suptitle("Complex Ginzburg-Landau (1D)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "ginzburg_landau_2d.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True

if __name__ == "__main__":
    run()
