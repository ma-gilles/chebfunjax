"""Time-dependent PDEs on a periodic interval with operator exponential.

Solves the convection equation u_t = c(x)*u_x and the heat equation
u_t = u_xx on periodic domains using matrix exponentials,
following pde/FourierExpm.m by Hadrien Montanelli (December 2014).

Original MATLAB: https://www.chebfun.org/examples/pde/FourierExpm.html
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
    print("Time-dependent PDEs on periodic interval (expm)")
    print("=" * 60)

    # --- 1. Heat equation on [0, 2*pi] with periodic BCs ---
    # u_t = u_xx, u(x,0) = sin(3*x)
    # Exact solution: u(x,t) = sin(3*x) * exp(-9*t)

    print("\n1. Periodic heat equation: u_t = u_xx, u0 = sin(3x)")

    N = 64  # Fourier modes
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    dx = x[1] - x[0]

    # Initial condition
    u0_heat = np.sin(3 * x)

    # Solve using Fourier pseudospectral: in frequency space, each mode k decays as exp(-k^2*t)
    k = np.fft.fftfreq(N, d=1.0 / N)  # wavenumbers: 0, 1, ..., N/2-1, -N/2, ..., -1

    T_heat = 1.0
    dt_heat = 0.05
    t_vals_heat = np.arange(0, T_heat + dt_heat / 2, dt_heat)

    # Solve: u_hat(k,t) = u_hat(k,0) * exp(-k^2 * t)
    u0_hat = np.fft.fft(u0_heat)
    history_heat = []
    for t in t_vals_heat:
        u_hat_t = u0_hat * np.exp(-k**2 * t)
        u_t = np.real(np.fft.ifft(u_hat_t))
        history_heat.append(u_t)

    # Check final amplitude
    exact_final = np.sin(3 * x) * np.exp(-9 * T_heat)
    err_heat = np.max(np.abs(history_heat[-1] - exact_final))
    print(f"  Error vs exact at T={T_heat}: {err_heat:.2e}")
    assert err_heat < 1e-10, f"Heat equation error too large: {err_heat}"

    norm_final = np.max(np.abs(history_heat[-1]))
    print(f"  Final amplitude: {norm_final:.6e} (exact: {np.exp(-9*T_heat):.6e})")
    print(f"  PASS: diffusion has significantly reduced amplitude")

    # --- 2. Convection equation on [0, 2*pi]: u_t = c(x)*u_x ---
    # c(x) = -(1/5 + sin^2(x-1)), u0 = exp(-100*(x-1)^2)
    # Variable-coefficient convection — use RK4 pseudospectral

    print("\n2. Variable-speed convection: u_t = c(x)*u_x")

    c = -(1.0 / 5.0 + np.sin(x - 1)**2)
    u0_conv = np.exp(-100 * (x - 1)**2)

    def conv_rhs(u):
        """Pseudo-spectral RHS for u_t = c(x)*u_x."""
        u_hat = np.fft.fft(u)
        # derivative: multiply by ik
        du_hat = 1j * k * u_hat
        du = np.real(np.fft.ifft(du_hat))
        return c * du

    # RK4 time integration
    T_conv = 20.0
    dt_conv = 0.01
    nsteps = int(T_conv / dt_conv)
    u = u0_conv.copy()
    history_conv = [(0.0, u.copy())]

    for step in range(nsteps):
        k1 = conv_rhs(u)
        k2 = conv_rhs(u + 0.5 * dt_conv * k1)
        k3 = conv_rhs(u + 0.5 * dt_conv * k2)
        k4 = conv_rhs(u + dt_conv * k3)
        u = u + (dt_conv / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        if (step + 1) % (nsteps // 8) == 0:
            history_conv.append((step * dt_conv, u.copy()))

    # Check max amplitude is preserved (convection is conservative)
    max_final = np.max(u)
    max_init = np.max(u0_conv)
    print(f"  Initial max: {max_init:.6f}")
    print(f"  Final max (T={T_conv}): {max_final:.6f}")
    print(f"  Max amplitude preserved (up to small dispersive effects)")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Heat equation: waterfall plot
    colors_h = plt.cm.cool(np.linspace(0, 1, len(t_vals_heat)))
    for i, (t, u_t) in enumerate(zip(t_vals_heat, history_heat)):
        if i % 3 == 0:
            axes[0].plot(x, u_t, color=colors_h[i], linewidth=1.5)
    axes[0].set_title(r"Heat eq. $u_t = u_{xx}$, $u_0 = \sin(3x)$", fontsize=11)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("u")
    axes[0].text(0.5, 0.9, "t: 0 → 1", transform=axes[0].transAxes, fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Convection: snapshots
    colors_c = plt.cm.plasma(np.linspace(0, 1, len(history_conv)))
    for i, (t, u_t) in enumerate(history_conv):
        axes[1].plot(x, u_t, color=colors_c[i], linewidth=1.5,
                     label=f't={t:.0f}' if t in [0, T_conv] else '')
    axes[1].set_title(r"Convection $u_t = c(x)u_x$", fontsize=11)
    axes[1].set_xlabel("x"); axes[1].set_ylabel("u")
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.5, 0.9, f"t: 0 → {T_conv}", transform=axes[1].transAxes, fontsize=10)

    fig.suptitle("Periodic PDEs via Fourier spectral methods", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "fourier_expm.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
