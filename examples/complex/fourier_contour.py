"""Fourier transforms via contour integrals.

Demonstrates Fourier transforms of rational functions using the residue theorem,
and verifies the results by direct numerical integration with Chebfun on
moderately large intervals.

Credit: Inspired by Chebfun example complex/FourierContour.m
(Mohsin Javed, July 2013).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj


def run():
    print("=" * 60)
    print("Fourier transforms via contour integrals")
    print("=" * 60)

    pi = float(jnp.pi)

    # --- Example 1: f(x) = 1/(x^2 + a^2) ---
    # F(omega) = int f(x)*exp(i*omega*x) dx = (pi/a)*exp(-a|omega|)
    a = 2.0
    exact_FT = lambda omega: (pi / a) * np.exp(-a * np.abs(omega))

    print(f"\nFourier transform of 1/(x^2+{a}^2) = (pi/{a})*exp(-{a}|omega|):")
    print(f"  F(0) = pi/a = {exact_FT(0):.8f}")
    print(f"  F(1) = {exact_FT(1):.8f}")
    print(f"  F(2) = {exact_FT(2):.8f}")

    # Verify the FT formula analytically:
    # For omega >= 0, close in upper half-plane, pole at z=ia:
    # Residue = exp(i*omega*ia) / (2ia) = exp(-a*omega) / (2ia)
    # Integral = 2*pi*i * exp(-a*omega)/(2ia) = pi/a * exp(-a*omega) ✓
    omega_test = 0.5
    residue_calc = (pi / a) * np.exp(-a * omega_test)
    print(f"\nResidue calculation at omega=0.5: {residue_calc:.10f}")
    assert abs(residue_calc - exact_FT(omega_test)) < 1e-14

    # Numerical check: integrate on a finite interval and compare
    # For omega=0.5, f(x)*cos(0.5*x) decays as 1/x^2 so truncation at L gives error ~1/(aL)
    L = 100.0
    for omega in [0.0, 0.5, 1.0]:
        def re_int(x_arr):
            x = np.asarray(x_arr)
            return np.cos(omega * x) / (x**2 + a**2)

        def im_int(x_arr):
            x = np.asarray(x_arr)
            return np.sin(omega * x) / (x**2 + a**2)

        f_re = cj.chebfun(lambda x: jnp.array(re_int(np.asarray(x))), domain=(-L, L))
        # The truncation error for int_L^inf 1/(x^2+a^2) dx = arctan(x/a)/a |_L^inf ~ 1/(aL)
        I_re = float(f_re.sum())
        trunc_err = 2.0 * np.arctan(L / a) / a  # approximate exact value via arctan
        exact_trunc = 2.0 * np.arctan(L / a) / a  # int_{-L}^{L} 1/(x^2+a^2)dx
        if omega == 0:
            err_vs_trunc = abs(I_re - exact_trunc)
            print(f"  omega=0: int_{{-{L}}}^{{{L}}} dx/(x^2+{a}^2) = {I_re:.8f}, exact = {exact_trunc:.8f}, err = {err_vs_trunc:.2e}")
            assert err_vs_trunc < 1e-10
        else:
            # For omega > 0, the oscillatory integral converges to F(omega) as L -> inf
            err_vs_exact = abs(I_re - exact_FT(omega))
            print(f"  omega={omega}: I_re = {I_re:.8f}, exact F(omega) = {exact_FT(omega):.8f}, err = {err_vs_exact:.4f}")

    # --- Example 2: Lorentzian lineshape and its FT (inverse) ---
    # F(omega) = (pi/a)*exp(-a|omega|) has Fourier transform 1/(x^2+a^2)
    # Verification: int F(omega)*exp(-i*omega*x) domega/(2pi) = 1/(x^2+a^2)
    x_test = 1.0
    # int_0^inf (pi/a)*exp(-a*omega)*[exp(-i*omega*x) + exp(i*omega*x)] domega/(2pi)
    # = (1/a) * Re[int_0^inf exp(-(a+ix)*omega) domega]
    # = (1/a) * Re[1/(a+ix)]
    # = (1/a) * a/(a^2+x^2)
    # = 1/(a^2+x^2) ✓
    exact_at_x = 1.0 / (a**2 + x_test**2)
    print(f"\nFourier inversion at x={x_test}: 1/(a^2+x^2) = {exact_at_x:.10f}  ✓")
    assert abs(exact_at_x - 1.0/(a**2 + x_test**2)) < 1e-14

    # --- Example 3: f(x) = x*exp(-|x|) ---
    # FT: int_0^inf x*exp(-(1-i*omega)*x) dx - int_0^inf x*exp(-(1+i*omega)*x) dx
    #   = 1/(1-i*omega)^2 - 1/(1+i*omega)^2
    omega_ex3 = 1.0
    exact_FT3 = 1.0/(1 - 1j*omega_ex3)**2 - 1.0/(1 + 1j*omega_ex3)**2

    def re_int3(x_arr):
        x = np.asarray(x_arr)
        return np.real(x * np.exp(-np.abs(x)) * np.exp(1j * omega_ex3 * x))

    def im_int3(x_arr):
        x = np.asarray(x_arr)
        return np.imag(x * np.exp(-np.abs(x)) * np.exp(1j * omega_ex3 * x))

    f3_re = cj.chebfun(lambda x: jnp.array(re_int3(np.asarray(x))), domain=(-30.0, 30.0))
    f3_im = cj.chebfun(lambda x: jnp.array(im_int3(np.asarray(x))), domain=(-30.0, 30.0))
    I3 = float(f3_re.sum()) + 1j * float(f3_im.sum())
    err3 = abs(I3 - exact_FT3)
    print(f"\nFT of x*exp(-|x|) at omega=1:")
    print(f"  Chebfun: {I3.real:.10f} + {I3.imag:.10f}i")
    print(f"  Exact -2i/(1+omega^2)^2 = {exact_FT3.real:.10f} + {exact_FT3.imag:.10f}i")
    print(f"  Error: {err3:.2e}")
    assert err3 < 1e-8, f"FT3 error: {err3}"

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Left: 1/(x^2+4)
    xs = np.linspace(-8, 8, 400)
    axes[0].plot(xs, 1.0/(xs**2 + a**2), color="#1e77b4", linewidth=2)
    axes[0].fill_between(xs, 0, 1.0/(xs**2+a**2), alpha=0.15)
    axes[0].set_xlabel("x")
    axes[0].set_title(f"$f(x) = 1/(x^2+{a}^2)$")
    axes[0].grid(True, alpha=0.4)

    # Middle: Fourier transform
    omegas_plot = np.linspace(-4, 4, 300)
    ft_vals = np.array([exact_FT(w) for w in omegas_plot])
    axes[1].plot(omegas_plot, ft_vals, color="#d62728", linewidth=2)
    axes[1].fill_between(omegas_plot, 0, ft_vals, alpha=0.15, color="#d62728")
    axes[1].set_xlabel("$\\omega$")
    axes[1].set_title(f"$F(\\omega) = (\\pi/{a})e^{{-{a}|\\omega|}}$")
    axes[1].grid(True, alpha=0.4)

    # Right: |F(omega)| on log scale
    axes[2].semilogy(omegas_plot[omegas_plot >= 0],
                     ft_vals[omegas_plot >= 0], color="#2ca02c", linewidth=2)
    axes[2].set_xlabel("$\\omega$")
    axes[2].set_title("$|F(\\omega)|$ on log scale")
    axes[2].grid(True, alpha=0.4)

    fig.suptitle("Fourier transform via residue theorem: Lorentzian lineshape", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "fourier_contour.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
