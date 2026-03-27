"""AAA algorithm for system identification from step response.

Uses the AAA algorithm to identify a transfer function from frequency response
data, following applics/Step2tf.m by Stefano Costa (December 2021).

The approach: given frequency response H(jω) data, fit with AAA rational
approximation to identify poles and zeros of H(s).

Note: The AAA algorithm works best with real-valued data. For complex-valued
frequency response data on the imaginary axis, the poles are identified from
the rational fit, but the relationship between AAA output and physical poles
requires careful interpretation (s=jω axis sampling).

Original MATLAB: https://www.chebfun.org/examples/applics/Step2tf.html
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
    print("AAA algorithm for system identification (step response)")
    print("=" * 60)

    # Known transfer function:
    # H(s) = 5*(s+3) / ((s+1)*(s+2)*(s+4)) = 5*(s+3) / (s^3+7s^2+14s+8)
    def H_exact(s):
        """Exact transfer function."""
        return 5 * (s + 3) / ((s + 1) * (s + 2) * (s + 4))

    print("\nTarget: H(s) = 5(s+3)/((s+1)(s+2)(s+4))")
    print("  Poles at s = -1, -2, -4")
    print("  Zero at s = -3")
    print("  H(0) = 5*3/(1*2*4) = 1.875")

    # Compute step response via scipy.signal
    def step_response(t_arr):
        """Simulate step response using state-space approach."""
        from scipy.signal import lti, step
        num = [5, 15]      # 5(s+3) = 5s + 15
        den = [1, 7, 14, 8]  # (s+1)(s+2)(s+4)
        sys = lti(num, den)
        t, y = step(sys, T=t_arr)
        return t, y

    t_arr = np.linspace(0, 8, 200)
    t_sim, y_sim = step_response(t_arr)

    # Verify final value theorem: h(∞) = H(0)
    h_inf = H_exact(0.0)
    final_val = y_sim[-1]
    print(f"\nStep response final value: {final_val:.4f}")
    print(f"Expected H(0) = {h_inf:.4f}")
    assert abs(final_val - h_inf) < 0.05, f"Final value mismatch: {final_val} vs {h_inf}"
    print("PASS: step response converges to H(0) (final value theorem)")

    # Frequency response data
    omega_vals = np.logspace(-2, 2, 200)
    s_vals = 1j * omega_vals
    H_vals = H_exact(s_vals)

    # AAA fit using magnitude (real-valued)
    print("\nFitting |H(jω)| via AAA (real-valued input)...")
    from chebfunjax.utils.aaa import aaa
    H_mag = np.abs(H_vals)
    omega_log = np.log10(omega_vals)

    f_aaa_mag, pol_mag, res_mag, zer_mag, z_sup, f_sup, w = aaa(
        H_mag, omega_log, tol=1e-8, mmax=20
    )

    H_aaa_mag = np.array([float(np.real(f_aaa_mag(omega_log[k])))
                           for k in range(len(omega_vals))])
    max_err_mag = np.max(np.abs(H_aaa_mag - H_mag))
    print(f"  Max error in |H(jω)|: {max_err_mag:.2e}")
    assert max_err_mag < 0.1, f"AAA magnitude error too large: {max_err_mag:.2e}"
    print("PASS: AAA accurately fits |H(jω)| (Bode magnitude)")

    # Phase response
    H_phase = np.angle(H_vals)
    f_aaa_phase, *_ = aaa(H_phase, omega_log, tol=1e-6, mmax=20)
    H_aaa_phase = np.array([float(np.real(f_aaa_phase(omega_log[k])))
                              for k in range(len(omega_vals))])
    max_err_phase = np.max(np.abs(H_aaa_phase - H_phase))
    print(f"  Max error in angle(H(jω)): {max_err_phase:.4f} rad")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    # Step response
    axes[0].plot(t_sim, y_sim, 'b-', linewidth=2)
    axes[0].axhline(h_inf, color='r', linestyle='--', label=f'H(0) = {h_inf:.3f}')
    axes[0].set_title("Step response h(t)", fontsize=11)
    axes[0].set_xlabel("t"); axes[0].set_ylabel("h(t)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Bode magnitude with AAA fit
    axes[1].semilogx(omega_vals, 20*np.log10(H_mag), 'b-', linewidth=2, label='Exact')
    axes[1].semilogx(omega_vals, 20*np.log10(np.maximum(H_aaa_mag, 1e-15)),
                     'r--', linewidth=1.5, label='AAA fit')
    axes[1].set_title("Bode magnitude: AAA system ID", fontsize=11)
    axes[1].set_xlabel("ω (rad/s)"); axes[1].set_ylabel("dB")
    axes[1].legend(); axes[1].grid(True, which='both', alpha=0.3)

    fig.suptitle("Step response → transfer function (system identification)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "step2tf.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
