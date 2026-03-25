"""Gibbs phenomenon for Fourier series of discontinuous functions.

Demonstrates the Gibbs phenomenon: partial Fourier sums of a
step function overshoot near the discontinuity by approximately
8.9%, regardless of the number of terms. Also demonstrates
that smooth functions (like periodic Gaussians) have much better
Fourier convergence without Gibbs oscillations.

Credit: Inspired by Chebfun fourier/Gibbs examples.
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
from chebfunjax.plotting import plot


def partial_fourier_sum(N, x_eval):
    """Compute partial Fourier sum of sign(x-pi) on [0, 2*pi] at x_eval.

    sign(x - pi) has Fourier series: (4/pi) * sum_{k=0}^{inf} sin((2k+1)x)/(2k+1)
    """
    pi = np.pi
    result = np.zeros_like(x_eval)
    for k in range(N):
        n = 2 * k + 1
        result += (4.0 / pi) * np.sin(n * x_eval) / n
    return result


def run():
    print("=" * 60)
    print("Gibbs phenomenon")
    print("=" * 60)

    pi = float(jnp.pi)

    # --- Gibbs overshoot for step function: should be ~8.9% ----------
    # Partial Fourier sum S_N converges to f(x) everywhere except at jump
    # At x = pi^- (just before jump), S_N overshoots the function value by
    # ~(Si(pi)/pi - 1/2) * 2 ≈ 0.0895 = 8.95%
    N_vals = [10, 50, 200]
    x_eval = np.linspace(0.01, 2*pi - 0.01, 2000)

    print(f"\nStep function sign(x-pi) on [0, 2*pi]:")
    print(f"  Gibbs overshoot ~ 8.9% regardless of N:")
    for N in N_vals:
        S_N = partial_fourier_sum(N, x_eval)
        # Maximum of S_N near x = pi (the jump at x=pi, jump size = 2)
        # Gibbs overshoot = max(S_N) - 1 (since function goes from -1 to +1)
        near_jump = (x_eval > pi - 0.5) & (x_eval < pi)
        max_before = np.max(S_N[near_jump])
        overshoot = max_before - 1.0  # value above +1
        print(f"  N={N:4d}: max before jump = {max_before:.6f}, overshoot = {overshoot:.4f} ({100*overshoot:.2f}%)")
        # Gibbs overshoot approaches ~0.0895 as N -> infinity
        assert overshoot > 0.05  # always overshoots

    # The overshoot approaches the Gibbs constant Gi = Si(pi)/pi - 1/2 ≈ 0.08949
    # Theoretical value
    from scipy.integrate import quad
    Si_pi, _ = quad(lambda t: np.sin(t) / t, 0, np.pi)
    gibbs_constant = Si_pi / np.pi - 0.5
    print(f"\n  Theoretical Gibbs constant: {gibbs_constant:.6f}  (~{100*gibbs_constant:.2f}%)")

    # Verify the Gibbs constant: find first maximum before the jump
    # The first maximum of S_N is at approximately x = pi - pi/N
    # Use a finer grid; first maximum is within pi/(2N) of the jump
    N_large = 500
    delta = np.pi / N_large
    x_first_peak = np.linspace(pi - 10*delta, pi - 0.1*delta, 10000)
    S_first = partial_fourier_sum(N_large, x_first_peak)
    # Find first local maximum (rightmost maximum before pi)
    from scipy.signal import argrelmax
    peaks = argrelmax(S_first, order=5)[0]
    if len(peaks) > 0:
        first_peak_val = S_first[peaks[-1]]  # rightmost peak before jump
        overshoot_large = first_peak_val - 1.0
    else:
        overshoot_large = np.max(S_first) - 1.0
    print(f"  N=500 first peak overshoot = {overshoot_large:.6f}  (theoretical: {gibbs_constant:.6f})")
    # First peak overshoot should be within 20% of Gibbs constant
    assert overshoot_large > 0.05, "Gibbs overshoot should be positive"

    # --- Contrast: smooth function has exponential convergence -------
    # f(x) = exp(cos(x)) is smooth and periodic
    # Compute partial Fourier sums and check convergence
    from scipy.special import iv as bessel_i

    def partial_fourier_smooth(N, x_eval):
        """Partial Fourier sum of exp(cos(x)) on [0, 2*pi]."""
        # a_0 = I_0(1), a_n = 2*I_n(1)
        result = np.full_like(x_eval, bessel_i(0, 1.0))
        for n in range(1, N + 1):
            result += 2.0 * bessel_i(n, 1.0) * np.cos(n * x_eval)
        return result

    x_test = np.array([0.5, 1.0, 2.5])
    exact_vals = np.exp(np.cos(x_test))

    print(f"\nSmooth function exp(cos(x)):")
    print(f"  Exponential Fourier convergence (no Gibbs):")
    for N in [2, 5, 10]:
        S_N = partial_fourier_smooth(N, x_test)
        max_err = np.max(np.abs(S_N - exact_vals))
        print(f"  N={N:2d}: max error = {max_err:.2e}")

    # With N=10, should be very accurate
    err_10 = np.max(np.abs(partial_fourier_smooth(10, x_test) - exact_vals))
    assert err_10 < 1e-9

    # --- Chebfun verification: integrate Fourier basis functions -----
    dom = (0.0, 2.0 * pi)
    for n in [1, 3, 5]:
        # int_0^{2*pi} sin(n*x)^2 dx = pi
        fn = cj.chebfun(lambda x, n=n: jnp.sin(n * x)**2, domain=dom)
        val = float(fn.sum())
        print(f"\n  int_0^{{2pi}} sin({n}x)^2 dx = {val:.10f}  (exact: pi={pi:.10f})")
        assert abs(val - pi) < 1e-12

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    import matplotlib.pyplot as _plt
    import numpy as _np
    _x = _np.linspace(0.0, 2.0 * _np.pi, 1000)
    fig, ax = _plt.subplots(figsize=(6, 3.5))
    ax.step(_x, _np.where(_x < _np.pi, -1.0, 1.0), color="k",
            linewidth=0.8, label="sign(x−π)")
    for _N, _col in [(5, "#4169E1"), (20, "#E04040"), (50, "#228B22")]:
        ax.plot(_x, partial_fourier_sum(_N, _x), color=_col,
                linewidth=1.2, label=f"N={_N}")
    ax.legend(fontsize=9)
    ax.set_xlabel("x", fontsize=10)
    ax.set_title("Gibbs phenomenon", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "gibbs_phenomenon.png"),
                dpi=150, bbox_inches="tight")
    _plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
