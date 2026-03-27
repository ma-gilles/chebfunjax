"""Spike integral: integrating a function with very narrow spikes.

Demonstrates the adaptive capability of Chebfun by integrating a
function with spikes of widths 1/10, 1/100, and 1/1000.

Credit: Inspired by Chebfun example quad/SpikeIntegral.m
(Nick Hale, October 2010).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()



def run():
    print("=" * 60)
    print("Spike integral")
    print("=" * 60)

    # The spike function: sum of sech^k spikes with widths 10, 100, 1000
    def spike_func(x):
        return (
            (1.0 / jnp.cosh(10.0  * (x - 0.2)))**2 +
            (1.0 / jnp.cosh(100.0 * (x - 0.4)))**4 +
            (1.0 / jnp.cosh(1000.0* (x - 0.6)))**6 +
            (1.0 / jnp.cosh(1000.0* (x - 0.8)))**8
        )

    dom = (0.0, 1.0)
    f = cj.chebfun(spike_func, domain=dom)
    I = float(f.sum())
    print(f"\nIntegral of spike function on [0,1] = {I:.15f}")

    # Reference: integrate each spike separately
    # int sech(a*(x-c))^n dx = integral of each peak
    # For sech(a*(x-c))^2 on (-inf,inf): 2/a
    # For sech(a*(x-c))^4 on (-inf,inf): 4/(3a)
    # For sech(a*(x-c))^6 on (-inf,inf): 16/(15a)
    # For sech(a*(x-c))^8 on (-inf,inf): 32/(35a) * something..
    # Let's use numerical reference with large domain
    f_ref = cj.chebfun(lambda x: (1.0/jnp.cosh(10.0*(x-0.2)))**2 + 0*x,
                        domain=(-0.5, 0.9))
    I1_ref = float(f_ref.sum())  # ≈ 2/10 = 0.2
    f2_ref = cj.chebfun(lambda x: (1.0/jnp.cosh(100.0*(x-0.4)))**4 + 0*x,
                         domain=(0.35, 0.45))
    I2_ref = float(f2_ref.sum())  # ≈ 4/(3*100)
    f3_ref = cj.chebfun(lambda x: (1.0/jnp.cosh(1000.0*(x-0.6)))**6 + 0*x,
                         domain=(0.595, 0.605))
    I3_ref = float(f3_ref.sum())  # ≈ 16/(15*1000)
    f4_ref = cj.chebfun(lambda x: (1.0/jnp.cosh(1000.0*(x-0.8)))**8 + 0*x,
                         domain=(0.795, 0.805))
    I4_ref = float(f4_ref.sum())

    I_ref = I1_ref + I2_ref + I3_ref + I4_ref
    print(f"Sum of individual spike integrals = {I_ref:.15f}")
    print(f"Chebfun length (polynomial degree): {len(f)}")

    err = abs(I - I_ref)
    print(f"Difference = {err:.2e}  (should be small, <0.01)")
    assert err < 0.01, f"Spike integral error too large: {err}"

    # Individual spike widths
    exact_I1 = 2.0 / 10.0
    exact_I2 = 4.0 / (3.0 * 100.0)
    exact_I3 = 16.0 / (15.0 * 1000.0)
    print(f"\nIndividual contributions:")
    print(f"  sech(10*(x-0.2))^2:   I1 ≈ {I1_ref:.8f}  exact ≈ {exact_I1:.8f}")
    print(f"  sech(100*(x-0.4))^4:  I2 ≈ {I2_ref:.8f}  exact ≈ {exact_I2:.8f}")
    print(f"  sech(1000*(x-0.6))^6: I3 ≈ {I3_ref:.8f}")
    print(f"  sech(1000*(x-0.8))^8: I4 ≈ {I4_ref:.8f}")

    assert abs(I1_ref - exact_I1) < 0.001
    assert abs(I2_ref - exact_I2) < 0.0001

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    xs = np.linspace(0.0, 1.0, 2000)
    ys = np.array(spike_func(jnp.array(xs)))
    axes[0].plot(xs, ys, color="#1e77b4", linewidth=1.5)
    axes[0].fill_between(xs, 0, ys, alpha=0.2)
    axes[0].set_xlabel("x")
    axes[0].set_title(f"Spike function  (Chebfun length = {len(f)})")
    axes[0].grid(True, alpha=0.4)
    axes[0].annotate(f"I = {I:.6f}", xy=(0.5, 0.5), xycoords="axes fraction",
                     fontsize=11, ha="center")

    # Zoom on narrowest spike
    xs_zoom = np.linspace(0.595, 0.605, 1000)
    ys_zoom = np.array(spike_func(jnp.array(xs_zoom)))
    axes[1].plot(xs_zoom, ys_zoom, color="#d62728", linewidth=1.5)
    axes[1].fill_between(xs_zoom, 0, ys_zoom, alpha=0.2, color="#d62728")
    axes[1].set_xlabel("x")
    axes[1].set_title("Narrowest spike: sech(1000*(x-0.6))^6")
    axes[1].grid(True, alpha=0.4)

    fig.suptitle("Spike integral: adaptive Chebfun quadrature", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "spike_integral.png"), dpi=150, bbox_inches="tight")
    _docs = os.path.join(_here, "..", "..", "docs", "images", "quad")
    os.makedirs(_docs, exist_ok=True)
    fig.savefig(os.path.join(_docs, "spike_integral.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
