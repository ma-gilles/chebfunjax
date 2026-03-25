"""Arc length in the complex plane.

Computes the arc length of curves in the complex plane by integrating
|dz/dt| dt, demonstrating Chebfun's ability to work with complex-valued
functions of a real variable.

Credit: Inspired by Chebfun example complex/ComplexArcLength.m
(Kuan Xu, November 2012).
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


def arc_length(z_func, dom):
    """Compute arc length of curve z(t) on domain dom using Chebfun."""
    # |dz/dt| = sqrt((dx/dt)^2 + (dy/dt)^2)
    # We compute this by differentiating the real and imaginary parts
    f_re = cj.chebfun(lambda t: jnp.real(jnp.array(z_func(np.asarray(t)))), domain=dom)
    f_im = cj.chebfun(lambda t: jnp.imag(jnp.array(z_func(np.asarray(t)))), domain=dom)
    df_re = f_re.diff()
    df_im = f_im.diff()
    speed = cj.chebfun(
        lambda t: jnp.sqrt(jnp.array(df_re(jnp.array(t)))**2 +
                           jnp.array(df_im(jnp.array(t)))**2),
        domain=dom
    )
    return float(speed.sum())


def run():
    print("=" * 60)
    print("Arc length in the complex plane")
    print("=" * 60)

    pi = float(jnp.pi)

    # --- Example 1: Unit circle arc length = 2*pi ---------------------
    def z_circle(t):
        return np.exp(1j * t)

    L1 = arc_length(z_circle, (0.0, 2.0 * pi))
    print(f"\nUnit circle arc length:")
    print(f"  Computed: {L1:.10f}")
    print(f"  Exact 2*pi = {2*pi:.10f}")
    assert abs(L1 - 2*pi) < 1e-8, f"Circle length error: {L1 - 2*pi}"

    # --- Example 2: Straight line z(t) = t + it, t in [0,1] ----------
    # |dz/dt| = |1+i| = sqrt(2), length = sqrt(2)
    def z_line(t):
        return t * (1 + 1j)

    L2 = arc_length(z_line, (0.0, 1.0))
    print(f"\nLine z(t) = (1+i)*t arc length:")
    print(f"  Computed: {L2:.10f}")
    print(f"  Exact sqrt(2) = {float(jnp.sqrt(jnp.array(2.0))):.10f}")
    assert abs(L2 - float(jnp.sqrt(jnp.array(2.0)))) < 1e-10

    # --- Example 3: Ellipse z(t) = a*cos(t) + i*b*sin(t) -------------
    # Arc length = 4a * E(e) where E is complete elliptic integral and e = sqrt(1-b^2/a^2)
    a, b = 2.0, 1.0

    def z_ellipse(t):
        return a * np.cos(t) + 1j * b * np.sin(t)

    L3 = arc_length(z_ellipse, (0.0, 2.0 * pi))
    from scipy.special import ellipe
    ecc = float(jnp.sqrt(jnp.array(1.0 - (b/a)**2)))
    exact_L3 = 4 * a * float(ellipe(ecc**2))
    print(f"\nEllipse (a={a}, b={b}) arc length:")
    print(f"  Computed: {L3:.10f}")
    print(f"  Exact (elliptic): {exact_L3:.10f}")
    assert abs(L3 - exact_L3) < 1e-6, f"Ellipse length error: {L3 - exact_L3}"

    # --- Example 4: Spiral z(t) = t*exp(it), t in [0, 10] ------------
    # |dz/dt| = |exp(it) + it*exp(it)| = |1 + it| * |exp(it)| = sqrt(1+t^2)
    # Length = int_0^10 sqrt(1+t^2) dt = exact value
    def z_spiral(t):
        return t * np.exp(1j * t)

    L4 = arc_length(z_spiral, (0.0, 10.0))
    # Exact: int_0^10 sqrt(1+t^2) dt = [t*sqrt(1+t^2)/2 + arcsinh(t)/2]_0^10
    T = 10.0
    exact_L4 = (T * np.sqrt(1 + T**2) + np.arcsinh(T)) / 2.0
    print(f"\nSpiral z(t) = t*exp(it), t in [0,10] arc length:")
    print(f"  Computed: {L4:.10f}")
    print(f"  Exact: {exact_L4:.10f}")
    assert abs(L4 - exact_L4) < 1e-6, f"Spiral length error: {L4 - exact_L4}"

    # --- Example 5: Keyhole top edge arc length ----------------------
    # From the KeyholeContour example: top edge from -R+ei to -r+ei
    R, r, e = 2.0, 0.2, 0.1
    c1 = -R + e*1j
    c2 = -r + e*1j
    L5_exact = abs(c2 - c1)  # just the length of the straight segment

    def z_leg(t):
        return c1 + t * (c2 - c1)

    L5 = arc_length(z_leg, (0.0, 1.0))
    print(f"\nKeyhole top leg arc length:")
    print(f"  Computed: {L5:.10f}")
    print(f"  Exact (straight line): {L5_exact:.10f}")
    assert abs(L5 - L5_exact) < 1e-10

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: various curves
    ts_circle = np.linspace(0, 2*pi, 200)
    ts_ellipse = np.linspace(0, 2*pi, 200)
    ts_spiral = np.linspace(0, 10, 400)

    z_c = z_circle(ts_circle)
    z_e = z_ellipse(ts_ellipse)
    z_s = z_spiral(ts_spiral)

    axes[0].plot(np.real(z_c), np.imag(z_c), color="#1e77b4", linewidth=2,
                 label=f"Circle L={L1:.3f}")
    axes[0].plot(np.real(z_e), np.imag(z_e), color="#d62728", linewidth=2,
                 label=f"Ellipse L={L3:.3f}")
    axes[0].set_aspect('equal')
    axes[0].set_xlabel("Re(z)")
    axes[0].set_ylabel("Im(z)")
    axes[0].set_title("Curves and their arc lengths")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.4)

    # Right: spiral
    axes[1].plot(np.real(z_s), np.imag(z_s), color="#2ca02c", linewidth=2,
                 label=f"Spiral L={L4:.3f}")
    axes[1].set_aspect('equal')
    axes[1].set_xlabel("Re(z)")
    axes[1].set_ylabel("Im(z)")
    axes[1].set_title("Spiral $z(t) = te^{it}$, $t\\in[0,10]$")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.4)

    fig.suptitle("Arc length in the complex plane", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "complex_arc_length.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
