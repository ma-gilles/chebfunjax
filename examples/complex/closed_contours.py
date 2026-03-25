"""Integrals over closed contours using periodic chebfuns.

Demonstrates computing complex contour integrals by parameterizing closed
contours as periodic functions and verifying Cauchy's theorem and the
residue theorem.

Credit: Inspired by Chebfun example complex/ClosedContours.m
(Mohsin Javed, June 2014).
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


def contour_integral(f_complex, r, n_pts=2000):
    """Compute int_|z|=r f(z) dz using trapezoidal rule (spectral accuracy for periodic)."""
    ts = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    zs = r * np.exp(1j * ts)
    dzs = 1j * r * np.exp(1j * ts) * (2 * np.pi / n_pts)
    fs = f_complex(zs)
    return np.sum(fs * dzs)


def run():
    print("=" * 60)
    print("Integrals over closed contours (residue theorem)")
    print("=" * 60)

    pi = float(jnp.pi)

    # --- Example 1: f(z) = (1-2z) / (z(z-1)(z-3)) on |z| = 2 -----------
    # Poles inside |z|=2: z=0 and z=1
    # Residue at z=0: (1-0)/((0-1)(0-3)) = 1/3
    # Residue at z=1: (1-2)/((1)(1-3)) = (-1)/(-2) = 1/2
    # By residue theorem: integral = 2*pi*i*(1/3 + 1/2) = 2*pi*i*5/6
    def f1(z):
        return (1.0 - 2.0*z) / (z * (z - 1.0) * (z - 3.0))

    I1 = contour_integral(f1, 2.0)
    exact1 = 2j * pi * (1.0/3.0 + 1.0/2.0)  # = 5*pi*i/3
    err1 = abs(I1 - exact1)
    print(f"\nExample 1: int_(|z|=2) (1-2z)/(z(z-1)(z-3)) dz")
    print(f"  Computed: {I1.real:.10f} + {I1.imag:.10f}i")
    print(f"  Exact 5*pi*i/3: {exact1.real:.10f} + {exact1.imag:.10f}i")
    print(f"  Error: {err1:.2e}")
    assert err1 < 1e-8, f"Error {err1}"

    # Chebfun integration
    def chebfun_circle_integral(f_complex, r, dom=(0.0, 2.0*float(jnp.pi))):
        """Integrate using Chebfun for spectral accuracy."""
        def re_int(t):
            z = r * jnp.exp(1j * jnp.array(t))
            dz = 1j * r * jnp.exp(1j * jnp.array(t))
            fz = jnp.array(f_complex(complex(float(z.real), float(z.imag))))
            return jnp.real(fz * dz)

        def im_int(t):
            z = r * jnp.exp(1j * jnp.array(t))
            dz = 1j * r * jnp.exp(1j * jnp.array(t))
            fz = jnp.array(f_complex(complex(float(z.real), float(z.imag))))
            return jnp.imag(fz * dz)

        # Use numpy-vectorized integrands for efficiency
        def re_int_vec(t_arr):
            t_np = np.asarray(t_arr)
            zs = r * np.exp(1j * t_np)
            dzs = 1j * r * np.exp(1j * t_np)
            fs = f_complex(zs)
            return np.real(fs * dzs)

        def im_int_vec(t_arr):
            t_np = np.asarray(t_arr)
            zs = r * np.exp(1j * t_np)
            dzs = 1j * r * np.exp(1j * t_np)
            fs = f_complex(zs)
            return np.imag(fs * dzs)

        f_re = cj.chebfun(lambda t: jnp.array(re_int_vec(np.asarray(t))), domain=dom)
        f_im = cj.chebfun(lambda t: jnp.array(im_int_vec(np.asarray(t))), domain=dom)
        return float(f_re.sum()) + 1j * float(f_im.sum())

    I1_cheb = chebfun_circle_integral(f1, 2.0)
    err1_cheb = abs(I1_cheb - exact1)
    print(f"  Chebfun: {I1_cheb.real:.12f} + {I1_cheb.imag:.12f}i, err = {err1_cheb:.2e}")
    assert err1_cheb < 1e-8

    # --- Example 2: sinc function (removable singularity at 0) -----------
    # sin(5z)/(5z) is analytic everywhere (removable singularity at 0)
    # By Cauchy's theorem: integral over any closed contour = 0
    def f2(z):
        return np.where(np.abs(z) < 1e-10, np.ones_like(z),
                        np.sin(5.0 * z) / (5.0 * z))

    I2 = contour_integral(f2, 1.0)
    print(f"\nExample 2: int_(|z|=1) sin(5z)/(5z) dz")
    print(f"  Computed: {I2.real:.2e} + {I2.imag:.2e}i  (should be 0 by Cauchy's theorem)")
    assert abs(I2) < 1e-10, f"sinc integral should be 0, got {abs(I2)}"

    # --- Example 3: exp(1/z)*sin(1/z) on unit circle -------------------
    # This has an essential singularity at z=0.
    # The Laurent series contains a 1/(2*pi*i) coefficient of 1/z term = 1.
    # So integral = 2*pi*i.
    def f3(z):
        return np.exp(1.0 / z) * np.sin(1.0 / z)

    I3 = contour_integral(f3, 1.0)
    exact3 = 2j * pi
    err3 = abs(I3 - exact3)
    print(f"\nExample 3: int_(|z|=1) exp(1/z)*sin(1/z) dz")
    print(f"  Computed: {I3.real:.10f} + {I3.imag:.10f}i")
    print(f"  Exact 2*pi*i: {exact3.real:.10f} + {exact3.imag:.10f}i")
    print(f"  Error: {err3:.2e}")
    assert err3 < 1e-6, f"Essential singularity integral error {err3}"

    # --- Example 4: 1/(z-z0) = 2*pi*i if z0 inside, 0 if outside -----
    def f4_in(z):
        z0 = 0.5 + 0.3j
        return 1.0 / (z - z0)

    def f4_out(z):
        z0 = 2.0 + 0.0j
        return 1.0 / (z - 0.5 - 0.3j)  # pole INSIDE |z|=1

    I4_in = contour_integral(f4_in, 1.0)  # z0 inside
    I4_out = contour_integral(lambda z: 1.0/(z - 3.0), 1.0)  # pole outside
    print(f"\nExample 4: Cauchy's integral formula")
    print(f"  z0 inside |z|=1: int 1/(z-z0) dz = {I4_in.real:.2e} + {I4_in.imag:.10f}i")
    print(f"  Exact 2*pi*i: 0 + {2*pi:.10f}i")
    print(f"  z0 outside |z|=1: int 1/(z-z0) dz = {I4_out.real:.2e} + {I4_out.imag:.2e}i (should be ~0)")
    assert abs(I4_in - 2j*pi) < 1e-8
    assert abs(I4_out) < 1e-10

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Left: the contour and poles for Example 1
    theta = np.linspace(0, 2*pi, 200)
    axes[0].plot(2*np.cos(theta), 2*np.sin(theta), 'b-', linewidth=2, label="|z|=2")
    poles_in = [0, 1]
    poles_out = [3]
    for p in poles_in:
        axes[0].plot(p, 0, 'rx', markersize=10, markeredgewidth=2, label="pole (inside)")
    for p in poles_out:
        axes[0].plot(p, 0, 'bs', markersize=8, markerfacecolor='none', label="pole (outside)")
    axes[0].set_aspect('equal')
    axes[0].set_xlim(-3, 4)
    axes[0].set_ylim(-3, 3)
    axes[0].axhline(0, color='k', linewidth=0.5)
    axes[0].axvline(0, color='k', linewidth=0.5)
    axes[0].set_title("Example 1: poles inside/outside")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # Middle: |f3(z)| on unit circle
    ts = np.linspace(0, 2*pi, 500)
    zs = np.exp(1j * ts)
    f3_vals = f3(zs)
    axes[1].plot(ts, np.real(f3_vals), label="Re")
    axes[1].plot(ts, np.imag(f3_vals), label="Im")
    axes[1].set_xlabel("t")
    axes[1].set_title("$e^{1/z}\\sin(1/z)$ on $|z|=1$")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Right: winding number illustration
    z0s = np.array([0.5 + 0.3j, 2.0 + 0.5j])
    labels = ["inside", "outside"]
    colors = ["r", "b"]
    axes[2].plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
    for z0, lbl, col in zip(z0s, labels, colors):
        axes[2].plot(z0.real, z0.imag, 'o', color=col, markersize=10, label=f"z0={lbl}")
    axes[2].set_aspect('equal')
    axes[2].set_xlim(-1.5, 2.5)
    axes[2].set_ylim(-1.5, 1.5)
    axes[2].legend(fontsize=8)
    axes[2].set_title("Cauchy's theorem: winding number")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Closed contour integrals and the residue theorem", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "closed_contours.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
