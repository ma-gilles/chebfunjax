"""A keyhole contour integral.

Integrates f(z) = log(z)*tanh(z) around a keyhole contour that avoids
the branch cut on the negative real axis.  The exact answer is
4*pi*i*log(pi/2).

Credit: Inspired by Chebfun example complex/KeyholeContour.m
(Nick Trefethen and Nick Hale, October 2010, revised June 2019).
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
    print("Keyhole contour integral")
    print("=" * 60)

    pi = float(jnp.pi)

    # Integrate f(z) = log(z) * tanh(z) around a keyhole contour.
    # The contour avoids the branch cut of log on the negative real axis.
    # r = inner radius, R = outer radius, e = half-width of keyhole
    r = 0.2
    R = 2.0
    e = 0.1

    # Corner points of the keyhole
    c1 = -R + e * 1j   # top-left
    c2 = -r + e * 1j   # top-right
    c3 = -r - e * 1j   # bottom-right
    c4 = -R - e * 1j   # bottom-left

    # Integrate each segment using Chebfun.
    # f(z) = log(z)*tanh(z).  Use principal branch of log.

    def f_of_z(z_re, z_im):
        """f(z) = log(z)*tanh(z), returns (re, im)."""
        # Principal branch: log(z) = log|z| + i*arg(z), arg in (-pi, pi)
        # Avoid the negative real axis (arg = ±pi there, choose +pi for upper edge)
        arg_z = np.arctan2(z_im, z_re)
        log_z = np.log(np.sqrt(z_re**2 + z_im**2)) + 1j * arg_z
        tanh_z = np.tanh(z_re + 1j * z_im)
        fz = log_z * tanh_z
        return fz.real, fz.imag

    def integrate_line(z_start, z_end, n=500):
        """Integrate f(z) dz along straight segment from z_start to z_end."""
        ts = np.linspace(0.0, 1.0, n)
        dz = z_end - z_start
        zs = z_start + ts * dz
        re, im = f_of_z(zs.real, zs.imag)
        fz = re + 1j * im
        # Trapezoidal
        integrand = fz * dz
        return np.trapezoid(integrand, ts)

    def integrate_arc(z_center, z_start, z_end, ccw=True, n=500):
        """Integrate f(z) dz along a circular arc from z_start to z_end."""
        R_arc = abs(z_start - z_center)
        arg_start = np.angle(z_start - z_center)
        arg_end = np.angle(z_end - z_center)
        # Determine direction
        if ccw:
            if arg_end < arg_start:
                arg_end += 2 * pi
        else:
            if arg_end > arg_start:
                arg_end -= 2 * pi
        ts = np.linspace(arg_start, arg_end, n)
        zs = z_center + R_arc * np.exp(1j * ts)
        dzs = 1j * R_arc * np.exp(1j * ts)
        re, im = f_of_z(zs.real, zs.imag)
        fz = re + 1j * im
        integrand = fz * dzs
        darg = (arg_end - arg_start) / (n - 1)
        return np.trapezoid(integrand, ts)

    # The keyhole contour goes:
    # (1) Top horizontal leg: c1 -> c2
    # (2) Inner small arc: c2 -> c3 going clockwise (CW) through the right
    #     Actually, the inner arc connects -r+ei to -r-ei by going CW
    #     (counterclockwise when viewed from outside the keyhole)
    # (3) Bottom horizontal leg: c3 -> c4
    # (4) Outer large arc: c4 -> c1 going CCW

    I1 = integrate_line(c1, c2, n=1000)
    # Inner arc: from c2 to c3 clockwise, i.e. going through angle ~pi to ~-pi
    # Arc from c2 (at angle ~pi-small) to c3 (at angle ~-(pi-small)) going CW
    # = going from angle pi-e/r downward to -(pi-e/r), i.e. CW = decreasing angle
    I2 = integrate_arc(0+0j, c2, c3, ccw=False, n=1000)
    I3 = integrate_line(c3, c4, n=1000)
    # Outer arc: from c4 to c1 going CCW (increasing angle from ~-pi to ~pi)
    I4 = integrate_arc(0+0j, c4, c1, ccw=True, n=1000)

    I_total = I1 + I2 + I3 + I4
    I_exact = 4j * pi * np.log(pi / 2.0)

    print(f"\nKeyhole contour integral of log(z)*tanh(z):")
    print(f"  Computed (trapezoidal): {I_total.real:.8f} + {I_total.imag:.8f}i")
    print(f"  Exact 4*pi*i*log(pi/2): {I_exact.real:.8f} + {I_exact.imag:.8f}i")
    error = abs(I_total - I_exact)
    print(f"  Error: {error:.4f}  (trapezoidal is O(1e-3) accurate here)")

    # For higher accuracy, use Chebfun for each segment
    def cheb_integrate_line(z_start, z_end):
        dc = complex(z_end - z_start)
        def re_int(t_arr):
            t = np.asarray(t_arr)
            zs = complex(z_start) + t * dc
            z_re, z_im = zs.real, zs.imag
            arg_z = np.arctan2(z_im, z_re)
            log_z_re = np.log(np.sqrt(z_re**2 + z_im**2))
            log_z_im = arg_z
            # tanh(z_re + i*z_im) -- use real formula
            tanh_r = np.tanh(z_re)
            tanh_i = np.tan(z_im)
            denom = 1 + tanh_r**2 * tanh_i**2
            f_re = (tanh_r + tanh_r * tanh_i**2) / denom
            f_im = (tanh_i - tanh_r**2 * tanh_i) / denom
            # log(z)*tanh(z)
            fz_re = log_z_re * f_re - log_z_im * f_im
            fz_im = log_z_re * f_im + log_z_im * f_re
            # multiply by dz = dc
            return fz_re * dc.real - fz_im * dc.imag

        def im_int(t_arr):
            t = np.asarray(t_arr)
            zs = complex(z_start) + t * dc
            z_re, z_im = zs.real, zs.imag
            arg_z = np.arctan2(z_im, z_re)
            log_z_re = np.log(np.sqrt(z_re**2 + z_im**2))
            log_z_im = arg_z
            tanh_r = np.tanh(z_re)
            tanh_i = np.tan(z_im)
            denom = 1 + tanh_r**2 * tanh_i**2
            f_re = (tanh_r + tanh_r * tanh_i**2) / denom
            f_im = (tanh_i - tanh_r**2 * tanh_i) / denom
            fz_re = log_z_re * f_re - log_z_im * f_im
            fz_im = log_z_re * f_im + log_z_im * f_re
            return fz_re * dc.imag + fz_im * dc.real

        fr = cj.chebfun(lambda t: jnp.array(re_int(np.asarray(t))), domain=(0.0, 1.0))
        fi = cj.chebfun(lambda t: jnp.array(im_int(np.asarray(t))), domain=(0.0, 1.0))
        return float(fr.sum()) + 1j * float(fi.sum())

    I1c = cheb_integrate_line(c1, c2)
    I3c = cheb_integrate_line(c3, c4)

    # For arcs use Chebfun on the angle parameter
    def cheb_integrate_arc(z_center, z_start, z_end, ccw=True):
        """Integrate f(z) dz along arc from z_start to z_end.
        Uses substitution s in [0,1] mapping to angle, so domain is always (0,1)."""
        R_arc = abs(z_start - z_center)
        arg_s = np.angle(z_start - z_center)
        arg_e = np.angle(z_end - z_center)
        if ccw:
            if arg_e < arg_s:
                arg_e += 2 * pi
        else:
            if arg_e > arg_s:
                arg_e -= 2 * pi
        d_arg = arg_e - arg_s  # total angular sweep

        def arc_re_int(s_arr):
            s = np.asarray(s_arr)
            t = arg_s + s * d_arg  # angle
            zs = complex(z_center) + R_arc * np.exp(1j * t)
            dzs = 1j * R_arc * np.exp(1j * t) * d_arg  # dz/ds = dz/dt * dt/ds
            z_re, z_im = zs.real, zs.imag
            arg_z = np.arctan2(z_im, z_re)
            log_z_re = np.log(np.sqrt(z_re**2 + z_im**2))
            log_z_im = arg_z
            tanh_r = np.tanh(z_re)
            tanh_i = np.tan(z_im)
            denom = 1 + tanh_r**2 * tanh_i**2
            f_re = (tanh_r + tanh_r * tanh_i**2) / denom
            f_im = (tanh_i - tanh_r**2 * tanh_i) / denom
            fz_re = log_z_re * f_re - log_z_im * f_im
            fz_im = log_z_re * f_im + log_z_im * f_re
            dz_re, dz_im = dzs.real, dzs.imag
            return fz_re * dz_re - fz_im * dz_im

        def arc_im_int(s_arr):
            s = np.asarray(s_arr)
            t = arg_s + s * d_arg
            zs = complex(z_center) + R_arc * np.exp(1j * t)
            dzs = 1j * R_arc * np.exp(1j * t) * d_arg
            z_re, z_im = zs.real, zs.imag
            arg_z = np.arctan2(z_im, z_re)
            log_z_re = np.log(np.sqrt(z_re**2 + z_im**2))
            log_z_im = arg_z
            tanh_r = np.tanh(z_re)
            tanh_i = np.tan(z_im)
            denom = 1 + tanh_r**2 * tanh_i**2
            f_re = (tanh_r + tanh_r * tanh_i**2) / denom
            f_im = (tanh_i - tanh_r**2 * tanh_i) / denom
            fz_re = log_z_re * f_re - log_z_im * f_im
            fz_im = log_z_re * f_im + log_z_im * f_re
            dz_re, dz_im = dzs.real, dzs.imag
            return fz_re * dz_im + fz_im * dz_re

        fr = cj.chebfun(lambda t: jnp.array(arc_re_int(np.asarray(t))), domain=(0.0, 1.0))
        fi = cj.chebfun(lambda t: jnp.array(arc_im_int(np.asarray(t))), domain=(0.0, 1.0))
        return float(fr.sum()) + 1j * float(fi.sum())

    I2c = cheb_integrate_arc(0+0j, c2, c3, ccw=False)
    I4c = cheb_integrate_arc(0+0j, c4, c1, ccw=True)

    I_cheb = I1c + I2c + I3c + I4c
    error_cheb = abs(I_cheb - I_exact)
    print(f"\nChebfun integration result:")
    print(f"  Computed: {I_cheb.real:.12f} + {I_cheb.imag:.12f}i")
    print(f"  Exact:    {I_exact.real:.12f} + {I_exact.imag:.12f}i")
    print(f"  Error:    {error_cheb:.2e}")
    assert error_cheb < 1e-6, f"Chebfun keyhole error {error_cheb}"

    print(f"\nSegment contributions:")
    print(f"  I1 (top leg):    {I1c.real:.6f} + {I1c.imag:.6f}i")
    print(f"  I2 (inner arc):  {I2c.real:.6f} + {I2c.imag:.6f}i")
    print(f"  I3 (bottom leg): {I3c.real:.6f} + {I3c.imag:.6f}i")
    print(f"  I4 (outer arc):  {I4c.real:.6f} + {I4c.imag:.6f}i")

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    # Left: the keyhole contour
    ts_200 = np.linspace(0, 1, 200)

    # Draw contour
    z_top = c1 + ts_200 * (c2 - c1)
    # inner arc: from c2 to c3 CW
    arg_c2 = np.angle(c2)
    arg_c3 = np.angle(c3)
    if arg_c3 > arg_c2:
        arg_c3 -= 2 * pi
    ts_arc = np.linspace(arg_c2, arg_c3, 200)
    z_inner = r * np.exp(1j * ts_arc)
    z_bot = c3 + ts_200 * (c4 - c3)
    # outer arc: from c4 to c1 CCW
    arg_c4 = np.angle(c4)
    arg_c1 = np.angle(c1)
    if arg_c1 < arg_c4:
        arg_c1 += 2 * pi
    ts_arc_out = np.linspace(arg_c4, arg_c1, 200)
    z_outer = R * np.exp(1j * ts_arc_out)

    for z_seg, col, lbl in [(z_top, "#1e77b4", "top"), (z_inner, "#d62728", "inner arc"),
                             (z_bot, "#2ca02c", "bottom"), (z_outer, "#ff7f0e", "outer arc")]:
        axes[0].plot(np.real(z_seg), np.imag(z_seg), color=col, linewidth=2, label=lbl)
    axes[0].plot([-R-0.3, 0], [0, 0], 'r--', linewidth=1.2, label="branch cut")
    axes[0].set_aspect('equal')
    axes[0].set_title("Keyhole contour")
    axes[0].legend(fontsize=7, loc='upper right')

    # Right: phase plot of log(z)*tanh(z)
    xs_g = np.linspace(-2.5, 2.5, 300)
    ys_g = np.linspace(-2.5, 2.5, 300)
    XX, YY = np.meshgrid(xs_g, ys_g)
    ZZ = XX + 1j * YY
    mask = np.abs(ZZ) > 0.05
    FZ = np.where(mask, np.log(ZZ) * np.tanh(ZZ), np.nan+0j)
    phase = np.angle(FZ)
    axes[1].pcolormesh(xs_g, ys_g, phase, cmap='hsv', shading='auto',
                       vmin=-pi, vmax=pi, alpha=0.8)
    for z_seg, col in [(z_top, "w"), (z_inner, "w"), (z_bot, "w"), (z_outer, "w")]:
        axes[1].plot(np.real(z_seg), np.imag(z_seg), color=col, linewidth=1.5)
    axes[1].set_aspect('equal')
    axes[1].set_title("Phase of $\\log(z)\\tanh(z)$ with contour")
    axes[1].set_xlim(-2.5, 2.5)
    axes[1].set_ylim(-2.5, 2.5)

    fig.suptitle(f"Keyhole contour: $\\int \\log(z)\\tanh(z)\\,dz = 4\\pi i \\log(\\pi/2)$",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "keyhole_contour.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
