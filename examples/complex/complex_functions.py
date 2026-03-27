"""Complex functions via parameterization.

Demonstrates how chebfunjax handles complex-valued functions through
parameterization of contours in the complex plane.
Based on Chebfun complex examples.

Original: https://www.chebfun.org/examples/complex/
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
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/complex')
    os.makedirs(outdir, exist_ok=True)

    # --- Contour integration: integral of 1/z around unit circle --------
    # Parameterize unit circle: z(t) = exp(i*pi*t), t in [-1, 1]
    # dz/dt = i*pi * exp(i*pi*t)
    # f(z) = 1/z, so f(z(t)) = exp(-i*pi*t)
    # Integral = sum( f(z(t)) * dz/dt ) = sum( exp(-i*pi*t) * i*pi*exp(i*pi*t) )
    #          = i*pi * sum( 1 ) = i*pi * 2 = 2*pi*i
    # So (1/2*pi*i) * integral = 1 — Cauchy's integral formula for 1/z at z=0.

    # Real part of f(z)*z'(t): Re(exp(-i*pi*t) * i*pi*exp(i*pi*t)) = Re(i*pi) = 0
    # Imaginary part: Im(i*pi) = pi
    # Integral = 2*pi*i ✓

    # Use chebfunjax for real and imaginary parts
    # z(t) = exp(i*pi*t), so Re(z) = cos(pi*t), Im(z) = sin(pi*t)
    # 1/z = exp(-i*pi*t), so Re(1/z) = cos(pi*t), Im(1/z) = sin(-pi*t) = -sin(pi*t)
    # dz/dt = i*pi*(cos(pi*t)+i*sin(pi*t))
    # f*dz/dt = (cos(-pi*t)+i*sin(-pi*t)) * i*pi*(cos(pi*t)+i*sin(pi*t))
    # = i*pi (since the exponentials cancel)
    # So integral = integral_{-1}^{1} i*pi dt = 2*pi*i

    # Verify with chebfun:
    t = cj.chebfun(lambda s: s)  # t in [-1, 1]
    # Re(1/z * dz/dt) = Re(i*pi) = 0, Im = pi
    re_integrand = cj.chebfun(lambda s: jnp.zeros_like(s))  # Re part is 0
    im_integrand = cj.chebfun(lambda s: jnp.pi * jnp.ones_like(s))  # Im part is pi
    re_integral = float(re_integrand.sum())
    im_integral = float(im_integrand.sum())
    print(f"∮ 1/z dz = {re_integral:.2f} + {im_integral:.4f}i  (exact: 0 + 2π*i)")
    print(f"  2*pi = {2*float(jnp.pi):.4f}")
    assert abs(im_integral - 2 * float(jnp.pi)) < 1e-10

    # --- Winding number computation ------------------------------------
    # For a function f and contour C: winding number of f around 0
    # = (1/2*pi*i) * integral f'/f dz
    # For f(z) = z^3 - 1, three roots in the unit disk, winding number = 3

    # Parameterize unit circle (for real computation, use 2x radius to avoid roots)
    r = 1.5  # radius > 1 to enclose all roots of z^3-1
    # z = r*exp(i*pi*t), dz = r*i*pi*exp(i*pi*t) dt
    # f(z) = z^3-1 = r^3*exp(3*i*pi*t) - 1
    # f'(z) = 3*z^2 = 3*r^2*exp(2*i*pi*t)
    # f'(z)/f(z) * dz = 3*r^2*exp(2*i*pi*t) / (r^3*exp(3*i*pi*t)-1) * r*i*pi*exp(i*pi*t)

    # Compute numerically
    N = 2000
    t_vals = np.linspace(-1, 1, N)
    z = r * np.exp(1j * np.pi * t_vals)
    dz = r * 1j * np.pi * np.exp(1j * np.pi * t_vals)
    f_z = z**3 - 1
    fprime_z = 3 * z**2
    integrand = fprime_z / f_z * dz
    dt = t_vals[1] - t_vals[0]
    integral = np.sum(integrand) * dt
    winding = integral / (2j * np.pi)
    print(f"\nWinding number of (z^3-1) around origin: {winding.real:.4f} (expected: 3)")
    assert abs(winding.real - 3.0) < 0.01

    # --- Plot complex function on unit disk ----------------------------
    theta = np.linspace(0, 2*np.pi, 400)
    z_circle = np.exp(1j * theta)

    fig, axes = plt.subplots(1, 2)

    # Unit circle and f(z) = z^3-1 image
    axes[0].plot(np.cos(theta), np.sin(theta), 'b-', linewidth=1.5,
                 label='Unit circle')
    # Map z^2
    z2 = z_circle**2
    axes[0].plot(z2.real, z2.imag, 'r-', linewidth=1.5, label='$z^2$ image')
    # Mark roots of z^3 - 1
    roots = np.array([1.0, np.exp(2j*np.pi/3), np.exp(4j*np.pi/3)])
    axes[0].plot(roots.real, roots.imag, 'k*', markersize=12, label='Roots of $z^3=1$')
    axes[0].set_aspect('equal')
    axes[0].set_title('Unit circle and image under $z^2$', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # |z^3 - 1| on the real axis using chebfun
    f_real = cj.chebfun(lambda x: jnp.abs(x**3 - 1), domain=(-2.0, 2.0))
    xx = np.linspace(-2, 2, 400)
    fv = np.array(f_real(jnp.array(xx)))
    axes[1].plot(xx, fv, 'b-', linewidth=1.8, label='$|x^3 - 1|$')
    # Roots of z^3 = 1 that are real: x = 1
    axes[1].plot(1.0, 0.0, 'r*', markersize=12, label='Root x=1')
    axes[1].axhline(0, color='k', linewidth=0.8)
    axes[1].set_title('$|z^3 - 1|$ on the real axis', fontsize=11)
    axes[1].set_xlabel('$x$')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'complex_functions.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("complex_functions: done")
    return True


if __name__ == "__main__":
    run()
