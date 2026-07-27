"""An ellipse rolling around another ellipse.

Two contact points z1(t), z2(t) trace the rims of two ellipses (solved as
complex ODEs); the coupled midpoint w(t) = z1 - z2*diff(z1)/diff(z2) is a
complex Chebfun.  tfinal is a root of imag(w) on [5, 7.5] and the trajectory
length is norm(diff(w), 1).  Faithful port of geom/Ellipses.m.

Original: https://www.chebfun.org/examples/geom/Ellipses.html
Author: Nick Trefethen, October 2011
"""

import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)

    L1, L2 = 3.0, 2.0
    tmax = 7.5

    def theta1(z):
        return np.arctan2(np.imag(z), np.real(z) / L1)

    def theta2(z):
        return np.arctan2(np.imag(z), np.real(z) / L2)

    # Complex ODEs (real 2-component form for scipy; MATLAB uses chebfun.ode113
    # on the complex field directly, with abstol = reltol = 1e-13).
    def ode1(t, y):
        z = y[0] + 1j * y[1]
        th = theta1(z)
        dz = (-L1 * np.sin(th) + 1j * np.cos(th)) / np.sqrt(
            L1**2 * np.sin(th)**2 + np.cos(th)**2)
        return [np.real(dz), np.imag(dz)]

    def ode2(t, y):
        z = y[0] + 1j * y[1]
        th = theta2(z)
        dz = (L2 * np.sin(th) - 1j * np.cos(th)) / np.sqrt(
            L2**2 * np.sin(th)**2 + np.cos(th)**2)
        return [np.real(dz), np.imag(dz)]

    s1 = solve_ivp(ode1, (0.0, tmax), [L1 / 2, 0.0], method='DOP853',
                   rtol=1e-13, atol=1e-13, dense_output=True)
    s2 = solve_ivp(ode2, (0.0, tmax), [-L2 / 2, 0.0], method='DOP853',
                   rtol=1e-13, atol=1e-13, dense_output=True)

    def _cplx(sol):
        def f(t):
            y = sol.sol(np.asarray(t))
            return jnp.asarray(y[0] + 1j * y[1])
        return cj.chebfun(f, domain=(0.0, tmax))

    z1 = _cplx(s1)
    z2 = _cplx(s2)

    # Midpoint trajectory (complex Chebfun).
    w = z1 - z2 * z1.diff() / z2.diff()

    # tfinal: root of imag(w) restricted to [5, 7.5]; MATLAB w{5,7.5}.
    r = np.asarray(w.restrict(5.0, 7.5).imag().roots()).ravel()
    tfinal = float(r[0])
    print(f"tfinal = {tfinal:.15f}")

    # Trajectory length = norm(diff(w{0,tfinal}), 1).
    trajectory_length = float(w.restrict(0.0, tfinal).diff().norm(1))
    print(f"trajectory_length = {trajectory_length:.15f}")

    # --- Plot ------------------------------------------------------------
    tt = np.linspace(0.0, tfinal, 800)
    w_vals = np.asarray(w(jnp.array(tt)))
    z1v = np.asarray(z1(jnp.array(tt)))
    z2v = np.asarray(z2(jnp.array(tt)))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(np.real(w_vals), np.imag(w_vals), 'k-', lw=2)
    axes[0].set_aspect('equal')
    axes[0].set_title('Midpoint trajectory w(t)', fontsize=11)
    axes[1].plot(np.real(z1v), np.imag(z1v), color='#0072BD', lw=2, label='z1 (big)')
    axes[1].plot(np.real(z2v), np.imag(z2v), color='#D95319', lw=2, label='z2 (small)')
    axes[1].plot(np.real(w_vals), np.imag(w_vals), 'k-', lw=2, label='w(t)')
    axes[1].set_aspect('equal')
    axes[1].legend(fontsize=9)
    axes[1].set_title(f'length = {trajectory_length:.4f}', fontsize=10)
    fig.suptitle('Ellipse rolling around ellipse', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'ellipses_rolling.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("ellipses_rolling: done")
    return True


if __name__ == "__main__":
    run()
