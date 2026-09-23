"""The fractal structure of the Lorenz attractor, via rational approximation.

Translation of ode-nonlin/LorenzAttractor.m: integrate the Lorenz
system to t = 5 at tight tolerance, then use ratinterp to study the
complex-t singularity structure of the trajectory components. The poles
of the rational approximants line up just above and below the real axis,
and the three components agree on where they are.

Original: https://www.chebfun.org/examples/ode-nonlin/LorenzAttractor.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax import ratinterp
from chebfunjax.chebfun1d.chebfun import ode113
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]
D = (0.0, 5.0)


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"LorenzAttractor_{FIG[0]:02d}.png"))
    plt.close(fig)


def _msort(z):
    """MATLAB sort of a complex vector: by modulus, then by angle."""
    z = np.asarray(z, dtype=complex)
    return z[np.lexsort((np.angle(z), np.abs(z)))]


def _c(z):
    """One MATLAB ``format short`` complex entry."""
    sg = '-' if z.imag < 0 else '+'
    return f"{z.real:9.4f} {sg} {abs(z.imag):6.4f}i"


def _rat(a, b):
    """Numerator and denominator chebfuns [p, q] of ratinterp on D."""
    return (cj.chebfun(jnp.asarray(a), coeffs=True, domain=list(D)),
            cj.chebfun(jnp.asarray(b), coeffs=True, domain=list(D)))


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    def fun(t, u):
        return np.array([10 * (u[1] - u[0]),
                         28 * u[0] - u[1] - u[0] * u[2],
                         u[0] * u[1] - (8 / 3) * u[2]])

    u = ode113(fun, D, np.array([-14.0, -15.0, 20.0]),
               rtol=1e-13, atol=1e-13)
    u1, u2, u3 = u

    # 3D trajectory
    t = np.linspace(*D, 20000)
    fig = plt.figure(figsize=(7.6, 6.2))
    ax = fig.add_subplot(projection="3d")
    ax.plot(np.asarray(u1(t)), np.asarray(u2(t)), np.asarray(u3(t)),
            lw=1.6)
    ax.grid(True)
    ax.view_init(elev=20, azim=20 - 90)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-40, 40)
    ax.set_zlim(5, 45)
    ax.set_xlabel("x(t)")
    ax.set_ylabel("y(t)")
    ax.set_zlabel("z(t)")
    ax.set_title("A 3D Trajectory of the Lorenz Attractor", fontsize=14)
    _save(fig)

    # the components as scalar functions of t
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for f in (u1, u2, u3):
        ax.plot(t, np.asarray(f(t)), lw=1.6)
    ax.grid(True)
    ax.set_xlabel("t")
    ax.set_ylabel("x(t), y(t), z(t)")
    ax.set_title("Solution to the Lorenz Attractor as Scalar Functions",
                 fontsize=14)
    _save(fig)

    rats = []
    for f, m, NN in ((u1, 221, 444), (u2, 241, 484), (u3, 236, 473)):
        rh, a, b, mu, nu, poles, res = ratinterp(
            f, m, 40, NN, None, 1e-12, domain=D)
        rats.append((*_rat(a, b), _msort(poles)))

    xx = np.linspace(-0.5, 5.5, 200)
    yy = np.linspace(-0.5, 0.5, 200)
    XX, YY = np.meshgrid(xx, yy)
    z = (XX + 1j * YY).ravel()
    fig, axes = plt.subplots(3, 1)
    for k, (p, q, poles) in enumerate(rats):
        ax = axes[k]
        rz = np.abs(np.asarray(p(z)) / np.asarray(q(z))).reshape(XX.shape)
        ax.contour(xx, yy, rz, levels=np.arange(0, 151, 5))
        ax.grid(True)
        ax.set_title(f"r{k + 1}(t)")
        ax.plot(poles.real, poles.imag, "xk", markersize=16 * 0.6, mew=1.6)
        ax.plot([0, 5], [0, 0], "k", lw=1.6)
    _save(fig)

    # format short
    poles1, poles2, poles3 = (r[2] for r in rats)
    diffpoles = np.sort(np.column_stack([np.abs(poles1 - poles2),
                                         np.abs(poles1 - poles3),
                                         np.abs(poles2 - poles3)]), axis=0)
    print("   poles in x         poles in y         poles in z"
          "         max. difference")
    for row in zip(poles1, poles2, poles3, diffpoles[:, 0]):
        print("".join(_c(complex(v)) for v in row))

    print("ans =")
    v = 0.5 * diffpoles[:, 0]
    for c0 in range(0, len(v), 7):
        c1 = min(c0 + 7, len(v))
        print(f"  Columns {c0 + 1} through {c1}" if c1 - c0 > 1
              else f"  Column {c1}")
        print("".join(f"{x:10.4f}" for x in v[c0:c1]))

    rh, a, b, mu, nu, poles, res = ratinterp(
        u1, 221, 40, 444, None, 0.0, domain=D)
    p, q = _rat(a, b)
    poles = _msort(poles)
    print("poles =")
    for zp in poles:
        print(_c(complex(zp)))

    fig, ax = plt.subplots()
    ax.plot(poles.real, poles.imag, 'or', ms=8 * 0.6, mfc='r')
    ax.grid(True)
    ax.set_xlabel('Re(t)')
    ax.set_ylabel('Im(t)')
    zr = np.asarray(p.roots(complex_roots=True))
    ax.plot(zr.real, np.zeros(len(zr)), 'ok', ms=12 * 0.6, mfc='none')
    ax.set_title('Poles and Zeros of the Rational Interpolant', fontsize=14)
    _save(fig)


if __name__ == "__main__":
    run()
