"""The three-body problem: the figure-eight orbit and its singularities.

Translation of ode-nonlin/ThreeBodyProblem.m: the celebrated
figure-of-eight choreography of three equal masses, integrated with
ode113 in complex arithmetic, then studied through ratinterp -- the
poles of a rational approximant to one body's trajectory mark the
complex-time singularities associated with close encounters.

Original: https://www.chebfun.org/examples/ode-nonlin/ThreeBodyProblem.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt

# uses-numpy: host plotting arrays and formatted reporting; native ODE callbacks use JAX.
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax import ratinterp
from chebfunjax.chebfun1d.chebfun import ode113
from chebfunjax.plotting import chebfun_style, matlab_plot, plotregion
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(
        _IMG, f"ThreeBodyProblem_{FIG[0]:02d}.png"),
        layout="matlab" if len(fig.axes) == 1 else None)
    plt.close(fig)


def _at(u, t, cols):
    """u(t, cols): values of the quasimatrix columns at time t."""
    return np.array([complex(u[k](jnp.asarray(float(t)))) for k in cols])


def _quiver(ax, z, w, scale):
    """Native numeric fitting and readable compatibility open arrowheads."""
    from chebfunjax import quiver_points

    return quiver_points(z.real, z.imag, w.real, w.imag, ax=ax,
                         autoscale_factor=scale, color="k", linewidth=2)


def _cplx(z):
    """MATLAB format long display of one complex entry."""
    sign = "-" if z.imag < 0 else "+"
    return f"{z.real:19.15f} {sign} {abs(z.imag):.15f}i"


def _ratinterp(v, m, n, tol, dom, outputs):
    """Call ratinterp with the same explicit output count as MATLAB."""
    return ratinterp(
        v, m, n, None, None, tol, domain=dom, outputs=outputs
    )


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    dom = (0.0, 4 * np.pi)
    a = 6.32591398 / (2 * np.pi)     # scaling factor to give period 2pi

    def fun(t, u):
        return a * jnp.array([
            u[3], u[4], u[5],
            ((u[1] - u[0]) / abs(u[1] - u[0]) ** 3
             + (u[2] - u[0]) / abs(u[2] - u[0]) ** 3),
            ((u[0] - u[1]) / abs(u[0] - u[1]) ** 3
             + (u[2] - u[1]) / abs(u[2] - u[1]) ** 3),
            ((u[0] - u[2]) / abs(u[0] - u[2]) ** 3
             + (u[1] - u[2]) / abs(u[1] - u[2]) ** 3)])

    u = ode113(fun, dom, np.array([
        0.540508553669932 + 0.345263318559681j,
        0.540508532338285 - 0.345263317862853j,
        -1.081017086008497 - 0.000000000697245j,
        -1.097122372968180 - 0.233604741427372j,
        1.097122377013713 - 0.233604786311327j,
        -0.000000004046108 + 0.467209527738458j]),
        atol=1e-13, rtol=1e-13)

    fig, ax = plt.subplots()
    matlab_plot(u[0], ax=ax, lw=2)
    ax.set_title("Figure of Eight Solution to Three Body Problem")
    ax.set_aspect("equal")
    ax.grid(True)
    z0 = _at(u, 0, range(3))
    ax.plot(z0.real, z0.imag, "ok", markerfacecolor="k", markersize=7)
    _quiver(ax, z0, _at(u, 0, range(3, 6)), 0.4)
    _save(fig)

    v = u[2]
    print("v =")
    print(repr(v))

    p, q, rh, mu, nu, poles = _ratinterp(v, 151, 150, 1e-12, dom, 6)
    print("mu =")
    print(f"{mu:6d}")
    print("nu =")
    print(f"{nu:6d}")

    tl = jnp.linspace(0, 4 * np.pi, 100)
    print("ans =")
    print(f"     {float(np.max(np.abs(rh(tl) - np.asarray(v(tl))))):.15e}")

    print("poles =")
    for z in poles:
        print(_cplx(z))

    print("ans =")
    for x in np.real(poles) * 3 / np.pi:
        print(f"{x:20.15f}")

    fig, ax = plt.subplots()
    zq = np.asarray(q.roots(all_roots=True))
    matlab_plot(zq.real, zq.imag, "o", ax=ax, markersize=4, color="b",
                markerfacecolor="b")
    zp = np.asarray(p.roots(complex_roots=True))
    ax.plot(zp.real, zp.imag, "ok", markersize=5, markerfacecolor="none")
    ax.set_aspect("equal")
    ax.grid(True)
    plotregion(v, ax=ax)
    ax.set_title("Poles, Zeros of Rational Interpolant and Bernstein "
                 "Ellipse For v")
    ax.plot([0, 4 * np.pi], [np.finfo(float).eps] * 2, "-r")
    _save(fig)

    c = [1, 2, 4, 5]
    t = [np.pi / 3 * cj_ for cj_ in c]
    fig, axs = plt.subplots(2, 2)
    for j, ax in enumerate(axs.ravel()):
        matlab_plot(v, ax=ax, lw=2)
        ax.set_aspect("equal")
        ax.grid(True)
        z = _at(u, t[j], range(3))
        ax.plot(z[:2].real, z[:2].imag, "ok", markersize=7,
                markerfacecolor="none")
        ax.plot(z[2].real, z[2].imag, "ok", markerfacecolor="k",
                markersize=7)
        _quiver(ax, z, _at(u, t[j], range(3, 6)), 0.4)
        ax.set_title(f"Configuration At t={c[j]:d}pi/3")
    _save(fig)

    p, q, rh, mu, nu, poles, res = _ratinterp(
        v, 157, 156, 0.0, dom, 7
    )
    fig, ax = plt.subplots()
    ax.plot([0, 4 * np.pi], [np.finfo(float).eps] * 2, "-r")
    zq = np.asarray(q.roots(complex_roots=True))
    matlab_plot(zq.real, zq.imag, "o", ax=ax, markersize=4, color="b",
                markerfacecolor="b")
    zp = np.asarray(p.roots(complex_roots=True))
    ax.plot(zp.real, zp.imag, "ok", markersize=5, markerfacecolor="none")
    ax.set_aspect("equal")
    ax.grid(True)
    plotregion(v, ax=ax)
    ax.set_title("Without Robustness")
    _save(fig)


if __name__ == "__main__":
    run()
