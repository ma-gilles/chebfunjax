"""The three-body problem: the figure-eight orbit and its singularities.

Faithful replica of ode-nonlin/ThreeBodyProblem.m: the celebrated
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

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax import ratinterp
from chebfunjax.chebfun1d.chebfun import ode113
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]
D = (0.0, 4 * np.pi)


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"ThreeBodyProblem_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _bernstein_ellipse(ax, f):
    """Bernstein ellipse of f's coefficient decay, mapped to D."""
    c = np.abs(np.asarray(f.funs[0].tech.coeffs))
    c = c[c > 0]
    n = len(c)
    # decay rate rho from the coefficient envelope
    k = np.arange(n)
    mask = c > 1e-13 * c.max()
    rho = np.exp(-np.polyfit(k[mask], np.log(c[mask]), 1)[0])
    th = np.linspace(0, 2 * np.pi, 400)
    z = 0.5 * (rho * np.exp(1j * th) + np.exp(-1j * th) / rho)
    mid, h = 0.5 * (D[0] + D[1]), 0.5 * (D[1] - D[0])
    ax.plot(mid + h * z.real, h * z.imag, "-g", lw=1)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    a = 6.32591398 / (2 * np.pi)      # scaling: period 2*pi

    def fun(t, u):
        p, v = u[:3], u[3:]
        acc = [sum((p[j] - p[i]) / abs(p[j] - p[i])**3
                   for j in range(3) if j != i) for i in range(3)]
        return a * np.array([v[0], v[1], v[2], acc[0], acc[1], acc[2]])

    y0 = np.array([0.540508553669932 + 0.345263318559681j,
                   0.540508532338285 - 0.345263317862853j,
                   -1.081017086008497 - 0.000000000697245j,
                   -1.097122372968180 - 0.233604741427372j,
                   1.097122377013713 - 0.233604786311327j,
                   -0.000000004046108 + 0.467209527738458j])
    u = ode113(fun, D, y0, rtol=1e-13, atol=1e-13)
    v = u[2]

    # the figure-eight
    tt = np.linspace(*D, 4000)
    zv = np.asarray(v(tt))
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(zv.real, zv.imag, lw=1.6)
    p0 = [complex(u[k](np.float64(0.0))) for k in range(3)]
    vel0 = [complex(u[k](np.float64(0.0))) for k in range(3, 6)]
    ax.plot([z.real for z in p0], [z.imag for z in p0], "ok",
            markersize=7, markerfacecolor="k")
    ax.quiver([z.real for z in p0], [z.imag for z in p0],
              [w.real for w in vel0], [w.imag for w in vel0],
              width=0.004, color="k")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Figure of Eight Solution to Three Body Problem")
    _save(fig)

    print(f"v = chebfun, length {len(v)}, complex values, "
          f"interval [0, 4*pi]")

    # robust rational approximation
    rh, p, q, mu, nu, poles = ratinterp(
        v, 151, 150, None, None, 1e-12, domain=D)[:6]
    print(f"mu =\n   {mu}")
    print(f"nu =\n     {nu}")
    tl = np.linspace(0, 4 * np.pi, 100)
    err = float(np.max(np.abs(np.asarray(rh(tl)) - np.asarray(v(tl)))))
    print(f"max|rh - v| =\n     {err:.15e}")
    pol = np.asarray(poles)
    pol = pol[np.argsort(pol.real)]
    print("poles =")
    for z in pol:
        print(f"   {z.real:.6f} {z.imag:+.6f}i")
    print("real(poles)*3/pi =")
    print("  ", np.round(np.real(pol) * 3 / np.pi, 4))

    # poles and the Bernstein ellipse
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(pol.real, pol.imag, "o", markersize=5, color="b",
            markerfacecolor="b")
    ax.plot([0, 4 * np.pi], [0, 0], "-r", lw=1.4)
    _bernstein_ellipse(ax, v)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Poles of Rational Interpolant and Bernstein "
                 "Ellipse For v")
    _save(fig)

    # configurations at t = c*pi/3
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6))
    for ax, c in zip(axes.ravel(), (1, 2, 4, 5)):
        t0 = np.pi / 3 * c
        ax.plot(zv.real, zv.imag, lw=2)
        pos = [complex(u[k](np.float64(t0))) for k in range(3)]
        vel = [complex(u[k](np.float64(t0))) for k in range(3, 6)]
        ax.plot([z.real for z in pos[:2]], [z.imag for z in pos[:2]],
                "ok", markersize=7)
        ax.plot(pos[2].real, pos[2].imag, "ok", markersize=7,
                markerfacecolor="k")
        ax.quiver([z.real for z in pos], [z.imag for z in pos],
                  [w.real for w in vel], [w.imag for w in vel],
                  width=0.005, color="k")
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_title(f"Configuration At t={c}pi/3")
    _save(fig)

    # tol = 0: spurious poles appear
    rh0, p0_, q0_, mu0, nu0, poles0 = ratinterp(
        v, 157, 156, None, None, 0.0, domain=D)[:6]
    pol0 = np.asarray(poles0)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot([0, 4 * np.pi], [0, 0], "-r", lw=1.4)
    ax.plot(pol0.real, pol0.imag, "o", markersize=4, color="b",
            markerfacecolor="b")
    _bernstein_ellipse(ax, v)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("tol = 0: spurious poles join the genuine ones")
    _save(fig)
    print(f"tol=0: mu={mu0} nu={nu0} npoles={len(pol0)}")


if __name__ == "__main__":
    run()
