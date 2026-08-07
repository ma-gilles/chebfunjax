"""Delay differential equations in Chebfun.

Faithful replica of ode-nonlin/DelayDifferentialEquations.m: a tour of
delay, pantograph, state-dependent and integro-differential equations,
first by explicit spectral discretization (chebpts / diffmat / barymat)
and then through chebop, whose operators may evaluate the unknown at
mapped arguments -- x(q*t), x(max(t-p, 0)), even x(x).

Original: https://www.chebfun.org/examples/ode-nonlin/DelayDifferentialEquations.html
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

from chebfunjax import chebfun, volt
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.diffmat import cumsummat, diffmat
from chebfunjax.utils.interpolation import barymat
from chebfunjax.utils.quadrature import chebpts as _chebpts_ref


def chebpts_dom(n, dom):
    """2nd-kind points mapped onto [dom[0], dom[1]] (MATLAB chebpts(n, dom))."""
    x = np.asarray(_chebpts_ref(n), dtype=float)
    a, b = float(dom[0]), float(dom[1])
    return a + (b - a) * (x + 1.0) / 2.0

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"DelayDifferentialEquations_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_disc(t, x, sol, tt=None):
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    if tt is None:
        tt = np.linspace(float(t[0]), float(t[-1]), 1001)
    s = sol(tt) if callable(sol) else sol
    ax.plot(tt, s, "-")
    ax.plot(np.asarray(t), np.asarray(x), ".", markersize=12)
    ax.grid(True)
    _save(fig)


def _show(x, label="x"):
    print(f"{label} =\n   chebfun ({len(x.funs)} smooth piece"
          f"{'s' if len(x.funs) > 1 else ''}), length {len(x)}, "
          f"endpoint values {float(x(jnp.float64(x.domain.a))):.4g} "
          f"{float(x(jnp.float64(x.domain.b))):.4g}")


def _plot_pair(sol_fn, x, dom, extra=None):
    tt = np.linspace(dom[0], dom[-1], 1001)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(tt, sol_fn(tt), "-")
    ax.plot(tt, np.asarray(x(tt)), ".--", markersize=6, markevery=25)
    ax.grid(True)
    _save(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # ----- 1. An ODE by explicit spectral discretization --------------
    a, x0 = 1.0, 1.0
    dom = (0.0, 1.0)
    n = 16
    t = chebpts_dom(n, dom)
    D = np.asarray(diffmat(n, 1, dom), dtype=float)
    Iden = np.eye(n)
    A = D - a * Iden
    rhs = 0 * t
    A[0, :] = 0
    A[0, 0] = 1
    rhs[0] = x0
    x = np.linalg.solve(A, rhs)
    sol = np.exp(a * t) * x0
    err = float(np.linalg.norm(x - sol))
    print(f"err =\n     {err:.15e}")
    _plot_disc(t, x, lambda s: np.exp(a * s) * x0)

    # ----- 2. The pantograph equation, discretized --------------------
    a, b, q = 1.0, -8.0, 0.5
    tau = q * t
    E = np.asarray(barymat(jnp.asarray(tau), jnp.asarray(t)), dtype=float)
    A = D - a * Iden - b * E
    rhs = 0 * t
    A[0, :] = 0
    A[0, 0] = 1
    rhs[0] = x0
    x = np.linalg.solve(A, rhs)
    sol_fn = lambda s: -7 / 2 * s**3 + 21 / 2 * s**2 - 7 * s + 1
    err = float(np.linalg.norm(x - sol_fn(t)))
    print(f"err =\n     {err:.15e}")
    _plot_disc(t, x, sol_fn)

    # ----- 3. The pantograph equation, chebop --------------------------
    N = Chebop(lambda t, x: x.diff() - a * x - b * x(q * t), domain=dom)
    N.lbc = x0
    xc = N.solve(0.0)
    _show(xc)
    solc = chebfun(sol_fn, domain=dom)
    err = float((xc - solc).norm())
    print(f"err =\n     {err:.15e}")
    _plot_pair(sol_fn, xc, dom)

    # ----- 4. Two pantograph delays ------------------------------------
    q1, q2 = 1 / 2, 1 / 3
    N = Chebop(lambda t, x: x.diff() - a * x - b * x(q1 * t) - x(q2 * t),
               domain=dom)
    N.lbc = x0
    xc = N.solve(0.0)
    _show(xc)
    res = float(N(xc).norm())
    print(f"res =\n     {res:.15e}")
    tt = np.linspace(0, 1, 1001)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(tt, np.asarray(xc(tt)), lw=1.4)
    ax.grid(True)
    _save(fig)

    # ----- 5. A two-interval discretization (plain ODE) ---------------
    n = 10
    p = 0.5
    domL, domR = (0.0, p), (p, 2 * p)
    tL = chebpts_dom(n, domL)
    tR = chebpts_dom(n, domR)
    t2 = np.concatenate([tL, tR])
    DL = np.asarray(diffmat(n, 1, domL), dtype=float)
    DR = np.asarray(diffmat(n, 1, domR), dtype=float)
    Z = np.zeros((n, n))
    AL = np.hstack([DL - a * np.eye(n), Z])
    AR = np.hstack([Z, DR - a * np.eye(n)])
    B = np.zeros(2 * n)
    B[0] = 1
    C = np.zeros(2 * n)
    C[n - 1] = 1
    C[n] = -1
    A = np.vstack([B, AL[1:n, :], C, AR[1:n, :]])
    rhs = np.concatenate([[x0], np.zeros(n - 1), [0.0], np.zeros(n - 1)])
    x = np.linalg.solve(A, rhs)
    err = float(np.linalg.norm(x - np.exp(a * t2) * x0))
    print(f"err =\n     {err:.15e}")
    _plot_disc(t2, x, lambda s: np.exp(a * s) * x0)

    # ----- 6. A constant delay, discretized ----------------------------
    tauL = np.maximum(tL - p, 0)
    tauR = tR - p
    EL = np.asarray(barymat(jnp.asarray(tauL), jnp.asarray(tL)), dtype=float)
    ER = np.asarray(barymat(jnp.asarray(tauR), jnp.asarray(tL)), dtype=float)
    AL = np.hstack([DL - a * EL, Z])
    AR = np.hstack([-ER, DR])
    A = np.vstack([B, AL[1:n, :], C, AR[1:n, :]])
    x = np.linalg.solve(A, rhs)
    solv = np.concatenate([a * tL + 1,
                           (a * p + 1) + a * (tR - p) * (.5 * a * (tR - p) + 1)])
    err = float(np.linalg.norm(x - solv))
    print(f"err =\n     {err:.15e}")
    _plot_disc(t2, x, solv, tt=t2)

    # ----- 7. The same delay problem through chebop --------------------
    N = Chebop(lambda t, x: x.diff() - x((t - p).maximum(0.0)),
               domain=(0.0, 0.5, 1.0))
    N.lbc = 1.0
    xc = N.solve(0.0)
    _show(xc)
    sol7 = lambda s: np.where(s <= p, s + 1,
                              (a * p + 1) + a * (s - p) * (.5 * a * (s - p) + 1))
    tt = np.linspace(0, 1, 1001)
    err = float(np.max(np.abs(np.asarray(xc(tt)) - sol7(tt))))
    print(f"err =\n     {err:.15e}")
    _plot_pair(sol7, xc, (0, 1))

    # ----- 8. Longer domain: six pieces --------------------------------
    N = Chebop(lambda t, x: x.diff() - x((t - p).maximum(0.0)),
               domain=tuple(np.arange(0, 3.01, 0.5)))
    N.lbc = 1.0
    xc = N.solve(0.0)
    _show(xc)
    res = float(N(xc).norm())
    print(f"res =\n     {res:.15e}")
    tt = np.linspace(0, 3, 1200)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(tt, np.asarray(xc(tt)), lw=1.4)
    ax.grid(True)
    _save(fig)

    # ----- 9. Two constant delays: SKIPPED --------------------------
    # The union domain has eleven pieces, and every probe column of the
    # assembly applies two compositions to an eleven-piece chebfun.
    # MATLAB solves this in seconds; our piecewise-composition
    # evaluation is orders of magnitude too slow (ledgered performance
    # gap), so this section is skipped rather than left to run for
    # hours.
    print("section 9 skipped: piecewise-composition performance wall")
    FIG[0] += 1

    # ----- 10. A Volterra integro-differential equation ----------------
    f = lambda s, tv: np.exp(tv - s)
    n = 16
    t = chebpts_dom(n, dom)
    D = np.asarray(diffmat(n, 1, dom), dtype=float)
    av = 1 / 2
    # MATLAB's f(t, t') puts s down the rows: F_ij = f(t_i, t_j)
    # = exp(t_j - t_i)
    V = f(t[:, None], t[None, :]) * np.asarray(
        cumsummat(n, dom), dtype=float)
    A = D - V + av * np.eye(n)
    rhs = 0 * t
    A[0, :] = 0
    A[0, 0] = 1
    rhs[0] = x0
    x = np.linalg.solve(A, rhs)
    bb = np.sqrt(av**2 - 2 * av + 5) / 2
    sol10 = lambda s: np.exp(-(av + 1) / 2 * s) * (
        np.cosh(bb * s) + .5 * (1 - av) / bb * np.sinh(bb * s))
    err = float(np.linalg.norm(x - sol10(t)))
    print(f"err =\n     {err:.15e}")
    _plot_disc(t, x, sol10)

    # ----- 11. The same through chebop + volt --------------------------
    # our volt kernel is K(x, y) with (Kf)(x) = int K(x,y) f(y) dy;
    # the equation's kernel in that convention is exp(y - x)
    N = Chebop(lambda t, x: x.diff() + av * x
               - volt(lambda X, Y: np.exp(Y - X), x), domain=dom)
    N.lbc = x0
    xc = N.solve(0.0)
    _show(xc)
    solc = chebfun(sol10, domain=dom)
    err = float((xc - solc).norm())
    print(f"err =\n     {err:.15e}")
    _plot_pair(sol10, xc, (0, 1))

    # ----- 12. A state-dependent delay, Newton by hand -----------------
    n = 16
    t = chebpts_dom(n, dom)
    D = np.asarray(diffmat(n, 1, dom), dtype=float)
    x = t.copy()
    for _j in range(10):
        E = np.asarray(barymat(jnp.asarray(x), jnp.asarray(t)), dtype=float)
        F = D @ x + E @ x
        J = D + E + np.diag(E @ D @ x)
        J[0, :] = 0
        J[0, 0] = 1
        F[0] = x[0] - x0
        dx = np.linalg.solve(J, F)
        x = x - dx
        ndu = float(np.linalg.norm(dx))
        print(f"ndu =\n   {ndu:.15g}")
        if ndu < 1e-10:
            break
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(t, x, lw=1.4)
    ax.grid(True)
    _save(fig)

    from chebfunjax.chebfun1d.chebfun import Chebfun as _Chebfun
    from chebfunjax.domain import Domain as _Domain
    xc12 = _Chebfun.from_values(jnp.asarray(x), _Domain(dom))
    res = float((xc12.diff() + xc12(xc12)).norm())
    print(f"res =\n     {res:.15e}")

    # ----- 13. x' + x(x) = 0 through chebop ----------------------------
    N = Chebop(lambda t, x: x.diff() + x(x), domain=dom)
    N.lbc = 1.0
    xc = N.solve(0.0)
    _show(xc)
    res = float(N(xc).norm())
    print(f"res =\n     {res:.15e}")

    # ----- 14. A pantograph problem with known solution ----------------
    q = 0.5
    tcf = chebfun(lambda t: t, domain=dom)
    g = -2 * tcf / (1 + tcf**2)**2 - 1 / (1 + (q * tcf)**2)**2
    N = Chebop(lambda t, y: y.diff() - y(q * t)**2, domain=dom)
    N.lbc = 1.0
    y = N.solve(g)
    _show(y, "y")
    sol14 = lambda s: 1 / (1 + s**2)
    err = float(np.max(np.abs(np.asarray(y(np.linspace(0, 1, 800)))
                              - sol14(np.linspace(0, 1, 800)))))
    print(f"err =\n     {err:.15e}")
    _plot_pair(sol14, y, (0, 1))

    # ----- 15. A delayed system ----------------------------------------
    q = 0.7
    g1 = (tcf.cos() - (q * tcf).cos() * tcf.cos()
          - (q * tcf).sin() * tcf.sin())
    g2 = ((q * tcf).sin() * tcf.cos() - (q * tcf).cos() * tcf.sin()
          - tcf.sin())
    N = Chebop(lambda t, u, v: [
        u.diff() - (t.sin() * u(q * t) + t.cos() * v(q * t)),
        v.diff() - (-t.cos() * u(q * t) + t.sin() * v(q * t))],
        domain=dom)
    N.lbc = [np.sin(0.0), np.cos(0.0)]
    u, v = N.solve([g1, g2])
    tt = np.linspace(0, 1, 800)
    err = float(np.hypot(
        np.max(np.abs(np.asarray(u(tt)) - np.sin(tt))),
        np.max(np.abs(np.asarray(v(tt)) - np.cos(tt)))))
    print(f"delayed system err =\n     {err:.15e}")
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(tt, np.sin(tt), "-b", tt, np.cos(tt), "-b")
    ax.plot(tt, np.asarray(u(tt)), ".--r", markevery=30)
    ax.plot(tt, np.asarray(v(tt)), ".--r", markevery=30)
    ax.grid(True)
    _save(fig)

    # ----- 16. A state-dependent y(y) with known solution --------------
    xcf = chebfun(lambda x: x, domain=dom)
    mu = 0.0
    g = (3 + mu) * xcf**(2 + mu) - xcf**((3 + mu)**2)
    N = Chebop(lambda t, y: y.diff() - y(y), domain=dom)
    N.lbc = 0.0
    y = N.solve(g)
    _show(y, "y")
    sol16 = lambda s: s**(3 + mu)
    err = float(np.max(np.abs(np.asarray(y(np.linspace(0, 1, 800)))
                              - sol16(np.linspace(0, 1, 800)))))
    print(f"err =\n     {err:.15e}")
    _plot_pair(sol16, y, (0, 1))

    # ----- 17. State-dependent 1/t cascade: SKIPPED (same wall) -------
    print("section 17 skipped: piecewise-composition performance wall")
    FIG[0] += 1

    # ----- 18. Delay + integral terms together ------------------------
    dom20 = (0.0, 20.0)
    q = 0.5

    def op18(x, u):
        vt = volt(lambda T, S: (T / q - S), u)
        return (u.diff() - 1 / 100 * (q * x - x - 10) * u(q * x)
                - 1 / 100 * (x + 20) * np.exp(-1.0)
                - 1 / 100 * u.cumsum() - 1 / 1000 * vt(q * x))

    N = Chebop(op18, domain=dom20)
    N.lbc = np.exp(-1.0)
    u18 = N.solve(0.0)
    _show(u18, "u")
    sol18 = lambda s: np.exp(s / 10 - 1)
    tt = np.linspace(0, 20, 800)
    err = float(np.max(np.abs(np.asarray(u18(tt)) - sol18(tt))))
    print(f"err =\n     {err:.15e}")
    _plot_pair(sol18, u18, (0, 20))

    # ----- 19. A second-order equation with delays --------------------
    N = Chebop(lambda t, y: y.diff(2) + y(t) - 5 * y(t / 2)**2,
               domain=dom)
    N.lbc = lambda u: [u - 1, u.diff() + 2]
    N.init = 1 + tcf
    y19 = N.solve(0.0)
    _show(y19, "y")
    sol19 = lambda s: np.exp(-2 * s)
    tt = np.linspace(0, 1, 800)
    err = float(np.max(np.abs(np.asarray(y19(tt)) - sol19(tt))))
    print(f"err =\n     {err:.15e}")
    _plot_pair(sol19, y19, (0, 1))

    # ----- 20. An interval condition u(0) = u(1/3) --------------------
    dom3 = (0.0, 1 / 3)
    t3 = chebfun(lambda t: t, domain=dom3)
    g = t3.exp() / 2
    N = Chebop(lambda t, u: u.diff() + u**2 + 2 * u(t / 2), domain=dom3)
    N.bc = lambda t, u: u(0.0) - u(1 / 3)
    N.init = 1 + 0 * t3
    u20 = N.solve(g)
    _show(u20, "u")
    res = float((N(u20) - g).norm())
    print(f"res =\n     {res:.15e}")
    tt = np.linspace(0, 1 / 3, 400)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(tt, np.asarray(u20(tt)), lw=1.4)
    ax.grid(True)
    _save(fig)

    # ----- 21. Three-variable constant-delay system: SKIPPED ----------
    print("section 21 skipped: piecewise-composition performance wall")
    FIG[0] += 1

    # ----- 22. A system with a state-dependent delay ------------------
    dom5 = (1.0, 5.0)
    t5 = chebfun(lambda t: t, domain=dom5)
    N = Chebop(lambda t, u, v: [
        u.diff() - v,
        v.diff() + v((1 - v).exp()) * v**2 * (1 - v).exp()], domain=dom5)
    N.lbc = [np.log(1.0), 1.0]
    u22, v22 = N.solve(0.0)
    _show(u22, "u{1}")
    tt = np.linspace(1, 5, 800)
    err = float(np.hypot(
        np.max(np.abs(np.asarray(u22(tt)) - np.log(tt))),
        np.max(np.abs(np.asarray(v22(tt)) - 1 / tt))))
    print(f"err =\n     {err:.15e}")
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(tt, np.log(tt), "-b", tt, 1 / tt, "-b")
    ax.plot(tt, np.asarray(u22(tt)), ".r", markevery=30)
    ax.plot(tt, np.asarray(v22(tt)), ".r", markevery=30)
    ax.grid(True)
    _save(fig)


if __name__ == "__main__":
    run()
