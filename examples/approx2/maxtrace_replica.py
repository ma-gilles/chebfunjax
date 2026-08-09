"""Maximum trace problems.

Faithful replica of approx2/MaxTrace.m (Hashemi, 2016): maximizing
trace(G' f G) over infinity-by-k orthonormal quasimatrices G via the
spectral decomposition of the symmetric kernel f -- the solution is
the k eigenfunctions of largest eigenvalue, computed from the
chebfun2 SVD with the sign trick of the NearestPSDKernel example.

Original: https://www.chebfun.org/examples/approx2/MaxTrace.html
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

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx2')

# shared Gauss-Legendre rule on [-1, 1]
_TQ, _WQ = np.polynomial.legendre.leggauss(600)


def _ip(u, v):
    """L2(-1,1) inner product of two 1D chebfuns."""
    return float(np.sum(_WQ * np.asarray(u(_TQ)) * np.asarray(v(_TQ))))


def ev(f, k):
    """Eigenfunctions of the k algebraically largest eigenvalues of the
    symmetric chebfun2 f (MATLAB subfunction ev in MaxTrace.m)."""
    U, S, V = f.svd(full=True)
    s = np.array([_ip(U[i], V[i]) for i in range(len(U))])
    lam = np.sign(s) * np.asarray(S)
    idx = np.argsort(lam)
    ind = idx[-k:][::-1]
    return [U[i] for i in ind], lam[ind]


def _display(name, F):
    xa, xb, ya, yb = F.domain
    cv = [float(F(np.array([x]), np.array([y]))[0])
          for (x, y) in [(xa, ya), (xb, ya), (xa, yb), (xb, yb)]]
    g = np.linspace(-1, 1, 151)
    X, Y = np.meshgrid(g, g)
    vs = float(np.max(np.abs(np.asarray(F(X, Y)))))
    print(f"{name} =")
    print("   chebfun2 object")
    print("       domain                 rank       corner values")
    print(f"[{xa:4.0f},{xb:4.0f}] x [{ya:4.0f},{yb:4.0f}]"
          f"     {int(F.rank):4d}     "
          f"[{cv[0]:.2g} {cv[1]:.2g} {cv[2]:.2g} {cv[3]:.2g}]")
    print(f"vertical scale = {vs:.2g}")


def _plot_eigfuns(Us, title, fname):
    xs = np.linspace(-1, 1, 600)
    fig = plt.figure(figsize=(8.0, 5.6))
    ax = fig.add_subplot(projection="3d")
    for j, u in enumerate(Us, start=1):
        ax.plot(np.full_like(xs, j), xs, np.asarray(u(xs)), lw=1.6)
    ax.set_title(title)
    ax.view_init(48, -37 - 90)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=140, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # --- square peg (cheb.gallery2('squarepeg')) ---
    def squarepeg(x, y):
        return 1 / ((1 + (2 * x)**20) * (1 + (2 * y)**20))

    f = Chebfun2.from_function(squarepeg)
    _display("f", f)
    U1, _ = ev(f, 1)
    g = np.linspace(-1, 1, 260)
    X, Y = np.meshgrid(g, g)
    fig, ax = plt.subplots(figsize=(7.4, 5.6),
                           subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, np.asarray(f(X, Y)), cmap="summer",
                    rstride=1, cstride=1, linewidth=0)
    ax.set_zlim(0, 1)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "MaxTrace_repl_01.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    xs = np.linspace(-1, 1, 600)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(xs, np.asarray(U1[0](xs)), lw=1.6)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "MaxTrace_repl_02.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- tilted peg ---
    print("ff = ")
    print("    @(x,y)1./((1+(x+y).^20).*(1+(y-x).^20))")

    def tilted(x, y):
        return 1 / ((1 + (x + y)**20) * (1 + (y - x)**20))

    f = Chebfun2.from_function(tilted)
    U4, _ = ev(f, 4)
    _plot_eigfuns(U4, "4 eigenfunctions of the tiltedpeg",
                  "MaxTrace_repl_03.png")

    # --- waffle (cheb.gallery2('waffle')) ---
    def waffle(x, y):
        return 1 / (1 + 1e3 * ((x**2 - .25)**2 * (y**2 - .25)**2))

    f = Chebfun2.from_function(waffle)
    _display("f", f)
    print("ff = ")
    print("    @(x,y)1./(1+1e3*((x.^2-.25).^2.*(y.^2-.25).^2))")
    U5, _ = ev(f, 5)
    _plot_eigfuns(U5, "5 eigenfunctions of the waffle",
                  "MaxTrace_repl_04.png")

    # --- multiquadric ---
    c = 0.6

    def mq(x, y):
        return np.sqrt((x**2 + y**2) + c**2)

    f = Chebfun2.from_function(mq)
    _display("f", f)
    U5, lam5 = ev(f, 5)
    _plot_eigfuns(U5, "5 eigenfunctions of the multiquadric",
                  "MaxTrace_repl_05.png")

    # trace(U'*f*U) via quadrature: (fU_j)(x) = int f(x,y) U_j(y) dy
    TX, TY = np.meshgrid(_TQ, _TQ)  # (y, x)
    FV = mq(TX.T, TY.T)             # FV[i, j] = f(x_i, y_j)

    def trace_quad(Us):
        tot = 0.0
        for u in Us:
            uv = np.asarray(u(_TQ))
            fu = FV @ (_WQ * uv)         # (f u)(x_i)
            tot += float(np.sum(_WQ * uv * fu))
        return tot

    optimal = trace_quad(U5)
    print("optimal =")
    print(f"   {optimal:.15f}")

    # U2 = qr(x.^(1:5)): orthonormalize [x, x^2, ..., x^5] in L2(-1,1)
    Vdm = np.stack([_TQ**j for j in range(1, 6)], axis=1)
    Q, _ = np.linalg.qr(np.sqrt(_WQ)[:, None] * Vdm)
    Q = Q / np.sqrt(_WQ)[:, None]

    tot = 0.0
    for j in range(5):
        uv = Q[:, j]
        fu = FV @ (_WQ * uv)
        tot += float(np.sum(_WQ * uv * fu))
    print("leg_trace =")
    print(f"   {tot:.15f}")


if __name__ == "__main__":
    run()
