"""Spread option in 2D Black-Scholes.

Translation of applics/BlackScholes2D.m (Glau, Hashemi,
Mahlstedt, Poetz, 2017): the spread-option price in the 2D
Black-Scholes model as a trivariate function of (T, K, rho),
approximated by a Chebfun3 at tolerance 1e-5 -- each sample is a 2D
Clenshaw-Curtis quadrature of the payoff against the correlated
bivariate normal density -- with the accuracy/speedup check on a
15^3 evaluation grid.

Original: https://www.chebfun.org/examples/applics/BlackScholes2D.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.quadrature import chebpts, chebweights

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'applics')

S01, S02 = 1.0, 1.0
R = 0.0
X01, X02 = np.log(S01), np.log(S02)
SIG1, SIG2 = 0.3, 0.3
DOM = (0.3, 2.0, 0.3, 2.0, -0.9, 0.9)


def _disp_chebfun3(f):
    """MATLAB @chebfun3/disp."""
    r1, r2, r3 = f.rank
    m, n, p = f.length()
    w = max(len(str(r)) for r in (r1, r2, r3))
    dom = ", ".join(f"{v:.3g}" for v in f.domain).split(", ")
    print("   chebfun3 object ")
    print(f"   cols: Inf x {r1:<{w}d} chebfun")
    print(f"   rows: Inf x {r2:<{w}d} chebfun")
    print(f"  tubes: Inf x {r3:<{w}d} chebfun")
    print(f"   core: {r1} x {r2} x {r3}")
    print(f" length: {m}, {n}, {p}")
    print(f" domain: [{dom[0]}, {dom[1]}] x [{dom[2]}, {dom[3]}] x "
          f"[{dom[4]}, {dom[5]}]")
    print(f" vertical scale = {f.vscale():.2g}")


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # 150-point Clenshaw-Curtis rule on [-10, 10] in each variable.
    xg = np.asarray(chebpts(150)) * 10.0
    wg = np.asarray(chebweights(150, kind=2)) * 10.0
    XX, YY = np.meshgrid(xg, xg)

    def payoff(K, x1, x2):
        return np.maximum(np.exp(X01 + x1) - np.exp(X02 + x2) - K, 0.0)

    def normdensity_2d(s1, s2, rho, x1, x2, mu1, mu2):
        det = (s1**2) * (s2**2) * (1 - rho**2)
        q = ((s2**2) * (x1 - mu1)**2
             - 2 * rho * s1 * s2 * (x1 - mu1) * (x2 - mu2)
             + (s1**2) * (x2 - mu2)**2)
        return np.exp(-0.5 * q / det) / (np.sqrt(det) * 2 * np.pi)

    def price(T, K, rho):
        F = payoff(K, XX, YY) * normdensity_2d(
            SIG1 * np.sqrt(T), SIG2 * np.sqrt(T), rho, XX, YY,
            (R - 0.5 * SIG1**2) * T, (R - 0.5 * SIG2**2) * T)
        return float(np.exp(-R * T) * (wg @ F @ wg))

    t0 = time.time()
    chebPrice = chebfun3(
        lambda T, K, rho: np.vectorize(price)(T, K, rho),
        domain=DOM, tol=1e-5)
    print("chebPrice =")
    _disp_chebfun3(chebPrice)
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")

    # slice(chebPrice, 2, 0.3, -0.9); campos([-10 10 10])
    fig, ax = chebPrice.slice(2.0, 0.3, -0.9)
    ax.view_init(elev=35, azim=142)
    ax.set_xlabel('T')
    ax.set_ylabel('K')
    ax.set_zlabel(r'$\rho$')
    _savefig(fig, os.path.join(_IMG, "BlackScholes2D_01.png"))
    plt.close(fig)

    # Accuracy and speedup check on a 15^3 grid.
    M = 15
    T = np.linspace(DOM[0], DOM[1], M)
    K = np.linspace(DOM[2], DOM[3], M)
    rho = np.linspace(DOM[4], DOM[5], M)
    V = np.zeros((M, M, M))
    t0 = time.time()
    for i in range(M):
        for j in range(M):
            for k in range(M):
                V[i, j, k] = price(T[i], K[j], rho[k])
    print("time_price =")
    print(f"    {time.time() - t0:.4f}")

    TT, KK, RR = np.meshgrid(T, K, rho, indexing="ij")
    t0 = time.time()
    VCheb = np.asarray(chebPrice(TT, KK, RR))
    print("time_ChebPrice =")
    print(f"    {time.time() - t0:.4f}")

    err = float(np.max(np.abs(V - VCheb)))
    print("err =")
    print(f"   {err:.4e}")


if __name__ == "__main__":
    run()
