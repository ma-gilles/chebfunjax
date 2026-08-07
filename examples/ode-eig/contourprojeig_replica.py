"""Eigenvalues of differential operators by contour integral projection.

Faithful replica of ode-eig/ContourProjEig.m by Anthony Austin (May
2013): a FEAST-like algorithm computing the eigenvalues of
L = -d^2/dx^2 on [0, pi] inside [0, 10] by projecting three arbitrary
functions onto the eigenspace with an 8-point contour quadrature (each
node a complex shifted BVP solve), then a 3x3 Rayleigh-Ritz problem.

The three "arbitrary functions" are chebfuns built from random data at
32 Chebyshev points; the data (MATLAB rng(67714070), 2*randn(32,3)) is
inlined verbatim since MATLAB's and numpy's randn streams differ.

Original: https://www.chebfun.org/examples/ode-eig/ContourProjEig.html
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

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun, Domain
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-eig')

YDATA = [
    [1.321832265766146,
     -2.2038026174084235,
     1.2896539568433243,
     -1.9213113860136342,
     1.4194840793490016,
     1.9848066828548527,
     -5.910765678968401,
     1.7102217167746556,
     -1.0859339806236827,
     -3.268818474007753,
     -1.4377765989838451,
     0.3976908692018349,
     2.1847819599379803,
     3.153301820594084,
     0.6118886144706419,
     -0.5466976639802201,
     -0.8486070897325596,
     -1.6685057408853234,
     2.6437936987560198,
     1.256422124189088,
     -3.132898590560586,
     -1.6522578689920226,
     -1.715537657110907,
     -1.4458670722584477,
     0.7141143810567787,
     -0.599324360731327,
     0.523764051591279,
     0.49932620545542455,
     -1.6972561765036833,
     5.270521163943548,
     2.43504000458663,
     -1.5885409040683176],
    [0.8526095842820595,
     -1.4061146337621473,
     -1.469773804550658,
     0.9929466429477789,
     7.157689649007411,
     1.1256587809057892,
     5.203495906067041,
     -1.139358356771769,
     -0.285253014753592,
     0.9874431832998773,
     0.1992676181171017,
     -0.7456194247068845,
     -0.7485867164707447,
     -1.5071050888565753,
     0.2654727179539723,
     -2.7178353209597614,
     1.1458039698218416,
     -2.9519163476883103,
     1.0795684226496836,
     -1.3872774624372142,
     1.057599465079778,
     2.4639206254626926,
     -3.554704151815086,
     0.5873051021711939,
     0.6992173292302674,
     -1.5734464292342636,
     -0.4261741089846725,
     2.4176064017672103,
     2.3495631450128562,
     0.7915480163242729,
     -1.0958298554428239,
     -2.959580194544783],
    [1.4047942126223765,
     -0.7632992227467811,
     0.4492889056826361,
     -1.791188629189427,
     -1.7824409765015141,
     1.271595431710643,
     1.5670825950426865,
     -2.4439537483625844,
     -0.9063702969720815,
     -2.5079066372764856,
     -1.9922990201478685,
     2.288688670243047,
     0.694456929962654,
     0.8360091853580479,
     -1.1101119159112514,
     -0.9411518237324088,
     -2.6437273449300447,
     1.6177710314221878,
     2.0910432172951023,
     -1.3245847001928859,
     1.6977768330794005,
     -5.233774207964326,
     -2.2055015868178716,
     -5.242226530396801,
     -2.7846041622499356,
     0.9127523172408534,
     -1.8056831824271262,
     3.0517306472059778,
     -6.5680514725263395,
     -3.287697708604017,
     -0.03177498682579377,
     1.3866486402141467]]


def _plot(fs, title, fname, ylim=None):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    xx = np.linspace(0, np.pi, 2000)
    for f in fs:
        ax.plot(xx, np.asarray(f(xx)).real, lw=2)
    ax.set_xlim(0, np.pi)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=14)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    L = Chebop(lambda x, u: -u.diff(2), domain=(0, np.pi))
    L.lbc = 0.0
    L.rbc = 0.0
    print(L)

    t0 = time.time()
    dom = Domain((0.0, np.pi))
    Y = [Chebfun.from_values(jnp.asarray(np.array(c)), dom) for c in YDATA]
    _plot(Y, "Three arbitrary functions", "ContourProjEig_repl_01.png")

    # 8-point midpoint-rule contour: circle of radius 5 about 5.
    K = 8
    omega_k = np.exp(2j * np.pi * np.arange(K) / K + 1j * np.pi / K)
    zk = 5 * omega_k + 5
    wk = omega_k / 5

    W = [0.0 * Y[0] for _ in range(3)]
    for k in range(K // 2):
        Ls = Chebop(lambda x, u, _z=zk[k]: _z * u + u.diff(2),
                    domain=(0, np.pi))
        Ls.lbc = 0.0
        Ls.rbc = 0.0
        for j in range(3):
            sol = Ls.solve(Y[j])
            W[j] = W[j] + 2 * 5 * (wk[k] * sol).real()
    W = [w * (1.0 / K) for w in W]

    # Rayleigh-Ritz: 3x3 generalized eigenproblem with function inner
    # products.
    LW = [L(w) for w in W]
    A3 = np.array([[float((W[i] * LW[j]).sum()) for j in range(3)]
                   for i in range(3)])
    B3 = np.array([[float((W[i] * W[j]).sum()) for j in range(3)]
                   for i in range(3)])
    import scipy.linalg as sla
    D, V3 = sla.eig(A3, B3)
    T = time.time() - t0
    print("T =")
    print(f"   {T:.9f}")

    print("ans =")
    for d in D.real:
        print(f"   {d:.15f}")

    # L2-normalized eigenfunctions.
    F = []
    for j in range(3):
        f = W[0] * V3[0, j].real + W[1] * V3[1, j].real \
            + W[2] * V3[2, j].real
        F.append(f * (1.0 / float(f.norm(2))))
    _plot(F, "Eigenfunctions of L = -d^2/dx^2",
          "ContourProjEig_repl_02.png", ylim=(-1, 1))

    # Comparison with eigs.
    t0 = time.time()
    lam = np.sort(np.asarray(L.eigs(k=3)).real)
    print("ans =")
    for d in lam:
        print(f"   {d:.15f}")
    print("T =")
    print(f"   {time.time() - t0:.9f}")


if __name__ == "__main__":
    run()
