"""Eigenfunctions of the Laplacian on the disk.

Faithful replica of disk/Eigenfunctions.m by Heather Wilber, January
2017: cylindrical harmonics as Laplacian eigenfunctions, their
orthonormality, Neumann variants, and an eigenfunction expansion of a
smooth function.

Original: https://www.chebfun.org/examples/disk/Eigenfunctions.html
Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'disk')


def _release():
    # Cumulative XLA cache entries from many distinct-shape harmonic
    # constructions slow later compiles ~100x (guide17 pattern); release
    # between sections.
    import gc

    import jax
    jax.clear_caches()
    gc.collect()


def _surf(u, title, stem):
    th = np.linspace(-np.pi, np.pi, 120)
    r = np.linspace(0, 1, 60)
    T, R = np.meshgrid(th, r)
    V = np.asarray(u(jnp.asarray(T.ravel()),
                     jnp.asarray(R.ravel()))).reshape(T.shape)
    X, Y = R * np.cos(T), R * np.sin(T)
    fig = plt.figure(figsize=(5.2, 4.4))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, np.real(V), cmap="viridis", linewidth=0)
    ax.set_axis_off()
    ax.set_title(title)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, stem + ".png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    _release()


def run():
    os.makedirs(_IMG, exist_ok=True)

    u42 = Diskfun.harmonic(4, 2)
    print("u42 =")
    print(repr(u42))
    _surf(u42, r"$u_{4,2}$", "Eigenfunctions_repl_01")

    lam = float(np.asarray(cj.chebfun(
        lambda x: jnp.asarray(jv(4, np.asarray(x))),
        domain=[10, 13]).roots())[0])
    resid = u42.lap() + u42 * (lam ** 2)
    th = np.linspace(-np.pi, np.pi, 400)
    r = np.linspace(0, 1, 100)
    T, R = np.meshgrid(th, r)
    rv = np.asarray(resid(jnp.asarray(T.ravel()),
                          jnp.asarray(R.ravel())))
    print("ans =")
    print(f"     {float(np.max(np.abs(rv))):.15e}")

    for (Lm, n), stem in [((0, 1), "02"), ((0, 2), "03"),
                          ((0, 3), "04"), ((0, 4), "05"),
                          ((-2, 1), "06"), ((-3, 2), "07"),
                          ((3, 3), "08"), ((11, 7), "09")]:
        _surf(Diskfun.harmonic(Lm, n),
              rf"$u_{{{Lm},{n}}}$", f"Eigenfunctions_repl_{stem}")

    uN21 = Diskfun.harmonic(2, 1, "neumann")
    uN34 = Diskfun.harmonic(3, 4, "neumann")
    _surf(uN21, "u21 with Neumann bc", "Eigenfunctions_repl_10")
    _surf(uN34, "u34 with Neumann bc", "Eigenfunctions_repl_11")

    u01 = Diskfun.harmonic(0, 1)
    u02 = Diskfun.harmonic(0, 2)
    u03 = Diskfun.harmonic(0, 3)
    v22 = Diskfun.harmonic(-3, 2)
    u117 = Diskfun.harmonic(11, 7)
    print("int1 =")
    print(f"     {float((u01 * u02).sum2()):.15e}")
    print("int2 =")
    print(f"    {float((v22 * u117).sum2()):.15e}")
    print("int3 =")
    print(f"   {float((u03 * u03).sum2()):.15f}")

    # -- Eigenfunction expansion of a smooth function -----------------
    f = Diskfun.from_function(
        lambda t, r: 20 * (1 - r ** 2) ** 2
        * jnp.exp(-6 * (r * jnp.cos(t) + 0.25) ** 2
                  - 6 * (r * jnp.sin(t) - 0.2) ** 2))
    N = NN = 7
    rows = []
    for m in range(0, N + 1):
        for n in range(1, NN + 1):
            reps = 1 if m == 0 else 2
            for j in range(1, reps + 1):
                sgn = (-1) ** j
                H = Diskfun.harmonic(sgn * m, n)
                rows.append((float((f * H).sum2()), sgn * m, n))
                if len(rows) % 10 == 0:
                    _release()
    coeffs = np.array(rows)

    fig = plt.figure(figsize=(6.0, 4.6))
    ax = fig.add_subplot(projection="3d")
    ax.stem(coeffs[:, 1], coeffs[:, 2], np.abs(coeffs[:, 0]))
    ax.set_zscale("log") if hasattr(ax, "set_zscale") else None
    ax.set_xlabel("m")
    ax.set_ylabel("n")
    ax.set_zlabel("abs. value of coeffs")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Eigenfunctions_repl_12.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    sel = coeffs[(np.abs(coeffs[:, 1]) < 6) & (np.abs(coeffs[:, 2]) < 6)]
    fproj = None
    for k, (c, m, n) in enumerate(sel):
        H = Diskfun.harmonic(int(m), int(n))
        term = H * float(c)
        fproj = term if fproj is None else fproj + term
        if k % 10 == 9:
            _release()
    diff = f - fproj
    dv = np.asarray(diff(jnp.asarray(T.ravel()),
                         jnp.asarray(R.ravel()))).reshape(T.shape)
    # L2 norm by tensor Gauss quadrature (Diskfun.norm-of-difference
    # NaN bug ledgered)
    from numpy.polynomial.legendre import leggauss
    xg, wg = leggauss(80)
    rq = 0.5 * (xg + 1.0)
    wq = 0.5 * wg
    thq = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    Tq, Rq = np.meshgrid(thq, rq)
    d2 = np.abs(np.asarray(diff(jnp.asarray(Tq.ravel()),
                                jnp.asarray(Rq.ravel()))
                           ).reshape(Tq.shape)) ** 2
    errf = float(np.sqrt((2 * np.pi / len(thq))
                         * float(wq @ (d2 * Rq).sum(axis=1))))
    print("errf =")
    print(f"   {errf:.15f}")

    _surf(fproj, "f (projection)", "Eigenfunctions_repl_13")
    _surf(f, "f", "Eigenfunctions_repl_14")
    return True


if __name__ == "__main__":
    run()
