"""Eigenfunctions of the Laplacian on the disk.

Translation of disk/Eigenfunctions.m by Heather Wilber, January
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.plotting import _coerce_cmap, chebfun_style, plot_disk
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'disk')
FIG = [0]


def _release():
    # Cumulative XLA cache entries from many distinct-shape harmonic
    # constructions slow later compiles ~100x (guide17 pattern); release
    # between sections.
    import gc

    import jax
    jax.clear_caches()
    gc.collect()


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(_IMG, f"Eigenfunctions_{FIG[0]:02d}.png"))
    plt.close(fig)
    _release()


def _plot(ax, u, title, view=(0, 90), colorbar=False):
    """plot(u), axis off, view(az,el), title(...): the diskfun surface,
    seen from above by default (MATLAB @diskfun/plot.m ends with view(2))."""
    plot_disk(u, ax=ax, n_theta=161, n_r=60)
    for ln in list(ax.lines):             # no base circle in MATLAB's plot
        ln.remove()
    az, el = view
    ax.view_init(elev=el, azim=az - 90)     # MATLAB azimuth -> matplotlib
    ax.set_axis_off()
    ax.set_title(f"${title}$" if "_" in title else title)   # TeX subscripts
    if colorbar:
        th = np.linspace(-np.pi, np.pi, 161)
        r = np.linspace(0, 1, 60)
        T, R = np.meshgrid(th, r)
        V = np.asarray(u(jnp.asarray(T.ravel()), jnp.asarray(R.ravel())))
        sm = plt.cm.ScalarMappable(cmap=_coerce_cmap(None),
                                   norm=plt.Normalize(V.min(), V.max()))
        ax.get_figure().colorbar(sm, ax=ax, shrink=0.8)


def _pair(left, right):
    fig = plt.figure()
    for i, args in enumerate((left, right)):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        _plot(ax, *args)
    _save(fig)


def _single(u, title, colorbar=False):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    _plot(ax, u, title, colorbar=colorbar)
    _save(fig)


def _disp(name, v):
    """MATLAB ``format long`` display of a real scalar."""
    print(f"{name} =")
    if v != 0 and not 1e-3 <= abs(v) < 1e3:
        print(f"{v:26.15e}")
    else:
        print(f"{v:20.15f}")


def _bessel_roots(L, a, b):
    """roots(chebfun(@(x) besselj(L,x), [a b]))."""
    x = cj.chebfun('x', domain=[a, b])
    return np.asarray(x.besselj(L).roots())


def run():
    os.makedirs(_IMG, exist_ok=True)

    u42 = Diskfun.harmonic(4, 2)
    print("u42 =")
    print(u42.disp())
    _single(u42, "u_{4,2}")

    lam = float(_bessel_roots(4, 10, 13)[0])
    _disp("ans", float((u42.lap() + lam**2 * u42).norm()))

    a, b = -100.4, 51.6
    u01 = Diskfun.harmonic(0, 1)
    u02 = Diskfun.harmonic(0, 2)
    _pair((u01, "u_{0,1}", (a, b)), (u02, "u_{0,2}", (a, b)))
    u03 = Diskfun.harmonic(0, 3)
    u04 = Diskfun.harmonic(0, 4)
    _pair((u03, "u_{0,3}", (a, b)), (u04, "u_{0,4}", (a, b)))

    v21 = Diskfun.harmonic(-2, 1)
    v22 = Diskfun.harmonic(-3, 2)
    _pair((v21, "v_{2,1}", (-99.5, 60.3)), (v22, "v_{2,2}", (-1.1e2, 75)))
    u33 = Diskfun.harmonic(3, 3)
    u117 = Diskfun.harmonic(11, 7)
    _pair((u33, "u_{3,3}"), (u117, "u_{11,7}"))

    uN21 = Diskfun.harmonic(2, 1, "neumann")
    uN34 = Diskfun.harmonic(3, 4, "neumann")
    _pair((uN21, "u21 with Neumann bc", (-1.2e2, 50)),
          (uN34, "u34 with Neumann bc", (a, b)))

    _disp("int1", float((u01 * u02).sum2()))
    _disp("int2", float((v22 * u117).sum2()))
    _disp("int3", float((u03 * u03).sum2()))

    # diskfun(@(x,y) ...): Cartesian input, x = r cos(t), y = r sin(t).
    f = Diskfun.from_function(
        lambda t, r: 20 * (1 - r ** 2) ** 2
        * jnp.exp(-6 * (r * jnp.cos(t) + 0.25) ** 2
                  - 6 * (r * jnp.sin(t) - 0.2) ** 2))
    _single(f, "f", colorbar=True)

    N = NN = 7
    rows = []
    for m in range(0, N + 1):
        for n in range(1, NN + 1):
            for j in range(1, 2 + int(m != 0)):
                H = Diskfun.harmonic((-1) ** j * m, n)
                rows.append((float((f * H).sum2()), (-1) ** j * m, n))
                if len(rows) % 10 == 0:
                    _release()
    coeffs = np.array(rows)

    # stem3(coeffs(:,2),coeffs(:,3),abs(coeffs(:,1)),'filled'), log z
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    zc = np.log10(np.abs(coeffs[:, 0]))
    zb = np.floor(zc.min())
    for xm, yn, z in zip(coeffs[:, 1], coeffs[:, 2], zc):
        ax.plot([xm, xm], [yn, yn], [zb, z], color='#0072BD', lw=0.8)
    ax.scatter(coeffs[:, 1], coeffs[:, 2], zc, color='#0072BD', s=12,
               depthshade=False)
    zt = np.arange(zb, np.ceil(zc.max()) + 1, 5)
    ax.set_zticks(zt)
    ax.set_zticklabels([f"$10^{{{int(t)}}}$" for t in zt])
    ax.view_init(elev=38, azim=1.205e2 - 90)
    ax.set_xlabel('m')
    ax.set_ylabel('n')
    ax.set_zlabel('abs. value of coeffs')
    _save(fig)

    coeffs = coeffs[(np.abs(coeffs[:, 1]) < 6) & (np.abs(coeffs[:, 2]) < 6)]
    fproj = None
    for k, (c, m, n) in enumerate(coeffs):
        term = float(c) * Diskfun.harmonic(int(m), int(n))
        fproj = term if fproj is None else fproj + term
        if k % 10 == 9:
            _release()

    _disp("errf", float((f - fproj).norm()))

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    _plot(ax, fproj, "f", colorbar=True)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    _plot(ax, f - fproj, "f: error", colorbar=True)
    _save(fig)

    coeffs = coeffs[(np.abs(coeffs[:, 1]) < 4) & (np.abs(coeffs[:, 2]) < 4)]
    csz = len(coeffs)
    NN = N = 3

    broots = np.zeros((csz, 3))
    Jzero = _bessel_roots(0, np.sqrt((3 / 4)**2 * np.pi**2), NN * np.pi)
    broots[:NN] = np.column_stack([Jzero, np.zeros(NN), np.arange(1, NN + 1)])
    k = NN
    for L in range(1, N + 1):
        Jzero = _bessel_roots(L, np.sqrt((3 / 4)**2 * np.pi**2 + L**2),
                              (NN + L / 2) * np.pi)
        broots[k:k + 2 * NN:2] = np.column_stack(
            [Jzero, -L * np.ones(NN), np.arange(1, NN + 1)])
        broots[k + 1:k + 2 * NN + 1:2] = np.column_stack(
            [Jzero, L * np.ones(NN), np.arange(1, NN + 1)])
        k += 2 * NN

    # The MATLAB loop never resets T, so u{i} = sum_{i'<=i} sum_k
    # cos(broots(k)*tm(i'))*H_k; accumulate the cosine weights per mode
    # (same sum, one diskfun per harmonic) for the snapshots shown.
    tm = np.linspace(0, 4, 41)
    H = [c * Diskfun.harmonic(int(m), int(n)) for c, m, n in coeffs]
    _release()

    def snapshot(i):          # MATLAB u{i}, 1-based
        w = np.cos(np.outer(tm[:i], broots[:, 0])).sum(axis=0)
        T = None
        for wk, Hk in zip(w, H):
            T = float(wk) * Hk if T is None else T + float(wk) * Hk
        return T

    # (the animation loop over u{j} is overwritten by the subplots below)
    _pair((snapshot(5), "t=.4"), (snapshot(13), "t=1.2"))
    _pair((snapshot(15), "t=1.4"), (snapshot(25), "t=2.4"))
    return True


if __name__ == "__main__":
    run()
