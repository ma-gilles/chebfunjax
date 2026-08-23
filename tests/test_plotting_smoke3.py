"""Coverage smokes for the MATLAB-style plotting dispatchers (Fable 5).

Exercises the argument-parsing and limit-policy paths added for the
plot-test ports so the plotting module stays inside the CI coverage
gate.  Rendering correctness is pinned by the MATLAB port tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")

import chebfunjax as cj
import chebfunjax.plotting as P

jax.config.update("jax_enable_x64", True)


def _close():
    import matplotlib.pyplot as plt
    plt.close("all")


def test_matlab_plot_paths():
    f = cj.chebfun(jnp.sin)
    g = cj.chebfun(jnp.cos)
    P.matlab_plot(f)
    P.matlab_plot(f, "r-")
    P.matlab_plot(f, g)
    P.matlab_plot([f, g])
    P.matlab_plot(f, numpts=50, interval=[-0.5, 0.5])
    xd = np.linspace(-1, 1, 5)
    P.matlab_plot(xd, np.sin(xd), "o", f)
    fig, ax = P.matlab_plot(f)
    P.matlab_plot(g, ax=ax)  # hold-union path
    _close()


def test_matlab_plot3_surf_comet():
    f = cj.chebfun(jnp.sin)
    g = cj.chebfun(jnp.cos)
    h = cj.chebfun(jnp.exp)
    P.matlab_plot3(f, g, h)
    P.matlab_surf_quasi([f, g, h])
    P.matlab_surf_quasi([f, g], mode="mesh")
    P.matlab_surf_quasi([f, g], mode="surfc")
    P.comet(f)
    P.comet(f, g)
    P.comet3(f, g, h)
    P.comet3([f, g, h])
    P.plotcoeffs(f)
    P.plotcoeffs([f, g], loglog=True)
    P.plotcoeffs(f, fmt=".--")
    P.waterfall([f, g], [0.0, 1.0], linewidth=2, FaceColor="r")
    _close()


def test_chebfun2_and_3_plots():
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3
    f2 = Chebfun2.from_function(lambda x, y: x * y)
    P.contour(f2, levels=[0.0, 0.0], pivots="r.")
    xg, yg = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
    P.contour(f2, xx=xg, yy=yg, filled=True, levels=[0.0, 0.0])
    P.surf(f2, f2)
    P.surf(f2, f2, f2)
    P.waterfall_chebfun2(f2, "-")
    f3 = Chebfun3.from_function(lambda x, y, z: x + y + z)
    P.slice_chebfun3(f3, n_pts=17)
    P.scan_chebfun3(f3, dim=2, n_frames=2, n_pts=17)
    P.isosurface_chebfun3(f3, n_pts=13)
    _close()


def test_disk_sphere_plots():
    from chebfunjax.diskfun.diskfun import Diskfun
    from chebfunjax.spherefun.spherefun import Spherefun
    fd = Diskfun.from_function(lambda t, r: r * jnp.cos(t))
    fd.plot(".-")
    P.contour3_disk(fd, levels=[0.2, 0.2], pivots="r.", n_pts=40)
    P.contour_disk(fd, levels=[0.2, 0.2], fmt="k-")
    fs = Spherefun.sphharm(2, 1)
    fs.plot(".-")
    fs.contour3(levels=5, n_pts=40)
    _close()


def test_spherefun_new_api():
    from chebfunjax.spherefun.spherefun import Spherefun

    def f(lam, th):
        return jnp.cos(th) + jnp.sin(th) * jnp.cos(lam)

    g = Spherefun.from_function(f)
    m, n = g.length()
    assert m > 0 and n > 0
    assert np.asarray(g.sample(12, 9)).shape == (9, 12)
    U, D, V = g.sample_cdr(12, 9)
    S = np.asarray(U) @ np.asarray(D) @ np.asarray(V).T
    assert np.allclose(S, np.asarray(g.sample(12, 9)), atol=1e-10)
    mM = np.asarray(g.minandmax2est())
    assert mM[0] < mM[1]
    g.cosh(), g.sinh(), g.tanh()
    assert float((Spherefun.combine(*g.partition()) - g).norm()) < 1e-10
    F = Spherefun.vertcat(g, g, g)
    assert len(F.components) == 3
