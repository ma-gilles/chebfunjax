"""Smoke tests for the heavier plotting entry points (Agg backend).

The MATLAB-figure-parity of these renderers is pinned by the examples
pipeline (docs/images vs chebfun.org); these tests only exercise the
code paths in the core CI job — each call must produce a figure
without raising.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import chebfunjax as cj
from chebfunjax import plotting

jax.config.update("jax_enable_x64", True)


def _close(fig):
    plt.close(fig if fig is not None else "all")


class TestPlot1D:
    def test_plot_and_coeffs(self, tmp_path):
        f = cj.chebfun(lambda t: jnp.sin(3 * t), domain=(-1.0, 1.0))
        fig, ax = plt.subplots()
        plotting.plot_1d(f, ax=ax)
        plotting._apply_style(ax, title="t", xlabel="x", ylabel="y")
        plotting.save_chebfun_figure(fig, str(tmp_path / "f.png"))
        _close(fig)

        fig, ax = plt.subplots()
        plotting.plotcoeffs(f, ax=ax)
        _close(fig)

    def test_phaseplot(self):
        f = cj.chebfun(lambda t: jnp.exp(1j * jnp.pi * t), domain=(-1.0, 1.0))
        fig, ax = plt.subplots()
        plotting.phaseplot(f, ax=ax)
        _close(fig)


class TestPlot2D:
    def _f2(self):
        from chebfunjax.chebfun2d.chebfun2 import Chebfun2
        return Chebfun2.from_function(
            lambda x, y: jnp.cos(2 * x) * jnp.sin(2 * y))

    def test_surf_contour(self):
        f2 = self._f2()
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        plotting.surf(f2, ax=ax)
        _close(fig)

        fig, ax = plt.subplots()
        plotting.contour(f2, ax=ax)
        _close(fig)

    def test_quiver_2d(self):
        from chebfunjax.chebfun2d.chebfun2 import Chebfun2
        fx = Chebfun2.from_function(lambda x, y: -y)
        fy = Chebfun2.from_function(lambda x, y: x)
        fig, ax = plt.subplots()
        plotting.quiver_2d(fx, fy, ax=ax)
        _close(fig)


class TestPlotSphereDisk:
    def test_plot_disk(self):
        from chebfunjax.diskfun import Diskfun
        f = Diskfun.from_function(
            lambda th, r: 1.0 + 0.1 * r * jnp.cos(th))
        # plot_disk renders on a 3-D axis (MATLAB surf-style disk).
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        plotting.plot_disk(f, ax=ax)
        _close(fig)

    def test_plot_sphere(self):
        from chebfunjax.spherefun import Spherefun
        f = Spherefun.from_function(
            lambda lam, th: 1.0 + 0.05 * jnp.cos(3 * lam) * jnp.sin(th))
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        plotting.plot_sphere(f, ax=ax)
        _close(fig)


class TestWaterfall:
    def test_waterfall(self):
        xs = np.linspace(0, 1, 12)
        rows = [cj.chebfun(lambda t, k=k: jnp.sin(t + 0.1 * k),
                           domain=(0.0, 2 * np.pi)) for k in range(4)]
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        plotting.waterfall(rows, ax=ax)
        _close(fig)
        del xs


class TestQuiverFamilies:
    def test_quiver_disk(self):
        from chebfunjax.diskfun import Diskfun
        from chebfunjax.diskfun.diskfunv import Diskfunv
        fx = Diskfun.from_function(lambda th, r: -r * jnp.sin(th))
        fy = Diskfun.from_function(lambda th, r: r * jnp.cos(th))
        fv = Diskfunv(fx, fy)
        fig = plt.figure()
        ax = fig.add_subplot()
        plotting.quiver_disk(fv, ax=ax)
        _close(fig)

    def test_quiver_sphere_and_gradient(self):
        from chebfunjax.spherefun import Spherefun
        f = Spherefun.from_function(
            lambda lam, th: jnp.cos(lam) * jnp.sin(th))
        fv = f.gradient()
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        plotting.quiver_sphere(fv, ax=ax)
        _close(fig)

    def test_quiver_sphere_cartesian(self):
        from chebfunjax.spherefun import Spherefun
        fx = Spherefun.from_function(lambda lam, th: -jnp.sin(lam))
        fy = Spherefun.from_function(lambda lam, th: jnp.cos(lam))
        fz = Spherefun.from_function(lambda lam, th: 0.0 * lam)
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        plotting.quiver_sphere_cartesian(fx, fy, fz, ax=ax)
        _close(fig)


class TestBallRenderers:
    def _ball(self):
        from chebfunjax.ballfun.ballfun import Ballfun
        return Ballfun.from_function(
            lambda x, y, z: 1.0 + 0.2 * x * y + 0.1 * z)

    def test_plot_ball_slices(self):
        bf = self._ball()
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        plotting.plot_ball_slices(bf, ax=ax)
        _close(fig)

    def test_quiver_ball(self):
        from chebfunjax.ballfun.ballfun import Ballfun
        from chebfunjax.ballfun.ballfunv import Ballfunv
        vx = Ballfun.from_function(lambda x, y, z: -y)
        vy = Ballfun.from_function(lambda x, y, z: x)
        vz = Ballfun.from_function(lambda x, y, z: 0.0 * z)
        bfv = Ballfunv(vx, vy, vz)
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        plotting.quiver_ball(bfv, ax=ax)
        _close(fig)
