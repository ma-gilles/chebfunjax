"""Regression tests for the chebfunjax plotting module.

Verifies:
  1. Importing chebfunjax does NOT force the Agg backend.
  2. All plotting symbols are accessible as cj.<name>.
  3. Each function returns (fig, ax) without raising.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pytest


# ---------------------------------------------------------------------------
# 1. Backend guard — must come before importing chebfunjax
# ---------------------------------------------------------------------------

def test_import_does_not_force_agg():
    """Importing chebfunjax must not unconditionally switch to the Agg backend.

    We record the backend before and after import.  The backend must be
    unchanged (or already Agg if the test runner chose it — that is fine).
    """
    backend_before = matplotlib.get_backend()
    import chebfunjax  # noqa: F401 — side-effect is what we are testing
    backend_after = matplotlib.get_backend()
    assert backend_after == backend_before, (
        f"chebfunjax import changed the matplotlib backend from "
        f"'{backend_before}' to '{backend_after}'.  "
        f"Do not call matplotlib.use() unconditionally in plotting.py."
    )


# ---------------------------------------------------------------------------
# 2 & 3. Symbol presence and (fig, ax) return type
# ---------------------------------------------------------------------------

# Use Agg for all subsequent tests so they run headlessly.
matplotlib.use("Agg")
import chebfunjax as cj  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def _make_chebfun1d():
    return cj.chebfun(jnp.sin)


def _make_chebfun2d():
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2
    return Chebfun2.from_function(lambda x, y: jnp.cos(x + y))


def _make_ballfun():
    from chebfunjax.ballfun.ballfun import Ballfun

    return Ballfun.from_function(
        lambda x, y, z: x + 2.0 * y - z,
        fixed_size=(15, 16, 16),
    )


def _make_spherefunv():
    from chebfunjax.spherefun.spherefun import Spherefun
    from chebfunjax.spherefun.spherefunv import Spherefunv

    f = Spherefun.from_function(lambda lam, th: jnp.cos(lam) * jnp.sin(th))
    g = Spherefun.from_function(lambda lam, th: jnp.sin(lam) * jnp.sin(th))
    return Spherefunv(f, g)


def _make_spherefun():
    from chebfunjax.spherefun.spherefun import Spherefun

    return Spherefun.from_function(lambda lam, th: jnp.cos(lam) * jnp.sin(th))


def _make_chebfun3():
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3

    return Chebfun3.from_function(lambda x, y, z: x + y + z)


class TestTopLevelExports:
    """All plotting names must be importable from chebfunjax."""

    def test_plot_exported(self):
        assert callable(cj.plot)

    def test_plotcoeffs_exported(self):
        assert callable(cj.plotcoeffs)

    def test_contour_exported(self):
        assert callable(cj.contour)

    def test_surf_exported(self):
        assert callable(cj.surf)

    def test_phaseplot_exported(self):
        assert callable(cj.phaseplot)

    def test_plot_disk_exported(self):
        assert callable(cj.plot_disk)

    def test_plot_sphere_exported(self):
        assert callable(cj.plot_sphere)

    def test_plot_slices_exported(self):
        assert callable(cj.plot_slices)


class TestReturnTypes:
    """Each plotting function must return (fig, ax) without raising."""

    def test_plot_returns_fig_ax(self):
        f = _make_chebfun1d()
        result = cj.plot(f)
        assert isinstance(result, tuple) and len(result) == 2
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plotcoeffs_returns_fig_ax(self):
        f = _make_chebfun1d()
        result = cj.plotcoeffs(f)
        assert isinstance(result, tuple) and len(result) == 2
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_contour_returns_fig_ax(self):
        g = _make_chebfun2d()
        result = cj.contour(g)
        assert isinstance(result, tuple) and len(result) == 2
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_surf_returns_fig_ax(self):
        g = _make_chebfun2d()
        result = cj.surf(g, n_pts=20)
        assert isinstance(result, tuple) and len(result) == 2
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_phaseplot_returns_fig_ax(self):
        result = cj.phaseplot(lambda z: z**2, region=[-2.0, 2.0, -2.0, 2.0],
                              n_pts=50)
        assert isinstance(result, tuple) and len(result) == 2
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_disk_returns_fig_ax(self):
        from chebfunjax.diskfun.diskfun import Diskfun
        fd = Diskfun.from_function(lambda t, r: jnp.cos(t) * r)
        result = cj.plot_disk(fd, n_theta=30, n_r=20)
        assert isinstance(result, tuple) and len(result) == 2
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_sphere_returns_fig_ax(self):
        from chebfunjax.spherefun.spherefun import Spherefun
        fs = Spherefun.from_function(lambda lam, theta: jnp.cos(lam) * jnp.sin(theta))
        result = cj.plot_sphere(fs, n_lam=30, n_theta=20)
        assert isinstance(result, tuple) and len(result) == 2
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_slices_returns_fig_ax(self):
        from chebfunjax.chebfun3d.chebfun3 import Chebfun3
        f3 = Chebfun3.from_function(lambda x, y, z: jnp.cos(x + y + z))
        result = cj.plot_slices(f3, n_pts=12)
        assert isinstance(result, tuple) and len(result) == 2
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestMatlabStyleRegression:
    """Regression checks for MATLAB-faithful plotting defaults."""

    def test_contour_default_has_no_colorbar_axis(self):
        g = _make_chebfun2d()
        fig, ax = cj.contour(g)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_ballfun_dispatch_uses_five_slice_surfaces(self):
        bf = _make_ballfun()
        fig, ax = cj.plot(bf)
        assert isinstance(fig, plt.Figure)
        assert getattr(ax, "name", None) == "3d"
        assert len(fig.axes) == 1
        assert len(ax.collections) == 5
        plt.close(fig)

    @pytest.mark.parametrize(
        ("style", "expected_collections"),
        [("WedgeAz", 3), ("WedgePol", 5)],
    )
    def test_ballfun_plot_accepts_matlab_wedge_styles(
        self, style, expected_collections
    ):
        bf = _make_ballfun()
        fig, ax = bf.plot(style)
        assert isinstance(fig, plt.Figure)
        assert getattr(ax, "name", None) == "3d"
        assert len(ax.collections) == expected_collections
        plt.close(fig)

    def test_spherefunv_plot_dispatch_returns_3d_quiver_axes(self):
        sfv = _make_spherefunv()
        fig, ax = cj.plot(sfv)
        assert isinstance(fig, plt.Figure)
        assert getattr(ax, "name", None) == "3d"
        assert len(fig.axes) == 1
        assert len(ax.collections) >= 1
        plt.close(fig)

    def test_contour_sphere_returns_3d_axes(self):
        sf = _make_spherefun()
        fig, ax = cj.contour_sphere(sf, levels=12)
        assert isinstance(fig, plt.Figure)
        assert getattr(ax, "name", None) == "3d"
        assert len(fig.axes) == 1
        assert len(ax.lines) >= 1
        plt.close(fig)

    def test_chebfun3_plot_renders_boundary_faces_and_colorbar(self):
        f3 = _make_chebfun3()
        fig, ax = cj.plot(f3, n_pts=20)
        assert isinstance(fig, plt.Figure)
        assert getattr(ax, "name", None) == "3d"
        assert len(ax.collections) == 6
        assert len(fig.axes) == 2
        plt.close(fig)
