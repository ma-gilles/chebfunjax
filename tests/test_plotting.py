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
