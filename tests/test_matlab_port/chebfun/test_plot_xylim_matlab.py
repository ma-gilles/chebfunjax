"""Port of MATLAB Chebfun tests/chebfun/test_plot_xylim.m (Fable 5).

MATLAB's ``plot``/``hold on`` map to
:func:`chebfunjax.plotting.matlab_plot` (holding = passing the same
axes); ``get(gca, 'xlim')`` maps to ``ax.get_xlim()``, and the
'xlimmode'/'ylimmode' manual/auto checks map to matplotlib's
``get_autoscalex_on``/``get_autoscaley_on``.  Assertions that MATLAB
itself skips (commented out upstream after R2025a behaviour changes)
are skipped here too.

Provenance
----------
MATLAB source : tests/chebfun/test_plot_xylim.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib

matplotlib.use("Agg")

import chebfunjax as cj
import chebfunjax.plotting as P

jax.config.update("jax_enable_x64", True)

TOL = 1e-4


class TestChebfunPlotXYLim:
    def test_all_matlab_assertions(self):
        import matplotlib.pyplot as plt

        # %% Finite functions on bounded domains
        dom1 = (0.0, float(np.pi))
        x = cj.chebfun(lambda t: t, domain=dom1)

        fig, ax = P.matlab_plot(x)
        assert np.linalg.norm(np.array(dom1) - ax.get_xlim()) < TOL
        plt.close(fig)

        sinx = cj.chebfun(lambda t: jnp.sin(t), domain=dom1)
        fig, ax = P.matlab_plot(
            [0.62 * sinx, 0.0 * sinx, -0.62 * sinx])
        assert ax.get_autoscaley_on()          # ylimmode auto
        plt.close(fig)

        fig, ax = P.matlab_plot(sinx)
        P.matlab_plot(-sinx, ax=ax)            # hold on
        assert ax.get_autoscaley_on()
        plt.close(fig)

        fig, ax = P.matlab_plot(0.62 * sinx)
        P.matlab_plot(-0.62 * sinx, ax=ax)
        assert ax.get_autoscaley_on()
        plt.close(fig)

        # %% Finite functions on unbounded domains
        dom2 = (-jnp.inf, 0.0)
        f = cj.chebfun(lambda t: jnp.exp(t), domain=dom2)
        fig, ax = P.matlab_plot(f)
        assert not ax.get_autoscalex_on()      # xlimmode manual
        assert ax.get_autoscaley_on()          # ylimmode auto
        assert np.linalg.norm(np.array(ax.get_xlim())
                              - np.array([-10.0, 0.0])) < TOL
        plt.close(fig)

        g = cj.chebfun(lambda t: -0.62 * jnp.exp(t), domain=dom2)
        fig, ax = P.matlab_plot(g)
        assert ax.get_autoscaley_on()
        yl = ax.get_ylim()
        assert abs(yl[0] - 0.62) > 0.05
        plt.close(fig)

        dom3 = (-20.0, 20.0)
        h = cj.chebfun(lambda t: jnp.cos(t), domain=dom3)
        fig, ax = P.matlab_plot(g)
        P.matlab_plot(h, "r", ax=ax)           # hold on
        assert np.linalg.norm(np.array(ax.get_xlim())
                              - np.array(dom3)) < TOL
        plt.close(fig)

        fig, ax = P.matlab_plot(h, "g")
        P.matlab_plot(g, ax=ax)
        assert np.linalg.norm(np.array(ax.get_xlim())
                              - np.array(dom3)) < TOL
        plt.close(fig)

        dom4 = (-2.0, 2.0)
        h4 = cj.chebfun(lambda t: jnp.cos(t), domain=dom4)
        fig, ax = P.matlab_plot(g)
        P.matlab_plot(h4, "r", ax=ax)
        assert np.linalg.norm(np.array(ax.get_xlim())
                              - np.array([-10.0, 2.0])) < TOL
        plt.close(fig)

        fig, ax = P.matlab_plot(h4, "g")
        P.matlab_plot(g, ax=ax)
        assert np.linalg.norm(np.array(ax.get_xlim())
                              - np.array([-10.0, 2.0])) < TOL
        plt.close(fig)

        # %% Functions that blow up
        f1 = 11.3 * sinx
        f2 = cj.chebfun(lambda t: 1.0 / t, domain=dom1, exps=(-1.0, 0.0))

        fig, ax = P.matlab_plot(f2)
        assert not ax.get_autoscalex_on()      # xlimmode manual
        assert not ax.get_autoscaley_on()      # ylimmode manual
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        assert abs(yl[0] - 1.0 / xl[1]) < TOL and yl[1] < 10
        P.matlab_plot(f1, "r", ax=ax)          # hold on
        yl = ax.get_ylim()
        assert abs(yl[0]) < TOL and yl[1] > 10
        plt.close(fig)

        # Reverse plotting order.
        fig, ax = P.matlab_plot(f1, "g")
        yl_old = ax.get_ylim()
        P.matlab_plot(f2, ax=ax)
        yl_new = ax.get_ylim()
        assert abs(yl_old[0] - yl_new[0]) < TOL and yl_new[1] > 10
        plt.close(fig)

        # Symmetry of x on [-inf, inf].
        fsym = cj.chebfun(lambda t: t, domain=(-jnp.inf, jnp.inf))
        fig, ax = P.matlab_plot(fsym)
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        assert abs(xl[0] + xl[1]) < 1e-10
        assert abs(yl[0] + yl[1]) < 1e-10
        plt.close(fig)
