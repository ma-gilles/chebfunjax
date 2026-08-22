"""Port of MATLAB Chebfun tests/chebfun/test_plot.m (Fable 5).

MATLAB's ``plot``/``plot3``/``surf``/``surfc``/``surface``/``mesh`` on
chebfuns map to :func:`chebfunjax.plotting.matlab_plot`,
:func:`~chebfunjax.plotting.matlab_plot3` and
:func:`~chebfunjax.plotting.matlab_surf_quasi`; positional MATLAB
options ('numpts', 100) map to Python keywords.  Array-valued chebfuns
and quasimatrices both map to lists of column chebfuns (the established
port convention).  As in MATLAB, the assertions only check that nothing
crashes.

Provenance
----------
MATLAB source : tests/chebfun/test_plot.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")

import chebfunjax as cj
import chebfunjax.plotting as P
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain

jax.config.update("jax_enable_x64", True)


def _pw(fns, brks):
    """Piecewise chebfun from callables on consecutive subintervals."""
    funs = []
    for fn, a, b in zip(fns, brks[:-1], brks[1:]):
        funs.extend(cj.chebfun(fn, domain=(a, b)).funs)
    return Chebfun(funs=funs, domain=Domain(tuple(brks)))


def _does_not_crash(fn):
    import matplotlib.pyplot as plt
    try:
        fn()
        return True
    finally:
        plt.close("all")


class TestChebfunPlot:
    def test_all_matlab_assertions(self):
        # Real scalar functions (with breakpoints).
        fsr1 = _pw([jnp.sin, jnp.sin], [-1.0, 0.0, 1.0])
        fsr2 = _pw([jnp.cos, jnp.cos], [-1.0, 0.5, 1.0])
        fsr3 = _pw([jnp.exp, jnp.exp], [-1.0, -0.5, 1.0])

        # Array-valued functions / quasimatrices -> lists of columns.
        far1 = [_pw([jnp.sin] * 2, [-1.0, 0.0, 1.0]),
                _pw([jnp.cos] * 2, [-1.0, 0.0, 1.0]),
                _pw([jnp.exp] * 2, [-1.0, 0.0, 1.0])]
        far2 = [_pw([jnp.sin] * 2, [-1.0, -0.5, 1.0]),
                _pw([lambda x: 2 * jnp.cos(x)] * 2, [-1.0, -0.5, 1.0]),
                _pw([lambda x: 3 * jnp.exp(x)] * 2, [-1.0, -0.5, 1.0])]
        far3 = [_pw([lambda x: 3 * jnp.sin(x)] * 2, [-1.0, 0.5, 1.0]),
                _pw([lambda x: 2 * jnp.cos(x)] * 2, [-1.0, 0.5, 1.0]),
                _pw([jnp.exp] * 2, [-1.0, 0.5, 1.0])]
        fqr1, fqr2, fqr3 = far1, far2, far3

        # Complex functions.
        fsc = _pw([lambda x: jnp.exp(1j * x)] * 2, [-1.0, 0.0, 1.0])
        fac = [_pw([lambda x, k=k: k * jnp.exp(1j * x)] * 2,
                   [-1.0, 0.5, 1.0]) for k in (1.0, 2.0, 3.0)]
        fqc = fac

        # Singular function 1/x with interior blow-up.
        s1 = cj.chebfun(lambda x: 1.0 / x, domain=(-1.0, 0.0),
                        exps=(0.0, -1.0))
        s2 = cj.chebfun(lambda x: 1.0 / x, domain=(0.0, 1.0),
                        exps=(-1.0, 0.0))
        fsing = Chebfun(funs=list(s1.funs) + list(s2.funs),
                        domain=Domain((-1.0, 0.0, 1.0)))

        # Discontinuous function {sin, exp}.
        fdc = _pw([jnp.sin, jnp.exp], [-1.0, 0.0, 1.0])

        # Unbounded domains.
        fub1 = cj.chebfun(lambda x: jnp.exp(-x ** 2) * jnp.sin(2 * jnp.pi * x),
                          domain=(0.0, jnp.inf))
        fub2 = cj.chebfun(lambda x: jnp.exp(-x ** 2) * jnp.cos(2 * jnp.pi * x),
                          domain=(0.0, jnp.inf))
        fub3 = cj.chebfun(lambda x: jnp.exp(-x ** 2) * x,
                          domain=(0.0, jnp.inf))

        # Delta functions: dirac(x) - dirac(x-.5) + dirac(x+.5).
        z = cj.chebfun(lambda x: 0.0 * x)
        fdel = Chebfun(funs=z.funs, domain=z.domain,
                       deltas=((0.0, 1.0), (0.5, -1.0), (-0.5, 1.0)))
        x = cj.chebfun(lambda x: x)

        # Real scalar functions.
        assert _does_not_crash(lambda: P.matlab_plot(fsr1))            # 1
        assert _does_not_crash(lambda: P.matlab_plot(fsr1, fsr2))      # 2
        assert _does_not_crash(lambda: P.matlab_plot3(fsr1, fsr2, fsr3))  # 3

        # Array-valued functions.
        assert _does_not_crash(lambda: P.matlab_plot(far1))            # 4
        assert _does_not_crash(lambda: P.matlab_plot(far1, far2))      # 5
        assert _does_not_crash(lambda: P.matlab_plot3(far1, far2, far3))  # 6

        # Quasimatrices.
        assert _does_not_crash(lambda: P.matlab_plot(fqr1))            # 7
        assert _does_not_crash(lambda: P.matlab_plot(fqr1, fqr2))      # 8
        assert _does_not_crash(lambda: P.matlab_plot3(fqr1, fqr2, fqr3))  # 9

        # Mixes of array-valued functions and quasimatrices.
        assert _does_not_crash(lambda: P.matlab_plot(far1, fqr2))      # 10
        assert _does_not_crash(lambda: P.matlab_plot(fqr1, far2))      # 11
        for trip in [(far1, far2, fqr3), (far1, fqr2, far3),
                     (fqr1, far2, far3), (fqr1, fqr2, far3),
                     (fqr1, far2, fqr3), (far1, fqr2, fqr3)]:  # 12-17
            assert _does_not_crash(lambda t=trip: P.matlab_plot3(*t))

        # Quasimatrices against scalar functions.
        assert _does_not_crash(lambda: P.matlab_plot(fsr1, fqr2))      # 18
        assert _does_not_crash(lambda: P.matlab_plot(fqr1, fsr2))      # 19
        assert _does_not_crash(lambda: P.matlab_plot(fsr1, far2))      # 20
        assert _does_not_crash(lambda: P.matlab_plot(far1, fsr2))      # 21

        # Complex functions.
        assert _does_not_crash(lambda: P.matlab_plot(fsc))             # 22
        assert _does_not_crash(lambda: P.matlab_plot(fac))             # 23
        assert _does_not_crash(lambda: P.matlab_plot(fqc))             # 24

        # Singular function.
        assert _does_not_crash(lambda: P.matlab_plot(fsing))           # 25

        # Unbounded domains.
        assert _does_not_crash(lambda: P.matlab_plot(fub1))            # 26
        assert _does_not_crash(lambda: P.matlab_plot(fub1, fub2))      # 27
        assert _does_not_crash(lambda: P.matlab_plot3(fub1, fub2, fub3))  # 28

        # Plot flags and options.
        assert _does_not_crash(lambda: P.matlab_plot(fsr1, numpts=100))  # 29
        assert _does_not_crash(
            lambda: P.matlab_plot(fsr2, interval=[-0.5, 0.5]))         # 30
        assert _does_not_crash(
            lambda: P.matlab_plot(fsr2, np.array([-0.5, 0.5])))        # 31
        assert _does_not_crash(
            lambda: P.matlab_plot(fdc, jumpline="r-"))                 # 32
        assert _does_not_crash(
            lambda: P.matlab_plot(fdc, jumpline="none"))               # 33
        assert _does_not_crash(
            lambda: P.matlab_plot(fdc, jumpline={"LineStyle": "--"}))  # 34
        assert _does_not_crash(
            lambda: P.matlab_plot3(fdc, fsr1, fsr2, jumpline="r-"))    # 35
        assert _does_not_crash(lambda: P.matlab_plot(fdel, "r"))       # 36
        assert _does_not_crash(
            lambda: P.matlab_plot(fdel, deltaline="--ro"))             # 37
        assert _does_not_crash(lambda: P.matlab_plot(x, fdel))         # 38
        assert _does_not_crash(
            lambda: P.matlab_plot(fqr1, linewidth=2))                  # 39

        # Discrete data alongside chebfuns.
        xd = np.linspace(-1, 1, 10)
        y1 = np.column_stack([np.asarray(c(jnp.asarray(xd)))
                              for c in far1])
        y3 = np.column_stack([np.asarray(c(jnp.asarray(xd)))
                              for c in far3])
        assert _does_not_crash(
            lambda: P.matlab_plot(far1, "b", far2, "r", xd, y1, "om",
                                  xd, y3, "-ok"))                      # 40

        # SURF, SURFACE, SURFC, MESH.
        assert _does_not_crash(lambda: P.matlab_surf_quasi(far1))      # 41
        assert _does_not_crash(lambda: P.matlab_surf_quasi(fqr1))      # 42
        assert _does_not_crash(lambda: P.matlab_surf_quasi(fqr1))      # 43
        assert _does_not_crash(
            lambda: P.matlab_surf_quasi(far1, mode="surfc"))           # 44
        assert _does_not_crash(
            lambda: P.matlab_surf_quasi(fqr1, mode="surfc"))           # 45
        assert _does_not_crash(
            lambda: P.matlab_surf_quasi(far1, mode="mesh"))            # 46
        assert _does_not_crash(
            lambda: P.matlab_surf_quasi(fqr1, mode="mesh"))            # 47

        # Abbreviated styles (map to matplotlib keywords).
        assert _does_not_crash(
            lambda: P.matlab_plot(fsr1, linewidth=15))                 # 48
        assert _does_not_crash(
            lambda: P.matlab_plot(fsr1, linestyle="--"))               # 49
        assert _does_not_crash(
            lambda: P.matlab_plot(fsr1, markersize=10))                # 50

        # Complex constant.
        assert _does_not_crash(
            lambda: P.matlab_plot(cj.chebfun(lambda t: 1j + 0.0 * t),
                                  "o"))                                # 51

        # Doubles alongside chebfuns.
        fexp = cj.chebfun(jnp.exp)
        xe = np.linspace(-1, 1, 5)
        fe = np.exp(xe)
        assert _does_not_crash(
            lambda: P.matlab_plot(fexp, "b", xe, fe, "or"))            # 52
        assert _does_not_crash(
            lambda: P.matlab_plot(xe, fe, "or", fexp))                 # 53
