"""chebfun(f, exps=...) endpoint-singularity construction (Fable 5).

Wires Singfun into the chebfun factory (MATLAB's 'exps' flag), closing
the chebfun-level blowup gap for single-interval domains.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

import chebfunjax as cj


class TestChebfunExps:
    def test_inverse_sqrt_weight(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = cj.chebfun(lambda x: 1.0 / jnp.sqrt(1 - x * x),
                           exps=(-0.5, -0.5))
        xs = np.linspace(-0.999, 0.999, 60)
        npt.assert_allclose(np.asarray(f(jnp.asarray(xs))),
                            1 / np.sqrt(1 - xs ** 2), rtol=1e-11)
        npt.assert_allclose(float(f.sum()), np.pi, atol=1e-15)

    def test_shifted_domain_right_singularity(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = cj.chebfun(lambda x: (7.0 - x) ** (-0.3) * jnp.cos(x),
                           domain=(-2.0, 7.0), exps=(0.0, -0.3))
        xs = np.linspace(-1.9, 6.9, 40)
        npt.assert_allclose(np.asarray(g(jnp.asarray(xs))),
                            (7 - xs) ** (-0.3) * np.cos(xs), atol=1e-12)

    def test_fractional_root_left(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = cj.chebfun(lambda x: (1 + x) ** 0.5 * jnp.exp(x),
                           exps=(0.5, 0.0))
        xs = np.linspace(-0.99, 0.99, 40)
        npt.assert_allclose(np.asarray(f(jnp.asarray(xs))),
                            (1 + xs) ** 0.5 * np.exp(xs), atol=1e-12)

    def test_exps_conflicts_raise(self):
        with pytest.raises(ValueError, match="exps"):
            cj.chebfun(lambda x: x, exps=(0.0, -0.5), trig=True)


class TestChebfunBlowup:
    def test_blowup_autodetect_fractional(self):
        # 'blowup' with no exps autodetects a branch point at the left end.
        pow = -0.5
        f = cj.chebfun(lambda x: (1 + x) ** pow * jnp.sin(x),
                       domain=(-1.0, 1.0), blowup=2)
        assert f.funs[0].tech.exponents[0] == pytest.approx(pow, abs=1e-9)
        xs = np.linspace(-0.99, 0.99, 40)
        npt.assert_allclose(np.asarray(f(jnp.asarray(xs))),
                            (1 + xs) ** pow * np.sin(xs), atol=1e-12)

    def test_singtype_pole(self):
        # singType 'pole' forces the integer pole-order finder.
        f = cj.chebfun(lambda x: (1 + x) ** (-1.0) * jnp.cos(x),
                       domain=(-1.0, 1.0), blowup=2,
                       singType=["pole", "none"])
        assert f.funs[0].tech.exponents == (-1.0, 0.0)

    def test_singtype_none_stays_smooth(self):
        # singType 'none' at both ends -> exponent 0 -> a smooth piece.
        from chebfunjax.fun.singfun import Singfun
        f = cj.chebfun(jnp.sin, domain=(-1.0, 1.0), blowup=1,
                       singType=["none", "none"])
        assert not isinstance(f.funs[0].tech, Singfun)

    def test_isinf_isfinite(self):
        pole = cj.chebfun(lambda x: jnp.sin(x) / (x + 1.0),
                          domain=(-1.0, 1.0), exps=(-1.0, 0.0))
        assert pole.isinf() and not pole.isfinite()
        smooth = cj.chebfun(jnp.sin)
        assert smooth.isfinite() and not smooth.isinf()


class TestExpsParsing:
    def test_parse_exps_conventions(self):
        from chebfunjax.chebfun1d.chebfun import _parse_exps
        # 2 values on a single interval -> the two endpoints.
        assert _parse_exps((-0.5, 0.0), 1) == [(-0.5, 0.0)]
        # 2 values on 3 intervals -> only the outer domain endpoints.
        assert _parse_exps((0.5, -0.5), 3) == [
            (0.5, 0.0), (0.0, 0.0), (0.0, -0.5)]
        # 1 value broadcasts everywhere.
        assert _parse_exps((-1.0,), 2) == [(-1.0, -1.0), (-1.0, -1.0)]
        # n_int+1 values: one shared per breakpoint.
        assert _parse_exps((-1.0, -1.0, -1.0, -1.0), 3) == [
            (-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0)]
        # 2*n_int values: per-interval pairs.
        assert _parse_exps((0.0, 0.0, -1.0, 0.0, 0.0, 0.0), 3) == [
            (0.0, 0.0), (-1.0, 0.0), (0.0, 0.0)]

    def test_parse_exps_bad_count(self):
        from chebfunjax.chebfun1d.chebfun import _parse_exps
        with pytest.raises(ValueError, match="exponents"):
            _parse_exps((0.0, 0.0, 0.0), 3)

    def test_multi_interval_poles(self):
        # tan on a domain broken at its poles: a simple pole per breakpoint.
        from chebfunjax.fun.singfun import Singfun
        dom = tuple(np.pi * np.arange(-2.5, 3.0, 1.0))
        f = cj.chebfun(jnp.tan, domain=dom, exps=tuple([-1.0] * 6))
        assert len(f.funs) == 5
        for p in f.funs:
            assert isinstance(p.tech, Singfun)
            assert p.tech.exponents == (-1.0, -1.0)


class TestSqrtSingular:
    def test_sqrt_boundary_roots(self):
        from chebfunjax.fun.singfun import Singfun
        f = cj.chebfun(lambda x: 1.0 - x ** 2, domain=(-1.0, 1.0))
        g = f.sqrt()
        assert isinstance(g.funs[0].tech, Singfun)
        assert g.funs[0].tech.exponents == (0.5, 0.5)
        xs = jnp.asarray(np.linspace(-0.98, 0.98, 40))
        npt.assert_allclose(np.asarray(g(xs)),
                            np.sqrt(1 - np.asarray(xs) ** 2), atol=1e-13)

    def test_sqrt_positive_stays_smooth(self):
        from chebfunjax.fun.singfun import Singfun
        f = cj.chebfun(lambda x: 2.0 + jnp.sin(x))
        g = f.sqrt()
        assert not isinstance(g.funs[0].tech, Singfun)
        xs = jnp.asarray(np.linspace(-0.99, 0.99, 40))
        npt.assert_allclose(np.asarray(g(xs)),
                            np.sqrt(2 + np.sin(np.asarray(xs))), atol=1e-13)


class TestSingularAbsRealRepr:
    """abs/real/repr on 'exps' chebfuns keep the singular structure."""

    def _pole(self):
        # 1/(1-x^2): simple poles at both endpoints.
        return cj.chebfun(lambda x: 1.0 / (1.0 - x ** 2),
                          exps=[-1, -1])

    def test_abs_keeps_exponents_and_values(self):
        from chebfunjax.fun.singfun import Singfun
        f = self._pole()
        g = abs(f)
        assert isinstance(g.funs[0].tech, Singfun)
        assert g.funs[0].tech.exponents == f.funs[0].tech.exponents
        xs = jnp.asarray(np.linspace(-0.9, 0.9, 21))
        npt.assert_allclose(np.asarray(g(xs)),
                            1.0 / (1.0 - np.asarray(xs) ** 2), rtol=1e-12)

    def test_abs_negative_smooth_part(self):
        f = -self._pole()
        g = abs(f)
        xs = jnp.asarray(np.linspace(-0.9, 0.9, 21))
        npt.assert_allclose(np.asarray(g(xs)),
                            1.0 / (1.0 - np.asarray(xs) ** 2), rtol=1e-12)

    def test_sum_abs_pole_infinite(self):
        assert np.isposinf(float(abs(self._pole()).sum()))

    def test_real_structural_on_singfun(self):
        from chebfunjax.fun.singfun import Singfun
        f = self._pole()
        g = f.real()
        assert isinstance(g.funs[0].tech, Singfun)
        xs = jnp.asarray(np.linspace(-0.9, 0.9, 21))
        npt.assert_allclose(np.asarray(g(xs)), np.asarray(f(xs)), rtol=1e-13)

    def test_repr_no_crash_inf_endpoints(self):
        r = repr(self._pole())
        assert "inf" in r.lower()

    def test_root_power_no_interior_roots(self):
        from chebfunjax.fun.singfun import Singfun
        g = abs(self._pole()) ** 0.5
        assert isinstance(g.funs[0].tech, Singfun)
        a, b = g.funs[0].tech.exponents
        assert (a, b) == (-0.5, -0.5)
        xs = jnp.asarray(np.linspace(-0.9, 0.9, 21))
        npt.assert_allclose(np.asarray(g(xs)),
                            (1.0 / (1.0 - np.asarray(xs) ** 2)) ** 0.5,
                            rtol=1e-12)


class TestBlowupSplitting:
    """Automatic interior-pole detection with blowup + splitting."""

    def test_single_interior_pole(self):
        from chebfunjax.fun.singfun import Singfun
        f = cj.chebfun(lambda x: 1.0 / (x - 0.25), domain=[-1, 1],
                       blowup=True, splitting=True)
        assert len(f.funs) == 2
        assert abs(f.funs[0].interval[1] - 0.25) < 1e-10
        assert isinstance(f.funs[0].tech, Singfun)
        xs = jnp.asarray(np.linspace(-0.9, 0.9, 21))
        xs = xs[np.abs(np.asarray(xs) - 0.25) > 0.05]
        npt.assert_allclose(np.asarray(f(xs)),
                            1.0 / (np.asarray(xs) - 0.25), rtol=1e-9)

    def test_find_blowup_locates_pole(self):
        from chebfunjax.chebfun1d.chebfun import _find_blowup
        edge = _find_blowup(lambda x: 1.0 / (x - 0.3) ** 2, -1.0, 1.0, 1.0)
        assert edge is not None and abs(edge - 0.3) < 1e-12

    def test_find_blowup_rejects_smooth(self):
        from chebfunjax.chebfun1d.chebfun import _find_blowup
        assert _find_blowup(jnp.cos, -1.0, 1.0, 1.0) is None

    def test_no_splitting_flag_unchanged(self):
        f = cj.chebfun(lambda x: 1.0 / (1.0 - x ** 2), exps=[-1, -1])
        assert len(f.funs) == 1
