"""All six native Chebyshev-Pade predicates, using public Chebfun norms.

MATLAB source: tests/chebfun/test_chebpade.m; @chebfun/chebpade.m.
Chebfun commit: 7574c77.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = np.finfo(float).eps
DOM = (-1.0, 3.0)
XS = jnp.asarray(np.linspace(-0.99, 2.99, 80))


class TestChebfunChebpade:
    @pytest.fixture(autouse=True)
    def _native_warning_state(self):
        # Native warning('off'), with the previous state restored after tests.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield

    def _setup(self):
        P = cj.chebfun(
            jnp.asarray([0.5045, -1.3813, 2.1122, 0.0558, -0.6817]), domain=DOM, coeffs=True
        )
        Q = cj.chebfun(jnp.asarray([1, 0.1155, -0.8573, -0.2679, 0.5246]), domain=DOM, coeffs=True)
        R = P / Q
        return P, Q, R

    def test_maehly_no_reduction(self):
        P, Q, R = self._setup()
        p, q, _ = cj.chebpade(R, 4, 4, "maehly")
        err = float((P - p).norm()) + float((Q - q).norm())
        bound = 100 * R.vscale * EPS
        print({"native_clause": 1, "continuous_L2_sum": err, "source_bound": bound}, flush=True)
        assert err < bound
        # Retain prior sampled checks as supplements, at their original bounds.
        sampled_error = float(jnp.max(jnp.abs(p(XS) - P(XS)))) + float(
            jnp.max(jnp.abs(q(XS) - Q(XS)))
        )
        assert sampled_error < 100 * float(jnp.max(jnp.abs(R(XS)))) * EPS

    def test_maehly_degree_reduction(self):
        P, Q, R = self._setup()
        p, q, _ = cj.chebpade(R, 6, 5, "maehly")
        err = float((P - p).norm()) + float((Q - q).norm())
        bound = 100 * R.vscale * EPS
        print({"native_clause": 2, "continuous_L2_sum": err, "source_bound": bound}, flush=True)
        assert err < bound
        sampled_error = float(jnp.max(jnp.abs(p(XS) - P(XS)))) + float(
            jnp.max(jnp.abs(q(XS) - Q(XS)))
        )
        assert sampled_error < 100 * float(jnp.max(jnp.abs(R(XS)))) * EPS

    def test_geddes(self):
        a = jnp.asarray(
            [
                -464 / 6375,
                -742 / 6375,
                349 / 12750,
                512 / 6375,
                13 / 3400,
                2129 / 51000,
                1333 / 8160,
                9703 / 34000,
            ]
        )
        b = jnp.asarray([-32 / 85, -28 / 85, 1.0])
        f = cj.chebfun(lambda x: jnp.polyval(a, x) / jnp.polyval(b, x))
        p, q, _ = cj.chebpade(f, 7, 2)
        err = float((f - p / q).norm(jnp.inf))
        print({"native_clause": 3, "continuous_supremum": err, "source_bound": 2e-15}, flush=True)
        assert err < 2e-15
        cp = np.asarray(p.chebcoeffs(len(p)))
        assert abs(cp[0] - 17 / 46) < 1e-13
        # Prior101-site check retained at the original bound as supplemental.
        xs = jnp.asarray(np.linspace(-1, 1, 101))
        assert float(jnp.max(jnp.abs(f(xs) - p(xs) / q(xs)))) < 2e-15

    def test_complex_scaling(self):
        f = cj.chebfun(jnp.exp)
        p1, _, r1 = cj.chebpade(f, 2, 3)
        p2, _, r2 = cj.chebpade(1j * f, 2, 3)
        ratios = np.asarray(
            [
                complex(np.asarray(p2(0.3))) / complex(np.asarray(p1(0.3))),
                complex(np.asarray(r2(-0.4))) / complex(np.asarray(r1(-0.4))),
            ]
        )
        assert np.linalg.norm(ratios - np.asarray([1j, 1j]), np.inf) < 1e-13

    def test_explicit_num_coeffs(self):
        f = cj.chebfun(jnp.exp)
        p, q, r = cj.chebpade(f, 2, 2, 10)
        err = float((f - p / q).norm())
        print({"native_clause": 6, "continuous_L2": err, "source_bound": 2e-4}, flush=True)
        assert err < 2e-4
        # Prior single-site1e-3 probe retained only as supplemental.
        assert abs(float(np.real(np.asarray(r(0.5)))) - np.exp(0.5)) < 1e-3
