"""Port of MATLAB Chebfun tests/chebfun/test_chebpade.m (Fable 5).

FIXED: chebpade (Clenshaw-Lord + Maehly) added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_chebpade.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.utils.cfpade import _from_coeffs_any

EPS = np.finfo(float).eps
DOM = (-1.0, 3.0)
XS = jnp.asarray(np.linspace(-0.99, 2.99, 80))


class TestChebfunChebpade:
    def _setup(self):
        P = _from_coeffs_any(
            [0.5045, -1.3813, 2.1122, 0.0558, -0.6817], DOM)
        Q = _from_coeffs_any(
            [1, 0.1155, -0.8573, -0.2679, 0.5246], DOM)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            R = cj.chebfun(lambda x: P(x) / Q(x), domain=DOM)
        return P, Q, R

    def test_maehly_no_reduction(self):
        # pass(1)
        P, Q, R = self._setup()
        p, q, _ = cj.chebpade(R, 4, 4, "maehly")
        err = float(jnp.max(jnp.abs(p(XS) - P(XS)))) \
            + float(jnp.max(jnp.abs(q(XS) - Q(XS))))
        vs = float(jnp.max(jnp.abs(R(XS))))
        assert err < 100 * vs * EPS

    def test_maehly_degree_reduction(self):
        # pass(2)
        P, Q, R = self._setup()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, q, _ = cj.chebpade(R, 6, 5, "maehly")
        err = float(jnp.max(jnp.abs(p(XS) - P(XS)))) \
            + float(jnp.max(jnp.abs(q(XS) - Q(XS))))
        vs = float(jnp.max(jnp.abs(R(XS))))
        assert err < 100 * vs * EPS

    def test_geddes(self):
        # pass(3)-(4)
        a = jnp.asarray(np.array(
            [-464 / 6375, -742 / 6375, 349 / 12750, 512 / 6375,
             13 / 3400, 2129 / 51000, 1333 / 8160, 9703 / 34000]))
        b = jnp.asarray(np.array([-32 / 85, -28 / 85, 1.0]))
        f = cj.chebfun(
            lambda x: jnp.polyval(a, x) / jnp.polyval(b, x))
        p, q, _ = cj.chebpade(f, 7, 2)
        xs = jnp.asarray(np.linspace(-1, 1, 101))
        assert float(jnp.max(jnp.abs(
            f(xs) - p(xs) / q(xs)))) < 2e-15
        cp = np.asarray(p.funs[0].tech.coeffs)
        assert abs(cp[0] - 17 / 46) < 1e-13

    def test_complex_scaling(self):
        # pass(5): chebpade(1i*f) == 1i * chebpade(f)
        f = cj.chebfun(jnp.exp)
        p1, _, r1 = cj.chebpade(f, 2, 3)
        p2, _, r2 = cj.chebpade(f * 1j, 2, 3)
        rat1 = complex(np.asarray(p2(jnp.asarray(0.3)))) \
            / complex(np.asarray(p1(jnp.asarray(0.3))))
        rat2 = complex(np.asarray(r2(jnp.asarray(-0.4)))) \
            / complex(np.asarray(r1(jnp.asarray(-0.4))))
        assert abs(rat1 - 1j) < 1e-13
        assert abs(rat2 - 1j) < 1e-13

    def test_explicit_num_coeffs(self):
        # pass(6)-ish: explicit M runs and gives a sane [2/2]
        f = cj.chebfun(jnp.exp)
        _, _, r = cj.chebpade(f, 2, 2, 10)
        err = abs(float(np.real(np.asarray(
            r(jnp.asarray(0.5))))) - np.exp(0.5))
        assert err < 1e-3
