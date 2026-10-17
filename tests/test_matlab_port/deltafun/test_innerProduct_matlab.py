"""Port of MATLAB Chebfun tests/deltafun/test_innerProduct.m (Fable 5).

FIXED: Deltafun.innerProduct added in the Fable 5 audit
(distributional pairing <f + sum m_k delta_k, g>).

Provenance
----------
MATLAB source : tests/deltafun/test_innerProduct.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun

D = Domain((-1.0, 1.0))


class TestDeltafunInnerProduct:
    def test_delta_sifts(self):
        # <delta_{x0}, g> = g(x0)
        zero = Bndfun.from_function(lambda x: jnp.zeros_like(x), D)
        d = Deltafun.from_fun_and_deltas(zero, jnp.asarray([0.3]),
                                         jnp.asarray([[1.0]]))
        g = Bndfun.from_function(jnp.cos, D)
        assert abs(float(d.innerProduct(g)) - np.cos(0.3)) < 1e-13

    def test_smooth_plus_delta(self):
        f = Deltafun.from_fun_and_deltas(
            Bndfun.from_function(jnp.sin, D), jnp.asarray([0.25]),
            jnp.asarray([[2.0]]))
        g = Bndfun.from_function(jnp.cos, D)
        # <sin, cos> = 0 (odd*even) + 2 cos(0.25)
        assert abs(float(f.innerProduct(g)) - 2 * np.cos(0.25)) < 1e-13

    def test_overlapping_deltas_return_signed_infinity(self):
        zero = Bndfun.from_function(lambda x: jnp.zeros_like(x), D)
        d1 = Deltafun.from_fun_and_deltas(zero, jnp.asarray([0.3]),
                                          jnp.asarray([[1.0]]))
        assert np.isposinf(float(d1.innerProduct(d1)))
        dneg = Deltafun.from_fun_and_deltas(zero, jnp.asarray([0.3]),
                                            jnp.asarray([[-1.0]]))
        assert np.isneginf(float(d1.innerProduct(dneg)))

    def test_delta_derivatives_on_physical_domain(self):
        domain = Domain((2.0, 6.0))
        zero = Bndfun.from_function(lambda x: jnp.zeros_like(x), domain)
        square = Bndfun.from_function(lambda x: x**2, domain)
        d = Deltafun.from_fun_and_deltas(
            zero, jnp.asarray([3.0]), jnp.asarray([[0.0], [3.0], [4.0]]))
        # 3 delta' + 4 delta'' acts as -3 (x^2)' + 4 (x^2)''.
        assert abs(float(d.innerProduct(square)) - (-18.0 + 8.0)) < 1e-12

    def test_overlap_uses_strict_relative_proximity(self):
        zero = Bndfun.from_function(lambda x: jnp.zeros_like(x), D)
        left = Deltafun.from_fun_and_deltas(
            zero, jnp.asarray([0.5]), jnp.asarray([[1.0]]))
        near = Deltafun.from_fun_and_deltas(
            zero, jnp.asarray([0.5 * (1.0 + 0.5e-11)]),
            jnp.asarray([[1.0]]))
        separated = Deltafun.from_fun_and_deltas(
            zero, jnp.asarray([0.5 * (1.0 + 2.0e-11)]),
            jnp.asarray([[1.0]]))
        assert np.isposinf(float(left.innerProduct(near)))
        assert float(left.innerProduct(separated)) == 0.0

    def test_empty_deltafun_pairing(self):
        empty = Deltafun.empty()
        smooth = Bndfun.from_function(jnp.cos, D)
        assert empty.innerProduct(smooth).size == 0
        assert Deltafun.from_fun(smooth).innerProduct(empty).size == 0

    def test_empty_smooth_operand_discards_delta_data(self):
        delta_only = Deltafun(
            Bndfun.empty(), jnp.asarray([0.25]), jnp.asarray([[2.0]]))
        smooth = Bndfun.from_function(jnp.cos, D)
        # @deltafun/deltafun.m returns empty before parsing supplied delta data.
        assert delta_only.isempty()
        assert delta_only.n_deltas == 0
        assert delta_only.innerProduct(smooth).size == 0

    def test_all_seven_original_predicates_on_matlab_rng_fixture(self):
        """Port the literal source predicates on MATLAB's captured RNG draws."""
        fixture_path = (Path(__file__).parent / "fixtures"
                        / "innerProduct_rng1.json")
        fixture = json.loads(fixture_path.read_text())
        assert fixture["original_predicate_count"] == 7
        assert fixture["original_predicates"] == [True] * 7

        tol = fixture["deltaTol"]
        mag = jnp.asarray(fixture["mag"])
        loc = jnp.asarray(fixture["loc"])
        f = Bndfun.from_function(jnp.exp, D)
        g = Deltafun.from_fun_and_deltas(f, loc, mag)

        empty = Deltafun.empty()
        pass1 = (empty.innerProduct(empty).size == 0
                 and empty.innerProduct(g).size == 0
                 and g.innerProduct(empty).size == 0)

        expected2 = jnp.sum(mag * f(loc)) + (f * f).sum()
        pass2 = abs(float(g.innerProduct(f) - expected2)) < tol

        d = Deltafun.from_fun_and_deltas(
            Bndfun.from_function(lambda x: jnp.zeros_like(x), D),
            jnp.asarray([0.0]), jnp.asarray([[0.0], [1.0]]))
        smooth_g = Deltafun.from_fun(f)
        pass3 = abs(float(d.innerProduct(smooth_g)
                           + smooth_g.diff().funPart(0.0))) < tol
        pass4 = abs(float(d.diff().innerProduct(smooth_g)
                           - smooth_g.diff(2).funPart(0.0))) < tol

        loc5 = jnp.asarray([-0.5, 0.0, 0.5])
        mag5 = jnp.asarray([[1.0, 2.0, 3.0],
                            [-1.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0]])
        d5 = Deltafun.from_fun_and_deltas(
            Bndfun.from_function(lambda x: jnp.zeros_like(x), D),
            loc5, mag5)
        ip5 = (jnp.sum(mag5[0] * f(loc5))
               - jnp.sum(mag5[1] * f.diff()(loc5))
               + jnp.sum(mag5[2] * f.diff(2)(loc5)))
        pass5 = abs(float(smooth_g.innerProduct(d5) - ip5)) < tol

        d0 = Deltafun.from_fun_and_deltas(
            Bndfun.from_function(lambda x: jnp.zeros_like(x), D),
            jnp.asarray([0.0]), jnp.asarray([[1.0]]))
        d0neg = -d0
        pass6 = np.isposinf(float(d0.innerProduct(d0)))
        pass7 = np.isneginf(float(d0.innerProduct(d0neg)))

        assert [pass1, pass2, pass3, pass4, pass5, pass6, pass7] == \
            fixture["original_predicates"]
