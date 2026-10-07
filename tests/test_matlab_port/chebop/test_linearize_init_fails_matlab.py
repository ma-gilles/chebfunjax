"""Complete port of MATLAB Chebfun test_linearize_init_fails.m.

Preserves all seven source clauses, exception identifiers, domains and bounds.
Default scalar norms are continuous L2. The coupled residual uses the source
chebmatrix default Frobenius norm, expressed as the square root of the sum of block L2 squares.

Provenance
----------
MATLAB source : tests/chebop/test_linearize_init_fails.m,
    @chebmatrix/norm.m, @chebfun/norm.m, @chebop/linearize.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Original: Copyright 2017 by The University of Oxford
    and The Chebfun Developers. See https://www.chebfun.org/.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)
ID = "CHEBFUN:CHEBOP:linearize:invalidInitialGuess"


def _assert_matlab_id(exc):
    assert str(exc.value).startswith(ID + ":")


def _matlab_chebmatrix_frobenius_norm(residuals):
    """Pinned @chebmatrix/norm.m default for scalar-valued blocks."""
    return jnp.sqrt(sum(component.norm() ** 2 for component in residuals))


class TestChebopLinearizeInitFails:
    def test_clause_1_first_order_ivp_continuous_l2(self):
        N = Chebop(lambda u: u.diff() - u.sqrt(), domain=(0.0, 1.0))
        N.lbc = 1.0
        u = N.solve(1.0)
        assert float((N(u) - 1.0).norm()) < 1e-10

    def test_clause_2_first_order_general_bc_bvp(self):
        N = Chebop(lambda u: u.diff() - u.sqrt(), domain=(0.0, 1.0))
        N.bc = lambda u: u(0.0) - 1.0
        with pytest.raises(ValueError) as exc:
            N.solve(1.0)
        _assert_matlab_id(exc)

    def test_clause_3_direct_solvebvp(self):
        N = Chebop(lambda u: u.diff() - u.sqrt(), domain=(0.0, 1.0))
        N.lbc = 1.0
        with pytest.raises(ValueError) as exc:
            N.solvebvp(1.0)
        _assert_matlab_id(exc)

    def test_clause_4_second_order_sqrt(self):
        N = Chebop(lambda u: u.diff(2) - u.sqrt(), domain=(0.0, 1.0))
        N.bc = 1.0
        with pytest.raises(ValueError) as exc:
            N.solve(1.0)
        _assert_matlab_id(exc)

    def test_clause_5_second_order_reciprocal(self):
        N = Chebop(lambda u: u.diff(2) - 1.0 / u, domain=(0.0, 1.0))
        N.bc = 1.0
        with pytest.raises(ValueError) as exc:
            N.solve(1.0)
        _assert_matlab_id(exc)

    def test_clause_6_coupled_system_ivp_frobenius_norm(self):
        N = Chebop(
            lambda x, u, v: [u.diff(2) - v.sqrt(), v.diff(2) + 1.0 / u],
            domain=(0.0, 1.0),
        )
        N.lbc = lambda u, v: [u - 1.0, u.diff() - 0.1,
                              v - 2.0, v.diff() + 0.2]
        uv = N.solve([1.0, 2.0])
        residual = N(uv)
        # MATLAB `norm(chebmatrix)` is sqrt(sum(norm(block,2)**2)); each
        # scalar chebfun's `norm(block,2)` equals its default continuous L2.
        source_frobenius = _matlab_chebmatrix_frobenius_norm(
            [residual[0] - 1.0, residual[1] - 2.0]
        )
        assert float(source_frobenius) < 1e-8

    def test_clause_7_coupled_system_bvp_invalid_initial_guess(self):
        N = Chebop(
            lambda x, u, v: [u.diff(2) - v.sqrt(), v.diff(2) + 1.0 / u],
            domain=(0.0, 1.0),
        )
        N.bc = lambda x, u, v: [u(0.0) - 1.0, u(1.0) + 1.0,
                                v(0.0) - 2.0, v(0.0) - 3.0]
        with pytest.raises(ValueError) as exc:
            N.solve(1.0)
        _assert_matlab_id(exc)
