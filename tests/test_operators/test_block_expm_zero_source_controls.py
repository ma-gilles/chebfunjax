"""Zero-time semigroup preserves source initial-condition objects and lengths.

Provenance
----------
MATLAB source : @linop/expm.m, tests/linop/test_expm.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain
from chebfunjax.operators.blocklinop import BlockLinop, linop
from chebfunjax.operators.blocks import primitive_operators
from chebfunjax.operators.chebmatrix import ChebMatrix


@pytest.mark.parametrize("disc", ["chebcolloc2", "chebcolloc1", "ultraS"])
def test_zero_time_preserves_all_initial_entries_without_discretization(monkeypatch, disc):
    domain = (-1., 1.)
    _, _, deriv, _, _ = primitive_operators(domain)
    operator = linop(deriv**2)
    # Deliberate trailing zero coefficients and isolated complex point values
    # make resampling or simplify observably change source state.
    f = Chebfun.from_coeffs(jnp.array([1.+2.j, 3., 0., 0.]), Domain(domain))
    f = f.set_point_values(jnp.array([7.+4.j, 8.-2.j])).transpose()

    def forbidden(*args, **kwargs):
        pytest.fail("MATLAB zero-time branch returns original blocks")

    monkeypatch.setattr(BlockLinop, "_expm_at", forbidden)
    monkeypatch.setattr(BlockLinop, "_expm_altdisc", forbidden)
    for initial in [f, [f], ChebMatrix([[f]], domain=domain)]:
        result = operator.expm(0, initial, discretization=disc)
        assert result[0] is f
        assert len(result[0]) == 4
        assert result[0].is_transposed
        assert bool(jnp.array_equal(result[0]._point_values, f._point_values))
