"""Five literal MATLAB tests/domain/test_poly.m predicates.

Provenance
----------
Chebfun pin7574c77. Domain arguments are native Domain objects; procedural
poly dispatch is the Python adapter for MATLAB domain/POLY. Bounds, scalar
signed errors, matrix spectral norm, lengths and column counts are native.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import poly
from chebfunjax.domain import Domain

EPS = np.finfo(float).eps


def _vs(f):
    return float(np.max(np.atleast_1d(np.asarray(f.vscale))))


class TestDomainPoly:
    @pytest.mark.parametrize("clause", range(1, 6))
    def test_matlab_clause(self, clause):
        domain = Domain((-1, 2) if clause == 5 else (-1, 1))
        if clause <= 2:
            x = 0.0 if clause == 1 else 0.1
            f = poly(x, domain)
            if clause == 2:
                assert len(f) == 2
            assert float(f(x)) < 100 * EPS * _vs(f)
        elif clause == 3:
            x = np.array([-0.1, 0.1])
            f = poly(x, domain)
            assert len(f) == 3
            assert np.linalg.norm(np.asarray(f(jnp.asarray(x)))) < 100 * EPS * _vs(f)
        else:
            x = np.array([[-0.1, 0.1], [-0.5, 0.9]])
            f = poly(x, domain)
            values = np.column_stack(
                [
                    np.asarray(f.extract_columns(0)(jnp.asarray(x[:, 0]))),
                    np.asarray(f.extract_columns(1)(jnp.asarray(x[:, 1]))),
                ]
            )
            assert len(f) == 3 and f.size(2) == 2
            assert np.linalg.norm(values, ord=2) < 100 * EPS * _vs(f)
