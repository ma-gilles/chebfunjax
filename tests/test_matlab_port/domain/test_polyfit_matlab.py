"""Eight literal MATLAB tests/domain/test_polyfit.m predicates.

Provenance
----------
Chebfun pin7574c77. Domain arguments use Domain objects; procedural dispatch
is the Python adapter. Python 1D arrays represent native scalar data columns.
Vector 2-norms, array matrix spectral norm and all source bounds are retained.
Seed7681 uniform transformed stream previously matched archived MATLAB
atanh source capture; direct polyfit random y capture remains separate.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import polyfit
from chebfunjax.domain import Domain
from chebfunjax.utils.quadrature import chebpts

EPS = np.finfo(float).eps


def _vs(f):
    return float(np.max(np.atleast_1d(np.asarray(f.vscale))))


class TestDomainPolyfit:
    @pytest.mark.parametrize("clause", range(1, 9))
    def test_matlab_clause(self, clause):
        degree = {1: 2, 2: 3, 3: 4, 4: 4, 5: 12, 6: 3, 7: 2, 8: 3}[clause]
        count = {1: 1000, 2: 100, 3: 50, 4: 1000, 5: 10, 6: 1000, 7: 1000, 8: 10}[clause]
        domain = Domain((0, 1000) if clause == 7 else (-1, 1))
        x = (
            np.asarray(chebpts(count))
            if clause == 4
            else np.linspace(0, 1000, count)
            if clause == 7
            else np.linspace(-1, 1, count)
        )
        if clause == 2:
            y = x**3
        elif clause in (3, 4):
            y = 2 * x + 4 * x**2 + x**4
        elif clause == 5:
            y = -5 + 10 * np.random.RandomState(7681).rand(10)
        elif clause == 8:
            y = np.column_stack([x**2, x**3])
        else:
            y = x**2
        f = polyfit(x, y, degree, domain)
        residual = np.asarray(f(jnp.asarray(x))) - y
        error = np.linalg.norm(residual, ord=2)
        factor = 100 if clause in (1, 6, 7) else 5e5 if clause in (2, 3, 4) else 10
        assert error < factor * EPS * _vs(f)
