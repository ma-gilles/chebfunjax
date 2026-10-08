"""Literal diagnostic contracts from MATLAB diffmat.m at 7574c77.

Provenance
----------
MATLAB source : diffmat.m, parseInputs and boundary-condition dispatch
Chebfun commit: 7574c77
Python ValueError/UserWarning text carries the MATLAB identifier as a prefix.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.utils.diffmat import diffmat


@pytest.mark.parametrize('args,identifier,message', [
    ((8, -1), 'wrongInput', 'The order of differentiation matrix must be non-negative.'),
    (((6, 8), 2, 'periodic'), 'wrongInput',
     'Rectangular Fourier differentiation matrices are not supported.'),
    ((8, 'periodic', 'rect'), 'wrongInput',
     'Rectangular Fourier differentiation matrices are not supported.'),
    ((8, 'chebkind1', 'chebkind2', 'leg'), 'unknown', 'Too many inputs for grid type.'),
    ((8, 'banana'), 'unknown', 'Unknown input banana'),
    ((8, 3, 'dirichlet', 'neumann', 'sum'), 'unknown',
     'Too many inputs for boundary condition. Use curly brackets to group left and right '
     'boundary conditions, if multiple boundary conditions are considered at one boundary.'),
    ((8, 3, ['dirichlet'], ['neumann'], ['sum']), 'unknown', 'Unrecognized boundary condition.'),
    ((8, 2, 'dirichlet'), 'wrongBC',
     'The number of boundary conditions must match differentiation order p.'),
    ((8, 1, ['banana']), 'wrongBC', 'Unknown type of boundary conditions.'),
])
def test_source_error_identifiers_and_text(args, identifier, message):
    with pytest.raises(ValueError) as caught:
        diffmat(*args)
    assert str(caught.value) == f'CHEBFUN:diffmat:{identifier}: {message}'


def test_source_breakpoint_warning_and_endpoint_selection():
    with pytest.warns(UserWarning) as caught:
        matrix = diffmat(8, 1, (-2, 0, 7))
    assert len(caught) == 1
    assert str(caught[0].message) == (
        'CHEBFUN:diffmat:noBreaks: DIFFMAT does not support domains with breakpoints.')
    assert jnp.array_equal(matrix, diffmat(8, 1, (-2, 7)))
