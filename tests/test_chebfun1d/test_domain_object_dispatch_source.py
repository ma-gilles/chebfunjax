"""Domain-object dispatch compatibility, sources @domain/poly[fit].m7574c77."""

import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import poly, polyfit
from chebfunjax.domain import Domain


@pytest.mark.parametrize("constructor", [poly, polyfit])
def test_tuple_and_list_domains_retained(constructor):
    args = (
        (np.array([-0.1, 0.1]),)
        if constructor is poly
        else (np.array([-1.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0]), 2)
    )
    expected = constructor(*args, Domain((-1.0, 1.0)))
    for domain in [(-1.0, 1.0), [-1.0, 1.0]]:
        got = constructor(*args, domain)
        np.testing.assert_array_equal(
            np.asarray(got.funs[0].coeffs), np.asarray(expected.funs[0].coeffs)
        )
        assert got.domain == expected.domain
