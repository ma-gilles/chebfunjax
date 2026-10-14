"""Literal n=2 mapping, including canonical-interval rounding.

Provenance
----------
MATLAB source : legpts.m, lines 118-124
Chebfun commit: 7574c77
Canonical nodes captured with MATLAB R2025b Update 2.
"""

import jax
import numpy as np
import pytest

from chebfunjax.utils.quadrature import legpts


@pytest.mark.parametrize("interval", [None, (-1., 1.), (-10., 3.), (0., 1.)])
def test_source_two_point_mapping_eager_and_jit(interval):
    a, b = (-1., 1.) if interval is None else interval
    x0 = np.asarray([-1., 1.]) / np.sqrt(3.)
    expected = ((b-a)/2)*(x0+1)+a
    eager = legpts(2, interval, newtheta=True)
    compiled = jax.jit(lambda: legpts(2, interval, newtheta=True))()
    for result in (eager, compiled):
        np.testing.assert_array_equal(result[0], expected)
        np.testing.assert_array_equal(result[1], np.ones(2)*(b-a)/2)
        np.testing.assert_array_equal(result[2], [1., -1.])
        np.testing.assert_allclose(result[3], np.arccos(x0), rtol=0, atol=4e-16)
    if interval is None:
        np.testing.assert_array_equal(eager[0],
                                      [-0.57735026918962584, 0.57735026918962573])
