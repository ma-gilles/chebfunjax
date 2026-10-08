# uses-numpy: analytic host reference assertions; numerical bounds unchanged.
"""Bounded source vscale controls; Chebfun 7574c77.

Provenance: @separableApprox/vscale.m, @separableApprox/length.m,
@spherefun/sample.m, @separableApprox/cdr.m. Complex factor case deliberately
exposes the inherited public sample real-part omission; do not xfail it.
"""
import math

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.tech.trigtech import Trigtech


class Protocol:
    def __init__(self, lengths, values, empty=False):
        self.lengths, self.values, self.empty = lengths, values, empty
        self.calls = []

    def isempty(self):
        self.calls.append("isempty")
        return self.empty

    def length(self):
        self.calls.append("length")
        return self.lengths

    def sample(self, m, n):
        self.calls.append(("sample", m, n))
        return jnp.asarray(self.values)


def test_empty_precedes_length_and_sample():
    f = Protocol(None, None, empty=True)
    assert Spherefun.vscale(f) == 0.0
    assert f.calls == ["isempty"]


@pytest.mark.parametrize("lengths, expected", [
    ((1, 1), (9, 9)), ((3, 17), (9, 17)),
    ((23, 4), (23, 9)), ((2001, 31), (2000, 31)),
    ((41, 4000), (41, 2000)), ((2000, 9), (2000, 9)),
])
def test_source_rows_first_independent_clamps(lengths, expected):
    f = Protocol(lengths, [[-3., 2.], [1., -4.]])
    assert Spherefun.vscale(f) == 4.0
    assert f.calls == ["isempty", "length", ("sample", *expected)]


@pytest.mark.parametrize("values, expected", [
    ([[3+4j, -2j]], 5.), ([[0., 0.]], 0.), ([[float("inf")]], float("inf")),
])
def test_magnitude_reduction_protocol(values, expected):
    # Tests reduction after sample, not correctness of inherited sample itself.
    f = Protocol((1, 1), values)
    assert Spherefun.vscale(f) == expected


def field(column_coefficients, row_coefficients=(1.,), pivot=1., real=True):
    c = Trigtech.from_coeffs(jnp.asarray(column_coefficients), is_real=real)
    r = Trigtech.from_coeffs(jnp.asarray(row_coefficients), is_real=True)
    return Spherefun(cols=[c], rows=[r], pivots=jnp.asarray([pivot]),
                     idx_plus=(0,), idx_minus=())


@pytest.mark.parametrize("value", [0., -4., 2.5])
def test_real_constant_source_value(value):
    assert field([value]).vscale() == abs(value)


def test_theta_cosine_endpoint_source_value():
    # 2 + cos(theta), scaled by row2 / pivot2; max3 at physical north pole.
    np.testing.assert_allclose(field([.5, 2., .5], (2.,), 2.).vscale(),
                               3., rtol=0., atol=8*np.finfo(float).eps)


def test_source_sample_takes_real_of_each_factor():
    # Source sample explicitly real(C) and real(R), giving abs(2)=2.
    assert field([2.+3.j], real=False).vscale() == 2.


def test_all_nan_pivot_retained():
    assert math.isnan(field([1.], pivot=float("nan")).vscale())
