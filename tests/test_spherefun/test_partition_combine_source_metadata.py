"""Independent representation and dtype controls for source partition/combine.

Provenance
----------
MATLAB source : @spherefun/partition.m; @spherefun/combine.m
Chebfun commit: 7574c77
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.tech.trigtech import Trigtech


def _object(pivots, plus, minus, locations=(), poles=False):
    terms = [Trigtech.from_coeffs(jnp.asarray([i + 1.0]), is_real=True)
             for i in range(len(pivots))]
    return Spherefun(cols=terms, rows=terms, pivots=jnp.asarray(pivots),
                     idx_plus=plus, idx_minus=minus,
                     pivot_locations=locations, nonzero_poles=poles)


def test_partition_preserves_locations_and_pole_semantics():
    f = _object([2.0, 3.0, 5.0], (2, 0), (1,),
                ((1., 2.), (3., 4.), (5., 6.)), True)
    even, odd = f.partition()
    assert even.pivot_locations == ((5., 6.), (1., 2.))
    assert odd.pivot_locations == ((3., 4.),)
    assert even.nonzero_poles and not odd.nonzero_poles
    assert even.idx_plus == (0, 1) and even.idx_minus == ()
    assert odd.idx_minus == (0,) and odd.idx_plus == ()
    assert jnp.array_equal(even.pivots, jnp.array([5., 2.]))
    assert even.cols[0] is f.cols[2] and odd.rows[0] is f.rows[1]


@pytest.mark.parametrize("disabled", [False, True])
def test_complex_partition_jit_and_derivative(disabled):
    @jax.jit
    def action(x):
        f = _object(jnp.array([1 + 2j, 3 - 4j]) * x, (0,), (1,))
        even, odd = f.partition()
        return even.pivots[0] + 2 * odd.pivots[0]

    with jax.disable_jit(disabled):
        assert action(2.) == 14 - 12j
        assert jax.jvp(action, (2.,), (1.,))[1] == 7 - 6j


def test_combine_reversed_parity_order_and_source_location_quirk():
    odd = _object([3 - 4j], (), (0,), ((9., 8.),))
    even = _object([1 + 2j], (0,), (), ((7., 6.),), True)
    result = Spherefun.combine(odd, even)
    assert jnp.array_equal(result.pivots, jnp.array([1 + 2j, 3 - 4j]))
    assert result.cols[0] is even.cols[0] and result.rows[1] is odd.rows[0]
    assert result.idx_plus == (0,) and result.idx_minus == (1,)
    # Source locations deliberately remain in argument order.
    assert result.pivot_locations == ((9., 8.), (7., 6.))
    assert result.nonzero_poles


@pytest.mark.parametrize("left,right", [(False, False), (False, True),
                                         (True, False), (True, True)])
def test_combine_pole_flag_is_logical_or(left, right):
    a = _object([2.], (0,), (), poles=left)
    b = _object([3.], (0,), (), poles=right)
    result = Spherefun.combine(a, b)
    assert result.nonzero_poles == (left or right)
    # Despite source prose, its implementation permits equal strict parity.
    assert result.idx_plus == (0, 1) and result.idx_minus == ()


def test_source_error_identifiers_and_empty_identity():
    f = _object([2., 3.], (0,), (1,))
    with pytest.raises(ValueError, match="CHEBFUN:SPHEREFUN:combine:parity"):
        Spherefun.combine(f, f)
    with pytest.raises(TypeError, match="CHEBFUN:SPHEREFUN:combine:unknown"):
        Spherefun.combine(f, 1)
    assert Spherefun.combine(f, Spherefun.empty()) is f
    assert Spherefun.combine(Spherefun.empty(), f) is f
