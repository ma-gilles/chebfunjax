"""Source branches absent from the default-constructor original test route.

Native pin7574c77 @chebfun3/chebfun3classic.m. Original scientific predicates
remain in the unchanged test_constructor.m port, run under classic defaults.
"""

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d._chebfun3classic import aca2, aca3, construct, tolerance3
from chebfunjax.chebfun3d._chebfun3double import tensor_axes
from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech2


def test_factor_limited_aca2_preserves_failure_and_pivots():
    c, p, r, indices, failed = aca2(jnp.eye(6), 0.0, 2.0)
    assert failed and c.shape == (6, 3) and r.shape == (6, 3)
    np.testing.assert_array_equal(p, np.ones(3))
    np.testing.assert_array_equal(indices, [[0, 0], [1, 1], [2, 2]])
    _c, p, _r, indices, failed = aca2(jnp.eye(6), 0.0, 0.0)
    assert not failed and p.size == 6
    np.testing.assert_array_equal(indices, np.column_stack((np.arange(6), np.arange(6))))


def test_factor_limited_aca3_retains_both_failure_flags():
    tensor = jnp.zeros((6, 6, 6)).at[jnp.arange(6), jnp.arange(6), jnp.arange(6)].set(1.0)
    domain = (-1.0, 1.0) * 3
    skeleton = aca3(tensor, tensor_axes(tensor.shape, domain, Chebtech2), domain, 0.0, 0.0)
    assert skeleton.fail3 and not skeleton.fail2
    np.testing.assert_array_equal(skeleton.pivots, np.ones(3))
    np.testing.assert_array_equal(skeleton.indices3, [[0, 0, 0], [1, 1, 1], [2, 2, 2]])


def test_rectangular_phase1_keeps_explicit_grid_in_tolerance():
    domain = (-1.0, 1.0) * 3
    values = jnp.ones((5, 5, 3))
    axes = tensor_axes(values.shape, domain, Chebtech2)
    relative, absolute = tolerance3(values, axes, domain, np.finfo(float).eps, 3)
    expected = 2 * 3**0.8 * np.finfo(float).eps
    assert abs(relative - expected) <= np.spacing(expected)
    assert float(absolute) == relative
    assert relative < 2 * 5**0.8 * np.finfo(float).eps


def test_classic_uses_factory_grid_and_scalar_zero_rank():
    pref = ChebfunPref()
    pref.cheb3Prefs.constructor = "classic"
    pref.minSamples = 33
    pref.maxLength = 5
    trace = []
    f = construct(lambda x, y, z: x * y * z, (-1.0, 1.0) * 3, pref, trace=trace)
    assert trace[0]["grid_phase1"] == 17
    np.testing.assert_allclose(
        float(f(0.2, 0.3, 0.4)), 0.024, rtol=0, atol=100 * np.finfo(float).eps
    )
    z = chebfun3(lambda x, y, z: jnp.zeros_like(x), "classic")
    assert z.rank == (0, 0, 0)
    assert z.length() == (1, 1, 1)


def test_numeric_classic_selector_precedes_callable_dispatch():
    # Native constructor.m's double branch precedes classic selection.
    values = jnp.ones((3, 3, 3))
    f = chebfun3(values, "classic")
    assert abs(float(f(0.2, 0.3, 0.4)) - 1) < 100 * np.finfo(float).eps


def test_phase1_refines_slice_grid_without_changing_fiber_grid():
    pref = ChebfunPref()
    pref.cheb3Prefs.constructor = "classic"
    trace = []
    f = construct(
        lambda x, y, z: jnp.exp(x * y) * z, (-1.0, 1.0) * 3, pref, fiber_dim=3, trace=trace
    )
    shape = trace[0]["phase1_shape"]
    assert shape[0] > shape[2] and shape[0] == shape[1]
    assert trace[0]["grid_phase1"] == shape[2]
    assert abs(float(f(0.2, 0.3, 0.4)) - np.exp(0.06) * 0.4) < 100 * np.finfo(float).eps


def test_vector_check_repeats_probe_and_wraps_scalar_output():
    from chebfunjax.chebfun3d._chebfun3classic import vector_check

    calls = []

    def scalar(x, y, z):
        calls.append(x.shape)
        return 2.0

    op, vectorized = vector_check(scalar, (-1.0, 1.0) * 3)
    assert calls == [(2, 2, 2), (2, 2, 2)] and not vectorized
    np.testing.assert_array_equal(op(jnp.arange(3.0), 0.0, 0.0), [2.0, 2.0, 2.0])


def test_classic_auto_vectorizes_scalar_callback():
    import math

    import pytest

    with pytest.warns(RuntimeWarning, match="constructor:vectorize"):
        f = chebfun3(lambda x, y, z: math.cos(x), "classic")
    assert abs(float(f(0.2, 0.3, 0.4)) - math.cos(0.2)) < 100 * np.finfo(float).eps


def test_vector_check_reports_nonfinite_before_dimension_clustering():
    import pytest

    for value, identifier in [(jnp.inf, "constructor:inf"), (jnp.nan, "constructor:nan")]:
        with pytest.raises(ValueError, match=identifier):
            chebfun3(lambda x, y, z, v=value: jnp.full_like(x, v), "classic")
