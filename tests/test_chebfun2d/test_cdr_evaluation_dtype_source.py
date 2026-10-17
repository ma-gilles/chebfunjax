"""Evaluated CDR dtype controls, including complex-valued analytic extensions."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.separable_approx import SeparableApprox
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize('case', ['real_trig', 'complex_trig', 'complex_cheb',
                                  'complex_coord', 'mixed_real', 'late_complex',
                                  'complex_pivot', 'empty'])
def test_actual_slice_product_dtype(case, record_property):
    cos = Trigtech.from_function(lambda t: jnp.cos(jnp.pi*t))
    sin = Trigtech.from_function(lambda t: jnp.sin(jnp.pi*t))
    exponential = Trigtech.from_function(lambda t: jnp.exp(1j*jnp.pi*t))
    linear = Chebtech2.from_coeffs(jnp.array([1., .2]))
    complex_poly = Chebtech2.from_coeffs(jnp.array([1.+1j, .2-.1j]))
    cols, rows, weights = [cos], [sin], jnp.array([1.25])
    analytic = lambda x, y: 1.25*jnp.cos(jnp.pi*y)*jnp.sin(jnp.pi*x)  # noqa: E731
    if case == 'complex_trig':
        cols = [exponential]
        analytic = lambda x, y: 1.25*jnp.exp(1j*jnp.pi*y)*jnp.sin(jnp.pi*x)  # noqa: E731
    elif case == 'complex_cheb':
        cols, rows = [complex_poly], [linear]
        analytic = lambda x, y: 1.25*(1.+1j+(.2-.1j)*y)*(1.+.2*x)  # noqa: E731
    elif case == 'mixed_real':
        rows = [linear]
        analytic = lambda x, y: 1.25*jnp.cos(jnp.pi*y)*(1.+.2*x)  # noqa: E731
    elif case == 'late_complex':
        cols, rows, weights = [cos, exponential], [sin, linear], jnp.array([1.25, .3])
        analytic = lambda x, y: (1.25*jnp.cos(jnp.pi*y)*jnp.sin(jnp.pi*x)  # noqa: E731
                                 + .3*jnp.exp(1j*jnp.pi*y)*(1.+.2*x))
    elif case == 'complex_pivot':
        weights = jnp.array([1.25+.5j])
        analytic = lambda x, y: (1.25+.5j)*jnp.cos(jnp.pi*y)*jnp.sin(jnp.pi*x)  # noqa: E731
    elif case == 'empty':
        cols, rows, weights = [], [], jnp.empty(0)
        analytic = lambda x, y: jnp.zeros(jnp.broadcast_shapes(x.shape, y.shape),  # noqa: E731
                                          dtype=jnp.result_type(x, y))
    f = SeparableApprox(cols=cols, rows=rows, pivots=weights, domain=(-1., 1., -1., 1.))
    complex_coordinates = case in ('complex_coord', 'empty')
    x = jnp.asarray(.23+.07j if complex_coordinates else .23)
    y = jnp.asarray(-.31-.04j if complex_coordinates else -.31)
    cv, rv = [c(y) for c in cols], [r(x) for r in rows]
    actual, compiled = f(x, y), jax.jit(lambda a, b: f(a, b))(x, y)
    metadata = dict(coeff_dtypes=[str(s.coeffs.dtype) for s in (*cols, *rows)],
                    slice_real_flags=[getattr(s, 'is_real', None) for s in (*cols, *rows)],
                    slice_value_dtypes=[str(v.dtype) for v in (*cv, *rv)],
                    pivot_dtype=str(weights.dtype), pivot_values=repr(np.asarray(weights)),
                    coordinate_dtypes=[str(x.dtype), str(y.dtype)],
                    output_dtype=str(actual.dtype), jit_dtype=str(compiled.dtype),
                    output=repr(np.asarray(actual)))
    record_property('actual_metadata', json.dumps(metadata))
    expected = analytic(x, y)
    # Exercise the corrected fallback directly even when the public call can
    # use the separate homogeneous-Chebtech array kernel.
    traced = f._call_traced(x, y)
    assert jnp.iscomplexobj(traced) == jnp.iscomplexobj(expected)
    np.testing.assert_allclose(traced, expected, atol=2e-14, rtol=2e-14)
    assert jnp.iscomplexobj(actual) == jnp.iscomplexobj(expected)
    assert actual.dtype == compiled.dtype
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(compiled, expected, atol=2e-14, rtol=2e-14)
    xv = jnp.array([-.7, .1, .6]) + (0.02j if complex_coordinates else 0.)
    yv = jnp.array([-.6, .2, .5]) + (-0.01j if complex_coordinates else 0.)
    for xx, yy in [(xv, yv), jnp.meshgrid(xv, yv)]:
        for evaluator in [f, jax.jit(lambda a, b: f(a, b))]:
            got = evaluator(xx, yy)
            want = analytic(xx, yy)
            assert jnp.iscomplexobj(got) == jnp.iscomplexobj(want)
            np.testing.assert_allclose(got, want, atol=2e-14, rtol=2e-14)
    tangent = jnp.ones_like(x)
    _, got_dx = jax.jvp(lambda a: f._call_traced(a, y), (x,), (tangent,))
    _, want_dx = jax.jvp(lambda a: analytic(a, y), (x,), (tangent,))
    np.testing.assert_allclose(got_dx, want_dx, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize('construction', ['callback', 'randnfun2'])
def test_public_real_constructor_metadata(construction, record_property):
    import chebfunjax as cj
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2

    if construction == 'callback':
        f = Chebfun2.from_function(lambda x, y: jnp.cos(jnp.pi*y)*jnp.sin(jnp.pi*x), trig=True)
    else:
        f = cj.randnfun2(1., 'trig', seed=0)
    ap = f.approx
    x, y = jnp.asarray(.23), jnp.asarray(-.31)
    cols, rows = [c(y) for c in ap.cols], [r(x) for r in ap.rows]
    result = f(x, y)
    record_property('actual_metadata', json.dumps(dict(
        rank=len(ap.cols), coeff_dtypes=[str(s.coeffs.dtype) for s in (*ap.cols, *ap.rows)],
        slice_real_flags=[getattr(s, 'is_real', None) for s in (*ap.cols, *ap.rows)],
        actual_slice_value_dtypes=[str(v.dtype) for v in (*cols, *rows)],
        pivot_dtype=str(ap.pivots.dtype), pivot_values_dtype=(None if ap.pivot_values is None
                                                            else str(ap.pivot_values.dtype)),
        actual_pivots=repr(np.asarray(ap.pivots)), actual_raw_pivots=repr(np.asarray(ap.pivot_values)),
        output_dtype=str(result.dtype), output=repr(np.asarray(result)))))
    assert not any(jnp.iscomplexobj(v) for v in (*cols, *rows))
    assert not jnp.iscomplexobj(ap.pivots)
    assert not jnp.iscomplexobj(result)
