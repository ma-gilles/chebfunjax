"""Source-policy contracts, separate from strict end-to-end cube bound.

Provenance
----------
MATLAB source : @fun/detectEdge.m (findBlowup, zoomIn)
Chebfun commit: 7574c77
"""
import importlib

import jax.numpy as jnp


def locator():
    module = importlib.import_module("chebfunjax.chebfun1d.chebfun")
    method = getattr(module, "_source_find_blowup_bounded", None)
    assert method is not None, "source bounded singular locator is not implemented"
    return method


def test_zero_global_scale_does_not_reject_finite_function_peak():
    # Source rejects maxy < 1e5*vscale. A nonzero median replacement would
    # reject this bounded peak, but zero constructor scale must not do so.
    point = locator()(lambda x: 1-(x-.25)**2, -.5, .75, 0.)
    assert point is not None
    # Rounded 1-d^2 has a plateau near its maximum. Four conservative
    # unit-roundoff contributions bound value ambiguity by 4u=2eps;
    # inversion gives sqrt(2eps), with factor2 for bracket/endpoint rounding.
    # This new analytic control uses sqrt(8eps), not a source-case tolerance.
    assert abs(point-.25) < float(jnp.sqrt(8*jnp.finfo(jnp.float64).eps))


def test_positive_global_scale_rejects_same_bounded_peak():
    assert locator()(lambda x: 1-(x-.25)**2, -.5, .75, 1.) is None


def test_endpoint_zoom_retains_source_third_sample_bracket():
    batches = []

    def monotone(x):
        batches.append(x)
        return 2+jnp.asarray(x)

    locator()(monotone, 1., 2., 0.)
    # First call is two cached endpoints, then48 interior samples of50.
    # Source upper-end zoom starts at x(end-2), not x(end-1).
    first = batches[1]
    second = batches[2]
    expected_left = first[-2]
    expected_first_interior = expected_left + (2-expected_left)/49
    assert abs(float(second[0]-expected_first_interior)) <= 4*jnp.finfo(jnp.float64).eps
