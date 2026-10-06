"""Independent bounded singular outer-finalization contracts.

Provenance
----------
MATLAB source : @chebfun/chebfun.m; @chebfun/getValuesAtBreakpoints.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import numpy as np
import pytest


def panels():
    from chebfunjax.chebfun1d.chebfun import _build_exps_piece
    ends = [-2., -1., 0., 1., 2.]
    def op(x):
        return 1/(x+2)
    pieces = [_build_exps_piece(op, a, b, -1. if k == 0 else 0., 0.,
                               "pole", "none", extrapolate=True)
              for k, (a, b) in enumerate(zip(ends[:-1], ends[1:]))]
    return op, pieces, ends


@pytest.mark.parametrize("given,wanted", [([-2., 2.], [-2., 2.]),
                                           ([-2., 0., 2.], [-2., 0., 2.])])
def test_redundant_analytical_panels_and_given_boundaries(given, wanted):
    from chebfunjax.chebfun1d.chebfun import _finalize_bounded_singular
    op, pieces, _ = panels()
    out = _finalize_bounded_singular(pieces, given, op)
    assert tuple(out.domain.breakpoints) == tuple(wanted)
    xx = jnp.linspace(-1.9, 2., 31)
    assert np.max(np.abs(np.asarray(out(xx)-op(xx)))) < 2e-12
    assert out.funs[0].tech.exponents[0] == -1.


def test_nan_callback_falls_back_but_infinity_is_retained():
    from chebfunjax.chebfun1d.chebfun import _finalize_bounded_singular
    op, pieces, ends = panels()
    def callback(x):
        return jnp.where(x == 0., jnp.nan,
                         jnp.where(x == 2., jnp.inf, op(x)))
    out = _finalize_bounded_singular(pieces, ends, callback)
    values = np.asarray(out._point_values)
    assert abs(values[2]-.5) < 1e-14
    assert np.isposinf(values[-1])
    assert tuple(out.domain.breakpoints) == tuple(ends)
