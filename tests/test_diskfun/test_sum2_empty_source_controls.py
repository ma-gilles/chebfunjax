"""Native no-plus-term integral, including empty Diskfun.

MATLAB source: @diskfun/sum2.m, @diskfun/norm.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun


def test_empty_sum2_scalar_zero_preserves_other_empty_results():
    f = Diskfun.empty()
    assert f.isempty()
    result = np.asarray(f.sum2())
    assert result.shape == ()
    assert result.dtype == np.float64
    assert result == 0.0
    assert np.asarray(f.norm()).size == 0
    # Existing generic sum empty behavior is an independent regression.
    assert f.sum().isempty()


def test_minus_only_sum2_scalar_zero():
    f = Diskfun.from_function(lambda theta, r: r * jnp.cos(theta))
    assert f.idx_plus == ()
    assert np.asarray(f.sum2()).shape == ()
    assert float(f.sum2()) == 0.0


def test_nonempty_sum2_area():
    f = Diskfun.from_function(lambda theta, r: jnp.ones_like(r))
    assert abs(float(f.sum2()) - np.pi) < 100 * 1e-10
