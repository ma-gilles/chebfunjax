"""Source contract for Chebfun3 coefficient extraction with trig factors.

Provenance
----------
MATLAB source: @chebfun3/chebcoeffs3.m and @trigtech/chebcoeffs.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.tech.trigtech import Trigtech


def test_chebcoeffs3_trig_factor_uses_native_no_count_error():
    f = chebfun3(
        lambda x, y, z: jnp.cos(jnp.pi * x)
        * jnp.cos(jnp.pi * y)
        * jnp.cos(jnp.pi * z),
        trig=True,
    )
    assert isinstance(f.cols[0], Trigtech)
    assert isinstance(f.rows[0], Trigtech)
    assert isinstance(f.tubes[0], Trigtech)

    # chebcoeffs3 has no count parameter and calls chebcoeffs(factor) without
    # one. Native @trigtech/chebcoeffs rejects that call with this identifier.
    with pytest.raises(ValueError, match="CHEBFUN:TRIGTECH:chebcoeffs:input"):
        f.chebcoeffs3()

    # At the factor level, an explicit count is supported and converts to a
    # finite Chebyshev series instead of returning the Fourier storage.
    c = f.cols[0].chebcoeffs(12)
    assert c.shape == (12,)
    assert jnp.all(jnp.isfinite(c))
