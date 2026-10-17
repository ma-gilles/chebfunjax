"""Public inverse preserves native raw endpoint trials at default rangeCheck.

This supplemental native-output regression uses an identical retained forward
polynomial. It is separate from original tests/chebfun/test_inv.m predicates;
no inverse degree is prescribed and no existing tolerance is changed.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import Chebfun


def test_retained_polynomial_native_inverse_endpoint_bits():
    data = json.loads((Path(__file__).with_name('fixtures') /
                       'inverse_retained_flower_native_endpoints.json').read_text())

    def values(name):
        return np.frombuffer(bytes.fromhex(data[name]), dtype='<f8')

    forward = Chebfun.from_coeffs(jnp.asarray(values('coeffs_hex')),
                                  domain=tuple(values('domain_hex')))
    forward = forward.set_point_values(jnp.asarray(values('point_values_hex')))
    inverse = forward.inv()
    actual = np.asarray(inverse.point_values)
    assert actual.dtype == np.float64
    assert actual.tobytes() == bytes.fromhex(data['native_inverse_point_values_hex'])
