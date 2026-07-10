"""Port of MATLAB Chebfun tests/bndfun/test_createMap.m (Opus 4.8).

MATLAB ``bndfun.createMap(dom)`` returns a map struct with fields ``For``
(forward map [-1,1] -> dom), ``Inv`` (inverse map dom -> [-1,1]) and ``Der``
(derivative of the forward map).  chebfunjax stores this affine map inside
``Domain`` rather than a separate ``createMap`` static method; the equivalent
operations are ``Domain.forward_map``, ``Domain.inverse_map`` and
``Domain.map_derivative``.  These are exact for the affine bounded map.

Provenance
----------
MATLAB source : tests/bndfun/test_createMap.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.domain import Domain

DOM = (-2.0, 7.0)


class TestBndfunCreateMap:
    def test_forward_and_inverse_map(self):
        d = Domain(DOM)
        fwd = np.asarray(d.forward_map(jnp.asarray([-1.0, 1.0])))
        inv = np.asarray(d.inverse_map(jnp.asarray([DOM[0], DOM[1]])))
        assert np.all(fwd == np.array(DOM))
        assert np.all(inv == np.array([-1.0, 1.0]))

    def test_map_derivative(self):
        d = Domain(DOM)
        assert d.map_derivative() == (DOM[1] - DOM[0]) / 2
