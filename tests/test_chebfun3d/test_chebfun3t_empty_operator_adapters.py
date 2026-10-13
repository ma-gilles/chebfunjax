"""Controls for MATLAB empty operands and Python callable dispatch.

Provenance
----------
MATLAB source : ``@chebfun3t/{plus,rdivide,power}.m``,
    ``@chebfun3t/chebfun3t.m``
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3t import Chebfun3T, chebfun3t


def test_empty_string_and_empty_array_operands_map_before_scalar_dispatch():
    f = Chebfun3T.from_coeffs(jnp.array([[[2.0]]]), vscale=2.0)
    for empty in ("", [], np.empty((0, 1)), jnp.empty((0, 2, 1))):
        result = f + empty
        assert isinstance(result, Chebfun3T) and result.isempty()
        result = f / empty
        assert isinstance(result, Chebfun3T) and result.isempty()
        result = f ** empty
        assert isinstance(result, Chebfun3T) and result.isempty()


def test_empty_factory_returns_before_invalid_domain_or_options():
    assert chebfun3t("", domain=(1.0, 0.0), unknown_option=True).isempty()


def test_arithmetic_rejection_does_not_evaluate_python_callable():
    f = Chebfun3T.from_coeffs(jnp.array([[[2.0]]]), vscale=2.0)
    calls = []

    def callback(*args):
        calls.append(args)
        return 7.0

    with pytest.raises((TypeError, ValueError)):
        _ = f + callback
    assert calls == []
