"""Scalar left-division controls for the native Chebfun3T API.

Provenance
----------
MATLAB source : ``@chebfun3t/{ldivide,mldivide,rdivide}.m``
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d import Chebfun3T, ldivide, mldivide


def _constant(value, domain=(-1.0, 1.0) * 3):
    return Chebfun3T.from_coeffs(jnp.asarray([[[value]]]), domain=domain)


def test_named_left_division_is_distinct_from_python_right_division():
    f = _constant(3.0)

    # Native 2\f and ldivide(2,f) are f/2. Python 2/f continues to mean
    # native rdivide(2,f), which is a different pointwise operation.
    for result in (mldivide(2, f), ldivide(2, f)):
        np.testing.assert_allclose(np.asarray(result(0.0, 0.0, 0.0)), 1.5)
    np.testing.assert_allclose(np.asarray((2.0 / f)(0.0, 0.0, 0.0)), 2.0 / 3.0)
    np.testing.assert_allclose(np.asarray(ldivide(f, 2.0)(0.0, 0.0, 0.0)), 2.0 / 3.0)


def test_mldivide_type_zero_and_empty_dispatch():
    f = _constant(3.0)
    empty = Chebfun3T.empty()

    with pytest.raises(ValueError, match="Division must be by a scalar") as exc:
        mldivide(f, 2.0)
    assert exc.value.identifier == "CHEBFUN:CHEBFUN3T:mldivide:nonScalar"

    with pytest.raises(ValueError, match="Division by zero") as exc:
        mldivide(0.0, f)
    assert exc.value.identifier == "CHEBFUN:CHEBFUN3T:rdivide:divByZero"

    for nondouble in (jnp.asarray(2.0, dtype=jnp.float32),
                      jnp.asarray(2, dtype=jnp.int64),
                      jnp.asarray(True, dtype=jnp.bool_)):
        with pytest.raises(ValueError, match="Division must be by a scalar") as exc:
            mldivide(nondouble, f)
        assert exc.value.identifier == "CHEBFUN:CHEBFUN3T:mldivide:nonScalar"

    assert mldivide(empty, f) is empty
    assert mldivide(f, empty) is empty
    assert mldivide(None, f) is None
    assert mldivide("", f) == ""
    empty_left = jnp.empty((0,), dtype=jnp.float64)
    empty_right = jnp.empty((0,), dtype=jnp.float64)
    assert mldivide(empty_left, f) is empty_left
    assert mldivide(f, empty_right) is empty_right
    assert isinstance(ldivide(empty, f), Chebfun3T)
    assert ldivide(empty, f).isempty()
    assert isinstance(ldivide(None, f), Chebfun3T)
    assert ldivide(None, f).isempty()
    assert ldivide("", f).isempty()
    assert ldivide(empty_left, f).isempty()

    # These named operations are class adapters, not general numeric array
    # division functions when no Chebfun3T participates.
    with pytest.raises(TypeError, match="requires at least one Chebfun3T"):
        ldivide(2, 3)
    with pytest.raises(TypeError, match="requires at least one Chebfun3T"):
        mldivide(2, [])


def test_left_division_of_chebfuns_keeps_rdivide_domain_error():
    f = _constant(3.0)
    g = _constant(2.0, domain=(-2.0, 2.0) * 3)
    with pytest.raises(ValueError, match="Domains inconsistent") as exc:
        ldivide(f, g)
    assert exc.value.identifier == "CHEBFUN:CHEBFUN3T:rdivide:domains"
