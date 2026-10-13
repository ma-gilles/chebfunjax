"""Compare diffbarytrig to the guarded MATLAB capture in test fixtures."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.io import loadmat

from chebfunjax.utils.trigutils import diffbarytrig

FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "diffbarytrig_native_source_20261013"
    / "capture.mat"
)


def _records():
    payload = loadmat(FIXTURE, squeeze_me=True, struct_as_record=False)
    return payload["records"].ravel()


@pytest.mark.parametrize("index", range(11))
def test_native_values(index):
    record = _records()[index]
    actual = diffbarytrig(
        jnp.asarray(record.zz),
        jnp.asarray(record.zj).reshape((-1,)),
        jnp.asarray(record.fj).reshape((-1,)),
        jnp.asarray(record.wj).reshape((-1,)),
        int(record.order),
        str(record.form),
    )
    expected = np.asarray(record.values)
    np.testing.assert_array_equal(np.isnan(np.asarray(actual)), np.isnan(expected))
    np.testing.assert_allclose(
        np.asarray(actual), expected, rtol=5e-14, atol=5e-14, equal_nan=True
    )


def test_native_order_zero_error_message():
    record = _records()[11]
    with pytest.raises(ValueError, match="^Use revaltrig\\.m for function evaluation$"):
        diffbarytrig(
            jnp.asarray(record.zz),
            jnp.asarray(record.zj).reshape((-1,)),
            jnp.asarray(record.fj).reshape((-1,)),
            jnp.asarray(record.wj).reshape((-1,)),
            int(record.order),
            str(record.form),
        )


def test_invalid_form_is_rejected_instead_of_hanging():
    zj = jnp.array([0.2, 1.1])
    with pytest.raises(ValueError, match="form must begin"):
        diffbarytrig(jnp.array([0.4]), zj, jnp.ones(2), jnp.ones(2), 1, "bad")


def test_integral_float_order_is_accepted():
    zj = jnp.array([0.2, 1.1, 2.4, 4.6])
    fj = jnp.array([1.0, -0.7, 0.4, 1.2])
    wj = jnp.array([0.7, -1.2, 0.8, 0.3])
    zz = jnp.array([0.35, 1.7])
    npt = diffbarytrig(zz, zj, fj, wj, 1, "odd")
    fpt = diffbarytrig(zz, zj, fj, wj, 1.0, "odd")
    np.testing.assert_array_equal(np.asarray(fpt), np.asarray(npt))
