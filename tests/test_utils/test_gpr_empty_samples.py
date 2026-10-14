"""Source-specific empty-prior sample behavior for fixed-L GPR calls."""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.gpr import gpr, gpr_chebfun

FIXTURE = Path(__file__).parent / "data/gpr_fixed_native_hex.json"


@pytest.mark.parametrize("trig", [False, True])
def test_empty_fixed_l_prior_without_samples_is_constant(trig):
    options = {
        "domain": (-1.0, 1.0),
        "sigma": 1.5,
        "length_scale": 0.4,
        "trig": trig,
    }
    result = gpr([], [], **options)
    assert result["domain"] == (-1.0, 1.0)
    assert result["samples"] is None
    np.testing.assert_array_equal(np.asarray(result["mean"]), np.zeros(200))
    np.testing.assert_array_equal(np.asarray(result["variance"]), np.full(200, 2.25))

    mean, variance, samples = gpr_chebfun([], [], **options)
    query = jnp.asarray([-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(np.asarray(mean(query)), np.zeros(3))
    np.testing.assert_array_equal(np.asarray(variance(query)), np.full(3, 2.25))
    assert samples is None


def _native_empty_sample_error():
    document = json.loads(FIXTURE.read_text())
    case = next(item for item in document["cases"]
                if item["case_id"] == "empty_fixedL_samples")
    variables = case["variables"]
    assert variables["success"]["values"] == [False]
    identifier = variables["error_identifier"]["values"][0]
    message = variables["error_message"]["values"][0]
    return identifier, message


def test_empty_fixed_l_mapping_preserves_native_missing_sigmaf_error():
    identifier, message = _native_empty_sample_error()
    assert identifier == "MATLAB:nonExistentField"
    with pytest.raises(AttributeError) as caught:
        gpr([], [], domain=(-1.0, 1.0), sigma=1.5, length_scale=0.4,
            n_samples=1)
    assert str(caught.value) == message


def test_empty_fixed_l_tuple_preserves_native_missing_sigmaf_error():
    identifier, message = _native_empty_sample_error()
    assert identifier == "MATLAB:nonExistentField"
    with pytest.raises(AttributeError) as caught:
        gpr_chebfun([], [], domain=(-1.0, 1.0), sigma=1.5,
                    length_scale=0.4, n_samples=1)
    assert str(caught.value) == message
