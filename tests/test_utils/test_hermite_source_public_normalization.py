# uses-numpy: host reference assertions and captured MATLAB fixture inspection.
"""Exact expected-only source stages; Chebfun hermpts.m169-174, commit7574c77."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._gradual import gradual_positive_multiply

DATA = json.loads((Path(__file__).parent / "fixtures/hermite_source_public_normalization.json").read_text())

def bits(key):
    return np.asarray([int(x,16) for x in DATA[key]],dtype=np.uint64)

def scalar(key):
    return np.asarray(int(DATA[key],16),dtype=np.uint64).view(np.float64)

@pytest.mark.parametrize("disabled", [False, True])
def test_source_factor_two_separately_rounded_stages(disabled):
    helper=bits("helper_w_hex").view(np.float64)
    factor=scalar("factor_hex")
    prob_factor=scalar("prob_factor_hex")
    def stages(w,f,p):
        physical=gradual_positive_multiply(w,f)
        probabilist=gradual_positive_multiply(physical,p)
        return physical,probabilist
    with jax.disable_jit(disabled):
        physical,probabilist=jax.jit(stages)(jnp.asarray(helper),jnp.asarray(factor),jnp.asarray(prob_factor))
    np.testing.assert_array_equal(np.asarray(physical).view(np.uint64),bits("public_phys_hex"))
    np.testing.assert_array_equal(np.asarray(probabilist).view(np.uint64),bits("public_prob_hex"))

@pytest.mark.parametrize("disabled", [False, True])
def test_source_exact_physical_stage_probabilist_multiply(disabled):
    physical=bits("public_phys_hex").view(np.float64)
    with jax.disable_jit(disabled):
        actual=jax.jit(gradual_positive_multiply)(jnp.asarray(physical),jnp.asarray(scalar("prob_factor_hex")))
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint64),bits("public_prob_hex"))
