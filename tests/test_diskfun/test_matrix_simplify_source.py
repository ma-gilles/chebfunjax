"""Joint FUN simplification from literal native matrix controls, pin7574c77."""
import json
import math
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.diskfun._simplify import simplify_matrix
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech, trig_coeffs2vals
from chebfunjax.utils.misc import standard_chop
from chebfunjax.utils.transforms import coeffs2vals

RAW = json.loads((Path(__file__).parents[1]/"fixtures/disk_matrix_simplify_native_20261018.json").read_text())


def component(record, key):
    if not math.prod(record["shape"]):
        return np.empty(record["shape"])
    return np.asarray([struct.unpack(">d", bytes.fromhex(w))[0] for w in record[key]]).reshape(record["shape"], order="F")


def array(record):
    real = component(record, "re_hex")
    return real if record["is_real"] else real+1j*component(record, "im_hex")


def words(actual, expected):
    actual = np.asarray(actual)
    assert list(actual.shape) == expected["shape"]
    for values, key in [(actual.real, "re_hex"), (actual.imag, "im_hex")]:
        native = component(expected, key)
        np.testing.assert_array_equal(np.isnan(values), np.isnan(native))
        np.testing.assert_array_equal(np.isposinf(values), np.isposinf(native))
        np.testing.assert_array_equal(np.isneginf(values), np.isneginf(native))
        finite = np.isfinite(native)
        np.testing.assert_array_equal(np.ascontiguousarray(values[finite]).view(np.uint64),
                                      np.ascontiguousarray(native[finite]).view(np.uint64))


def techs(case):
    coefficients = jnp.asarray(array(case["input"]))
    return [(Chebtech2.from_coeffs(coefficients[:, j]) if case["kind"] == "cheb"
             else Trigtech.from_coeffs(coefficients[:, j], is_real=True))
            for j in range(coefficients.shape[1])]


@pytest.mark.parametrize("case", RAW["cases"], ids=lambda c: c["name"])
def test_captured_source_pairs_have_native_cutoffs(case):
    pairs = jnp.asarray(array(case["pairs"]))
    tolerances = array(case["tolerances"]).reshape(-1)
    actual = [standard_chop(pairs[:, j], float(tolerances[j])) for j in range(pairs.shape[1])]
    np.testing.assert_array_equal(actual, array(case["cutoffs"]).reshape(-1))


@pytest.mark.parametrize("case", [pytest.param(c, marks=pytest.mark.xfail(strict=True, reason="Inherited noisy FFT cutoff53 vs native54")) if c["name"] == "cheb_decay65" else c for c in RAW["cases"]], ids=lambda c: c["name"])
def test_joint_output_words_and_storage(case):
    result = simplify_matrix(techs(case), float(array(case["base"]).reshape(())))
    assert result is not None
    coefficients = jnp.stack([t.coeffs for t in result], axis=1)
    words(coefficients, case["output"])
    assert (not jnp.iscomplexobj(coefficients)) == case["output"]["is_real"]


@pytest.mark.parametrize("case", [pytest.param(c, marks=pytest.mark.xfail(strict=True, reason="Inherited FFT local-scale word")) if c["name"] == "cheb_decay64" else c for c in RAW["cases"]], ids=lambda c: c["name"])
def test_source_coefficient_scales(case):
    coefficients = jnp.asarray(array(case["input"]))
    values = coeffs2vals(coefficients) if case["kind"] == "cheb" else trig_coeffs2vals(coefficients)
    scales = jnp.max(jnp.abs(values), axis=0)[None, :]
    words(scales, case["scales"])


@pytest.mark.parametrize("case", RAW["cases"], ids=lambda c: c["name"])
def test_joint_layout_and_flags(case):
    original = techs(case)
    result = simplify_matrix(original, float(array(case["base"]).reshape(())))
    assert result is not None and len(result) == len(original)
    assert len({t.n for t in result}) == 1
    assert all(t.ishappy == case["ishappy_after"] for t in result)
    if case["kind"] == "trig":
        assert all(t.is_real for t in result)
    # Original coefficients are immutable; simplifying chooses a common prefix
    # or Fourier subspace from these, and never edits the source matrix.
    words(jnp.stack([t.coeffs for t in original], axis=1), case["input"])


def test_independent_polynomial_global_tolerance():
    coefficients = jnp.zeros((17, 3)).at[0, 0].set(1.).at[3, 1].set(2.**-20).at[8, 2].set(2.**-60)
    result = simplify_matrix([Chebtech2.from_coeffs(coefficients[:, j]) for j in range(3)], 2.**-52)
    # T3 remains above global absolute tolerance; the tiny T8 column has
    # local tolerance256, so standardChop returns1 for that column.
    assert [t.n for t in result] == [4, 4, 4]
    np.testing.assert_array_equal(jnp.stack([t.coeffs for t in result], axis=1), coefficients[:4])


def test_heterogeneous_unhappy_and_nonfinite_dispatch_retained():
    good = Chebtech2.from_coeffs(jnp.asarray([1., 0., 0.]))
    assert simplify_matrix([good, Chebtech2.from_coeffs(jnp.ones(2))], 2.**-52) is None
    assert simplify_matrix([Chebtech2.from_coeffs(jnp.ones(3), ishappy=False)], 2.**-52) is None
    assert simplify_matrix([Chebtech2.from_coeffs(jnp.asarray([jnp.nan, 1.]))], 2.**-52) is None
