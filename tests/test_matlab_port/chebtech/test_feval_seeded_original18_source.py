"""Original Chebtech feval predicates on the pinned MATLAB seeded nodes.

This supplements (and does not replace) ``test_feval_matlab.py``. The native
test seeds MATLAB's RNG with 7681, draws one 1000-by-1 vector, then reuses that
same vector for both Chebtech1 and Chebtech2 and predicates 1-8. Predicate 9
uses its literal 2-by-3 matrix. These tests preserve all 18 original
assertions and source tolerances while using the captured MATLAB draw.

Provenance
----------
Native source: tests/chebtech/test_feval.m
Native source SHA-256: 010d78a299d1bfab1cc31200f09578d584e5a4f550a765bfc20db98b999e2522
Native seed helper SHA-256: 2e6f80c9fcd2c511a783742cb688b22818c8084d457fd724ab471ea4e0d02e5c
Chebfun pin: 7574c77680d7e82b79626300bf255498271a72df
Fixture: ../chebfun/data/feval_seedRNG_7681_xr.mat, variable ``xr``
Fixture SHA-256: 5bda37bcb3a3994f75dfa2e624ffadc1c391fdd1968be1e5fec626ae906b513b
Fixture provenance SHA-256: ea04906ed7974534b89d89ebc1b562829b330eb2c9f12bb88fb08a8e571dba53
Capture helper SHA-256: 1d29f9db45c583ca3dd052b6516d739ebb77158c1145fe9d0931327a187407cd

The captured vector was produced by the same seed and first draw in the
native Chebfun test_feval.m. Both native tests call ``seedRNG(7681)`` and
immediately draw ``2 * rand(1000, 1) - 1`` before other random operations.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.io import loadmat

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)
BOTH = [Chebtech1, Chebtech2]
Z6 = np.exp(2 * np.pi * 1j / 6)

_MAT_PATH = (
    Path(__file__).parents[1]
    / "chebfun"
    / "data"
    / "feval_seedRNG_7681_xr.mat"
)
_MATLAB_X = np.asarray(loadmat(_MAT_PATH)["xr"], dtype=np.float64)
assert _MATLAB_X.shape == (1000, 1)
X = jnp.asarray(_MATLAB_X[:, 0])


def _max_norm(values) -> float:
    """Maximum absolute entry, matching MATLAB vector infinity norm here."""
    return float(np.max(np.abs(np.asarray(values))))


def _native_array_tolerance(f) -> float:
    """Scalar form of MATLAB ``max(vscale(f) * eps)``."""
    return 10.0 * float(np.max(np.asarray(f.vscale_columns))) * EPS


@pytest.mark.parametrize("Tech", BOTH)
def test_original_predicate_1_exp(Tech):
    f = Tech.from_function(lambda x: jnp.exp(x) - 1)
    exact = jnp.exp(X) - 1
    assert _max_norm(f(X) - exact) < 10 * f.vscale * EPS


@pytest.mark.parametrize("Tech", BOTH)
def test_original_predicate_2_lorentzian(Tech):
    f = Tech.from_function(lambda x: 1.0 / (1 + x**2))
    exact = 1.0 / (1 + X**2)
    assert _max_norm(f(X) - exact) < 10 * f.vscale * EPS


@pytest.mark.parametrize("Tech", BOTH)
def test_original_predicate_3_high_frequency(Tech):
    f = Tech.from_function(lambda x: jnp.cos(1e4 * x))
    exact = jnp.cos(1e4 * X)
    assert _max_norm(f(X) - exact) < 2e4 * f.vscale * EPS


@pytest.mark.parametrize("Tech", BOTH)
def test_original_predicate_4_complex_sinh(Tech):
    f = Tech.from_function(lambda t: jnp.sinh(t * Z6))
    exact = jnp.sinh(X * Z6)
    assert _max_norm(f(X) - exact) < 10 * f.vscale * EPS


@pytest.mark.parametrize("Tech", BOTH)
def test_original_predicate_5_row_vector(Tech):
    f = Tech.from_function(lambda t: jnp.sinh(t * Z6))
    x_row = X.reshape((1, 1000))
    err = f(x_row) - jnp.sinh(x_row * Z6)
    assert err.shape == (1, 1000)
    assert _max_norm(err) < 10 * f.vscale * EPS


@pytest.mark.parametrize("Tech", BOTH)
def test_original_predicate_6_matrix(Tech):
    f = Tech.from_function(lambda t: jnp.sinh(t * Z6))
    x_matrix = jnp.asarray(np.reshape(_MATLAB_X, (100, 10), order="F"))
    err = f(x_matrix) - jnp.sinh(x_matrix * Z6)
    assert err.shape == (100, 10)
    assert _max_norm(err) < 10 * f.vscale * EPS


@pytest.mark.parametrize("Tech", BOTH)
def test_original_predicate_7_three_dimensional_array(Tech):
    f = Tech.from_function(lambda t: jnp.sinh(t * Z6))
    x_tensor = jnp.asarray(np.reshape(_MATLAB_X, (10, 10, 10), order="F"))
    err = f(x_tensor) - jnp.sinh(x_tensor * Z6)
    assert err.shape == (10, 10, 10)
    assert _max_norm(err) < 10 * f.vscale * EPS


@pytest.mark.parametrize("Tech", BOTH)
def test_original_predicate_8_array_valued(Tech):
    f = Tech.from_function(
        lambda x: jnp.stack([jnp.sin(x), x**2, jnp.exp(1j * x)], axis=-1)
    )
    exact = jnp.stack([jnp.sin(X), X**2, jnp.exp(1j * X)], axis=-1)
    err = np.asarray(f(X) - exact)
    assert err.shape == (1000, 3)
    # Native ``max(abs(err))`` is columnwise, followed by ``all``; its RHS
    # ``10 * max(vscale(f) * eps)`` is one scalar shared by all columns.
    assert np.all(np.max(np.abs(err), axis=0) < _native_array_tolerance(f))


@pytest.mark.parametrize("Tech", BOTH)
def test_original_predicate_9_array_valued_matrix_argument(Tech):
    f = Tech.from_function(
        lambda x: jnp.stack(
            [jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x), jnp.exp(jnp.pi * x)],
            axis=-1,
        )
    )
    x2 = jnp.asarray([[-1.0, 0.0, 1.0], [0.25, 0.5, 0.75]])
    fx = np.asarray(f(x2))
    # Python stores components last; reorder only this comparison to native's
    # [rows, component blocks] layout.
    assert fx.shape == (2, 3, 3)
    native_layout = np.transpose(fx, (0, 2, 1)).reshape(2, 9)
    root2 = np.sqrt(2.0)
    exact = np.asarray(
        [
            [0, 0, 0, -1, 1, -1, np.exp(-np.pi), 1, np.exp(np.pi)],
            [1 / root2, root2 / root2, 1 / root2, 1 / root2, 0,
             -1 / root2, np.exp(np.pi * 0.25), np.exp(np.pi * 0.5),
             np.exp(np.pi * 0.75)],
        ]
    )
    # Native ``all(all(abs(fx-f_exact) < 10*max(vscale(f)*eps)))``.
    assert np.all(np.abs(native_layout - exact) < _native_array_tolerance(f))
