"""Literal clauses from tests/classicfun/test_mtimes.m.

MATLAB source: tests/classicfun/test_mtimes.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
Python @ is MATLAB mtimes; * remains pointwise times. Python literals are
MATLAB doubles and explicit uint8 keeps its class. Uniform RNG uses MT19937
seed 6178 (first 100 points independently captured in the logical fixture).
Native randn and the subsequent uniform stream await the capture fixture;
these clauses skip explicitly, with separate deterministic controls retained.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.unbndfun import Unbndfun

EPS = float(jnp.finfo(jnp.float64).eps)
DOM = Domain((-2.0, 7.0))
X = jnp.asarray(9 * np.random.RandomState(6178).rand(1000) - 2)
FIXTURE = Path(__file__).parents[2] / "fixtures/classic_mtimes_rng6178.json"


def array_op(x):
    return jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)


def unbounded_op(x):
    return jnp.stack([jnp.exp(x), x*jnp.exp(x), (1-jnp.exp(x))/x], axis=-1)


@pytest.fixture(scope="module")
def native():
    if not FIXTURE.exists():
        pytest.skip("Native MATLAB randn after rand(1000,1) and subsequent rand(100,1) fixture pending")
    data = json.loads(FIXTURE.read_text())
    assert data["source_commit"] == "7574c77680d7e82b79626300bf255498271a72df"
    assert jnp.array_equal(jnp.asarray(data["x_bounded"]), X)
    assert bool(jnp.all(jnp.asarray(data["pass"])))
    return (complex(*data["alpha"]), jnp.asarray(data["A"]), jnp.asarray(data["x_unbounded"]))


def check_error(action, identifier, message):
    with pytest.raises(ValueError) as caught:
        action()
    assert caught.value.identifier == identifier
    assert str(caught.value) == message


class TestClassicfunMtimes:
    def test_pass01_empty(self):
        f = Bndfun.from_function(jnp.sin, DOM)
        g = Bndfun.empty()
        assert all(jnp.size(value) == 0 for value in (f @ [], [] @ f, 2 @ g, g @ 2))

    def test_pass02_scalar_isequal(self, native):
        alpha, _, _ = native
        f = Bndfun.from_function(jnp.sin, DOM)
        assert (alpha @ f).isequal(f @ alpha)

    def test_pass03_scalar_values(self, native):
        alpha, _, _ = native
        f = Bndfun.from_function(jnp.sin, DOM)
        g1 = alpha @ f
        assert jnp.linalg.norm(g1(X) - alpha*jnp.sin(X), ord=jnp.inf) < 10*g1.vscale*EPS

    def test_pass04_scalar_zero(self):
        f = Bndfun.from_function(jnp.sin, DOM)
        assert jnp.all((0 @ f)(X) == 0)

    def test_pass05_array_isequal(self, native):
        alpha, _, _ = native
        f = Bndfun.from_function(array_op, DOM)
        assert (alpha @ f).isequal(f @ alpha)

    def test_pass06_array_values(self, native):
        alpha, _, _ = native
        g1 = alpha @ Bndfun.from_function(array_op, DOM)
        assert jnp.max(jnp.abs(g1(X) - alpha*array_op(X))) < 10*jnp.max(g1.vscale*EPS)

    def test_pass07_array_zero(self):
        f = Bndfun.from_function(array_op, DOM)
        assert jnp.all((0 @ f)(X) == jnp.zeros((X.size, 3)))

    def test_pass08_matrix(self, native):
        _, A, _ = native
        g = Bndfun.from_function(array_op, DOM) @ A
        assert jnp.max(jnp.abs(g(X) - array_op(X) @ A)) < 10*jnp.max(g.vscale*EPS)

    def test_pass09_left_size(self):
        f = Bndfun.from_function(jnp.exp, DOM)
        check_error(lambda: [[1, 2, 3]] @ f,
                    "CHEBFUN:CLASSICFUN:mtimes:size", "Inner matrix dimensions must agree.")

    def test_pass10_right_size(self):
        f = Bndfun.from_function(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1), DOM)
        check_error(lambda: f @ [[1], [2], [3]],
                    "CHEBFUN:CHEBTECH:mtimes:size2", "Inner matrix dimensions must agree.")

    def test_pass11_two_funs(self):
        f = Bndfun.from_function(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1), DOM)
        g = Bndfun.from_function(lambda x: x, DOM)
        check_error(lambda: f @ g,
                    "CHEBFUN:CLASSICFUN:mtimes:classicfunMtimesClassicfun",
                    "Use .* to multiply CLASSICFUN objects.")

    def test_pass12_uint8(self):
        f = Bndfun.from_function(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1), DOM)
        check_error(lambda: f @ jnp.uint8(128),
                    "CHEBFUN:CLASSICFUN:mtimes:classicfunMtimesUnknown",
                    "mtimes does not know how to multiply a CLASSICFUN and a uint8.")

    def test_pass13_unbounded_matrix_norm(self, native):
        _, A, x = native
        f = Unbndfun.from_function(unbounded_op, Domain((-jnp.inf, -3*jnp.pi)))
        g = f @ A
        # MATLAB norm(matrix, inf) is the maximum absolute row sum.
        assert jnp.linalg.norm(g(x) - unbounded_op(x) @ A, ord=jnp.inf) < 1e2*jnp.max(EPS*g.vscale)
