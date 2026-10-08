"""Deterministic adapter controls, NOT native RNG source qualification.

Original MATLAB source: tests/classicfun/test_mtimes.m, commit 7574c77.
The scalar/matrix/site stand-ins below retain historical numerical coverage.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.unbndfun import Unbndfun

EPS = float(np.finfo(np.float64).eps)
DOM = Domain((-2.0, 7.0))
X = jnp.asarray(np.linspace(-2.0, 7.0, 1000))
ALPHA = 0.3 + 0.7j
INF = np.inf

# Stand-in for MATLAB's ``A = randn(3, 3)``; any well-conditioned 3x3 real
# matrix exercises the same code path.
A = jnp.asarray(
    np.array(
        [
            [0.537667139546100, -2.258846861003648, 0.318765239858981],
            [1.833885014595086, 0.862173320368121, -1.307688296305273],
            [-2.258846861003648, 0.318765239858981, -0.433592022305684],
        ]
    )
)


class TestDeterministicMtimesControls:
    def test_scalar_left_equals_right(self):
        f = Bndfun.from_function(jnp.sin, DOM)
        g1 = ALPHA @ f
        g2 = f @ ALPHA
        assert g1.isequal(g2)

    def test_scalar_multiplication_values(self):
        f = Bndfun.from_function(jnp.sin, DOM)
        g1 = ALPHA @ f
        err = jnp.abs(jnp.asarray(g1(X)) - ALPHA * jnp.sin(X))
        assert float(jnp.max(err)) < 10 * g1.vscale * EPS

    def test_zero_scalar_gives_zero(self):
        f = Bndfun.from_function(jnp.sin, DOM)
        g = 0 @ f
        assert bool(jnp.all(jnp.asarray(g(X)) == 0))

    def test_array_valued_scalar_mult(self):
        # pass(5,6,7): scalar mtimes of an array-valued fun -- alpha*f == f*alpha,
        # values match, and 0*f is all-zero.
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) Bndfun.
        def fop(x):
            return jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)
        f = Bndfun.from_function(fop, DOM)
        g1 = ALPHA @ f
        g2 = f @ ALPHA
        assert g1.isequal(g2)
        err = jnp.abs(jnp.asarray(g1(X)) - ALPHA * fop(X))
        assert float(jnp.max(err)) < 10 * g1.vscale * EPS
        assert bool(jnp.all(jnp.asarray((0 @ f)(X)) == 0))

    def test_matrix_mtimes(self):
        # pass(8): f*A mixes the columns of an array-valued fun, tol
        # 10*max(vscale)*eps.  MATLAB's '*' is Python's '@'.
        def fop(x):
            return jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)
        f = Bndfun.from_function(fop, DOM)
        g = f @ A
        err = np.abs(np.asarray(g(X)) - np.asarray(fop(X)) @ np.asarray(A))
        assert float(np.max(err)) < 10 * float(np.max(np.asarray(g.vscale))) * EPS

    def test_unbndfun_matrix_mtimes(self):
        # pass(13): array-valued Unbndfun on [-Inf, -3*pi] times A,
        # tol 1e2*max(eps*vscale).
        dom = Domain((-INF, -3 * np.pi))
        def op(x):
            return jnp.stack(
                [jnp.exp(x), x * jnp.exp(x), (1 - jnp.exp(x)) / x], axis=-1
            )
        f = Unbndfun.from_function(op, dom)
        g = f @ A
        x = jnp.asarray(np.linspace(-1e6, -3 * np.pi, 100))
        gexact = np.asarray(op(x)) @ np.asarray(A)
        err = float(np.linalg.norm(np.asarray(g(x)) - gexact, ord=np.inf))
        assert err < 1e2 * EPS * float(np.max(np.asarray(g.vscale)))


@pytest.mark.parametrize("unbounded", [False, True])
def test_scalar_jit_ad_and_vmap(unbounded):
    from chebfunjax.tech.chebtech import Chebtech2

    coeffs = jnp.asarray([1., 2., 3.])
    tech = Chebtech2(coeffs=coeffs, ishappy=True)
    f = (Unbndfun(tech, Domain((-jnp.inf, -3*jnp.pi)), "left")
         if unbounded else Bndfun(tech, DOM))
    out = jax.jit(lambda a: a @ f)(2.)
    assert jnp.array_equal(out.onefun.coeffs, 2*coeffs)
    assert out.domain == f.domain
    if unbounded:
        assert out.mapping_type == f.mapping_type
    assert jax.grad(lambda a: jnp.sum((f @ a).onefun.coeffs))(2.) == jnp.sum(coeffs)
    assert jnp.array_equal(jax.vmap(lambda a: (a @ f).onefun.coeffs)(jnp.array([2., 3.])),
                           jnp.array([2., 3.])[:, None]*coeffs)


def test_uniform_seed_against_native_first100():
    fixture = Path(__file__).parents[1] / "test_matlab_port/chebfun/fixtures/logical_source_matlab.json"
    data = json.loads(fixture.read_text())
    captured = np.asarray(data["x"])
    assert np.array_equal(2*np.random.RandomState(6178).rand(100) - 1, captured)


def test_unbounded_dispatch_and_mapping():
    from chebfunjax.tech.chebtech import Chebtech2

    f = Unbndfun(Chebtech2(coeffs=jnp.array([1.]), ishappy=True),
                 Domain((-jnp.inf, -3*jnp.pi)), "left")
    for action, identifier, message in [
        (lambda: jnp.array([[1., 2.]]) @ f, "CHEBFUN:CLASSICFUN:mtimes:size",
         "Inner matrix dimensions must agree."),
        (lambda: f @ jnp.uint8(128), "CHEBFUN:CLASSICFUN:mtimes:classicfunMtimesUnknown",
         "mtimes does not know how to multiply a CLASSICFUN and a uint8."),
        (lambda: f @ Bndfun.from_function(jnp.sin, DOM),
         "CHEBFUN:CLASSICFUN:mtimes:classicfunMtimesClassicfun",
         "Use .* to multiply CLASSICFUN objects."),
    ]:
        with pytest.raises(ValueError) as caught:
            action()
        assert caught.value.identifier == identifier
        assert str(caught.value) == message
    assert (jnp.array(2.) @ f).isequal(f @ 2.)
    assert jnp.size(f @ []) == 0
    assert jnp.size([] @ f) == 0
