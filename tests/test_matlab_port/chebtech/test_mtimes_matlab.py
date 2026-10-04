"""Port of MATLAB Chebfun tests/chebtech/test_mtimes.m (Fable 5).

All12 MATLAB assertion groups are ported for both tech kinds. Python ``@``
spells MATLAB MTIMES; Python ``*`` retains pointwise TIMES, including numeric
row-vector column scaling. Error checks preserve both the source identifier
and the exact source message. Numeric-left and Tech-left empty arguments are
checked before shape/type dispatch.

MATLAB uses ``alpha = randn() + 1i*randn()``; a fixed complex scalar is
used here so the test is deterministic.

Provenance
----------
MATLAB source : tests/chebtech/test_mtimes.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-1.0, 1.0, 100))
ALPHA = 0.3 + 0.7j  # fixed complex scalar (MATLAB uses randn()+1i*randn())


@pytest.fixture(params=[Chebtech1, Chebtech2], ids=["chebtech1", "chebtech2"])
def cls(request):
    return request.param


class TestChebtechMtimes:
    def test_empty_cases(self, cls):
        # pass(n,1): isempty(f*[]) && isempty([]*f) && isempty(2*g) && isempty(g*2)
        f = cls.from_function(jnp.sin)
        e = cls.empty()
        assert (f @ []).isempty()
        assert ([] @ f).isempty()
        assert (2.0 @ e).isempty()
        assert (e @ 2.0).isempty()

    def test_scalar_left_equals_right(self, cls):
        # pass(n,2): isequal(alpha*f, f*alpha)
        f = cls.from_function(jnp.sin)
        assert (ALPHA @ f).isequal(f @ ALPHA)

    def test_scalar_multiplication_values(self, cls):
        # pass(n,3): norm(feval(alpha*f, x) - alpha*sin(x), inf)
        #            < 10*vscale(g1)*eps
        f = cls.from_function(jnp.sin)
        g1 = ALPHA @ f
        err = jnp.abs(g1(X) - ALPHA * jnp.sin(X))
        assert float(jnp.max(err)) < 10 * g1.vscale * EPS

    def test_zero_scalar_gives_zero(self, cls):
        # pass(n,4): all((0*f).coeffs == 0)
        f = cls.from_function(jnp.sin)
        g = 0 @ f
        assert bool(jnp.all(g.coeffs == 0))

    def test_array_valued_scalar_cases(self, cls):
        # pass(n,5)-(7): scalar * array-valued tech [sin cos exp].
        f = cls.from_function(
            lambda x: jnp.stack(
                [jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1))
        g1 = ALPHA @ f
        assert g1.isequal(f @ ALPHA)                       # pass(n,5)
        exact = ALPHA * jnp.stack(
            [jnp.sin(X), jnp.cos(X), jnp.exp(X)], axis=-1)
        assert float(jnp.max(jnp.abs(g1(X) - exact))) \
            < 10 * g1.vscale * EPS                          # pass(n,6)
        assert bool(jnp.all((0 @ f).coeffs == 0))           # pass(n,7)

    def test_array_valued_matrix_mtimes(self, cls):
        # pass(n,8): f*A mixes the columns (MATLAB mtimes -> @).
        rng = np.random.default_rng(6178)
        A = jnp.asarray(rng.standard_normal((3, 3)))
        f = cls.from_function(
            lambda x: jnp.stack(
                [jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1))
        g = f @ A
        exact = jnp.stack(
            [jnp.sin(X), jnp.cos(X), jnp.exp(X)], axis=-1) @ A
        assert float(jnp.max(jnp.abs(g(X) - exact))) \
            < 10 * g.vscale * EPS

    def test_nonscalar_double_on_the_left(self, cls):
        # pass(n,9): MTIMES maps to @, not pointwise *.
        f = cls.from_function(jnp.exp)
        with pytest.raises(ValueError) as exc:
            [1, 2, 3] @ f
        assert exc.value.identifier == "CHEBFUN:CHEBTECH:mtimes:size"
        assert str(exc.value) == "Inner matrix dimensions must agree."

    def test_matrix_dimension_mismatch(self, cls):
        # pass(n,10): f*[1;2;3] for a 2-column f raises
        # CHEBFUN:CHEBTECH:mtimes:size2 ('Inner matrix dimensions must
        # agree.'); chebfunjax f @ A raises on the same mismatch.
        f = cls.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        assert f.coeffs.shape[1] == 2
        with pytest.raises(ValueError) as exc:
            f @ jnp.asarray([[1.0], [2.0], [3.0]])
        assert exc.value.identifier == "CHEBFUN:CHEBTECH:mtimes:size2"
        assert str(exc.value) == "Inner matrix dimensions must agree."

    def test_tech_matrix_product_is_rejected(self, cls):
        # pass(n,11): MATLAB matrix *, represented here by @.
        f = cls.from_function(jnp.sin)
        g = cls.from_function(lambda x: x)
        with pytest.raises(ValueError) as exc:
            f @ g
        assert exc.value.identifier == "CHEBFUN:CHEBTECH:mtimes:chebtechMtimesChebtech"
        assert str(exc.value) == "Use .* to multiply CHEBTECH objects."

    def test_uint8_matrix_multiplier_is_rejected(self, cls):
        # pass(n,12): keep the explicit source numeric class.
        f = cls.from_function(jnp.sin)
        with pytest.raises(ValueError) as exc:
            f @ jnp.uint8(128)
        assert exc.value.identifier == "CHEBFUN:CHEBTECH:mtimes:chebtechMtimesUnknown"
        assert str(exc.value) == "mtimes does not know how to multiply a CHEBTECH and a uint8."
