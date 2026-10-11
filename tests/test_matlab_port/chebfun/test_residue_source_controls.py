"""Public residue source controls, including repeated-pole reconstruction.

Provenance
----------
MIT Chebfun wrapper: @chebfun/residue.m, commit 7574c77.
Numeric algorithm references: licensed MATLAB R2025b residue.m/mpoles.m/resi2.m.
Manufactured low-degree errors use 100eps; native predicate bounds unchanged.
"""

import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.utils.polyops import residue


def test_interleaved_repeats():
    (b, a) = residue(jnp.array([1.0, 2.0, 3.0]), jnp.array([1.0, 2.0, 1.0]), cj.chebfun())
    x = cj.chebfun("x")
    assert (b - (3 * x * x - 4 * x - 2)).normest() < 100 * jnp.finfo(jnp.float64).eps
    assert (a - (x * x * x - 4 * x * x + 5 * x - 2)).normest() < 100 * jnp.finfo(jnp.float64).eps


def test_array_guard():
    x = cj.chebfun("x")
    u = cj.Chebfun.horzcat([x, x * x])
    try:
        residue(u, x + 2)
    except ValueError as e:
        assert "CHEBFUN:CHEBFUN:residue:quasi" in str(e)
        return
    raise AssertionError("missing native array-valued guard")


def test_multiple_fun_guard():
    x = cj.chebfun("x").restrict([-1.0, 0.0, 1.0])
    try:
        residue(x, cj.chebfun("x") + 2)
    except ValueError as e:
        assert "CHEBFUN:CHEBFUN:residue:multipleFuns" in str(e)
        return
    raise AssertionError("missing native multiple-FUN guard")


def test_repeated_forward():
    from chebfunjax.utils._residue_source import forward

    (r, p, k) = forward(jnp.array([1.0]), jnp.array([1.0, -4.0, 4.0]))
    assert jnp.max(jnp.abs(p - 2)) < 100 * jnp.finfo(jnp.float64).eps
    assert jnp.max(jnp.abs(r - jnp.array([0.0, 1.0]))) < 100 * jnp.finfo(jnp.float64).eps
    assert k.size == 0


def test_complex_inverse():
    point = 2 + 1j
    r = jnp.array([1 + 1j, 2 - 1j])
    p = jnp.array([point, point])
    (b, a) = residue(r, p, cj.chebfun())
    x = cj.chebfun("x")
    assert (b - (r[0] * (x - point) + r[1])).normest() < 100 * jnp.finfo(jnp.float64).eps
    assert (a - (x - point) ** 2).normest() < 100 * jnp.finfo(jnp.float64).eps


def test_polynomial_and_domain():
    x = cj.chebfun("x", domain=[-2.0, 3.0])
    (r, p, k) = residue(x * x + 1, x + 4)
    assert k.domain.a == -2 and k.domain.b == 3
    assert (k - (x - 4)).normest() < 100 * jnp.finfo(jnp.float64).eps
    assert (
        jnp.abs(r[0] - 17) < 100 * jnp.finfo(jnp.float64).eps
        and jnp.abs(p[0] + 4) < 100 * jnp.finfo(jnp.float64).eps
    )
    (b, a) = residue(r, p, k)
    assert b.domain.a == -2 and b.domain.b == 3
    assert (b - (x * x + 1)).normest() < 100 * jnp.finfo(jnp.float64).eps
    assert (a - (x + 4)).normest() < 100 * jnp.finfo(jnp.float64).eps


def test_tiny_complex_part():
    from chebfunjax.utils._residue_source import inverse

    (b, a) = inverse(jnp.array([1 + 1e-30j]), jnp.array([2.0]), jnp.empty(0))
    assert jnp.imag(b[0]) == 1e-30


def test_near_pole_grouping():
    from chebfunjax.utils._residue_source import _mpoles

    (order, powers) = _mpoles(jnp.array([100.0, 100.05, 1.0]), True)
    assert jnp.array_equal(order, jnp.array([1, 0, 2])) and powers == [1, 2, 1]


def test_numeric_matrix_guard():
    for u, v in [(jnp.ones((2, 2)), jnp.ones(4)), (jnp.ones(4), jnp.ones((2, 2)))]:
        try:
            residue(u, v, cj.chebfun())
        except ValueError as e:
            assert "CHEBFUN:CHEBFUN:residue:quasi" in str(e)
        else:
            raise AssertionError("numeric quasi guard missing")


def test_quotient_column_major():
    from chebfunjax.utils._residue_source import inverse

    k = jnp.array([[1.0, 0.0], [2.0, 3.0]])
    (b, a) = inverse(jnp.array([1.0]), jnp.array([2.0]), k)
    expected = jnp.convolve(jnp.ravel(k, order="F"), jnp.array([1.0, -2.0])).at[-1].add(1)
    assert jnp.array_equal(b, expected)
