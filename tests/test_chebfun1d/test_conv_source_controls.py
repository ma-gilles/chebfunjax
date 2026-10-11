"""Additional convolution and moved-breakpoint contracts.

Provenance
----------
@chebfun/{conv,plus,overlap,tweakDomain}.m, @bndfun/conv.m.
Chebfun commit: 7574c77. Quadrature controller: MATLAB R2025b integralCalc.m.
These are supplemental analytic/API checks, not additional native clauses.
"""

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun1d.linalg import Quasimatrix

EPS = float(jnp.finfo(jnp.float64).eps)


@pytest.mark.parametrize("old", [False, True])
@pytest.mark.parametrize("row", [False, True])
def test_singleton_column_storage(old, row):
    f = cj.chebfun(jnp.ones((3, 1)))
    assert f.funs[0].tech.coeffs.ndim == 2
    if row:
        f = f.T
    out = f.conv(f, *(("old",) if old else ()))
    x = jnp.array([-1.0, 0.0, 1.0])
    assert jnp.max(jnp.abs(out(x).reshape(-1) - (2 - jnp.abs(x)))) < 10 * EPS
    assert out.is_transposed == row


def test_row_array_convolution():
    f = Chebfun.horzcat(cj.chebfun(1.0), cj.chebfun(2.0)).T
    out = f.conv(f)
    x = jnp.array([-1.0, 0.0, 1.0])
    expected = jnp.stack((2 - jnp.abs(x), 4 * (2 - jnp.abs(x))))
    assert out.is_transposed
    assert jnp.max(jnp.abs(out(x) - expected)) < 40 * EPS


def test_single_column_quasimatrix_convolution():
    col = cj.chebfun(1.0)
    f = Quasimatrix([col], col.domain)
    out = f.conv(f)
    assert abs(float(out(0.0)) - 2) < 10 * EPS


@pytest.mark.parametrize("row", [False, True])
@pytest.mark.parametrize("columns", [1, 2])
def test_plus_uses_aligned_stored_values(row, columns):
    point = 0.125
    moved = float(jnp.nextafter(jnp.float64(point), jnp.inf))
    f = cj.chebfun([1.0, 3.0], domain=(-1.0, point, 1.0))
    g = cj.chebfun([2.0, 4.0], domain=(-1.0, moved, 1.0))
    f = f.set_point_values(jnp.array([1.0, 7.0, 3.0]))
    g = g.set_point_values(jnp.array([2.0, 11.0, 4.0]))
    if columns == 2:
        g = Chebfun.horzcat(g, 2 * g)
    if row:
        f, g = f.T, g.T
    out = f + g
    expected = jnp.array(18.0) if columns == 1 else jnp.array([18.0, 29.0])
    assert jnp.array_equal(out.point_values[1], expected)
    assert out.is_transposed == row


@pytest.mark.parametrize("old", [False, True])
def test_complex_scaling_is_retained(old):
    f, g = cj.chebfun(lambda x: 1 + x), cj.chebfun(lambda x: 1 - x)
    options = ("old",) if old else ()
    real = f.conv(g, *options)
    actual = ((1 + 2j) * f).conv(g, *options)
    assert jnp.issubdtype(actual.funs[0].tech.coeffs.dtype, jnp.complexfloating)
    assert float((actual - (1 + 2j) * real).norm(jnp.inf)) < 100 * EPS * float(real.vscale)


def test_flags_empty_dimensions_and_orientation():
    f, g = cj.chebfun(lambda x: x), cj.chebfun(1.0)
    assert f.conv(cj.chebfun()).isempty()
    assert cj.chebfun().conv(g).isempty()
    assert float((f.conv(g, "FULL") - f.conv(g)).norm()) == 0
    assert f.T.conv(g.T).is_transposed
    assert f.conv(g, "SaMe").domain.breakpoints[0] == -1.0
    assert f.conv(g, "SaMe").domain.breakpoints[-1] == 1.0
    with pytest.raises(ValueError, match="conv:validFlag"):
        f.conv(g, "valid")
    with pytest.raises(ValueError, match="conv:badInput"):
        f.conv(g, "unrecognized")
    with pytest.raises(ValueError, match="conv:transposed"):
        f.conv(g.T)
    with pytest.raises(ValueError, match="conv:dimagree"):
        Chebfun.horzcat(f, f).conv(Chebfun.horzcat(g, g, g))


def test_explicit_quasimatrix_dispatch():
    f, g = cj.chebfun(lambda x: x), cj.chebfun(1.0)
    q = Quasimatrix([f, g], f.domain)
    result = q.conv(g)
    expected = Chebfun.horzcat(f.conv(g), g.conv(g))
    assert float((result - expected).norm()) < 10 * EPS


def test_independent_quadrature_actually_refines(monkeypatch):
    from chebfunjax.chebfun1d import _convolution_quadrature as q

    f, g = cj.chebpoly(64), cj.chebfun(1.0)
    panels = []
    original = q._panel_product

    def observed(*args):
        panels.append(args[-1].shape[1])
        return original(*args)

    monkeypatch.setattr(q, "_panel_product", observed)
    integral = q._integral_product(f, g, 0.0, -1.0, 1.0)
    assert len(panels) > 1
    assert panels[0] == 10
    assert abs(float(integral) - 2 / (1 - 64**2)) < 100 * EPS


def test_native_delta_prime_step_limitation():
    # Native cvd1: both orders return zero/no delta. Direct diff(step) does
    # contain the jump. Do not silently replace this native limitation with
    # the mathematical distributional convolution. Interior piece counts
    # differ between native operand orders and are not asserted here.
    x = cj.chebfun("x")
    step = cj.chebfun([0.0, 1.0], domain=(-1.0, 0.0, 1.0))
    prime = x.dirac().diff()
    for result in (prime.conv(step), step.conv(prime)):
        assert not result.deltas
        assert float(result.norm(jnp.inf)) == 0.0
        assert (result.domain.a, result.domain.b) == (-2.0, 2.0)
    derivative = step.diff()
    assert derivative.deltas
    assert sum(row[1] for row in derivative.deltas) == 1.0
    assert all(row[0] == 0.0 for row in derivative.deltas)
