"""Independent tiny native periodic stack and conversion controls.

Provenance
----------
MATLAB source: @trigcolloc/sum.m, @trigspec/convertOperator.m,
    @trigcolloc/toFunctionOut.m, @trigspec/toFunctionOut.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.operators import _native_values as native
from chebfunjax.operators._periodic_nonlocal import (
    PeriodicDisc,
    assemble,
    convert_operator,
    output_function,
    prepare,
)
from chebfunjax.operators.blocks import D, I, diag, sum_functional
from chebfunjax.operators.chebop import Chebop
from chebfunjax.tech.trigtech import Trigtech
from chebfunjax.utils.diffmat import diffmat
from chebfunjax.utils.quadrature import chebweights


@pytest.mark.parametrize('n', [7, 8])
def test_periodic_complex_stack(n):
    disc = PeriodicDisc(n, (2., 6.))
    x = disc.points
    wave = jnp.exp(.5j*jnp.pi*(x-2))
    block = D((2., 6.), 2)+I((2., 6.))*sum_functional((2., 6.))
    matrix = block._values_capability.realize(disc)
    assert jnp.max(jnp.abs(matrix@wave+.25*jnp.pi**2*wave)) < 2e-13
    assert jnp.max(jnp.abs(matrix@jnp.ones(n)-4)) < 2e-13


def test_complex_multiplication_composition():
    domain = (-1., 1.)
    disc = PeriodicDisc(9, domain)
    from chebfunjax.chebfun1d.chebfun import chebfun
    coefficient = chebfun(lambda x: 2+1j*jnp.sin(jnp.pi*x), trig=True)
    block = diag(coefficient)*D(domain)
    result = block._values_capability.realize(disc)@jnp.cos(jnp.pi*disc.points)
    expected = coefficient(disc.points)*(-jnp.pi*jnp.sin(jnp.pi*disc.points))
    assert jnp.max(jnp.abs(result-expected)) < 2e-13


@pytest.mark.parametrize('n', [7, 8])
def test_full_stack_fourier_conversion(n):
    disc = PeriodicDisc(n, (-1., 1.))
    block = D(order=2)+I()*sum_functional()
    result = convert_operator(block._values_capability.realize(disc))
    modes = jnp.arange(-(n//2), (n+1)//2)
    diagonal = -(jnp.pi*modes)**2
    diagonal = diagonal.at[n//2].set(2.)
    if n % 2 == 0:
        assert jnp.max(jnp.abs(result-jnp.diag(diagonal))) < 2e-12
    else:
        # Native C2V(M.T).T uses V.T, not V. For odd dimensions V != V.T;
        # preserve this native conversion quirk rather than assuming a
        # similarity transform. This is independently formed from the DFT.
        vandermonde = jnp.exp(1j*jnp.pi*disc.points[:, None]*modes[None, :])
        inverse = jnp.conj(vandermonde.T)/n
        values_matrix = block._values_capability.realize(disc)
        expected = inverse@values_matrix@vandermonde.T
        assert jnp.max(jnp.abs(result-expected)) < 2e-12
        assert jnp.max(jnp.abs(result-jnp.diag(diagonal))) > 1


def test_native_pure_imaginary_cleanup():
    # Native convertOperator sets M=imag(M), dropping the imaginary unit.
    result = convert_operator(1j*jnp.eye(8))
    assert jnp.max(jnp.abs(result-jnp.eye(8))) < 2e-13
    assert not jnp.iscomplexobj(result)


def test_native_small_entry_cleanup():
    coefficients = jnp.diag(jnp.asarray([1., 2., 3., 4., 5., 6., 1e-12]))
    values = Trigtech.coeffs2vals(Trigtech.vals2coeffs(coefficients.T).T)
    expected = coefficients.at[-1, -1].set(0.)
    assert jnp.max(jnp.abs(convert_operator(values)-expected)) < 2e-13


@pytest.mark.parametrize('backend', ['trigcolloc', 'trigspec'])
def test_even_output_native_tail_and_tech(backend):
    coefficients = jnp.asarray([2., 0., 0., 3., 0., 0.], dtype=jnp.complex128)
    values = Trigtech.coeffs2vals(coefficients) if backend == 'trigcolloc' else coefficients
    output = output_function(values, (-1., 1.), backend)
    expected = jnp.asarray([1., 0., 0., 3., 0., 0., 1.])
    assert isinstance(output.funs[0].tech, Trigtech)
    assert jnp.max(jnp.abs(output.funs[0].tech.coeffs-expected)) < 2e-13


def test_values_output_repeats_source_roundtrips(monkeypatch):
    original = Trigtech.vals2coeffs
    calls = []
    def recorded(values):
        calls.append(len(values))
        return original(values)
    monkeypatch.setattr(Trigtech, 'vals2coeffs', staticmethod(recorded))
    output_function(jnp.arange(6.), (-1., 1.), 'trigcolloc')
    assert calls == [6, 7, 7, 7, 7, 7]


@pytest.mark.parametrize('backend', ['trigcolloc', 'trigspec'])
def test_odd_cutoff(backend):
    coefficients = jnp.arange(7.)+1j*jnp.arange(7.)
    data = Trigtech.coeffs2vals(coefficients) if backend == 'trigcolloc' else coefficients
    output = output_function(data, (-1., 1.), backend, cutoff=2)
    assert len(output.funs[0].tech.coeffs) == 3
    assert jnp.max(jnp.abs(output.funs[0].tech.coeffs-coefficients[2:5])) < 2e-13


@pytest.mark.parametrize('order', [0, 1, 2])
def test_first_kind_derivative_exact_unchanged(order):
    disc = native.FirstKindDisc((5,), (-2., 3.))
    result = native.derivative((-2., 3.), order).realize(disc)
    assert jnp.array_equal(result, diffmat(5, order, domain=(-2., 3.), kind=1))


def test_first_kind_piecewise_integral_exact_unchanged():
    disc = native.FirstKindDisc((5, 6), (-2., 0., 3.))
    result = native.integral(disc.domain).realize(disc)
    assert jnp.array_equal(result, jnp.concatenate((chebweights(5, kind=1),
                                                   1.5*chebweights(6, kind=1))))


@pytest.mark.parametrize('backend', ['trigcolloc', 'trigspec'])
def test_source_operator_tiny_assembly(backend):
    op = Chebop(lambda u: u.diff(2)+u.sum())
    op.bc = 'periodic'
    data = prepare(op, lambda x: jnp.cos(jnp.pi*x))
    matrix, rhs = assemble(data, 8, backend)
    assert matrix.shape == (8, 8)
    assert rhs.shape == (8,)
    assert data.initial is None


def test_differential_and_nonlinear_route_not_selected():
    differential = Chebop(lambda u: u.diff(2)+u)
    nonlinear = Chebop(lambda u: u.diff(2)+u*u.sum())
    differential.bc = nonlinear.bc = 'periodic'
    assert prepare(differential, 1.) is None
    assert prepare(nonlinear, 1.) is None


@pytest.mark.parametrize('backend', [None, 'coeffs'])
def test_legacy_numeric_coordinate_callback_never_enters_ad(monkeypatch, backend):
    from chebfunjax.operators import _periodic_nonlocal
    def forbidden(*args, **kwargs):
        raise AssertionError('Differential legacy callback entered new AD adapter')
    monkeypatch.setattr(_periodic_nonlocal, 'prepare', forbidden)
    op = Chebop(lambda x, u: u.diff(2)+(2+jnp.cos(jnp.pi*x))*u)
    op.bc = 'periodic'
    def rhs(x):
        return (2-jnp.pi**2+jnp.cos(jnp.pi*x))*jnp.cos(jnp.pi*x)
    result = op.solve(rhs, n=8, discretization=backend)
    from chebfunjax.chebfun1d.chebfun import chebfun
    exact = chebfun(lambda x: jnp.cos(jnp.pi*x), trig=True)
    assert (result-exact).norm(jnp.inf) < 1e-10


def test_nonlinear_sum_marker_retains_legacy_route(monkeypatch):
    calls = []
    def legacy(self, *args, **kwargs):
        calls.append(kwargs)
        return 42
    monkeypatch.setattr(Chebop, '_solve_periodic_nonlinear', legacy)
    op = Chebop(lambda u: u.diff(2)+u*u.sum())
    op.bc = 'periodic'
    assert op.solve(1., n=8) == 42
    assert len(calls) == 1


@pytest.mark.parametrize('backend, requested', [(None, 'trigcolloc'), ('coeffs', 'trigspec')])
def test_sum_marker_selects_requested_backend(monkeypatch, backend, requested):
    from chebfunjax.operators import _periodic_nonlocal
    calls = []
    def selected(data, **kwargs):
        calls.append(kwargs)
        assert data.block._coeff_fn is None
        return 42
    monkeypatch.setattr(_periodic_nonlocal, 'solve', selected)
    op = Chebop(lambda u: u.diff(2)+u.sum())
    op.bc = 'periodic'
    assert op.solve(1., n=8, n_min=12, discretization=backend) == 42
    assert len(calls) == 1
    assert calls[0]['backend'] == requested
    assert calls[0]['n_min'] == 12


def test_nonzero_initial_scale_is_correction_scale(monkeypatch):
    from chebfunjax.operators._periodic_nonlocal import solve
    scales = []
    original = Trigtech.happiness_check
    def check(coeffs, values, **kwargs):
        scales.append(kwargs['vscale'])
        return original(coeffs, values, **kwargs)
    monkeypatch.setattr(Trigtech, 'happiness_check', staticmethod(check))
    op = Chebop(lambda u: u.diff(2)+u.sum())
    op.bc = 'periodic'
    op.init = 100.
    data = prepare(op, 1.)
    scales.clear()  # Exclude periodic RHS/initial constructor happiness calls.
    result = solve(data, backend='trigcolloc', n_min=8, n_max=8)
    # solvebvpLinear calls linsolve(L,rhs,pref) without an initial vscale.
    assert len(scales) == 1
    assert abs(scales[0]-99.5) < 2e-12
    assert (result-.5).norm(jnp.inf) < 2e-12


@pytest.mark.parametrize('factor', [.5, 1., 2., 1e12])
def test_real_projection_strict_continuous_threshold(factor):
    from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
    from chebfunjax.domain import Domain
    from chebfunjax.operators._periodic_nonlocal import real_if_small
    tolerance = 5e-13
    tech = Trigtech.from_coeffs(jnp.asarray([1+1j*factor*tolerance]), is_real=False)
    function = Chebfun(funs=[_Piece(tech=tech, interval=(-1., 1.))],
                      domain=Domain((-1., 1.)))
    result = real_if_small(function, tolerance)
    if factor < 1:
        assert result.isreal()
        assert result.imag().norm(jnp.inf) == 0
    else:
        assert result is function
        assert not result.isreal()
