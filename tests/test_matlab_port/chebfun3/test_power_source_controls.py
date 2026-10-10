"""Private source dispatch controls and bounded analytic power cases.

Provenance
----------
MATLAB source : @chebfun3/{power,isreal,domainCheck,vscale}.m,
    @chebfun3/private/singleSignTest.m
Chebfun commit: 7574c77
These are added controls, not original native test_power assertions (absent).
"""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d import _power as power
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

DOMAIN = (-1., 1., -1., 1., -1., 1.)


def field(coeffs=(2., 1.), *, complex_core=False, complex_factor=False, domain=DOMAIN):
    factor = Chebtech2.from_coeffs(jnp.asarray(coeffs, dtype=(jnp.complex128
                                                if complex_factor else jnp.float64)))
    one = Chebtech2.from_coeffs(jnp.ones(1))
    core = jnp.ones((1, 1, 1), dtype=jnp.complex128 if complex_core else jnp.float64)
    return Chebfun3([factor], [one], [one], core, domain)


@pytest.mark.parametrize('left,right', [(None, []), ([], object()),
                                      (Chebfun3.empty(), object()),
                                      (object(), Chebfun3.empty())])
def test_empty_precedes_validation(left, right):
    assert power.source_power(left, right).isempty()


@pytest.mark.parametrize('order', [True, [1., 2.], 'bad', jnp.int32(2), jnp.float32(2)])
def test_invalid_order(order):
    with pytest.raises(ValueError, match='power:inputs'):
        power.source_power(field(), order)


@pytest.mark.parametrize('core,factor,expected', [(False, False, True),
                                                (True, False, False),
                                                (False, True, False)])
def test_literal_complex_zero_storage(core, factor, expected):
    assert power._source_isreal(field(complex_core=core, complex_factor=factor)) is expected


@pytest.mark.parametrize('domain,expected', [(DOMAIN, True),
    ((-1., 1.+2e-15, -1., 1., -1., 1.), False),
    ((0., 0., -1., 1., -1., 1.), False)])
def test_strict_domain(domain, expected):
    reference = domain if domain[0:2] == (0., 0.) else DOMAIN
    assert power._domain_check(field(domain=reference), field(domain=domain)) is expected


@pytest.mark.parametrize('values,scale,expected', [([0.], 0., (False, True, False)),
    ([0., 1.], 1., (True, True, True)), ([-1., 0.], 1., (True, True, False)),
    ([-1., 1.], 1., (False, False, False))])
def test_strict_sign_predicate(monkeypatch, values, scale, expected):
    f = SimpleNamespace(sample=lambda: jnp.asarray(values))
    monkeypatch.setattr(power, '_source_vscale', lambda _: scale)
    assert power._single_sign_test(f) == expected


@pytest.mark.parametrize('complex_core', [False, True])
def test_real_storage_controls_fractional_guard(monkeypatch, complex_core):
    f = field((0., 1.), complex_core=complex_core)
    calls = []
    monkeypatch.setattr(power, '_single_sign_test', lambda _: (calls.append('sign') or False, False, False))
    monkeypatch.setattr(power, '_construct', lambda *a, **kw: 'constructed')
    if complex_core:
        assert power.source_power(f, .5) == 'constructed'
        assert calls == []
    else:
        with pytest.raises(ValueError, match='power:fractional'):
            power.source_power(f, .5)
        assert calls == ['sign']


def test_function_exponent_has_no_sign_guard(monkeypatch):
    monkeypatch.setattr(power, '_single_sign_test', lambda _: pytest.fail('unexpected sign guard'))
    monkeypatch.setattr(power, '_construct', lambda *a, **kw: 'constructed')
    assert power.source_power(field((0., 1.)), field((.5,))) == 'constructed'


def test_domain_error_precedes_constructor(monkeypatch):
    monkeypatch.setattr(power, '_construct', lambda *a, **kw: pytest.fail('unexpected construction'))
    with pytest.raises(ValueError, match='power:domain'):
        power.source_power(field(), field(domain=(-2., 2., -1., 1., -1., 1.)))


def test_first_column_tech_selects_all_vscale_grids(monkeypatch):
    f = SimpleNamespace(cols=[Chebtech1.from_coeffs(jnp.ones(1))],
                        domain=DOMAIN, length=lambda: (2, 20, 60))
    counts = []
    def grid(n, a, b, *, kind):
        counts.append((n, a, b, kind))
        return jnp.linspace(a, b, n)
    monkeypatch.setattr(power, 'chebpts_ab', grid)
    monkeypatch.setattr(power, '_evaluate', lambda *a: jnp.asarray([-3., 2.]))
    assert power._source_vscale(f) == 3.
    assert counts == [(9, -1., 1., 1), (20, -1., 1., 1), (41, -1., 1., 1)]


def test_current_preference_forwarded(monkeypatch):
    calls = []
    monkeypatch.setattr(power, 'ChebfunPref', lambda: SimpleNamespace(
        cheb3Prefs=SimpleNamespace(chebfun3eps=7e-13)))
    def build(cls, op, *, domain, tol):
        calls.append((domain, tol))
        return 'result'
    monkeypatch.setattr(Chebfun3, 'from_function', classmethod(build))
    assert power._construct(lambda x, y, z: x, DOMAIN) == 'result'
    assert calls == [(DOMAIN, 7e-13)]


@pytest.mark.parametrize('base,order,complex_result', [([-2., 2.], [2., .5], False),
                                                     ([-2., 2.], [.5, 2.], True)])
def test_paired_value_promotion(base, order, complex_result):
    result = power._value_power(jnp.asarray(base), jnp.asarray(order))
    assert jnp.iscomplexobj(result) is complex_result
    reference = jnp.asarray([4., jnp.sqrt(2.)]) if not complex_result else jnp.asarray([1j*jnp.sqrt(2.), 4.])
    assert jnp.max(jnp.abs(result-reference)) < 2e-15


@pytest.mark.parametrize('case', ['integer', 'zero', 'negative_integer', 'positive_fraction',
    'negative_fraction', 'complex_base', 'scalar_base', 'negative_scalar_varying',
    'complex_scalar', 'function_exponent', 'complex_exponent'])
def test_actual_analytic_power(case):
    f, exponent = field(), field((1.5, 1.))
    x = jnp.linspace(-1., 1., 19)
    base = 2.+x
    if case == 'integer':
        result, expected = f**3, base**3
    elif case == 'zero':
        result, expected = f**0, jnp.ones_like(x)
    elif case == 'negative_integer':
        result, expected = f**-1, 1/base
    elif case == 'positive_fraction':
        result, expected = f**.5, jnp.sqrt(base)
    elif case == 'negative_fraction':
        result, expected = field((-2., -1.))**.5, 1j*jnp.sqrt(base)
    elif case == 'complex_base':
        result = field((2.+1j, 1.), complex_factor=True)**2
        expected = (base+1j)**2
    elif case == 'scalar_base':
        result, expected = 2**exponent, 2**(1.5+x)
    elif case == 'negative_scalar_varying':
        # At the inherited constructor's x=.5 probe exponent is integer 2;
        # other points need the principal complex branch.
        result = (-2)**exponent
        expected = jnp.exp((1.5+x)*(jnp.log(2.)+1j*jnp.pi))
    elif case == 'complex_scalar':
        result, expected = (1+1j)**exponent, jnp.exp((1.5+x)*jnp.log(1+1j))
    elif case == 'function_exponent':
        result, expected = f**exponent, base**(1.5+x)
    else:
        exponent = field((1.+1j,), complex_factor=True)
        result, expected = f**exponent, jnp.exp((1.+1j)*jnp.log(base))
    assert jnp.max(jnp.abs(result(x, x, x)-expected)) < 1e-11
    assert result.domain == DOMAIN


@pytest.mark.parametrize('coeffs', [(0.,), (0., 1.)])
def test_actual_fractional_rejections(coeffs):
    with pytest.raises(ValueError, match='power:fractional'):
        field(coeffs)**.5


@pytest.mark.parametrize('imag_zero,preserve,expected', [(True, False, 'real'),
                                                       (True, True, 'complex'),
                                                       (False, False, 'complex')])
def test_component_constructor_representation_adapter(monkeypatch, imag_zero, preserve, expected):
    calls = []
    class Component:
        def iszero(self):
            return imag_zero
        def __mul__(self, scalar):
            assert scalar == 1j
            return self
        def __add__(self, other):
            return 'complex'
    re = Component()
    def build(cls, op, *, domain, tol):
        calls.append((op(jnp.asarray([0.]), 0., 0.), domain, tol))
        return re if len(calls) == 1 else Component()
    from chebfunjax.chebfun3d import _plus
    monkeypatch.setattr(_plus, '_assemble_components', lambda a, b: 'complex')
    monkeypatch.setattr(Chebfun3, 'from_function', classmethod(build))
    result = power._construct(lambda x, y, z: x+2j, DOMAIN,
                              complex_possible=True, preserve_complex=preserve)
    assert result is re if expected == 'real' else result == expected
    assert len(calls) == 2
    assert calls[0][0][0] == 0. and calls[1][0][0] == 2.
    assert all(call[1] == DOMAIN for call in calls)
    assert all(call[2] == power.ChebfunPref().cheb3Prefs.chebfun3eps for call in calls)


@pytest.mark.parametrize('axes', [(0, 1, 2), (2, 0, 1), (1, 2, 0), (1, 0, 2), None])
def test_unequal_rank_grid_and_vector_evaluation(axes):
    def polynomial(degree):
        return Chebtech2.from_coeffs(jnp.eye(3)[degree])
    f = Chebfun3([polynomial(0), polynomial(1)], [polynomial(0)],
                 [polynomial(0), polynomial(1), polynomial(2)],
                 jnp.arange(1., 7.).reshape(2, 1, 3), DOMAIN)
    if axes is None:
        x = jnp.linspace(-.8, .8, 7)
        y, z = x/2, -x
    else:
        vectors = (jnp.linspace(-.8, .8, 3), jnp.linspace(-.6, .6, 4),
                   jnp.linspace(-.4, .4, 5))
        ordered = [None]*3
        for coordinate, axis in enumerate(axes):
            ordered[axis] = vectors[coordinate]
        grid = jnp.meshgrid(*ordered, indexing='ij')
        x, y, z = (grid[axis] for axis in axes)
    expected = 1+2*z+3*(2*z*z-1)+x*(4+5*z+6*(2*z*z-1))
    assert jnp.max(jnp.abs(power._evaluate(f, x, y, z)-expected)) < 8e-15
