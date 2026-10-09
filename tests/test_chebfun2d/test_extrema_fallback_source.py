"""Unrun source seed/transaction controls; no active-set numerical surrogate."""

import warnings
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun2d import _extrema_fallback as source
from chebfunjax.chebfun2d.separable_approx import SeparableApprox
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech2

DOMAIN = (-2., 6., -4., 2.)
X = jnp.asarray([[-2., -4.], [6., 2.]])
Y = jnp.asarray([-42., 62.])


def result(x, value):
    return SimpleNamespace(x=jnp.asarray(x), fun=value, exitflag=0)


def objective(z):
    return z[0] + 10*z[1]


class ArrayValues:
    def __init__(self, values, domain):
        self.values = jnp.asarray(values, dtype=jnp.float64)
        self.domain = Domain(domain)
        self.points = None

    def __len__(self):
        return self.values.shape[0]

    def __call__(self, points):
        self.points = points
        return self.values


def test_seed_column_major_first_and_original_values():
    rows = ArrayValues([[1, 0], [0, 1], [1, 0]], (-2., 6.))
    cols = ArrayValues([[1, 2], [2, 1]], (-4., 2.))
    # A=[[1,2,1],[2,1,2]]: first F-min=(row0,col0), F-max=(row1,col0).
    y, x = source.source_seed(rows, cols, objective)
    assert jnp.array_equal(x, jnp.asarray([[-2., -4.], [-2., 2.]]))
    assert jnp.array_equal(y, jnp.asarray([-42., 18.]))
    assert float(jnp.max(jnp.abs(rows.points - jnp.asarray([-2., 2., 6.])))) <= 4*2.**-52*6
    assert jnp.array_equal(cols.points, jnp.asarray([-4., 2.]))


def test_original_cdr_not_conjugated_or_seed_reconstruction():
    def poly(c):
        return Chebtech2.from_coeffs(jnp.asarray(c, dtype=jnp.float64))
    approx = SeparableApprox(cols=[poly([1]), poly([0, 1])],
                             rows=[poly([0, 1]), poly([1])],
                             pivots=jnp.asarray([2., -3.]), domain=DOMAIN)
    fun = source.original_objective(approx)
    # Physical x=0 -> t=-.5; y=.5 -> s=.5; 2*t-3*s=-2.5.
    assert float(fun(jnp.asarray([0., .5]))) == -2.5
    assert float(fun(jnp.asarray([-2., -4.]))) == 1.


@pytest.mark.parametrize('a,b', [(2.**52+1, 2.**52+6), (-2., 6.)])
def test_literal_map_endpoints_and_no_clipping(a, b):
    assert float(source._inverse(jnp.asarray(a), a, b)) == -1.
    assert float(source._inverse(jnp.asarray(b), a, b)) == 1.
    assert float(source._forward(jnp.asarray(-1.), a, b)) == a
    assert float(source._forward(jnp.asarray(1.), a, b)) == b
    assert jnp.isnan(jnp.arcsin(source._inverse(jnp.asarray(b+8), a, b)))


def test_active_success_ignores_zero_exit_status_and_keeps_exact_options():
    calls = []
    def active(fun, x, lower, upper, options):
        calls.append((x, lower, upper, options, float(fun(jnp.asarray([0., 0.])))))
        return result([0., 1.] if len(calls) == 1 else [2., 0.], -7. if len(calls) == 1 else -9.)
    def forbidden(*args):
        raise AssertionError('Fallback must not be called')
    y, x = source.refine_seed(objective, Y, X, DOMAIN, active_solver=active,
                              fallback_solver=forbidden)
    assert len(calls) == 2
    assert calls[0][3] == source.ActiveOptions()
    assert jnp.array_equal(calls[0][1], jnp.asarray([-2., -4.]))
    assert jnp.array_equal(calls[0][2], jnp.asarray([6., 2.]))
    assert jnp.array_equal(y, jnp.asarray([-7., 9.]))
    assert jnp.array_equal(x, jnp.asarray([[0., 1.], [2., 0.]]))


@pytest.mark.parametrize('active_failure', [1, 2])
@pytest.mark.parametrize('fallback_failure', [1, 2, None])
def test_partial_assignment_sequence(active_failure, fallback_failure):
    counts = [0, 0]
    seed_inputs = []
    def active(fun, x, lower, upper, options):
        counts[0] += 1
        if counts[0] == active_failure:
            raise RuntimeError('injected active failure')
        return result([1., 0.], -7.)
    def fallback(fun, z):
        counts[1] += 1
        seed_inputs.append(z)
        if counts[1] == fallback_failure:
            raise RuntimeError('injected fallback failure')
        return result([0., 0.], -11. if counts[1] == 1 else -13.)
    y, x = source.refine_seed(objective, Y, X, DOMAIN,
                              active_solver=active, fallback_solver=fallback)
    expected_min = (-7. if active_failure == 2 else -42.) if fallback_failure == 1 else -11.
    assert float(y[0]) == expected_min
    assert float(y[1]) == (62. if fallback_failure else 13.)
    assert jnp.array_equal(x, X if fallback_failure else jnp.asarray([[2., -1.], [2., -1.]]))
    assert jnp.array_equal(seed_inputs[0], jnp.arcsin(jnp.asarray([-1., -1.])))
    assert counts == [active_failure, fallback_failure or 2]


def test_absent_active_capability_direct_fallback_and_status_zero():
    values = []
    def fallback(fun, z):
        values.append(float(fun(jnp.zeros(2))))
        return result([0., 0.], values[-1])
    y, x = source.refine_seed(objective, Y, X, DOMAIN, fallback_solver=fallback)
    assert values == [-8., 8.]
    assert jnp.array_equal(y, jnp.asarray([-8., -8.]))
    assert jnp.array_equal(x, jnp.asarray([[2., -1.], [2., -1.]]))


@pytest.mark.parametrize('stage', ['active', 'fallback'])
def test_resource_baseexception_escapes_and_warning_state_restored(stage):
    class ResourceStop(BaseException):
        pass
    def stop(*args):
        raise ResourceStop()
    saved = list(warnings.filters)
    with pytest.raises(ResourceStop):
        source.refine_seed(objective, Y, X, DOMAIN,
                           active_solver=stop if stage == 'active' else None,
                           fallback_solver=stop)
    assert warnings.filters == saved


def test_rank_limit_after_caller_conversion_before_grid():
    # No arrays/4000-square allocations: boundary is rank, not seed grid size.
    class StopGrid:
        def __len__(self):
            raise RuntimeError('grid reached')
    approx = SimpleNamespace(rank=4001)
    with pytest.raises(ValueError, match='Rank is too large'):
        source.source_higher_extrema(approx, StopGrid(), StopGrid())
    approx.rank = 4000
    with pytest.raises(RuntimeError, match='grid reached'):
        source.source_higher_extrema(approx, StopGrid(), StopGrid())


def test_real_fallback_kernel_quadratic_on_box():
    def quadratic(z):
        return z[0]**2 + z[1]**2
    y, x = source.refine_seed(quadratic, jnp.asarray([.125, 2.]),
                              jnp.asarray([[.25, -.25], [1., 1.]]),
                              (-1., 1., -1., 1.))
    # Independent exact extrema 0 at origin, 2 at corners. Unit quadratic
    # coefficients/Hessian2I; source eps stopping remains unchanged.
    assert float(jnp.max(jnp.abs(x[0]))) <= 1e-12
    assert float(jnp.abs(y[0])) <= 2e-24
    assert float(jnp.abs(y[1] - 2)) <= 16*2.**-52
    assert float(jnp.max(jnp.abs(jnp.abs(x[1]) - 1))) <= 16*2.**-52


def test_real_array_chebfun_seed_uses_domain_endpoints():
    rows = Chebfun.from_function(lambda x: jnp.stack((jnp.ones_like(x), x), axis=-1),
                                domain=Domain((-2., 6.)), n=3)
    cols = Chebfun.from_function(lambda y: jnp.stack((y, jnp.ones_like(y)), axis=-1),
                                domain=Domain((-4., 2.)), n=3)
    # Seed matrix is y+x; extrema occur at opposite domain corners.
    # Original objective x+10y differs and must supply the returned values.
    y, x = source.source_seed(rows, cols, objective)
    assert isinstance(rows.domain, Domain)
    assert jnp.array_equal(x, X)
    assert jnp.array_equal(y, jnp.asarray([-42., 26.]))
