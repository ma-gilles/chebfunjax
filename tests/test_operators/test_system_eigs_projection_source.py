"""Independent system eigs projection/multiplicity/complex recovery controls.

Native source @linop/eigs.m and tests/chebop/test_eigs_system{,2}.m,
7574c77680d7e82b79626300bf255498271a72df. No eigenvalue blacklist.
"""
import math

import jax.numpy as jnp
import pytest

from chebfunjax import chebfun
from chebfunjax.operators import _eigs_source as source
from chebfunjax.operators.chebop import Chebop, SystemSolution

TOL = 1e-9  # Native original Maxwell residual/eigenvalue bound, unchanged.


def maxwell(domain=(0., math.pi)):
    op = Chebop(lambda x, u, v: [-u + v.diff(), u.diff()], domain)
    op.lbc = lambda u, v: u
    op.rbc = lambda u, v: u
    return op


def max_residual(vectors, eigenvalues):
    residual = 0.
    for (u, v), value in zip(vectors, eigenvalues):
        residual = max(residual, float((-u + v.diff() - value*u).norm(jnp.inf)),
                       float((u.diff() - value*v).norm(jnp.inf)),
                       float(jnp.abs(u(0.))), float(jnp.abs(u(math.pi))))
    return residual


@pytest.mark.parametrize("domain", [(0., math.pi), (0., math.pi/2, math.pi)])
def test_complex_modes_satisfy_ode_and_boundary_conditions(domain):
    vectors, values = maxwell(domain).eigs(k=5, n=40, sigma=0)
    expected = jnp.asarray([0., 1., 1., 2., 2.])
    assert float(jnp.max(jnp.abs(jnp.sort(jnp.abs(values))-expected))) < 1e-10
    assert max_residual(vectors, values) < TOL
    points = jnp.linspace(0., math.pi, 23)
    assert any(float(jnp.max(jnp.abs(jnp.imag(v(points))))) > .01
               for (_u, v), value in zip(vectors, values) if abs(value) > .5)
    assert all(isinstance(column, SystemSolution) for column in vectors)
    assert all(tuple(f.domain.breakpoints) == domain for column in vectors for f in column)


def test_genuine_double_zero_and_nonzero_multiplicities_survive():
    op = Chebop(lambda x, u, v: [-u.diff(2), -v.diff(2)], (0., math.pi))
    op.lbc = lambda u, v: [u.diff(), v.diff()]
    op.rbc = lambda u, v: [u.diff(), v.diff()]
    vectors, values = op.eigs(k=6, n=40, sigma=0)
    expected = jnp.asarray([0., 0., 1., 1., 4., 4.])
    assert float(jnp.max(jnp.abs(jnp.sort(jnp.real(values))-expected))) < TOL
    zero_modes = [v for v, value in zip(vectors, values) if abs(value) < TOL]
    assert len(zero_modes) == 2
    # Two genuine zero eigenfunctions must span two component directions.
    columns = jnp.stack([jnp.asarray([u(.37), v(.37)]) for u, v in zero_modes], axis=1)
    assert float(jnp.linalg.svd(columns, compute_uv=False)[-1]) > .1


def test_public_complex_sigma_is_forwarded():
    _vectors, values = maxwell().eigs(k=2, n=40, sigma=1.8j)
    expected = jnp.asarray([-.5+jnp.sqrt(15.)/2*1j, -.5+jnp.sqrt(3.)/2*1j])
    remaining = list(values)
    for target in expected:
        index = min(range(len(remaining)), key=lambda j: float(jnp.abs(remaining[j]-target)))
        assert float(jnp.abs(remaining.pop(index)-target)) < TOL


def test_eigenproblem_does_not_depend_on_or_mutate_solve_initial_guess():
    op = maxwell()
    initial = [chebfun(lambda x: 1+x, domain=(0., math.pi)),
               chebfun(lambda x: 2-x, domain=(0., math.pi))]
    op.init = initial
    _vectors, values = op.eigs(k=3, n=32, sigma=0)
    assert op.init is initial
    assert float(jnp.max(jnp.abs(jnp.sort(jnp.abs(values))-jnp.asarray([0., 1., 1.])))) < TOL


@pytest.mark.parametrize("kind", ["nonlinear_operator", "nonlinear_boundary", "tiny_affine_operator", "tiny_affine_boundary"])
def test_linear_homogeneous_contract_is_guarded_exactly(kind):
    op = maxwell()
    if kind == "nonlinear_operator":
        op.op = lambda x, u, v: [-u + v.diff() + u*u, u.diff()]
    elif kind == "nonlinear_boundary":
        op.lbc = lambda u, v: u*u
    elif kind == "tiny_affine_operator":
        op.op = lambda x, u, v: [-u + v.diff() + 1e-100, u.diff()]
    else:
        op.lbc = lambda u, v: u + 1e-100
    with pytest.raises(ValueError, match="CHEBFUN:CHEBOP:eigs:nonlinear"):
        op.eigs(k=2, n=24, sigma=0)


@pytest.mark.parametrize("mixed", [False, True])
def test_explicit_input_grid_maps_to_actual_projected_equation_dimensions(monkeypatch, mixed):
    seen = []
    original = source.C2Pencil

    class ObservedPencil(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen.append((self.dimensions, self.orders, [disc.n for disc in self.discs],
                         self.constraint_count, self.A.shape, self.B.shape))

    monkeypatch.setattr(source, "C2Pencil", ObservedPencil)
    op = maxwell()
    if mixed:
        op = Chebop(lambda x, u, v: [-u.diff(2) + v, u-v.diff()], (0., math.pi))
        op.lbc = lambda u, v: [u, v]
        op.rbc = lambda u, v: u
    op.eigs(k=2, n=24, sigma=0)
    assert len(seen) == 1
    dims, orders, sizes, constraints, ashape, bshape = seen[0]
    assert orders == ([2, 1] if mixed else [1, 1])
    assert dims == ((22,) if mixed else (23,))
    assert sizes == ([24, 23] if mixed else [24, 24])
    assert constraints == (3 if mixed else 2)
    assert ashape == bshape == ((47, 47) if mixed else (48, 48))


def test_complex_coefficient_matrix_is_preserved():
    shift = .125j
    op = Chebop(lambda x, u, v: [(shift-1)*u + v.diff(), u.diff()+shift*v],
                (0., math.pi))
    op.lbc = lambda u, v: u
    op.rbc = lambda u, v: u
    vectors, values = op.eigs(k=5, n=32, sigma=shift)
    expected = jnp.asarray([shift, shift-.5+jnp.sqrt(3.)/2*1j,
                            shift-.5-jnp.sqrt(3.)/2*1j,
                            shift-.5+jnp.sqrt(15.)/2*1j,
                            shift-.5-jnp.sqrt(15.)/2*1j])
    remaining = list(values)
    for target in expected:
        index = min(range(len(remaining)), key=lambda j: float(jnp.abs(remaining[j]-target)))
        assert float(jnp.abs(remaining.pop(index)-target)) < TOL
    residuals = []
    for column, value in zip(vectors, values):
        residuals.extend((a-u*value).norm(2) for a, u in zip(op(column), column))
    assert float(jnp.sqrt(jnp.sum(jnp.asarray(residuals)**2))) < TOL


@pytest.mark.parametrize("entry", ["method", "wrapper"])
@pytest.mark.parametrize("requested", [None, 28])
def test_explicit_n_default_and_n_precedence_use_actual_input_grid(monkeypatch, entry, requested):
    from chebfunjax.chebfun1d.ode import eigs

    seen = []
    original = source.C2Pencil

    class ObservedPencil(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen.append((self.dimensions, [disc.n for disc in self.discs]))

    monkeypatch.setattr(source, "C2Pencil", ObservedPencil)
    op = maxwell()
    if entry == "method":
        _vectors, values = op.eigs(k=3, n=requested, n_default=24, sigma=0)
    else:
        _vectors, values = eigs(op.op, domain=op.domain, lbc=op._lbc_raw,
                               rbc=op._rbc_raw, k=3, n=requested, n_default=24, sigma=0)
    expected = 24 if requested is None else requested
    assert seen == [((expected-1,), [expected, expected])]
    assert float(jnp.max(jnp.abs(jnp.sort(jnp.abs(values))-jnp.asarray([0., 1., 1.])))) < TOL


def test_wrapper_omitted_grid_uses_native_adaptive_policy(monkeypatch):
    from chebfunjax.chebfun1d.ode import eigs

    seen = []
    original = source.C2Pencil

    class ObservedPencil(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen.append((self.dimensions, [disc.n for disc in self.discs]))

    monkeypatch.setattr(source, "C2Pencil", ObservedPencil)
    op = maxwell()
    eigs(op.op, domain=op.domain, lbc=op._lbc_raw, rbc=op._rbc_raw, k=3, sigma=0)
    # Native minimum equation dimension32; this smooth problem is happy there.
    assert seen == [((32,), [33, 33])]
