"""Scalar general-BC dispatch controls and literal Chebfun source predicates.

Provenance
----------
MATLAB source: tests/chebop/test_bc.m, tests/chebop/test_exactInitial.m,
    @chebop/linearize.m and @chebop/solvebvp.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""
import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop


def test_general_bc_public_source_route(monkeypatch):
    op = Chebop(lambda u: u.diff(2) + u.sin(), (-1., 1.))
    op.bc = lambda u: [u.diff()(0.), u.sum()]
    sentinel = chebfun(0.)
    calls = []

    def source(self, *args, **kwargs):
        calls.append(kwargs)
        return sentinel

    def legacy(self, *args, **kwargs):
        raise AssertionError('scalar general BC entered legacy system Newton')

    monkeypatch.setattr(Chebop, '_solve_nonlinear', source)
    monkeypatch.setattr(Chebop, '_solve_nonlinear_system', legacy)
    result = op.solve(0., tol=3e-12, n_min=16, n_max=256)
    assert float(result.norm()) == 0.
    assert len(calls) == 1
    assert calls[0]['tol'] == 3e-12
    assert calls[0]['n_min'] == 16 and calls[0]['n_max'] == 256


def test_scalar_linearity_includes_general_bc():
    # Independent classification control: the equation is linear, BC is not.
    # Explicit nonzero source seed avoids a zero derivative of u(-1)^2 at0.
    op = Chebop(lambda u: u.diff(2), (-1., 1.))
    op.bc = lambda u: [u(-1.)**2 - 1., u(1.) - 1.]
    op.init = chebfun(2.)
    assert not op._is_linear()


def test_linear_general_bc_classification_preserved():
    op = Chebop(lambda u: u.diff(2)+u, (-1., 1.))
    op.bc = lambda x, u: [u.diff()(0.), u.sum()-2.]
    assert op._is_linear()


def test_nonlinear_bc_with_x_classification():
    op = Chebop(lambda u: u.diff(2), (-1., 1.))
    op.bc = lambda x, u: [u(-1.)**2-1., u(1.)-1.]
    op.init = chebfun(2.)
    assert not op._is_linear()


def test_original_bc_nonlinear_9_10(record_property):
    dom = (-1., 1.)
    op = Chebop(lambda x, u: u.diff(2) + u.sin(), dom)
    op.bc = lambda x, u: [u.diff()(0.), u.sum()]
    x = chebfun(lambda x: x, domain=dom)
    rhs = x.sin()
    u, _ = op.solvebvp(rhs)
    errors = [float((op(x, u)-rhs).norm()),
              float(jnp.linalg.norm(jnp.asarray(op.bc(x, u))))]
    for slot, error in enumerate(errors, 9):
        record_property(f'source_bc_{slot}', error)
    assert all(error < 1e-10 for error in errors), errors


def test_nonlinear_general_bc_satisfying_explicit_init_solution():
    op = Chebop(lambda u: u.diff(2), (-1., 1.))
    op.bc = lambda u: [u(-1.)**2 - 1., u(1.) - 1.]
    # Nonconstant explicit guess satisfies BCs and has nonzero ODE residual.
    # The earlier unsatisfied-init2 diagnostic is preserved in controls_v1.
    op.init = chebfun(lambda x: 1+.1*(1-x*x))
    u, info = op.solvebvp(0.)
    assert float((u-1).norm()) < 1e-11
    assert float(jnp.linalg.norm(jnp.asarray(op.bc(u)))) < 1e-11
    assert info['normDelta'] and info['bvpTol'] == 5e-13
