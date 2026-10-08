"""Complex scalar-column and public mixed Newton regression controls.

Provenance
----------
MATLAB source: @chebop/linearize.m, @linop/linsolve.m, @chebmatrix/matrix.m.
Chebfun commit: 7574c77
"""
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.operators.chebop import Chebop


def test_complex_scalar_column_keeps_imaginary_derivative():
    parameter = ADChebfun(cj.chebfun(1.)).seed(2, (True, False))
    matrix = (1j * parameter).jacobian.matrix(8)[0]
    assert float(jnp.max(jnp.abs(matrix[:, -1] - 1j))) < 1e-14


def test_public_parameter_newton_keeps_complex_equation():
    x = cj.chebfun(lambda t: t)
    operator = Chebop(lambda x, u, a: u.diff(2) + u**2 - (x + 2)**2 + a - 1j)
    operator.lbc = lambda u, a: [u - 1, u.diff() - 1]
    operator.rbc = lambda u, a: u - 3
    operator.init = [x + 2, 0.]
    u, parameter = operator.solve(0., n=8)
    # Exact manufactured solution u=x+2, parameter=i. Discarding the imaginary
    # equation incorrectly leaves parameter=0 despite the nonzero residual.
    assert isinstance(parameter, complex)
    assert abs(parameter - 1j) < 1e-11
    assert float(operator.op(x, u, parameter).norm()) < 1e-10
    assert float((u - x - 2).norm()) < 1e-11


def test_scalar_seed_classification_includes_endpoint_linearity():
    x = cj.chebfun(lambda t: t)
    for nonlinear_op, nonlinear_bc in [(False, False), (True, False), (False, True)]:
        operator = Chebop(lambda x, u, a: u.diff(2) + (u**2 if nonlinear_op else u) + a)
        operator.init = [x + 2, 1.]
        operator.lbc = lambda u, a: u**2 - 1 if nonlinear_bc else u - 1
        operator.rbc = lambda u, a: [u - 3, u.diff() - 1]
        assert operator._system_is_linear() is not (nonlinear_op or nonlinear_bc)
