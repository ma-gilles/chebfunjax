"""Native C1 corrections for one function and numeric scalar parameters.

Source: Chebfun7574c77 @chebcolloc1/{functionPoints,toValues,toFunctionOut},
@chebcolloc/{diff,feval,reduce}, @valsDiscretization/{instantiate,rhs,mldivide},
@opDiscretization/{extractBlock,partition}, and @linop/linsolve.m.
"""
import jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.operators._native_values import FirstKindDisc
from chebfunjax.tech.chebtech import Chebtech1
from chebfunjax.utils.diffmat import _cheb1_barywts
from chebfunjax.utils.interpolation import barymat
from chebfunjax.utils.quadrature import chebpts


def _realize(block, disc):
    capability = getattr(block, '_values_capability', None)
    if capability is None:
        raise NotImplementedError('C1 parameter correction requires a native first-kind operator capability')
    return capability.realize(disc)


def to_function(values, domain, cutoff=None):
    """C1 toFunctionOut: coefficients, cutoff, values, kind1 constructor."""
    coefficients = Chebtech1.vals2coeffs(values)
    if cutoff is not None:
        coefficients = coefficients[:int(cutoff)]
    tech = Chebtech1.from_values(Chebtech1.coeffs2vals(coefficients))
    return Chebfun(funs=[_Piece(tech=tech, interval=tuple(domain))],
                   domain=Domain(tuple(domain)))


class FirstKindParameterCorrection:
    """One frozen C1 operator, including scalar columns and leading BCs."""

    backend = 'chebcolloc1'

    def __init__(self, differential, conditions, size, order, domain):
        self.size, self.order = int(size), int(order)
        self.domain = tuple(domain)
        self.unknown_size = self.size+self.order
        self.function_disc = FirstKindDisc((self.unknown_size,), self.domain)
        self.equation_disc = FirstKindDisc((self.size,), self.domain)
        self.points = self.equation_disc.points
        self.projection = barymat(
            chebpts(self.size, kind=1), chebpts(self.unknown_size, kind=1),
            _cheb1_barywts(self.unknown_size), Chebtech1.angles(self.size),
            Chebtech1.angles(self.unknown_size), True)
        blocks = differential.jacobian.blocks[0]
        self.parameter_count = len(blocks)-1
        # extractBlock applies the function's dimAdjust=order. Scalar
        # Chebfun columns have dimAdjust=0 and are evaluated on their own
        # n-point grid; no interpolation from the function's n+order grid.
        function = self.projection @ _realize(blocks[0], self.function_disc)
        parameters = [jnp.asarray(block(self.points) if callable(block) else
                                  jnp.full(self.size, block)).reshape(self.size, 1)
                      for block in blocks[1:]]
        equation = jnp.concatenate([function, *parameters], axis=1)
        boundary = []
        for condition in conditions:
            if isinstance(condition, ADChebfun):
                row = condition.jacobian.blocks[0]
                function = jnp.asarray(_realize(row[0], self.function_disc)).reshape(1, self.unknown_size)
                scalars = [jnp.asarray(block).reshape(1, 1) for block in row[1:]]
                boundary.append(jnp.concatenate([function, *scalars], axis=1))
            else:
                boundary.append(jnp.zeros((1, self.unknown_size+self.parameter_count)))
        self.matrix = jnp.concatenate([*boundary, equation], axis=0)
        if self.matrix.shape != (self.unknown_size+self.parameter_count,)*2:
            raise ValueError('C1 mixed parameter correction is not square')
        self.scaling = None
        self.factors = None

    @property
    def is_factored(self):
        # Native valsDiscretization/isFactored uses the OUTPUT dimension,
        # including one entry per scalar. Rectangular differential matrices
        # have extra function columns and therefore fail this exact check.
        return (self.factors is not None and self.factors[0].shape[0]
                == self.size+self.parameter_count)

    def solve(self, residual, conditions):
        """Native values RHS and frozen scaled LU; returns raw unknowns."""
        if not self.is_factored:
            self.scaling = 1/jnp.maximum(1, jnp.max(jnp.abs(self.matrix), axis=1))
            self.factors = lu_factor(self.scaling[:, None]*self.matrix)
        rhs = residual(self.points) if callable(residual) else jnp.full(self.size, residual)
        boundary = [c.func if isinstance(c, ADChebfun) else c for c in conditions]
        rhs = jnp.concatenate([jnp.asarray(boundary).reshape(-1), jnp.asarray(rhs)])
        return -lu_solve(self.factors, self.scaling*rhs)

    def project(self, vector):
        """Native P*v acts on function components; scalar identity retained."""
        return self.projection @ vector[:self.unknown_size], vector[self.unknown_size:]
