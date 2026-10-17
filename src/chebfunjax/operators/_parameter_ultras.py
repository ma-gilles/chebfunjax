"""Typed ultraspherical corrections for one function and scalar parameters.

Provenance
----------
MATLAB source: @coeffsDiscretization/{instantiate,extractBlock,getOutputSpace},
@ultraS/{quasi2diffmat,toValues,rhs,reduce,toFunctionOut},
@opDiscretization/{getConstraints,opDiscretization,partition} and @linop/linsolve.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""
import jax.numpy as jnp

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.discretization.ultras import convertmat, diffmat, multmat
from chebfunjax.domain import Domain
from chebfunjax.operators._linear_altdisc import _coefficients, _prolong
from chebfunjax.operators.blocks import ChebColloc2Disc
from chebfunjax.tech.chebtech import Chebtech2


def to_function(coefficients, domain, cutoff=None):
    """Native ultraS toFunctionOut constructs C2 directly from coefficients."""
    if cutoff is not None:
        coefficients = coefficients[:int(cutoff)]
    return Chebfun.from_coeffs(coefficients, Domain(tuple(domain)))


class UltrasParameterCorrection:
    """One frozen coefficient-space operator with genuine scalar columns."""

    backend = 'ultraS'
    is_factored = False  # Native opDiscretization default: never factored.

    def __init__(self, differential, conditions, size, order, domain):
        self.size, self.order = int(size), int(order)
        self.domain = tuple(domain)
        self.unknown_size = self.size+self.order
        self.output_space = self.order-1  # Inclusive convertmat upper index.
        blocks = differential.jacobian.blocks[0]
        self.parameter_count = len(blocks)-1
        field = blocks[0]
        if getattr(field, '_coeff_fn', None) is None:
            raise TypeError('COEFFSDISCRETIZATION:instantiate:fail -- Cannot represent this operator. Suggest you use VALSDISCRETIZATION.')
        matrix = jnp.zeros((self.unknown_size, self.unknown_size))
        # Native quasi2diffmat reverses the descending coefficient list.
        for derivative_order, coefficient in enumerate(reversed(field.coeff_list())):
            coefficients = _coefficients(coefficient, self.domain)
            derivative = ((2/(self.domain[-1]-self.domain[0]))**derivative_order
                          *diffmat(self.unknown_size, derivative_order))
            matrix = matrix + (
                convertmat(self.unknown_size, derivative_order, self.output_space)
                @multmat(self.unknown_size, coefficients, derivative_order)
                @derivative)
        conversion = convertmat(self.size, 0, self.output_space)
        # Scalar dimAdjust=0: truncate BEFORE conversion at n, not n+order.
        parameters = [(conversion@_prolong(_coefficients(block, self.domain), self.size)).reshape(self.size, 1)
                      for block in blocks[1:]]
        equation = jnp.concatenate([matrix[:self.size, :], *parameters], axis=1)
        boundary = []
        values_disc = ChebColloc2Disc(self.unknown_size, self.domain)
        for condition in conditions:
            if isinstance(condition, ADChebfun):
                row = condition.jacobian.blocks[0]
                values_row = jnp.asarray(row[0].matrix(values_disc)).reshape(1, self.unknown_size)
                # Literal coeffsDiscretization.convertFunctional schedule.
                # The source uses C2 values only to represent this functional.
                function = jnp.rot90(Chebtech2.coeffs2vals(jnp.rot90(values_row)), -1)
                scalars = [jnp.asarray(block).reshape(1, 1) for block in row[1:]]
                boundary.append(jnp.concatenate([function, *scalars], axis=1))
            else:
                boundary.append(jnp.zeros((1, self.unknown_size+self.parameter_count)))
        self.matrix = jnp.concatenate([*boundary, equation], axis=0)
        if self.matrix.shape != (self.unknown_size+self.parameter_count,)*2:
            raise ValueError('Ultraspherical mixed parameter correction is not square')
        self.conversion = conversion

    def solve(self, residual, conditions):
        """Native coefficient RHS and plain unscaled solve for every RHS."""
        equation = self.conversion@_prolong(_coefficients(residual, self.domain), self.size)
        boundary = [c.func if isinstance(c, ADChebfun) else c for c in conditions]
        rhs = jnp.concatenate([jnp.asarray(boundary).reshape(-1), equation])
        return -jnp.linalg.solve(self.matrix, rhs)

    def project(self, vector):
        """Native coefficient truncation and identity on numeric parameters."""
        return vector[:self.size], vector[self.unknown_size:]
