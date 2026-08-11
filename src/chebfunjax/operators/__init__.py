"""Operator infrastructure for spectral discretization of ODEs and PDEs.

Provides:
- ``OperatorBlock`` — linear map function -> function (e.g. differentiation)
- ``FunctionalBlock`` — linear map function -> scalar (e.g. evaluation, integral)
- ``ChebMatrix`` — 2-D block matrix assembling an ODE system
- ``ChebColloc2Disc`` — discretization descriptor (n points, domain)
- Factory functions: ``D``, ``I``, ``diag``, ``eval_at``, ``sum_functional``
- ``Linop`` — linear ODE/BVP operator with BCs (scalar)
- ``BlockLinop`` — block linear operator with side and continuity conditions
- ``Chebop`` — user-friendly nonlinear ODE/BVP operator
- ``Chebop2`` — 2D PDE operator on rectangles (Poisson, Helmholtz, etc.)
"""

from chebfunjax.operators.blocklinop import BlockLinop, addbc, linop
from chebfunjax.operators.blocks import (
    ChebColloc2Disc,
    D,
    FunctionalBlock,
    I,
    OperatorBlock,
    cumsum_op,
    diag,
    eval_at,
    fred_op,
    inner_functional,
    jump_at,
    jump_functional,
    merge_domains,
    mult,
    primitive_functionals,
    primitive_operators,
    sum_functional,
    to_coeff,
    to_function,
    volt_op,
    zero_functional,
    zeros_op,
)
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.operators.chebop import Chebop, deflate
from chebfunjax.operators.chebop2 import Chebop2
from chebfunjax.operators.linop import Linop

__all__ = [
    "BlockLinop",
    "ChebColloc2Disc",
    "ChebMatrix",
    "Chebop",
    "Chebop2",
    "deflate",
    "D",
    "FunctionalBlock",
    "I",
    "Linop",
    "OperatorBlock",
    "addbc",
    "cumsum_op",
    "diag",
    "eval_at",
    "fred_op",
    "inner_functional",
    "jump_at",
    "jump_functional",
    "linop",
    "merge_domains",
    "mult",
    "primitive_functionals",
    "primitive_operators",
    "sum_functional",
    "to_coeff",
    "to_function",
    "volt_op",
    "zero_functional",
    "zeros_op",
]
