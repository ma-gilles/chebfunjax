"""Operator infrastructure for spectral discretization of ODEs.

Provides:
- ``OperatorBlock`` — linear map function -> function (e.g. differentiation)
- ``FunctionalBlock`` — linear map function -> scalar (e.g. evaluation, integral)
- ``ChebMatrix`` — 2-D block matrix assembling an ODE system
- ``ChebColloc2Disc`` — discretization descriptor (n points, domain)
- Factory functions: ``D``, ``I``, ``diag``, ``eval_at``, ``sum_functional``
"""

from chebfunjax.operators.blocks import (
    ChebColloc2Disc,
    D,
    FunctionalBlock,
    I,
    OperatorBlock,
    diag,
    eval_at,
    sum_functional,
)
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.operators.chebop import Chebop
from chebfunjax.operators.linop import Linop

__all__ = [
    "ChebColloc2Disc",
    "ChebMatrix",
    "Chebop",
    "D",
    "FunctionalBlock",
    "I",
    "Linop",
    "OperatorBlock",
    "diag",
    "eval_at",
    "sum_functional",
]
