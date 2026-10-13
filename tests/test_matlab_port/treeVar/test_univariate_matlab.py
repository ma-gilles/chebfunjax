"""All49 original unary TreeVar predicates, with independent numeric oracles.

Provenance
----------
MATLAB source : tests/treeVar/test_univariate.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
The reference evaluates numeric operations independently of TreeVar dispatch.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import airy

from chebfunjax.operators.treevar import to_first_order

DOM = (0.0, 2.0)
RHS = 3.2
ALP = 2.0
T_ARG = 1.0
U_ARG = [0.4, 0.2]
METHODS = (
    "abs acos acosd acot acoth acsc acscd acsch airy asec asecd asech "
    "asin asind asinh atan atand atanh conj cos cosd cosh cot cotd coth "
    "csc cscd csch exp expm1 imag log log10 log1p log2 pow2 real sec secd "
    "sech sin sind sinh sqrt tan tand tanh uminus uplus"
).split()


def _numeric_reference(name, x):
    """Independent numeric definitions at the source real argument0.4.

    Reciprocal inverse functions leaving the real domain use the principal
    complex branch explicitly; the reference imports no TreeVar evaluator.
    """
    direct = {
        "abs": np.abs, "acos": np.arccos, "asin": np.arcsin,
        "asinh": np.arcsinh, "atan": np.arctan, "atanh": np.arctanh,
        "conj": np.conj, "cos": np.cos, "cosh": np.cosh,
        "exp": np.exp, "expm1": np.expm1, "imag": np.imag,
        "log": np.log, "log10": np.log10, "log1p": np.log1p,
        "log2": np.log2, "pow2": np.exp2, "real": np.real,
        "sin": np.sin, "sinh": np.sinh, "sqrt": np.sqrt,
        "tan": np.tan, "tanh": np.tanh,
    }
    if name in direct:
        return direct[name](x)
    radians = np.pi / 180.0
    if name in ("acosd", "asind", "atand"):
        return direct[name[:-1]](x) / radians
    if name in ("cosd", "sind", "tand"):
        return direct[name[:-1]](radians * x)
    reciprocal = {
        "cot": np.tan, "coth": np.tanh, "csc": np.sin,
        "csch": np.sinh, "sec": np.cos, "sech": np.cosh,
    }
    if name in reciprocal:
        return 1 / reciprocal[name](x)
    if name in ("cotd", "cscd", "secd"):
        return 1 / reciprocal[name[:-1]](radians * x)
    inverse = {
        "acot": np.arctan, "acoth": np.arctanh, "acsc": np.arcsin,
        "acsch": np.arcsinh, "asec": np.arccos, "asech": np.arccosh,
    }
    stem = name[:-1] if name in ("acscd", "asecd") else name
    if stem in inverse:
        if stem in ("acsc", "asec") and abs(1 / x) > 1:
            # MATLAB's real-input inverse-trig branch differs from NumPy's
            # complex positive-zero boundary value (native R2025b capture).
            # Source compares converted method to a numeric primitive using
            # the same backend. Write that RHS independently in JAX; retain
            # MATLAB-vs-backend residuals as a separate native diagnostic.
            height = jnp.arccosh(jnp.abs(jnp.asarray(1 / x)))
            if stem == "asec":
                value = 1j * height if x > 0 else jnp.pi - 1j * height
            else:
                value = jnp.sign(x) * (jnp.pi / 2 - 1j * height)
        else:
            argument = complex(1 / x) if stem == "acoth" else 1 / x
            value = inverse[stem](argument)
        return value / radians if name != stem else value
    if name == "airy":
        return airy(x)[0]
    if name == "uminus":
        return -x
    if name == "uplus":
        return +x
    raise AssertionError(name)


@pytest.mark.parametrize("name", METHODS)
def test_original_unary_method(name):
    def op(u):
        value = -u if name == "uminus" else +u if name == "uplus" else getattr(u, name)()
        return u.diff(2) + u.diff() + ALP * value

    converted, *_ = to_first_order(op, RHS, DOM)
    got = np.ravel(converted(T_ARG, U_ARG))
    expected = np.asarray([
        U_ARG[1], RHS - U_ARG[1] - ALP * _numeric_reference(name, U_ARG[0])])
    assert np.linalg.norm(got - expected) < 10 * np.finfo(float).eps
