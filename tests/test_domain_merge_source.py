"""Native tolerance loop and retained delay-equation domain regression.

Provenance
----------
MATLAB source: @domain/merge.m; @linop/linsolve.m83-86.
Chebfun commit: 7574c77
"""

import math
from types import SimpleNamespace

import pytest

from chebfunjax.domain import merge
from chebfunjax.operators.blocklinop import BlockLinop
from chebfunjax.operators.blocks import I


def test_retained_near_breaks_do_not_make_microintervals():
    data = {
        "domain": [
            0.0,
            0.3333333333333333,
            0.5,
            0.6666666666666666,
            1.0,
            1.3333333333333333,
            1.5,
            1.6666666666666665,
            2.0,
        ],
        "rhs_domains": [
            [
                0.0,
                0.3333333333333333,
                0.5,
                0.6666666666666665,
                0.8333333333333333,
                1.0,
                1.1666666666666667,
                1.333333333333333,
                1.5,
                1.6666666666666665,
                1.8333333333333333,
                2.0,
            ]
        ],
    }
    base, rhs = data["domain"], data["rhs_domains"][0]
    expected = [
        0.0,
        0.3333333333333333,
        0.5,
        0.6666666666666665,
        0.8333333333333333,
        1.0,
        1.1666666666666667,
        1.333333333333333,
        1.5,
        1.6666666666666665,
        1.8333333333333333,
        2.0,
    ]
    assert merge(base, rhs) == expected
    op = BlockLinop(I(tuple(base)), domain=tuple(base))
    actual = op._merged_domain([SimpleNamespace(domain=tuple(rhs))])
    assert actual == tuple(expected)
    assert min(b - a for a, b in zip(actual[:-1], actual[1:])) > 0.16


def test_native_representative_strict_boundary_and_no_transitive_cluster():
    # Source strict less-than, anchored at each retained minimum.
    assert merge([0.0, 0.5, 1.0], [0.0, 0.625, 0.75, 1.0], 0.25) == [0.0, 0.5, 0.75, 1.0]


def test_default_scale_is100eps_and_keeps_smallest():
    e = 2.220446049250313e-16
    assert merge([0.0, 0.5, 2.0], [0.0, 0.5 + 199 * e, 2.0]) == [0.0, 0.5, 2.0]
    assert merge([0.0, 0.5, 2.0], [0.0, 0.5 + 200 * e, 2.0]) == [0.0, 0.5, 0.5 + 200 * e, 2.0]


def test_native_infinite_scale_is_one():
    e = 2.220446049250313e-16
    assert merge([-math.inf, 1.0, math.inf], [-math.inf, 1.0 + 99 * e, math.inf]) == [
        -math.inf,
        1.0,
        math.inf,
    ]


def test_source_incompatible_endpoints_identifier():
    with pytest.raises(ValueError, match="CHEBFUN:DOMAIN:merge:incompat"):
        merge([0.0, 1.0], [0.0, 2.0])
