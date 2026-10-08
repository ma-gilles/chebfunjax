"""All6 original predicates from tests/chebfun/test_isfinite.m.

Provenance: MATLAB Chebfun7574c77680d7e82b79626300bf255498271a72df.
Exact source functions/domains/exponents/splitting and breakpoint mutation.
"""

import pytest

from ._finite_source import source_case


@pytest.mark.parametrize("case", range(1, 7))
def test_source_isfinite(case):
    f = source_case(case)
    assert bool(f.isfinite()) == (case in (1, 2, 5))
