"""All6 original predicates from tests/chebfun/test_isinf.m.

Provenance: MATLAB Chebfun7574c77680d7e82b79626300bf255498271a72df.
Exact source functions/domains/exponents/splitting and breakpoint mutation.
"""

import pytest

from ._finite_source import source_case


@pytest.mark.parametrize("case", range(1, 7))
def test_source_isinf(case):
    f = source_case(case)
    assert bool(f.isinf()) == (case in (3, 4, 6))
