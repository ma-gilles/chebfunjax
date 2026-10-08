"""All 15 literal clauses of tests/adchebfun/test_lintest_times.m.

Provenance: Chebfun 7574c77680d7e82b79626300bf255498271a72df.
Inputs x, u=x+2, v=x+3 and independent seed indices match the source.
"""
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import chebfun


@pytest.mark.parametrize("clause", range(15))
def test_source_linearity_clause(clause):
    x = chebfun(lambda x: x)
    u, v = ADChebfun(x)+2, ADChebfun(x)+3
    if clause >= 6:
        u, v = u.seed(1, 2), v.seed(2, 2)
    operations = [
        lambda: u*2,
        lambda: u*(x+2),
        lambda: u.sin()*(x+2),
        lambda: 2*u,
        lambda: x*u,
        lambda: u*(u+2),
        lambda: u*2,
        lambda: u*(x+2),
        lambda: u.sin()*2,
        lambda: 2*u,
        lambda: x*u,
        lambda: u*(u+2),
        lambda: u*v,
        lambda: u.sin()*v,
        lambda: u*(u*v),
    ]
    expected = [(True,), (True,), (False,), (True,), (True,), (False,), (True, True), (True, True), (False, True), (True, True), (True, True), (False, True), (False, False), (False, False), (False, False)]
    assert operations[clause]().linearity == expected[clause]
