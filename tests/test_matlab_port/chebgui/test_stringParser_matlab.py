"""Literal port of MATLAB tests/chebgui/test_stringParser.m.

Provenance
----------
MATLAB source : @stringParser/str2anon.m, tests/chebgui/test_stringParser.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import pytest

from chebfunjax.utils.string_parser import str2anon

FIXTURES = [
    ("bvp", "u(0) = 1", "@(u) feval(u,0)-1"),
    ("bvp", "u(-1) = 1", "@(u) feval(u,-1)-1"),
    ("bvp", "u(0)=1, u(2) = 3", "@(u) feval(u,0)-1;feval(u,2)-3"),
    ("bvp", "u = 1,v = 0,w = 3", "@(u,v,w) [u-1;v;w-3]"),
    ("bvp", "u(end),u(1,left)", "@(u) feval(u,'end');feval(u,1,'left')"),
    ("bvp", "diff(sin(u))", "@(u) diff(sin(u))"),
    ("bvp", "diff(sin(u),2)", "@(u) diff(sin(u),2)"),
    ("bvp", "diff(u^2) + sin(x)*u = 1", "@(x,u) diff(u.^2)+sin(x).*u-1"),
    ("bvp", "feval(u,sqrt(2))", "@(u) feval(u,sqrt(2))"),
    ("bvp", "feval(u,0,left)", "@(u) feval(u,0,'left')"),
    ("bvp", "sum(u) = 0", "@(u) sum(u)"),
    ("bvp", "sum(u,0,.5) = 0", "@(u) sum(u,0,.5)"),
    ("bvp", "volt(sin(x-y),u) = 0", "@(x,u) volt(@(x,y)sin(x-y),u)"),
    (
        "bvp",
        "fred(sin(x-y),u) = fred(cos(x-z),u)",
        "@(x,u) fred(@(x,y)sin(x-y),u)-fred(@(x,z)cos(x-z),u)",
    ),
    (
        "bvp",
        "u + fred(sin(2*pi*(y-x)),u) = feval(u,0,left)",
        "@(x,u) u+fred(@(x,y)sin(2.*pi.*(y-x)),u)-feval(u,0,'left')",
    ),
    ("eig", "u''-lambda*u = 0", ("@(u) diff(u,2)", "@(u) u")),
    ("eig", "u' + sin(x)*u = lambda*u", ("@(x,u) diff(u)+sin(x).*u", "@(x,u) u")),
    ("eig", "u''+u' = lambda*(u + u') + u", ("@(u) diff(u,2)+diff(u)-u", "@(u) u+diff(u)")),
    ("pde", "u_t+x*u'' = u'", "@(x,t,u) -x.*diff(u,2)+diff(u)"),
]


@pytest.mark.parametrize(("kind", "text", "expected"), FIXTURES)
def test_source_callback(kind, text, expected):
    assert str2anon(text, kind) == expected
