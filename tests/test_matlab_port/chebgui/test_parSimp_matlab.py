"""Port of MATLAB tests/chebgui/test_parSimp.m (48 exact fixtures).

Provenance
----------
MATLAB source : @stringParser/parSimp.m, tests/chebgui/test_parSimp.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import pytest

from chebfunjax.utils.string_parser import par_simp


@pytest.mark.parametrize("text", ["(u", "u)"])
def test_unbalanced_parentheses_preserve_source_diagnostic(text):
    with pytest.raises(ValueError, match="CHEBFUN:STRINGPARSER:parSimp:parenthSimplify") as caught:
        par_simp(text)
    assert caught.value.identifier == "CHEBFUN:STRINGPARSER:parSimp:parenthSimplify"


FIXTURES = [
    ("((0.02.*diff(u,2)+diff(u))+u)", "0.02.*diff(u,2)+diff(u)+u"),
    ("((0.01.*diff(u,2)-x.*u)-1)", "0.01.*diff(u,2)-x.*u-1"),
    ("((x.^(2).*diff(u,2)+x.*diff(u))+(x.^(2)-3.^(2)).*u)",
     "x.^2.*diff(u,2)+x.*diff(u)+(x.^2-3.^2).*u"),
    ("(((0.01.*diff(u,2)+2.*(1-x.^(2)).*u)+u.^(2))-1)",
     "0.01.*diff(u,2)+2.*(1-x.^2).*u+u.^2-1"),
    ("((diff(u,2)+(1.2+sign((10-abs(x)))).*u)-1)",
     "diff(u,2)+(1.2+sign((10-abs(x)))).*u-1"),
    ("((diff(u,2)+u)-u.^(2))", "diff(u,2)+u-u.^2"),
    ("(diff(u,4)-(diff(u).*diff(u,2)-u.*diff(u,3)))",
     "diff(u,4)-diff(u).*diff(u,2)+u.*diff(u,3)"),
    ("(diff(u,2)+.87.*exp(u))", "diff(u,2)+.87.*exp(u)"),
    ("((.0005.*diff(u,2)+x.*(x.^(2)-0.5).*diff(u))+3.*(x.^(2)-0.5).*u)",
     ".0005.*diff(u,2)+x.*(x.^2-0.5).*diff(u)+3.*(x.^2-0.5).*u"),
    ("((.01.*diff(u,2)+u.*diff(u))-u)", ".01.*diff(u,2)+u.*diff(u)-u"),
    ("(diff(u,2)+sin(u))", "diff(u,2)+sin(u)"),
    ("((0.05.*diff(u,2)+diff(u).^(2))-1)", "0.05.*diff(u,2)+diff(u).^2-1"),
    ("(diff(u,2)-8.*sinh(8.*u))", "diff(u,2)-8.*sinh(8.*u)"),
    ("(diff(u,2)-(u-1).*(1+diff(u).^(2)).^(1.5))",
     "diff(u,2)-(u-1).*(1+diff(u).^2).^1.5"),
    ("((diff(u,2)-x.*sin(u))-1)", "diff(u,2)-x.*sin(u)-1"),
    ("((diff(u,2)-(1-u.^(2)).*diff(u))+u)",
     "diff(u,2)-(1-u.^2).*diff(u)+u"),
    ("(diff(u,2)-sin(v))", "diff(u,2)-sin(v)"),
    ("(diff(u,2)-sin(v))", "diff(u,2)-sin(v)"),
    ("(cos(u)+diff(v,2))", "cos(u)+diff(v,2)"),
    ("+(0.1.*diff(u,2)+diff(u))", "0.1.*diff(u,2)+diff(u)"),
    ("+((.01.*diff(u,2)+u)-u.^(3))", ".01.*diff(u,2)+u-u.^3"),
    ("+(-diff(u.^(2))+.02.*diff(u,2))", "-diff(u.^2)+.02.*diff(u,2)"),
    ("+(-.003.*diff(u,4)+diff((u.^(3)-u),2))",
     "-.003.*diff(u,4)+diff(u.^3-u,2)"),
    ("+0.1.*diff(u,2)", "0.1.*diff(u,2)"),
    ("+(.02.*diff(u,2)+cumsum(u).*sum(u))",
     ".02.*diff(u,2)+cumsum(u).*sum(u)"),
    ("+((u.*diff(u)-diff(u,2))-0.006.*diff(u,4))",
     "u.*diff(u)-diff(u,2)-0.006.*diff(u,4)"),
    ("+((-u+(x+1).*v)+0.1.*diff(u,2))", "-u+(x+1).*v+0.1.*diff(u,2)"),
    ("+((u-(x+1).*v)+0.2.*diff(v,2))", "u-(x+1).*v+0.2.*diff(v,2)"),
    ("+(0.1.*diff(u,2)-100.*u.*v)", "0.1.*diff(u,2)-100.*u.*v"),
    ("+(.2.*diff(v,2)-100.*u.*v)", ".2.*diff(v,2)-100.*u.*v"),
    ("+(0.001.*diff(w,2)+200.*u.*v)", "0.001.*diff(w,2)+200.*u.*v"),
    ("+(diff(u,2)-v)", "diff(u,2)-v"),
    ("(diff(v,2)-u)", "diff(v,2)-u"),
    ("+(diff(u,2)+exp(-100.*x.^(2)).*sin(pi.*t))",
     "diff(u,2)+exp(-100.*x.^2).*sin(pi.*t)"),
    ("(diff(u,2)+diff(u))", "diff(u,2)+diff(u)"),
    ("(-diff(u,2)+1i.*x.^(2).*u)", "-diff(u,2)+1i.*x.^2.*u"),
    ("(-.1.*diff(u,2)+4.*(sign((x+1))-sign((x-1))).*u)",
     "-.1.*diff(u,2)+4.*(sign((x+1))-sign((x-1))).*u"),
    ("(-diff(u,2)+x.^(2).*u)", "-diff(u,2)+x.^2.*u"),
    ("(-diff(u,2)+5.*cos(2.*x).*u)", "-diff(u,2)+5.*cos(2.*x).*u"),
    ("((diff(u,2)+u.*x)+v)", "diff(u,2)+u.*x+v"),
    ("(diff(v,2)+sin(x).*u)", "diff(v,2)+sin(x).*u"),
    ("(diff(v,2)-1e-2*u)", "diff(v,2)-1e-2*u"),
    ("(diff(v,2)-1e+2*u)", "diff(v,2)-1e+2*u"),
    ("(diff(v,2)+1e-2*u)", "diff(v,2)+1e-2*u"),
    ("(diff(v,2)+1e+2*u)", "diff(v,2)+1e+2*u"),
    ("diff(y)-(2/3)/4", "diff(y)-2/3/4"),
    ("diff(y)-2/(3/4)", "diff(y)-2/(3/4)"),
    ("diff(y)-1/(2*(y-1))", "diff(y)-1/(2*(y-1))"),
]


@pytest.mark.parametrize(("source", "expected"), FIXTURES)
def test_par_simp_source_fixture(source, expected):
    assert par_simp(source) == expected


def test_par_simp_fixture_count():
    assert len(FIXTURES) == 48
