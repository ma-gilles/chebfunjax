"""Literal port of MATLAB tests/chebgui/test_multipleOutputs.m.

Provenance
----------
MATLAB source : @stringParser/str2anon.m, tests/chebgui/test_multipleOutputs.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import pytest

from chebfunjax.utils.string_parser import str2anon

FIXTURES = [
    (
        "bvp",
        {
            "input": "u(-1) = 1",
            "an_fun": "feval(u,-1)-1",
            "independent_vars": ["", ""],
            "variables": ["u"],
            "pde_variables": [],
            "eigenvalue_names": [],
            "comma_separated": False,
        },
    ),
    (
        "bvp",
        {
            "input": "u(0)=1, u(2) + x = 3",
            "an_fun": "feval(u,0)-1;feval(u,2)+x-3",
            "independent_vars": ["x", ""],
            "variables": ["u"],
            "pde_variables": [],
            "eigenvalue_names": [],
            "comma_separated": True,
        },
    ),
    (
        "bvp",
        {
            "input": "u = 1,v = 0,w = 3",
            "an_fun": "u-1;v;w-3",
            "independent_vars": ["", ""],
            "variables": ["u", "v", "w"],
            "pde_variables": [],
            "eigenvalue_names": [],
            "comma_separated": True,
        },
    ),
    (
        "bvp",
        {
            "input": "x + diff(sin(v))",
            "an_fun": "x+diff(sin(v))",
            "independent_vars": ["x", ""],
            "variables": ["v"],
            "pde_variables": [],
            "eigenvalue_names": [],
            "comma_separated": False,
        },
    ),
    (
        "bvp",
        {
            "input": "sum(u,0,.5) = 0",
            "an_fun": "sum(u,0,.5)",
            "independent_vars": ["", ""],
            "variables": ["u"],
            "pde_variables": [],
            "eigenvalue_names": [],
            "comma_separated": False,
        },
    ),
    (
        "bvp",
        {
            "input": "fred(sin(x-y),u) = fred(cos(x-z),u)",
            "an_fun": "fred(@(x,y)sin(x-y),u)-fred(@(x,z)cos(x-z),u)",
            "independent_vars": ["x", ""],
            "variables": ["u"],
            "pde_variables": [],
            "eigenvalue_names": [],
            "comma_separated": False,
        },
    ),
    (
        "eig",
        {
            "input": "v''-lambda*v = 0",
            "an_fun": ["diff(v,2)", "v"],
            "independent_vars": ["", ""],
            "variables": ["v"],
            "pde_variables": [],
            "eigenvalue_names": ["lambda"],
            "comma_separated": False,
        },
    ),
    (
        "eig",
        {
            "input": "u''+u' = lam*(u + u') + x*u",
            "an_fun": ["diff(u,2)+diff(u)-x.*u", "u+diff(u)"],
            "independent_vars": ["x", ""],
            "variables": ["u"],
            "pde_variables": [],
            "eigenvalue_names": ["lam"],
            "comma_separated": False,
        },
    ),
    (
        "pde",
        {
            "input": "u_t+x*u'' = u'",
            "an_fun": "-x.*diff(u,2)+diff(u)",
            "independent_vars": ["x", "t"],
            "variables": ["u"],
            "pde_variables": ["u_t"],
            "eigenvalue_names": [],
            "comma_separated": False,
        },
    ),
]


@pytest.mark.parametrize(("kind", "case"), FIXTURES)
def test_source_multiple_outputs(kind, case):
    expected = (
        tuple(case["an_fun"]) if isinstance(case["an_fun"], list) else case["an_fun"],
        tuple(case["independent_vars"]),
        tuple(case["variables"]),
        tuple(case["pde_variables"]),
        tuple(case["eigenvalue_names"]),
        case["comma_separated"],
    )
    result = str2anon(case["input"], kind, outputs=6)
    assert tuple(result) == expected
    assert result.an_fun == expected[0]
    assert result.independent_vars == expected[1]
    assert result.variable_names == expected[2]
    assert result.pde_variable_names == expected[3]
    assert result.eigenvalue_names == expected[4]
    assert result.comma_separated == expected[5]
