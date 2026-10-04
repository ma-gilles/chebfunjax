"""Source-derived parser contracts beyond the original Chebgui test files.

These expectations are derived from executable MATLAB source, not fresh MATLAB
runs. Original 19 callback/9 metadata fixtures live in the corresponding port
files. No numerical tolerances are involved.

Provenance
----------
MATLAB source : @stringParser/{str2anon,lexer,parser,splitTreeEIG,pref2inf}.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import pytest

from chebfunjax.utils.string_parser import str2anon

ERRORS = [
    (
        "x+1",
        "bvp",
        None,
        "CHEBFUN:STRINGPARSER:str2anon:depvars",
        "No dependent variables detected. Variable 'x' treated as independent.",
    ),
    (
        "x+1",
        "bvp",
        "INIT",
        "CHEBFUN:STRINGPARSER:str2anon:depvars",
        'No dependent variables detected in the initial field. Input must be of the form "u = x, v = 2*x, ...',
    ),
    (
        "u+x+t=0",
        "bvp",
        None,
        "CHEBFUN:STRINGPARSER:lexer:tooManyIndVars",
        "Too many independent variables in input.",
    ),
    (
        "u_t+v_t+u=0",
        "pde",
        None,
        "CHEBFUN:STRINGPARSER:str2anon:pdeVariables",
        "Only one time derivative per line is allowed",
    ),
    (
        "u%",
        "bvp",
        None,
        "CHEBFUN:STRINGPARSER:lexer:unknownType",
        "Invalid token '%' in input.\nChebgui does not support '%' in its input fields.",
    ),
    (
        "u~v",
        "bvp",
        None,
        "CHEBFUN:STRINGPARSER:lexer:UnsupportedOperator",
        "Unsupported operator ~.",
    ),
    ("sin(u,1)", "bvp", None, "CHEBFUN:Parse:func1", "Method 'sin' only takes one input argument."),
    (
        "u(0,1)",
        "bvp",
        None,
        "CHEBFUN:Parse:secondArg",
        "Invalid second argument to u(0,...) type of expression.",
    ),
    ("u+", "bvp", None, "CHEBFUN:Parse:terminal", "Unrecognized character in input field:$"),
    (
        "u_t+x+r=0",
        "pde",
        None,
        "CHEBFUN:STRINGPARSER:lexer:tooManyIndVars",
        "Too many independent variables in input.",
    ),
    (
        "x+",
        "bvp",
        None,
        "CHEBFUN:STRINGPARSER:str2anon:depvars",
        "No dependent variables detected. Variable 'x' treated as independent.",
    ),
    (
        "sin+u",
        "bvp",
        None,
        "CHEBFUN:Parse:parenths",
        "Need parenthesis when using functions in input fields.",
    ),
    (
        "besselj(u)",
        "bvp",
        None,
        "CHEBFUN:Parse:func2",
        "Method 'besselj' requires two input arguments.",
    ),
    ("(u", "bvp", None, "CHEBFUN:Parse:parenths", "Parenthesis imbalance in input fields."),
    ("u==0", "bvp", None, "CHEBFUN:Parse:end", "Input expression ended in unexpected manner."),
    (")u", "bvp", None, "CHEBFUN:Parse:start", "Input field started with unaccepted symbol."),
    # Lexer rejects the character before the final independent-coordinate check.
    (
        "u+x+r%",
        "bvp",
        None,
        "CHEBFUN:STRINGPARSER:lexer:unknownType",
        "Invalid token '%' in input.\nChebgui does not support '%' in its input fields.",
    ),
]


@pytest.mark.parametrize(("text", "kind", "field", "identifier", "message"), ERRORS)
def test_source_error_identifier_message_and_precedence(text, kind, field, identifier, message):
    with pytest.raises(ValueError) as caught:
        str2anon(text, kind, field)
    assert caught.value.identifier == identifier
    assert str(caught.value) == message


@pytest.mark.parametrize("literal", ["1e-3", "1E+3", ".5e-2", "1i", "1.2j", "2."])
def test_numeric_tokens_do_not_add_dependent_variables(literal):
    result = str2anon(f"u={literal}", "bvp", outputs=6)
    assert result.variable_names == ("u",)
    assert result.an_fun == f"u-{literal}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("u`(0)=1", "@(u) feval(diff(u),0)-1"),
        ('u"(0)=1', "@(u) feval(diff(u,2),0)-1"),
        ("u'''(1,left)=2", "@(u) feval(diff(u,3),1,'left')-2"),
        ("z+x*u+a=0", "@(x,a,u,z) [z+x.*u+a]"),
        ("u+1,", "@(u) u+1"),
        ("u--v", "@(u,v) [u+v]"),
        ("u^2^3", "@(u) u.^2.^3"),
        ("u.*2+u./3+u.^4", "@(u) u.*2+u./3+u.^4"),
    ],
)
def test_source_derivatives_ordering_and_grammar(text, expected):
    assert str2anon(text, "bvp") == expected


def test_source_initial_scalar_six_outputs():
    assert tuple(str2anon("1", "bvp", "INITSCALAR", outputs=6)) == (
        "1",
        ("", ""),
        (),
        (),
        (),
        False,
    )
    with pytest.raises(UnboundLocalError, match="anFunComplete"):
        str2anon("1", "bvp", "INITSCALAR")


@pytest.mark.parametrize("outputs", [2, 3, 4, 5])
def test_partial_source_output_arity(outputs):
    expected = ("feval(u,0)-1", ("", ""), ("u",), (), (), False)
    assert str2anon("u(0)=1", "bvp", outputs=outputs) == expected[:outputs]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("u_t*u", "Cannot multiply time derivative"),
        ("u/u_t", "Cannot divide with time derivatives"),
        ("u_t^2+u", "Cannot take powers with time derivative"),
        ("u_t'+u", "Cannot differentiate time derivative"),
        ("diff(u_t)+u", "Cannot use time derivative as function arguments."),
    ],
)
def test_source_pde_placement_errors(text, message):
    with pytest.raises(ValueError) as caught:
        str2anon(text, "pde")
    assert caught.value.identifier == "CHEBFUN:STRINGPARSER:parser:PDE"
    assert str(caught.value) == message


def test_source_repeated_pde_name_becomes_independent_token():
    result = str2anon("u_t+u_t+u=0", "pde", outputs=6)
    assert result.pde_variable_names == ("u_t",)
    assert result.variable_names == ("u",)
    assert result.an_fun == "-u_t-u"


def test_source_right_hand_pde_print_and_sign(capsys):
    result = str2anon("u''=u_t", "pde", outputs=6)
    assert result.an_fun == "diff(u,2)"
    assert capsys.readouterr().out == "PDE on right\n"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("u=lambda/(u+1)", ("u-0./(u+1)", "1")),
        ("lambda*u=u''", ("-diff(u,2)", "-u")),
    ],
)
def test_source_eigenvalue_lowering_quirks(text, expected):
    assert str2anon(text, "eig", outputs=6).an_fun == expected


@pytest.mark.parametrize("text", ["u=lambda*u*x", "u=lambda*u/2", "u=lambda*u+lambda*u"])
def test_source_eigenvalue_factor_limit(text):
    with pytest.raises(ValueError) as caught:
        str2anon(text, "eig")
    assert caught.value.identifier == "CHEBFUN:STRINGPARSER:splitTreeEIG:factors"


def test_source_kernel_variables_are_removed_only_for_two_argument_form():
    result = str2anon("fred(sin(x-y),u)=0", "bvp", outputs=6)
    assert result.variable_names == ("u",)
    assert result.an_fun == "fred(@(x,y)sin(x-y),u)"
    result = str2anon("fred(sin(x-y),u,1)=0", "bvp", outputs=6)
    assert result.variable_names == ("u", "y")
    assert result.an_fun == "fred(sin(x-y),u,1)"


def test_source_pde_without_explicit_coordinate_omits_subscript_callback_argument():
    # str2anon.m:151-159 preserves the empty first slot while suppressing both
    # independent callback args; the second metadata slot still contains t.
    result = str2anon("u_t+u=0", "pde", outputs=6)
    assert result.independent_vars == ("", "t")
    assert result.an_fun == "-u"
    assert str2anon("u_t+u=0", "pde") == "@(u) -u"
