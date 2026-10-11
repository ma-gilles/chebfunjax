"""Supplemental gallery3 source/API controls, Chebfun7574c77.

No standalone native gallery3 test exists. Literal source expressions below
are evaluated independently with NumPy/SciPy as host input oracles; they are
not captured MATLAB numerical outputs. Library callbacks remain JAX.
"""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import jv

from chebfunjax import cheb

ENTRIES = {
    "aurentz": {
        "expression": "exp(-((x-r*cos(pi*t)).^2 + (y-r*sin(pi*t)).^2)./0.1)",
        "domain": [-1, 1, -1, 1, -1, 1],
        "trig": False,
    },
    "barth10": {
        "expression": "8*(x.^2 - p^4*y.^2) .* (y.^2 - p^4*z.^2) .*             "
        "(z.^2 - p^4.*x.^2) .* (x.^4 + y.^4 + z.^4 - 2*x.^2.*y.^2 "
        "-            2*x.^2.*z.^2 - 2*y.^2.*z.^2) + a*(3 + 5*p) .* "
        "((x.^2 + y.^2             + z.^2 - a)).^2 .* ((x.^2 + y.^2 + "
        "z.^2 - (2-p)*a)).^2",
        "domain": [-1, 1, -1, 1, -1, 1],
        "trig": False,
    },
    "bessel": {
        "expression": "besselj(0,10*sqrt(x.^2+y.^2+z.^2))",
        "domain": [-1, 1, -1, 1, -1, 1],
        "trig": False,
    },
    "cassini": {
        "expression": "((x-a).^2 + y.^2) .* ((x+a).^2 + y.^2) - z.^4",
        "domain": [-1, 1, -1, 1, -1, 1],
        "trig": False,
    },
    "clebsch": {
        "expression": "81*(x.^3 + y.^3 + z.^3) - 189*(x.^2.*y + x.^2.*z "
        "+            y.^2.*x + y.^2.*z + z.^2.*x + z.^2.*y) + "
        "54*(x.*y.*z) +             126*(x.*y + x.*z + y.*z) - "
        "9*(x.^2 + y.^2 + z.^2) -             9*(x + y + z) + 1",
        "domain": [-1, 1, -1, 1, -1, 1],
        "trig": False,
    },
    "doublehelix": {
        "expression": "abs(((x+1i*y).*exp(1i*z)).^2-1).^2",
        "domain": [-1.5, 1.5, -1.5, 1.5, -6, 6],
        "trig": False,
    },
    "genz1": {
        "expression": "cos(pi + 5 * (x+y+z))",
        "domain": [-1, 1, -1, 1, -1, 1],
        "trig": False,
    },
    "genz2": {
        "expression": "1./(16 + 5 * (x+y+z))",
        "domain": [-1, 1, -1, 1, -1, 1],
        "trig": False,
    },
    "genz3": {
        "expression": "exp(-25*(x.^2+y.^2+z.^2))",
        "domain": [-1, 1, -1, 1, -1, 1],
        "trig": False,
    },
    "genz4": {
        "expression": "1./((0.04+x.^2) .* (0.04+y.^2) .* (0.04+z.^2))",
        "domain": [-1, 1, -1, 1, -1, 1],
        "trig": False,
    },
    "griewank": {
        "expression": "x.^2/4000 + y.^2/4000 + z.^2/4000 -            "
        "cos(x).*cos(y/sqrt(2)).*cos(z/sqrt(3)) + 1",
        "domain": [-50, 50, -50, 50, -50, 50],
        "trig": False,
    },
    "hoop": {
        "expression": "z.^2 + 0.5*(x.^2+y.^2-.5).^2",
        "domain": [-1, 1, -1, 1, -1, 1],
        "trig": False,
    },
    "kummer": {
        "expression": "x.^4 + y.^4 + z.^4 - 0.1*(x.^2 +y.^2 + z.^2)             - "
        "0.5*(x.^2 .* y.^2 + x.^2.*z.^2 + y.^2.*z.^2) + 0.1*x.*y.*z - "
        "1",
        "domain": [-1, 1, -1, 1, -1, 1],
        "trig": False,
    },
    "lattice": {
        "expression": "cos(2*pi*x).^2 + cos(2*pi*y).^2 + cos(2*pi*z).^2",
        "domain": [-1, 1, -1, 1, -1, 1],
        "trig": True,
    },
    "levy": {
        "expression": "(sin(pi*(1 + (x - 1)/4))).^2 + ((z - 1)/4).^2 .*             "
        "(1+(sin(2*pi*(z - 1)/4)).^2) + (((x - 1)/4)).^2 .*             "
        "(1+10*(sin(pi*(1 + (x - 1)/4)+1)).^2) +             (((y - "
        "1)/4)).^2 .* (1+10*(sin(pi*(1 + (y - 1)/4)+1)).^2)",
        "domain": [-10, 10, -10, 10, -10, 10],
        "trig": False,
    },
    "octant": {"expression": "sqrt(x.^2+y.^2+z.^2)", "domain": [0, 1, 0, 1, 0, 1], "trig": False},
    "rastrigin": {
        "expression": "30 + (x.^2-10*cos(2*pi*x)) +                "
        "(y.^2-10*cos(2*pi*y)) + (z.^2-10*cos(2*pi*z))",
        "domain": [-1, 1, -1, 1, -1, 1],
        "trig": False,
    },
    "rosenbrock": {
        "expression": "100*(y-x.^2).^2 + (x-1).^2 + 100*(z-y.^2).^2 + (y-1).^2",
        "domain": [-2, 2, -2, 2, -2, 2],
        "trig": False,
    },
    "runge": {
        "expression": "1./(1+x.^2+y.^2+z.^2)",
        "domain": [-1, 1, -1, 1, -1, 1],
        "trig": False,
    },
    "shubert": {
        "expression": "(cos(2*x+1) + 2*cos(3*x+2) + 3*cos(4*x+3) +             "
        "4*cos(5*x+4) + 5*cos(4*x+5)) .* (cos(2*y+1) + 2*cos(3*y+2) "
        "+             3*cos(4*y+3) + 4*cos(5*y+4) + 5*cos(4*y+5)) "
        ".*             (cos(2*z+1) + 2*cos(3*z+2)+ 3*cos(4*z+3) + "
        "4*cos(5*z+4) +             5*cos(4*z+5))",
        "domain": [-1, 1, -1, 1, -1, 1],
        "trig": False,
    },
    "toyfunction": {
        "expression": "3*x.^7.*z + y.*z + y.*z.^2 + log(2+y).*z.^3             - 2*z.^5",
        "domain": [-1, 1, -1, 1, -1, 1],
        "trig": False,
    },
    "wagon": {
        "expression": "exp(sin(50*x)) + sin(60*exp(y)).*sin(60*z) +             "
        "sin(70*sin(x)).*cos(10*z) + sin(sin(80*y)) - sin(10*(x+z)) "
        "+            (x.^2 + y.^2 + z.^2)/4",
        "domain": [-1, 1, -1, 1, -1, 1],
        "trig": False,
    },
}
EPS = np.finfo(np.float64).eps


def source_values(name, x, y, z):
    expression = ENTRIES[name]["expression"]
    expression = expression.replace(".^", "**").replace("^", "**")
    expression = expression.replace(".*", "*").replace("./", "/")
    expression = expression.replace("1i", "1j")
    environment = dict(
        x=x,
        y=y,
        z=z,
        t=z,
        r=0.6,
        a=1 if name == "barth10" else 0.4,
        p=(np.sqrt(5) + 1) / 2,
        pi=np.pi,
        sin=np.sin,
        cos=np.cos,
        exp=np.exp,
        sqrt=np.sqrt,
        log=np.log,
        abs=np.abs,
        besselj=jv,
    )
    return eval(expression, {"__builtins__": {}}, environment)


@pytest.mark.parametrize("name", ENTRIES)
def test_source_handle_formula_domain_and_mode(name, monkeypatch):
    module = importlib.import_module("chebfunjax.chebfun3d.chebfun3")
    sentinel = object()
    calls = []

    def spy(fa, **kwargs):
        calls.append((fa, kwargs))
        return sentinel

    monkeypatch.setattr(module, "chebfun3", spy)
    monkeypatch.setattr(module.Chebfun3, "from_function", staticmethod(spy))
    f, fa = cheb.gallery3(name.upper(), return_handle=True)
    assert f is sentinel and len(calls) == 1 and calls[0][0] is fa
    expected_kwargs = {}
    domain = tuple(ENTRIES[name]["domain"])
    if domain != (-1, 1, -1, 1, -1, 1):
        expected_kwargs["domain"] = domain
    if ENTRIES[name]["trig"]:
        expected_kwargs["trig"] = True
    assert calls[0][1] == expected_kwargs
    # Includes boundaries and a broadcast 3x2 grid, not just origin identities.
    x = np.asarray([domain[0], 0.17 * (domain[1] - domain[0]) + domain[0], domain[1]])[:, None]
    y = np.asarray([domain[2], 0.61 * (domain[3] - domain[2]) + domain[2]])[None, :]
    z = 0.43 * (domain[5] - domain[4]) + domain[4]
    expected = source_values(name, x, y, z)
    got = np.asarray(fa(jnp.asarray(x), jnp.asarray(y), jnp.asarray(z)))
    bound = 64 * EPS * max(1.0, float(np.max(np.abs(expected))))
    assert got.shape == expected.shape
    assert float(np.max(np.abs(got - expected))) < bound
    compiled = np.asarray(jax.jit(fa)(jnp.asarray(x), jnp.asarray(y), jnp.asarray(z)))
    assert float(np.max(np.abs(compiled - expected))) < bound
    point = (jnp.asarray(0.2), jnp.asarray(0.3), jnp.asarray(0.4))
    scalar = fa(*point)
    assert scalar.ndim == 0
    reference = source_values(name, 0.2, 0.3, 0.4)
    assert abs(float(scalar) - float(reference)) < 64 * EPS * max(1.0, abs(float(reference)))


def test_no_name_selection_and_single_output(monkeypatch):
    import random

    module = importlib.import_module("chebfunjax.chebfun3d.chebfun3")
    sentinel = object()
    choices = []

    def choose(names):
        choices.extend(names)
        return "hoop"

    monkeypatch.setattr(random, "choice", choose)
    monkeypatch.setattr(module, "chebfun3", lambda fa, **kwargs: sentinel)
    assert cheb.gallery3() is sentinel
    assert set(choices) == set(ENTRIES) and len(choices) == 22


def test_unknown_source_identifier():
    with pytest.raises(ValueError, match="CHEB:GALLERY3:unknown:unknownFunction"):
        cheb.gallery3("not_a_gallery_function")


@pytest.mark.parametrize("name", ["cassini", "hoop", "rosenbrock"])
def test_easy_actual_default_construction(name):
    module = importlib.import_module("chebfunjax.chebfun3d.chebfun3")
    f, fa = cheb.gallery3(name, return_handle=True)
    assert isinstance(f, module.Chebfun3)
    assert tuple(f.domain) == tuple(ENTRIES[name]["domain"])
    assert not f.isPeriodicTech()
    dom = ENTRIES[name]["domain"]
    fractions = jnp.asarray([0.0, 0.13, 0.47, 0.81, 1.0])
    x = dom[0] + fractions * (dom[1] - dom[0])
    y = dom[2] + fractions[::-1] * (dom[3] - dom[2])
    z = dom[4] + jnp.asarray([0.2, 0.6, 0.4, 0.9, 0.1]) * (dom[5] - dom[4])
    expected = fa(x, y, z)
    actual = f(x, y, z)
    bound = 1000 * EPS * max(1.0, float(jnp.max(jnp.abs(expected))))
    error = float(jnp.max(jnp.abs(actual - expected)))
    print({"supplement": name, "error": error, "bound": bound}, flush=True)
    assert error < bound


def test_actual_source_lattice_trig_route():
    f, fa = cheb.gallery3("lattice", return_handle=True)
    assert tuple(f.domain) == (-1, 1, -1, 1, -1, 1)
    assert f.isPeriodicTech()
    x = jnp.asarray([-1.0, -0.31, 0.0, 0.27, 1.0])
    y = jnp.asarray([-0.63, 0.24, 1.0, -0.15, 0.48])
    z = jnp.asarray([0.16, -1.0, -0.52, 1.0, 0.73])
    expected = fa(x, y, z)
    error = float(jnp.max(jnp.abs(f(x, y, z) - expected)))
    bound = 1000 * EPS * max(1.0, float(jnp.max(jnp.abs(expected))))
    print({"supplement": "lattice_periodic", "error": error, "bound": bound}, flush=True)
    assert error < bound
