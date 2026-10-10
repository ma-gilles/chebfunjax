"""Native make.m six operator/dispatch/equality routes, pin7574c77.

Native unseeded rand exponents use the port's retained deterministic values;
this qualifies factory routes, not MATLAB RNG/input parity.
"""
import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.fun.singfun import Singfun


class TestSingfunMake:
    def test_handle_only(self):
        a, b = .3, .4
        def fh(x):
            return jnp.sin(x)/((1+x)**a*(1-x)**b)
        f = Singfun.constructor(fh)
        assert f.isequal(f.make(fh))

    def test_handle_and_exponents(self):
        a, b = .3, .4
        def fh(x):
            return jnp.sin(x)*(1+x)**a*(1-x)**b
        data = {'exponents': (a, b)}
        f = Singfun.constructor(fh, data)
        assert f.isequal(f.make(fh, data))

    def test_handle_and_singtype(self):
        a, b = 3, 4
        def fh(x):
            return jnp.exp(x)/((1+x)**a*(1-x)**b)
        data = {'exponents': [], 'singType': ('pole', 'pole')}
        pref = ChebfunPref()
        f = Singfun.constructor(fh, data, pref)
        assert f.isequal(f.make(fh, data, pref))

    def test_handle_exponents_pref(self):
        a, b = .3, .4
        def fh(x):
            return jnp.exp(jnp.sin(x))/((1+x)**a*(1-x)**b)
        data = {'exponents': (-a, -b)}
        pref = ChebfunPref()
        f = Singfun.constructor(fh, data, pref)
        assert f.isequal(f.make(fh, data, pref))

    def test_handle_singtype_pref(self):
        a, b = .3, .4
        def fh(x):
            return jnp.sin(jnp.exp(jnp.cos(x)))*(1+x)**a*(1-x)**b
        data = {'exponents': [], 'singType': ('root', 'root')}
        pref = ChebfunPref()
        f = Singfun.constructor(fh, data, pref)
        assert f.isequal(f.make(fh, data, pref))

    def test_all_arguments(self):
        a, b = 2, 3
        def fh(x):
            return jnp.exp(jnp.sin(x**2))/((1+x)**a*(1-x)**b)
        data = {'exponents': (-a, -b), 'singType': ('pole', 'pole')}
        pref = ChebfunPref()
        f = Singfun.constructor(fh, data, pref)
        assert f.isequal(f.make(fh, data, pref))
