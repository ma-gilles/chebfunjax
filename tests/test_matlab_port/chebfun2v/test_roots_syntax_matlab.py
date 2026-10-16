"""Original Whidborne construction and five unsorted matrix-norm predicates.

Provenance
----------
MATLAB source: tests/chebfun2v/test_roots_syntax.m
Chebfun commit: 7574c77
"""
import numpy as np

from chebfunjax.chebfun2d import chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

from ._helpers import TOL


class TestChebfun2vRootsSyntax:
    def test_all_matlab_assertions(self):
        domain = (0.0, 2*np.pi, .01, 25.)
        phi = chebfun2(lambda phi, p: phi, domain)
        p = chebfun2(lambda phi, p: p, domain)
        q = 1.4
        A = ((q*np.cos(q*np.pi)*phi.cos()
              + (2*q**2-1)*np.sin(q*np.pi)*phi.sin())/(q**2-1)
             + np.pi*np.sin(q*np.pi)/p + np.cos(q*np.pi)/(q*p))
        B = ((-q*np.sin(q*np.pi)*phi.cos()
              + (2*q**2-1)*np.cos(q*np.pi)*phi.sin())/(q**2-1)
             + np.pi*np.cos(q*np.pi)/p - np.sin(q*np.pi)/(q*p))
        eq43 = (1-q*A*p*np.cos(2*np.pi*q)+q*B*p*np.sin(2*np.pi*q)
                - q**2*p*phi.cos()/(q**2-1))
        eq44 = (q**2*A*np.sin(2*np.pi*q)+q**2*B*np.cos(2*np.pi*q)
                + q**2*phi.sin()/(q**2-1))
        r1 = np.asarray(eq43.roots(eq44, method="ms"))
        r2 = np.asarray(eq43.roots(eq44))
        r3 = np.asarray(eq43.roots(eq44, method="resultant"))
        field = Chebfun2v([eq43.approx, eq44.approx])
        r4 = np.asarray(field.roots(method="ms"))
        r5 = np.asarray(field.roots())
        r6 = np.asarray(field.roots(method="resultant"))
        # Native matrix norm is spectral and retains row order.
        for actual in (r2, r3, r4, r5, r6):
            assert actual.shape == r1.shape
            assert np.linalg.norm(r1-actual, ord=2) < TOL
