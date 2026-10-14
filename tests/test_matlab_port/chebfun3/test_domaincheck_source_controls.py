"""Adversarial domainCheck controls beyond the native same-domain tests.

MATLAB sources: @chebfun3/domainCheck.m, @chebfun/domainCheck.m,
@chebfun/hscale.m; Chebfun commit 7574c77.
"""

from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3, domainCheck


def test_source_empty_domain_rules():
    empty = chebfun3()
    nonempty = chebfun3(lambda x, y, z: x)
    assert domainCheck(empty, empty)
    assert not domainCheck(empty, nonempty)
    assert not domainCheck(nonempty, empty)


def test_source_factor_hscale_tolerance():
    base_domain = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    close_shift = 5e-16
    close_domain = tuple(endpoint + close_shift for endpoint in base_domain)
    far_shift = 5e-14
    far_domain = tuple(endpoint + far_shift for endpoint in base_domain)
    base = chebfun3(lambda x, y, z: x, base_domain)
    close = chebfun3(lambda x, y, z: x, close_domain)
    far = chebfun3(lambda x, y, z: x, far_domain)
    assert domainCheck(base, close)
    assert not domainCheck(base, far)
    # Explicit tol remains a Python-only relative override.
    assert domainCheck(base, far, tol=1e-12)


def test_explicit_tolerance_retains_original_python_scale():
    base = chebfun3(lambda x, y, z: x, (0., .01, 0., .01, 0., .01))
    other = Chebfun3(cols=base.cols, rows=base.rows, tubes=base.tubes,
                    core=base.core,
                    domain=tuple(v + 5e-14 for v in base.domain))
    assert not domainCheck(base, other)
    assert domainCheck(base, other, tol=1e-12)
