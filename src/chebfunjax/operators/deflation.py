"""Source deflation residual for plain functions and Fréchet AD."""


def _roots(roots):
    from chebfunjax.chebfun1d.chebfun import Chebfun
    if isinstance(roots, Chebfun):
        return [roots] if roots.n_columns == 1 else [roots.extract_columns(k) for k in range(roots.n_columns)]
    if hasattr(roots, "blocks"):
        return [block for row in roots.blocks for block in row]
    return list(roots)


def deflation_fun(residual, u, roots, p, alp, norm_type="L2"):
    """Return the residual multiplied by the source deflation factor.

    Roots may be a Chebfun, a sequence, or ChebMatrix blocks. L2 and H1
    follow the source product of squared norms, raised to p/2.

    Provenance
    ----------
    MATLAB source: @chebfun/deflationFun.m, @chebmatrix/deflationFun.m,
        @adchebfun/adchebfun.m (deflationFun).
    Chebfun commit: 7574c77
    """
    from chebfunjax.autodiff.adchebfun import ADChebfun
    if isinstance(residual, ADChebfun):
        return residual.deflation_fun(u, roots, p, alp, norm_type)
    product = 1.
    for root in _roots(roots):
        delta = u-root
        squared = delta.norm("fro")**2
        if norm_type != "L2":
            squared = squared+delta.diff().norm("fro")**2
        product = product*squared
    return residual*(1/product**(p/2)+alp)
