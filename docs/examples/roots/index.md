# Rootfinding Examples

Chebfun computes roots of smooth functions by first building the Chebyshev
expansion, then finding eigenvalues of the companion matrix. These examples
illustrate rootfinding in one dimension.

| Example | Description |
|---------|-------------|
| [Roots of a Bessel function (replica)](BesselRoots.md) | Faithful replica: all roots of J0 on [0,100]; 318 roots counted near x = 10^6. |
| [Newton's method (replica)](NewtonRaphson.md) | Faithful replica: tangent-line visualization; quadratic and cubic convergence tables digit-for-digit. |
| [Complex roots near the real axis (replica)](RootsNearAxis.md) | Faithful replica: roots(f,'complex') ellipse-pruned roots — 32/degree-85 digit-for-digit. |
| [Speed and accuracy of Chebfun roots (replica)](RootsSpeed.md) | Faithful replica: 2001 roots, length 3284 exact, error one ulp. |
| [The tiger's tail (replica)](Tiger.md) | Faithful replica: f = round(f) stripes, 345 roots digit-for-digit. |
| [The white curves of Ortiz and Rivlin (replica)](WhiteCurves.md) | Faithful replica: Chebyshev and Legendre white-curve pictures. |
| [The mystery of Bernoulli polynomials](bernoulli_polynomials.md) | Bernoulli polynomials are a family of polynomials defined by the recurrence |
| [Does a Chebfun of degree n have n roots?](fundamental_theorem_algebra.md) | The Fundamental Theorem of Algebra states that a polynomial of degree ... has exactly ... roots in ... (counting mult... |
| [Roots of random polynomials](random_polynomials.md) | If ... is a monic polynomial with independent standard-normal coefficients $a_0, \ldots, |
| [Roots of a secular equation with poles](secular_roots.md) | A secular equation is a rational function of the form |
| [Extrema and Roots](extrema_and_roots.md) | The extrema of a Chebfun ... are the roots of its derivative .... Chebfun computes both with spectral accuracy. |
| [Polynomial Roots](polynomial_roots.md) | Chebfun finds roots by converting the Chebyshev expansion to a companion matrix and computing its eigenvalues. This i... |
