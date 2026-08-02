# Rootfinding Examples

Chebfun computes roots of smooth functions by first building the Chebyshev
expansion, then finding eigenvalues of the companion matrix. These examples
illustrate rootfinding in one dimension.

| Example | Description |
|---------|-------------|
| [Roots of a Bessel function (replica)](BesselRoots.md) | Faithful replica: all roots of J0 on [0,100]; 318 roots counted near x = 10^6. |
| [The mystery of Bernoulli polynomials](bernoulli_polynomials.md) | Bernoulli polynomials are a family of polynomials defined by the recurrence |
| [Does a Chebfun of degree n have n roots?](fundamental_theorem_algebra.md) | The Fundamental Theorem of Algebra states that a polynomial of degree ... has exactly ... roots in ... (counting mult... |
| [Newton's method](newton_raphson.md) | Newton's method is the most fundamental root-finding algorithm. Starting from an initial guess ..., it iterates |
| [Roots of random polynomials](random_polynomials.md) | If ... is a monic polynomial with independent standard-normal coefficients $a_0, \ldots, |
| [Complex roots near the real axis](roots_near_axis.md) | A chebfun may have no real roots while having complex roots very close to the real axis. These complex roots influenc... |
| [Speed and accuracy of Chebfun roots](roots_speed.md) | Chebfun's ... command uses the colleague matrix — the Chebyshev analogue of the companion matrix — whose eigenvalues ... |
| [Roots of a secular equation with poles](secular_roots.md) | A secular equation is a rational function of the form |
| [The tiger's tail](tiger.md) | A high-degree chebfun with geometrically decaying Chebyshev coefficients produces a striking visual pattern when its ... |
| [The white curves of Ortiz and Rivlin](white_curves.md) | In their 1983 article "Another look at the Chebyshev polynomials", Ortiz and Rivlin noticed that the graph of the fir... |
| [Extrema and Roots](extrema_and_roots.md) | The extrema of a Chebfun ... are the roots of its derivative .... Chebfun computes both with spectral accuracy. |
| [Newton–Raphson Method](newton_raphson.md) | Newton's method finds roots of ... by iterating: |
| [Polynomial Roots](polynomial_roots.md) | Chebfun finds roots by converting the Chebyshev expansion to a companion matrix and computing its eigenvalues. This i... |
| [Complex Roots Near the Real Axis](roots_near_axis.md) | A Chebfun approximation of a smooth function is a polynomial that is accurate inside a Bernstein ellipse in the compl... |
