# Approximation Theory Examples

Chebfun was born from approximation theory. These examples demonstrate
polynomial and rational approximation, convergence theory, and the
classical families of special functions.

## Core Introductions

| Example | Description |
|---------|-------------|
| [AAA Rational Approximation](AAAApprox.md) | Chebfun has a number of methods for rational approximation of a function on an interval. Version 5.6.0 introduced the... |
| [AAA Approximation of a Spline](AAASpline.md) | When AAA approximates a spline function, its poles cluster exponentially near the nodes of non-analytic behaviour — t... |
| [Rootfinding with the AAA Algorithm](AAAZeros.md) | The AAA algorithm returns not only function values but also explicit zeros and poles of the rational approximant. For... |
| [Absolute Value Approximations by Rationals](AbsoluteValue.md) | Peter Lax observed that one can approximate ... by applying Newton's method to the equation ..., starting from .... T... |
| [Absolute Value Approximations by Rationals II](AbsoluteValueScaled.md) | This follows up the AbsoluteValue example. The key idea is to use the identity ... combined with the scaled Newton it... |
| [Accuracy of Chebyshev Coefficients via Aliasing](AliasingCoefficients.md) | The Chebyshev coefficients of a degree-... polynomial interpolant ... of a function ... are related to the exact coef... |
| [Accuracy of Legendre Coefficients via Aliasing](AliasingCoefficientsLeg.md) | The same aliasing phenomenon arises for Legendre coefficients: the 0th and ...th coefficients of the degree-... inter... |
| [B-splines and Convolution](BSplineConv.md) | The degree-... B-spline is obtained by convolving the box function ... with itself ... times: |
| [Bernstein Polynomials](BernsteinPolys.md) | Weierstrass proved in 1885 that any continuous function on ... can be uniformly approximated by polynomials. Bernstei... |
| [Best Approximation with the REMEZ Command](BestApprox.md) | The best (minimax or Chebyshev) approximation of degree ... to a function ... minimizes the ...-norm of the error. Th... |
| [Best Polynomial Approximation in the L1 Norm](BestL1.md) | For a function ... on ..., there are three classic best polynomial approximation problems: |
| [Least-Squares Approximation in Chebfun](BestL2Approximation.md) | The best ... approximation of degree ... to ... is the polynomial ... minimizing .... It equals the orthogonal projec... |
| [CF Approximation 30 Years Ago](CF30.md) | The Caratheodory-Fejer (CF) method computes near-best rational approximants extremely fast using the SVD of a Hankel ... |
| [The FFT in Chebfun](ChebfunFFT.md) | Chebyshev points ... on ... are the real parts of equispaced nodes on the unit circle. This connection makes the disc... |
| [Approximation of the Checkmark Function](Checkmark.md) | The checkmark function ... on ... is piecewise linear with a kink at .... It was studied by Dragnev, Legg, and Orive ... |
| [Illustrating the Mathematics of Signal Processing](CommunicationSystem.md) | Chebfun arithmetic makes it easy to demonstrate the mathematics of AM radio. A message signal ... is modulated onto a... |
| [Summing a Divergent Series](DivergentSeries.md) | The function |
| [Edge Detection in Chebfun](EdgeDetection.md) | Chebfun's ... mode uses a recursive bisection algorithm (originally by Rodrigo Platte) to automatically detect where ... |
| [Eight Shades of Rational Approximation](EightShades.md) | Chebfun (and chebfunjax) offer several approaches to rational approximation: |
| [Chebyshev Interpolation of Oscillatory Entire Functions](Entire.md) | For the entire function ..., the Chebyshev interpolant degree grows linearly with ...: roughly ... terms are needed f... |
| [Convergence Bounds for Entire Functions](EntireBound.md) | If ... is analytic on the Bernstein ...-ellipse with ... there, then by Theorem 8.3 of Trefethen [1]: |
| [Chebfuns from Equispaced Data](EquispacedData.md) | Equispaced interpolation suffers from the Runge phenomenon for high-degree polynomials. Chebfun's ... flag (introduce... |
| [Rational Approximation of the Fermi-Dirac Function](FermiDirac.md) | The Fermi-Dirac distribution ... arises in quantum mechanics and electronic structure theory. It has a smooth but rap... |
| [Digital Filters via CF Approximation](FiltersCF.md) | An ideal low-pass filter has frequency response ... for ... and ... otherwise. This step function cannot be represent... |
| [Gallery and Gallerytrig](Galleries.md) | Like MATLAB's ... command for matrices, Chebfun's ... provides a collection of interesting 1D functions for testing a... |
| [The Gamma Function and Its Poles](GammaFun.md) | The gamma function ... has simple poles at the non-positive integers. Chebfun can represent it on ... as a piecewise ... |
| [A Greedy Algorithm for Choosing Interpolation Points](GreedyInterp.md) | Without any prior knowledge, we can choose effective interpolation points greedily: |
| [Halphen's Constant for Approximation of exp(x)](Halphen.md) | The best type ... rational approximation to ... on ... satisfies |
| [Polynomial Basis for Hermite Interpolation](HermiteBasis.md) | Hermite interpolation matches both function values ... and derivatives ... at given nodes. The basis consists of two ... |
| [L1 Inpainting in One Dimension](Inpainting1D.md) | In 1D inpainting, a smooth function is corrupted over several intervals and we must recover it from the uncorrupted p... |
| [Interactive Interpolation](InteractiveInterp.md) | The key insight: Chebyshev nodes cluster near the endpoints, which is why they avoid the Runge phenomenon and yield s... |
| [Lebesgue Functions and Lebesgue Constants](LebesgueConst.md) | The Lebesgue constant ... measures the worst-case amplification of data errors in polynomial interpolation: |
| [Local Complexity of a Function](Local.md) | A globally smooth function may have much more complexity in some regions than others. The piecewise Chebfun represent... |
| [Approximating the Square Root by Polynomials and Rational Functions](MinimaxSqrt.md) | The square root ... on ... has a branch-point singularity at .... Polynomial approximation converges only as ... (alg... |
| [Nearest Orthonormal Functions](NearestOrthFun.md) | Given a quasimatrix ... (a matrix-valued Chebfun), the nearest orthonormal quasimatrix ... in the Frobenius norm is .... |
| [Noisy Functions in Chebfun](Noisy.md) | When a function has noise at level ..., adaptive construction will resolve the noise, producing a polynomial of degre... |
| [Chebfuns of Noisy Functions with Discontinuities](NoisyNonsmooth.md) | When a function has both noise and discontinuities, the best strategy is: |
| [Odd and Even Best Approximations](OddEven.md) | If ... is even (odd), its best polynomial approximant is also even (odd). This means we can compute the even part and... |
| [Orthogonal Polynomials via the Gram-Schmidt Process](OrthPolys.md) | For any weight ..., we can build orthonormal polynomials via: |
| [Orthogonal Polynomials via the Lanczos Process](OrthPolysLanczos.md) | Any set of orthogonal polynomials satisfies a three-term recurrence: |
| [Approximations and Oscillation of Error](OscError.md) | - Interpolation at Chebyshev points: the error equioscillates between ... extreme values (Chebyshev equioscillation t... |
| [Prolate Spheroidal Wave Functions](Prolate.md) | A function is bandlimited with bandwidth ... if its Fourier transform is supported on .... Among all such functions, ... |
| [Approximating the pth Root by Composite Rational Functions](PthComposite.md) | For approximating ... on ..., a single rational function of type ... achieves accuracy ... (root-exponential). Composite |
| [Approximating Pushnitski's Reciprocal Log Function](Pushnitski.md) | The function ... is continuous on ... (with ...) but its Taylor-like expansion near 0 involves ..., which is harder for |
| [Rational Approximation of abs(x) with Minimax](RationalAbsx.md) | Newman (1964) showed that the best type ... rational approximation to ... on ... achieves accuracy ..., far better th... |
| [Rational Interpolation, Robust and Non-robust](RationalInterp.md) | Rational interpolation at ... points for a type ... approximant is generally ill-conditioned — small perturbations ca... |
| [Rational Approximation of Monomials](Rationalxn.md) | The monomial ... on ... looks like it would be easy, but it has a very sharp transition near 0 that requires high-deg... |
| [Resolution of Wiggly Functions](ResolutionWiggly.md) | The function ... on ... is one of the Chebfun team's favorites for testing. It requires a polynomial of degree about ... |
| [Restricted-Denominator Approximations](RestrictedDenominatorApproximations.md) | In some applications (e.g., numerical ODE solvers), the denominator of a rational approximant to ... must be a polyno... |
| [Rational Approximation to the Exponential in a Complex Region](ScalingAndSquaring.md) | The identity ... allows computing ... by: 1. Scale: compute ... (small norm). |
| [Smooth Functions of Compact Support](SmoothCompact.md) | An infinitely smooth function with compact support can be constructed by convolving a box function ... with itself ..... |
| [Splines](Splines.md) | Chebfun has a ... command analogous to MATLAB's. It constructs a piecewise cubic polynomial that interpolates given d... |
| [A Pathological Function of Weierstrass](WeierstrassFunction.md) | In 1872, Karl Weierstrass shocked the mathematical world by constructing |
| [A Wiggly Function and Its Best Approximations](WigglyApprox.md) | The wiggly function ... on ... has frequency that increases with ...: while ... has frequency ..., the term |
| [Absolute Value Approximations by Rationals](absolute_value_newton.md) | Peter Lax observed a beautiful approach to approximating ...: solve the equation ... by Newton's method starting from... |
| [Approximating Bessel Functions](bessel_approximation.md) | Bessel functions ... are solutions of Bessel's differential equation. They are entire functions and their Chebyshev a... |
| [Chebyshev Coefficient Decay](chebyshev_coefficients.md) | The rate at which Chebyshev coefficients ... decay encodes the smoothness of the function. For an analytic function w... |
| [The Gamma Function and Its Poles](gamma_function.md) | This example displays some of chebfunjax's capabilities for functions with singularities, using the gamma function ..... |
| [Hermite Interpolation](hermite_interpolation.md) | Hermite interpolation matches both the values and derivatives of a function at a set of nodes. It naturally produces ... |
| [Lebesgue Functions and Lebesgue Constants](lebesgue_constants.md) | Suppose we have ... interpolation nodes ... in ... and want to interpolate a function ... at these points by a degree... |
| [Orthogonal Polynomials via Gram-Schmidt](orthogonal_polynomials.md) | Orthogonal polynomials are polynomials that are orthogonal with respect to a weighted ... inner product: |
| [Piecewise Smooth Functions](piecewise_smooth.md) | Functions with kinks or jump discontinuities require piecewise Chebyshev representations. Chebfun can handle these by... |
| [Best Polynomial Approximation in the L1 Norm](polyfitL1.md) | A key property of L1 best approximants: the error is highly concentrated near the singularity (kink) of the function,... |
| [Polynomial Approximation](polynomial_approximation.md) | Polynomial approximation is one of the most fundamental topics in numerical analysis. Chebfun uses Chebyshev interpol... |
| [Chebyshev Interpolation of Oscillatory Entire Functions](polynomial_convergence.md) | In this example we explore the approximation properties of Chebyshev interpolation for entire functions — that is, fu... |
| [Rational-Like Convergence](rational_like_convergence.md) | Functions with algebraic singularities (poles, branch points) near ... have Chebyshev coefficients that decay algebra... |
| [Special Functions](special_functions.md) | Chebfun can approximate special functions (Airy, Bessel, gamma, etc.) by constructing Chebyshev interpolants via call... |
| [A Pathological Function of Weierstrass](weierstrass.md) | In the late nineteenth century, Karl Weierstrass shocked the mathematical community by constructing a function that i... |
