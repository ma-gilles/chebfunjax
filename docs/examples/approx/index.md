# Approximation Theory Examples

Chebfun was born from approximation theory. These examples demonstrate
polynomial and rational approximation, convergence theory, and the
classical families of special functions.

## Core Introductions

| Example | Description |
|---------|-------------|
| [Polynomial approximation basics](polynomial_approximation.md) | Getting started with chebfunjax |
| [Chebyshev coefficient decay](chebyshev_coefficients.md) | Rates for analytic and non-analytic functions |
| [Polynomial convergence](polynomial_convergence.md) | Geometric convergence for analytic functions |
| [Piecewise smooth functions](piecewise_smooth.md) | Functions with breakpoints |
| [Special functions](special_functions.md) | Bessel, Airy, elliptic functions |

## Polynomial Approximation

| Example | Description |
|---------|-------------|
| [Chebyshev interpolation of entire functions](Entire.md) | Oscillatory entire functions |
| [Convergence bounds for entire functions](EntireBound.md) | Bernstein ellipse estimates |
| [Resolution of wiggly functions](ResolutionWiggly.md) | How Chebfun handles oscillatory functions |
| [Odd and even best approximations](OddEven.md) | Symmetry in minimax approximation |
| [Oscillation of error](OscError.md) | Equioscillation in best approximation |
| [Best approximation with Remez](BestApprox.md) | The Remez algorithm |
| [Best L1 polynomial approximation](BestL1.md) | Minimax in the L1 norm |
| [Best L1 norm fitting](polyfitL1.md) | L1 polynomial fitting |
| [Least-squares approximation](BestL2Approximation.md) | L2 best approximation |
| [Minimax sqrt approximation](MinimaxSqrt.md) | Polynomial and rational approximation of sqrt |
| [Local complexity](Local.md) | Measuring local smoothness |
| [Checkmark function](Checkmark.md) | Approximation of a non-smooth function |
| [Wiggly function](WigglyApprox.md) | Best approximations of oscillatory functions |
| [Smooth functions of compact support](SmoothCompact.md) | Compactly supported smooth functions |

## Rational Approximation

| Example | Description |
|---------|-------------|
| [Absolute value by rationals](absolute_value_newton.md) | Newton's method converging to \|x\| |
| [Absolute value approximations](AbsoluteValue.md) | Rational approximation of \|x\| |
| [Absolute value approximations II](AbsoluteValueScaled.md) | Scaled rational approximation of \|x\| |
| [Rational approximation of \|x\| minimax](RationalAbsx.md) | Minimax rational approximation |
| [Rational approximation of monomials](Rationalxn.md) | Rational approx of x^n |
| [Rational interpolation](RationalInterp.md) | Robust and non-robust rational interpolation |
| [Rational-like convergence](rational_like_convergence.md) | Supergeometric convergence |
| [Eight shades of rational approximation](EightShades.md) | Survey of rational methods |
| [Halphen's constant](Halphen.md) | Best rational approximation of exp(x) |
| [Fermi-Dirac rational approximation](FermiDirac.md) | Rational approx of step functions |
| [Pth root composite rational](PthComposite.md) | Composite rational approximation of roots |
| [Restricted denominator approximations](RestrictedDenominatorApproximations.md) | Approximation with restricted denominators |
| [Pushnitski's reciprocal log](Pushnitski.md) | Approximation of 1/log(x) |
| [CF approximation 30 years ago](CF30.md) | Historical CF rational approximation |
| [Digital filters via CF](FiltersCF.md) | Filter design using CF approximation |
| [Scaling and squaring for exp](ScalingAndSquaring.md) | Rational approximation of exp in complex regions |

## AAA Rational Approximation

| Example | Description |
|---------|-------------|
| [AAA rational approximation](AAAApprox.md) | The AAA algorithm |
| [AAA approximation of a spline](AAASpline.md) | AAA applied to piecewise functions |
| [Rootfinding with AAA](AAAZeros.md) | Finding zeros via AAA rational approximants |

## Interpolation

| Example | Description |
|---------|-------------|
| [Hermite interpolation](hermite_interpolation.md) | Matching values and derivatives |
| [Hermite polynomial basis](HermiteBasis.md) | Basis functions for Hermite interpolation |
| [Greedy interpolation](GreedyInterp.md) | Adaptive greedy point selection |
| [Interactive interpolation](InteractiveInterp.md) | Runge phenomenon and Chebyshev nodes |
| [Equispaced data](EquispacedData.md) | Chebfuns from equispaced samples |
| [Lebesgue constants](lebesgue_constants.md) | Chebyshev vs equispaced nodes |
| [Lebesgue functions](LebesgueConst.md) | Lebesgue functions and constants |

## Orthogonal Polynomials and Special Functions

| Example | Description |
|---------|-------------|
| [Orthogonal polynomials](orthogonal_polynomials.md) | Gram-Schmidt construction |
| [Orthogonal polynomials via Gram-Schmidt](OrthPolys.md) | Classical orthogonal families |
| [Orthogonal polynomials via Lanczos](OrthPolysLanczos.md) | Lanczos three-term recurrence |
| [Nearest orthonormal functions](NearestOrthFun.md) | L2 nearest orthonormal approximation |
| [Bernstein polynomials](BernsteinPolys.md) | Bernstein basis and approximation |
| [The Gamma function](gamma_function.md) | Poles and special values |
| [The Gamma function and poles](GammaFun.md) | Rational approximation near poles |
| [Prolate spheroidal wave functions](Prolate.md) | PSWFs and their properties |
| [Gallery functions](Galleries.md) | Chebfun gallery and gallerytrig |

## Convergence and Error Analysis

| Example | Description |
|---------|-------------|
| [Aliasing of Chebyshev coefficients](AliasingCoefficients.md) | Aliasing in Chebyshev series |
| [Aliasing of Legendre coefficients](AliasingCoefficientsLeg.md) | Aliasing in Legendre series |
| [Divergent series summation](DivergentSeries.md) | Summing divergent series |
| [Communication system mathematics](CommunicationSystem.md) | Signal processing with Chebfun |
| [The FFT in Chebfun](ChebfunFFT.md) | Connection between Chebyshev and FFT |
| [Edge detection](EdgeDetection.md) | Detecting discontinuities |

## Noisy Data and Splines

| Example | Description |
|---------|-------------|
| [Noisy functions](Noisy.md) | Approximation of noisy data |
| [Noisy non-smooth functions](NoisyNonsmooth.md) | Chebfuns of noisy discontinuous data |
| [L1 inpainting in 1D](Inpainting1D.md) | Signal reconstruction via L1 minimization |
| [Splines](Splines.md) | B-splines in Chebfun |
| [B-splines and convolution](BSplineConv.md) | B-spline convolution properties |
| [Weierstrass function](weierstrass.md) | Continuous but nowhere differentiable |
| [Pathological Weierstrass function](WeierstrassFunction.md) | Nowhere-differentiable functions |
