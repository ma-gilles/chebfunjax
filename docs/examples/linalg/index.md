# Linear Algebra Examples

These examples treat functions as vectors in infinite-dimensional function
spaces, using inner products, norms, QR factorization, and SVD.

| Example | Description |
|---------|-------------|
| [Convergence of the SOR iteration (replica)](SOR.md) | Faithful replica: spectral-radius chebfun; Young's optimal omega digit-for-digit. |
| [Condition numbers of various bases (replica)](CondNos.md) | Faithful replica: quasimatrix condition numbers digit-for-digit (4.006 / 4.796 / 1.000 / 7244.534). |
| [Eigenvalue level repulsion (replica)](LevelRepulsion.md) | Faithful replica: eigenvalue curves of (1-t)A+tB avoid crossing; chebfun min of the gap. |
| [Eigenvalues via the determinant (replica)](EigsViaDet.md) | Faithful replica: tridiagonal det recurrence as chebfun; sign+edge detection reaches machine precision. |
| [QR factorization of a quasimatrix (replica)](QuasiQR.md) | Faithful replica: continuous Householder QR, residual at machine precision. |
| [Vandermonde with Arnoldi (replica)](VandermondeArnoldi.md) | Faithful replica: Vandermonde conds digit-for-digit; Arnoldi stabilizes degree-80 fitting. |
| [Field of values (replica)](FieldOfValues.md) | Faithful replica: fov boundary chebfun, numerical abscissa 15-digit consistency, polygon case via merge. |
| [Resolvent norm on the imaginary axis (replica)](ResolventNorm.md) | Faithful replica: eigenvalues digit-for-digit, max resolvent norm to 13 digits. |
| [A quiz about nonnormal matrices (replica)](NonnormalQuiz.md) | Faithful replica: transient growth maxima digit-for-digit. |
| [Transient growth in linear systems (replica)](TransientGrowth.md) | Faithful replica: maximum energy 358147.98785177 matches to the last digit. |
| [Crouzeix's conjecture (replica)](Crouzeix.md) | Faithful replica: Crouzeix ratios — Jordan block exactly 2, normal matrix exactly 1. |
| [Eigenvalue landscapes (replica)](EigLandscapes.md) | Faithful replica: chebfun2 eigenvalue surfaces; fixed-grid construction added for kinked symmetric case. |
| [Nonsmoothness of the field of values (replica)](NonsmoothFOV.md) | Faithful replica: boundary smoothness analysis; hard 5x5 case lengths 5585/3661 vs MATLAB 5704/3781. |
| [The analytic SVD (replica)](AnalyticSVD.md) | Faithful replica: kinked sorted singular values vs smooth analytic branches through crossings. |
| [Conditioning of the Vandermonde quasimatrix (replica)](CondVandermonde.md) | Faithful replica: cond to 12 digits; (1+sqrt(2))^n growth. |
| [Eigenvalue near-crossings and analyticity (replica)](CrossingsAnalyticity.md) | Faithful replica: AAA poles reveal the narrow strip of analyticity at near-crossings. |
| [Mercury-Earth conjunctions (replica)](MercuryEarthConjunctions.md) | Faithful replica: conjunction times as determinant roots. |
| [Constrained least squares (replica)](ConstrainedLeastSquares.md) | Faithful replica: generalized QR, constrained fits — solution digit-for-digit, residuals at machine precision. |
| [Chebfun Inner Products](chebfun_inner_products.md) | The inner product of two Chebfuns is computed via ..., which evaluates ... exactly using the |
| [Inner Products and Norms](inner_products.md) | Chebfunjax treats functions as elements of function spaces with inner products and norms. This example demonstrates t... |
| [Matrix Functions](matrix_functions.md) | Chebfun can evaluate matrix functions using Chebyshev interpolation of the scalar function on the spectrum of the mat... |
