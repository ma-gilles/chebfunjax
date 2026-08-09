# 2D Approximation (Chebfun2)

Faithful replicas of the published
[chebfun.org approx2 examples](https://www.chebfun.org/examples/approx2/).
Chebfun2 extends the Chebfun philosophy to functions of two variables
on rectangles, using a low-rank (Gaussian elimination) representation.

| Example | Description |
|---------|-------------|
| [Low-rank approximation and alignment with axes](Alignment.md) | Rank of tanh(k(cx+sy)) vs. rotation angle — axis-alignment decides low-rank compressibility. |
| [2D zero set example of Dmitry Belyaev](Belyaev.md) | Zero sets of random plane-wave sums at k = 8, 16, 32 via Chebfun2 roots; 23 components with arc lengths. |
| [The low-rank structure of a sum of bump functions](BumpFunction.md) | 100 random Gaussian bumps have numerical rank far below 100. |
| [Gibbs phenomenon in 2D](Gibbs2D.md) | Chebyshev and periodic interpolants of a square-block data matrix; 2D Gibbs overshoots to 15 digits. |
| [Combining Chebyshev and trigonometric](Hosepipe.md) | Mixed cheb/trig representations ('trigy'): a corrugated hosepipe surface and a function on an annulus. |
| [Low-rank approximation and localized singularities](Localization.md) | Localized spikes and near-corner singularities give dramatic rank compression. |
| [Maximum trace problems](MaxTrace.md) | Maximizing trace(G' f G) via eigenfunctions from the chebfun2 SVD. |
| [Nearest positive semidefinite kernel](NearestPSDKernel.md) | The PSD part of a symmetric kernel by dropping negative-eigenvalue terms. |
| [Chebfun2 objects on non-rectangular domains](Other2DDomains.md) | Green's-theorem integrals over curved regions, sector maps, Jacobians, the Klein bottle shadow. |
| [Padua points in Chebfun2](PaduaPoints.md) | The Padua grid, its Lissajous characterization, and total-degree interpolation via the 'padua' flag. |
| [Low-rank compression of square and round pegs](Pegs.md) | Tilted (rank 100), aligned (rank 1), and round pegs. |
| [Pretty functions approximated by Chebfun2](PrettyFunctions.md) | Contour/pivot plots, surfaces, and the waterfall of Franke's function. |
| [Random functions in 2D](Random2D.md) | randnfun2 at space scales 0.2, 0.1, 0.05, plus the periodic variant and zero contours. |
| [Random ponds in a 2D landscape](RandomPonds.md) | Level sets of a random landscape filled to height h; percolation as h grows. |
| [2D zero set example of Warwick Tucker](Tucker.md) | The elegant zero set of sin(cos x² + 10 sin y²) − y cos x on [−5,5]². |
| [Zebra plots](Zebra.md) | Plus/minus 'zebra' plots on the disk, sphere, and rectangle. |
