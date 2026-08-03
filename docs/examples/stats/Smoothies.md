# Smoothies: nowhere-analytic functions

*Nick Trefethen, May 2020*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/stats/Smoothies.html)

(Chebfun example stats/Smoothies.m)

A *smoothie* is a random function that is $C^\infty$ but nowhere
analytic: a random Fourier series whose coefficients decay
root-exponentially — fast enough for infinite differentiability, too
slowly for analyticity.

![Smoothies figure 1](../../images/stats/Smoothies_repl_01.png)

The Chebyshev coefficients show clean root-exponential decay:

![Smoothies figure 2](../../images/stats/Smoothies_repl_02.png)

A periodic smoothie and its (two-sided) Fourier coefficients:

![Smoothies figure 3](../../images/stats/Smoothies_repl_03.png)

![Smoothies figure 4](../../images/stats/Smoothies_repl_04.png)

A complex smoothie traces a nowhere-analytic curve in the plane:

![Smoothies figure 5](../../images/stats/Smoothies_repl_05.png)

Derivatives of a smoothie are smoothies too, growing roughly like
the powers of the wavenumber — here the first and second
derivatives:

![Smoothies figure 6](../../images/stats/Smoothies_repl_06.png)

(`randn` draws are not reproducible across systems; the smoothies are
our own draws from the same distribution.)

---

*Replicated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); original
example copyright The University of Oxford and The Chebfun Developers.*
