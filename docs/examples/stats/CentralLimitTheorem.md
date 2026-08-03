# Central limit theorem

*Nick Trefethen, June 2015*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/stats/CentralLimitTheorem.html)

(Chebfun example stats/CentralLimitTheorem.m)

The central limit theorem says that sums of independent identically
distributed random variables, renormalized, converge to a Gaussian.
Here is a triangular probability distribution:

![CentralLimitTheorem figure 1](../../images/stats/CentralLimitTheorem_repl_01.png)

Its mean and variance, from chebfun integrals — matching MATLAB's
printed values to the last digit:

```text
mu =
    8.326672684688674e-17
variance =
   0.222222222222223
```

The convolution `X.conv(X)`, renormalized, is already noticeably
closer to the Gaussian of the same variance:

![CentralLimitTheorem figure 3](../../images/stats/CentralLimitTheorem_repl_03.png)

And after three convolutions:

![CentralLimitTheorem figure 4](../../images/stats/CentralLimitTheorem_repl_04.png)

The discrete analogue: a biased-coin Bernoulli distribution is a pair
of Dirac deltas, and convolving it with itself repeatedly generates
the binomial distribution — the delta arithmetic is exact:

![CentralLimitTheorem figure 6](../../images/stats/CentralLimitTheorem_repl_06.png)

```text
ans =
     1
ans =
   1.000000000000000
mu =
     6
sigma =
   1.549193338482967
```

After ten tosses the binomial sticks hug the Gaussian envelope:

![CentralLimitTheorem figure 8](../../images/stats/CentralLimitTheorem_repl_08.png)

(All printed values digit-for-digit with the published MATLAB run.)

---

*Replicated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); original
example copyright The University of Oxford and The Chebfun Developers.*
