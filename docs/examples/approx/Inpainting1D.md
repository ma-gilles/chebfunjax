# L1 inpainting in one dimension

*Yuji Nakatsukasa and Nick Trefethen, July 2019*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/approx/Inpainting1D.html)

Python translation: [`examples/approx/inpainting1d.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/approx/inpainting1d.py)

> **Qualification:** This source computation uses public nonperiodic `cj.randnfun(.1, key=PRNGKey(1))`; its normal draws do not reproduce MATLAB `rng(1)`. All three public fits receive the same captured nine-piece corrupted function. The source Watson L1 iteration reached its unchanged 100-iteration limit and emitted the warning below; its actual recovery error is `2.166633346778370e-05`, so the native prose’s near-zero recovery is not established. L2 and minimax values use this different noise realization. Native prose/code are retained for comparison. All five figures are captured at the reference 600×270 sizes; curves, limits, ticks, grid and line styling differ. A MATLAB replay using the saved piece coefficients also reaches 100 iterations, with recovery error `7.061517343020836e-06`. The initial polynomial coefficients differ by at most `1.12e-16`; damping/root-count choices first diverge at iteration 27. This does not establish numerical or visual parity.

Here is a smooth function corrupted in three regions:

```matlab
tic
x = chebfun('x');
smooth = .3 + x^2 + exp(.3*x);
rng(1), noise = randnfun(.1);
corrupted = max(smooth,noise);
plot(corrupted), grid on
title('corrupted smooth function')
```

![Inpainting1D figure 01](../../images/approx/Inpainting1D_01.png)

If we fit the function by a low-order polynomial in the $L^1$ norm, we can eliminate the corruption! This is a 1D version of what is called *inpainting*.

```matlab
n = length(smooth)-3;
p1 = polyfitL1(corrupted,n);
plot(p1), grid on, title('L1 fit')
```

```text
RuntimeWarning: CHEBFUN:POLYFIT:MAXITER: The maximum number of iterations was reach. Answer may not be accurate.
```

![Inpainting1D figure 02](../../images/approx/Inpainting1D_02.png)

The error is very small and would in principle be zero if we used a polynomial of the same degree as the function being recovered:

```matlab
err1 = norm(p1-smooth,inf)
```

```text
err1 =
     2.166633346778370e-05
```

The 2-norm has no such magic. The fit looks pretty good to the eye,

```matlab
p2 = polyfit(corrupted,n-2);
plot(p2), grid on, title('L2 fit')
err2 = norm(p2-smooth,inf)
```

```text
err2 =
   0.088886081865430
```

![Inpainting1D figure 03](../../images/approx/Inpainting1D_03.png)

but now the error is actually far from zero:

```matlab
plot(p2-smooth,'k'), grid on, title('L2 error')
```

![Inpainting1D figure 04](../../images/approx/Inpainting1D_04.png)

The $\infty$-norm is useless for our purpose:

```matlab
pinf = minimax(corrupted,n-2);
plot(pinf), grid on, title('Linf fit')
errinf = norm(pinf-smooth,inf)
```

```text
errinf =
   0.535844974246717
```

![Inpainting1D figure 05](../../images/approx/Inpainting1D_05.png)

We've called this example "1D inpainting" because it is a 1D version of the famous "inpainting" problem in image analysis. The tools used for that problem are varied and powerful, using everything from function approximation to partial differential equations to machine learning; what we have done here is only a small indication of some of the mathematics that may come into play. Note that the issue at hand is "sparsity" of the difference between the corrupted signal and its inpainted polynomial approximation. The $L^1$ norm comes up in many problems related to sparsity -- famously in the area of compressed sensing -- since it is an approximation to the $L^0$ "norm" (not actually a norm).

Our $L^1$ computation was quite slow:

```matlab
toc
```

```text
Elapsed time is 109.583760 seconds.
```

(Virtually all the time was taken by `polyfitL1`; in comparison `polyfit` and `minimax` are almost instantaneous.) This is partly because $L^1$ fitting is challenging, but equally because Chebfun's `polyfitL1` command avoids the tool that could speed it up considerably, namely linear programming. This is because linear programming is not available in core Matlab.

For details of $L^1$ fitting in Chebfun, see [1], which is based on an algorithm by Watson [2], and also the Chebfun example "Best polynomial approximation in the $L^1$ norm.

[1] Y. Nakatsukasa and A. Townsend, Error localization of best L1 polynomial approximants, SIAM J. Numer. Anal, 59 (2021), 314--333.

[2] G. A. Watson. An algorithm for linear L1 approximation of continuous functions, *IMA J. Numer. Anal.*, 1 (1981), 157--167.

---

*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); prose and MATLAB code from the original example, copyright The University of Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*
