# Inserting breakpoints to resolve layers

*Nick Trefethen, January 2016*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/ode-linear/Breakpoints.html)

Python translation: [`examples/ode-linear/breakpoints.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/ode-linear/breakpoints.py)

## 1. Boundary layer example

By default, Chebfun solves BVPs with global grids -- Chebyshev collocation spectral methods -- and this generally works well even for problems with rapidly varying solutions. Trouble appears, however, when the variations are *very* rapid. For example, following the example `ode-linear/BoundaryLayer`, here are solutions to the linear advection-diffusion equation $$ -\varepsilon u'' - u' = 1,\qquad u(0) = u(1) = 0 , $$ with $\varepsilon = 10^{-1}, 10^{-2},\dots, 10^{-5}$.

```matlab
MS = 'markersize';
dom = [0,1];
L = @(ep) chebop(@(x,u) -ep*diff(u,2) - diff(u),dom,'dirichlet');
headings = '        ep      pos(max(u))    length(u)    time (secs.) ';
disp(headings)
fs = '%12.1e %14.9f %9d %14.2f\n';
for ep = 10.^(-1:-1:-5)
  tic, u = L(ep)\1; t = toc;
  [val,pos] = max(u);
  fprintf(fs, ep, pos, length(u), t)
  plot(u,'b'), hold on
end
grid on, axis([-0.03 1 0 1.03]), hold off
title('Boundary layers for \epsilon = 1e-1, 1e-2,..., 1e-5')
```

```text
        ep      pos(max(u))    length(u)    time (secs.)
     1.0e-01    0.230263049        32          13.22
     1.0e-02    0.046051702        57           8.73
     1.0e-03    0.006907755       256           4.94
     1.0e-04    0.000921035       512           5.52
     1.0e-05    0.000115147      2048          11.64
```

![Breakpoints figure 01](../../images/ode-linear/Breakpoints_01.png)

The lengths and timings are excellent for the first three values of $\varepsilon$ and not so bad for $\varepsilon = 10^{-4}$, but for $\varepsilon = 10^{-5}$, we need a grid with thousands of points and the method cannot be regarded as satisfactory. (This boundary layer is of width $O(\varepsilon)$, but because of the quadratic clustering of Chebyshev grids at boundaries, the length of the chebfuns only grows like $O(\varepsilon^{-1/2})$.)

There is a standard method used in scientific computing for such problems, adaptive grid refinement, but Chebfun does not have such a capability. For many problems, however, it is remarkable what one can achieve by a method we might regard as "poor man's grid refinement": simply add a Chebfun breakpoint or two near the region of rapid change. To make this happen, it is enough to define the domain of the chebop by a vector of three or more points in order, i.e., the endpoints of an interval plus one or more points in the interior. For example, one might pass to the chebop constructor the domain vector `dom = [0 0.01 1]` rather than simply `dom = [0 1]`.

If an ODE BVP is solved on a domain with breakpoints, separate Chebyshev grids are used on subintervals, and that may provide a more efficient representation of the solution, which will then be a chebfun with several pieces, i.e., several "funs". For a discussion of some of the mathematics, see [1].

This is a non-adaptive, a priori approach. It cannot cope with a full range of problems, but it can do very well with many of them. For example, here is the same problem as before with a single breakpoint introduced at $x_b = 40\varepsilon$. Just one curve is plotted, the one with $\varepsilon = 10^{-3}$.

```matlab
dom = @(ep) [0 min(0.5,40*ep) 1];
L = @(ep) chebop(@(x,u) -ep*diff(u,2) - diff(u),dom(ep),'dirichlet');
disp(headings)
for ep = 10.^(-1:-1:-8)
  tic, u = L(ep)\1; t = toc;
  fprintf(fs, ep, pos, length(u), t)
  [val,pos] = max(u);
  if ep == 1e-3
    plot(u,'b'), hold on
    breakpoint = u.ends(2);
    plot(breakpoint,u(breakpoint),'.r',MS,16)
  end
end
grid on, axis([-0.03 1 0 1.03]), hold off
title('The same computed with a breakpoint, \epsilon = 1e-3')
```

```text
        ep      pos(max(u))    length(u)    time (secs.)
     1.0e-01    0.000115147        37           1.02
     1.0e-02    0.230263049        45           1.07
     1.0e-03    0.046051702        43           0.84
     1.0e-04    0.006907755        44           0.85
     1.0e-05    0.000921034        45           0.82
     1.0e-06    0.000115129        43           0.83
     1.0e-07    0.000013816        45           0.78
     1.0e-08    0.000001612       106           0.92
```

![Breakpoints figure 02](../../images/ode-linear/Breakpoints_02.png)

Quite an amazing improvement! Notice that the breakpoint at $x_b = 40 \varepsilon$ is well out of the boundary layer. The reason for this choice is that the purpose of the breakpoint is not to optimize the representation of $u$ within the small region $[0, x_b]$, where a reasonable number of gridpoints will be required in any case, but rather to ensure that $u$ has no significant structure on a small length scale in the big interval $[x_b,1]$. In fact, the second piece of each chebfun constructed above, the representation of $u$ on $[x_b ,1]$, is just of length 2, i.e., a linear polynomial:

```matlab
u
```

```text
u =
   chebfun column (2 smooth pieces)
       interval       length     endpoint values
[       0,   4e-07]       42   3.7e-08        1
[   4e-07,       1]       64         1  3.1e-16
vertical scale =   1    Total length = 106
```

## 2. Interior layer example

As our second example, we consider a linear problem with an interior layer: $$ \varepsilon u'' + xu' + xu = 0, \quad x \in [-2,2], ~ y(-2) = -4, y(2) = 2 . $$ This has an interior layer of width $O(\sqrt{\varepsilon}\kern 1pt)$ at $x=0$, which we can expect to challenge Chebfun as much as in the previous example, since the layer is thicker but the grid is no longer clustered. An experiment confirms this prediction:

```matlab
dom = [-2,2];
L = @(ep) chebop(@(x,u) ep*diff(u,2)+x*diff(u)+x*u,dom,-4,2);
disp(headings)
for ep = 10.^(-1:-1:-4)
  tic, u = L(ep)\0; t = toc;
  [val,pos] = max(u);
  fprintf(fs, ep, pos, length(u), t)
  plot(u,'m'), hold on
end
grid on, axis([-2 2 -6 17]), hold off
title('Interior layers for \epsilon = 1e-1, 1e-2,..., 1e-4')
```

```text
        ep      pos(max(u))    length(u)    time (secs.)
     1.0e-01    0.456331114        64          11.80
     1.0e-02    0.188033044       157           4.46
     1.0e-03    0.073657588       512           2.68
     1.0e-04    0.027481095      1397           5.47
```

![Breakpoints figure 03](../../images/ode-linear/Breakpoints_03.png)

Inserting breakpoints on either side of $x=0$ improves matters greatly. Again we plot just one of the curves, the one with $\varepsilon = 10^{-4}$.

```matlab
dom = @(ep) [-2 -min(.5,10*sqrt(ep)) min(.5,10*sqrt(ep)) 2];
L = @(ep) chebop(@(x,u) ep*diff(u,2)+x*diff(u)+x*u,dom(ep),-4,2);
disp(headings)
for ep = 10.^(-1:-1:-8)
  tic, u = L(ep)\0; t = toc;
  [val,pos] = max(u);
  fprintf(fs, ep, pos, length(u), t)
  if ep == 1e-4
    plot(u,'m'), hold on
    breakpoints = u.ends(2:3);
    plot(breakpoints,u(breakpoints),'.k',MS,16)
  end
end
grid on, axis([-2 2 -6 17]), hold off
title('The same computed with two breakpoints \epsilon = 1e-4')
```

```text
        ep      pos(max(u))    length(u)    time (secs.)
     1.0e-01    0.456331114        84           1.66
     1.0e-02    0.188033044       129           0.66
     1.0e-03    0.073657588       153           1.56
     1.0e-04    0.027481095       187           1.52
     1.0e-05    0.009892469       222           1.51
     1.0e-06    0.003473237       272           1.52
     1.0e-07    0.001198204       151           1.49
     1.0e-08    0.000408122       213           1.93
```

![Breakpoints figure 04](../../images/ode-linear/Breakpoints_04.png)

Here we see the sizes of the three pieces:

```matlab
u
```

```text
u =
   chebfun column (3 smooth pieces)
       interval       length     endpoint values
[      -2,  -0.001]       11        -4    -0.54
[  -0.001,   0.001]       74     -0.54       15
[   0.001,       2]      128        15        2
vertical scale =  15    Total length = 213
```

## 3. A nonlinear example

"Poor man's mesh refinement" is not restricted to linear problems. For a nonlinear problem with an interior layer, one may not know the location of an interior layer a priori, but an approximation may be enough for the method to work. For example, here is a nonlinear problem with an interior layer adapted from one of the chebgui demos: $$ 0.005u'' + uu' = u, \quad u(0) = -7/6, ~ u(1) = 3/2. $$ A global solution succeeds, but very slowly, for big matrices are involved:

```matlab
N = chebop(@(u) 0.005*diff(u,2) + u*diff(u) - u, [0 1]);
N.lbc = -7/6; N.rbc = 3/2;
tic, u = N\0, t = toc;
plot(u), grid on
title(['Nonlinear problem: time ' num2str(t) ' secs'])
```

```text
u =
   chebfun column (1 smooth piece)
       interval       length     endpoint values
[       0,       1]      862      -1.2      1.5
vertical scale = 1.5
```

![Breakpoints figure 05](../../images/ode-linear/Breakpoints_05.png)

The transition occurs at about $x=1/3$, and if we put a single breakpoint there, the computations becomes five times faster.

```matlab
N = chebop(@(u) 0.005*diff(u,2) + u*diff(u) - u, [0 1/3 1]);
N.lbc = -7/6; N.rbc = 3/2;
tic, u = N\0, t = toc
plot(u), grid on
title(['Same but with one breakpoint: time ' num2str(t) ' secs'])
breakpoint = u.ends(2); hold on
plot(breakpoint,u(breakpoint),'.r',MS,16), hold off
```

```text
u =
   chebfun column (2 smooth pieces)
       interval       length     endpoint values
[       0,    0.33]      106      -1.2  2.2e-14
[    0.33,       1]      149   1.5e-14      1.5
vertical scale = 1.5    Total length = 255
t =
  57.403378009796143
```

![Breakpoints figure 06](../../images/ode-linear/Breakpoints_06.png)

Note that the matrix size is now a good smaller, which is the main reason for the speedup.

With two breakpoints, for this particular example, not much changes.

```matlab
N = chebop(@(u) 0.005*diff(u,2) + u*diff(u) - u, [0 .30 .36 1]);
N.lbc = -7/6; N.rbc = 3/2;
tic, u = N\0, t = toc
plot(u), grid on
title(['Same but with two breakpoints: time ' num2str(t) ' secs'])
breakpoints = u.ends(2:3); hold on
plot(breakpoints,u(breakpoints),'.r',MS,16), hold off
```

```text
u =
   chebfun column (3 smooth pieces)
       interval       length     endpoint values
[       0,     0.3]       38      -1.2    -0.79
[     0.3,    0.36]       38     -0.79      0.5
[    0.36,       1]      128       0.5      1.5
vertical scale = 1.5    Total length = 204
t =
  71.505528211593628
```

![Breakpoints figure 07](../../images/ode-linear/Breakpoints_07.png)

As the illustrations of this Example probably make clear, inserting breakpoints is bit of an art, and some experimentation is generally worthwhile.

## 4. Reference

[1] T. A. Driscoll and J. A. C. Weideman, Optimal domain splitting for interpolation by Chebyshev polynomials, *SIAM J. Numer. Anal.* 52 (2014), 1913-1927.

---

*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); prose and MATLAB code from the original example, copyright The University of Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*
