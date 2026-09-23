# Phase-locking in a Duffing-type equation

*Kevin Burrage and Nick Trefethen, May 2017*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/ode-random/PhaseLocking.html)

Python translation: [`examples/ode-random/phaselocking.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/ode-random/phaselocking.py)

Consider the bistable equation $y' = ty - y^3 + f$, where $f$ is a random term of fixed amplitude. The fixed points of the deterministic part of the equation, locally at a time $t$, are $\pm |t|^{1/2}$. For small $t$, noise easily crosses this gap, but as $t$ gets larger any trajectory eventually settles down to a choice that is (almost surely) fixed forever. First we use $\lambda = 0.2$.

```matlab
tic, dom = [0 6]; N = chebop(dom); rng(0)
N.lbc = 0; N.op = @(t,y) diff(y) - t*y + y^3;
for k = 1:6
  f = randnfun(0.2,dom,'big');
  y = N\f; plot(y), hold on
end
xlabel('t'), ylabel('y')
title('lambda = 0.2, 6 paths'), toc
```

```text
PhaseLocking_01.png: 162.8s
```

![PhaseLocking figure 01](../../images/ode-random/PhaseLocking_01.png)

Here's the same computation with $\lambda = 0.05$.

```matlab
tic, clf
for k = 1:6
  f = randnfun(0.05,dom,'big');
  y = N\f; plot(y), hold on
end
xlabel('t'), ylabel('y')
title('lambda = 0.05, 6 paths'), toc
```

```text
PhaseLocking_01.png: 162.8s
```

![PhaseLocking figure 02](../../images/ode-random/PhaseLocking_02.png)

Here's a much bigger sample.

```matlab
tic, clf
for k = 1:60
  f = randnfun(0.05,dom,'big');
  y = N\f; plot(y), hold on
end
xlabel('t'), ylabel('y')
title('lambda = 0.05, 60 paths'), toc
```

```text
PhaseLocking_01.png: 162.8s
```

![PhaseLocking figure 03](../../images/ode-random/PhaseLocking_03.png)

---

*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); prose and MATLAB code from the original example, copyright The University of Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*
