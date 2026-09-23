# Random level hopping

*Nick Trefethen, May 2017*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/ode-random/LevelHopping.html)

Python translation: [`examples/ode-random/levelhopping.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/ode-random/levelhopping.py)

The equation $y' = -2\sin(2\pi y)$ has stable fixed points when $y$ is an integer. Let us add some noise, so that we have $$ y' = -2\sin(2\pi y) + f, $$ where $f$ is a random function. This gives us a process that hops from one fixed point to another. We illustrate first for $t\in [0,100]$ with $\lambda = 0.4$.

```matlab
rng(0), dom = [0 100]; tic
N = chebop(dom);
lambda = 0.4; f = randnfun(lambda,dom,'norm');
N.op = @(y) diff(y) + 2*sin(2*pi*y); N.lbc = 0;
LW = 'linewidth'; FS = 'fontsize';
y = N\f; plot(y,LW,2), grid on
xlabel('t',FS,32), ylabel('y',FS,32)
```

![LevelHopping figure 01](../../images/ode-random/LevelHopping_01.png)

Here we cut $\lambda$ in half.

```matlab
lambda = lambda/2;
f = randnfun(lambda,dom,'norm');
y = N\f; plot(y,LW,1), grid on
xlabel('t',FS,32), ylabel('y',FS,32)
```

![LevelHopping figure 02](../../images/ode-random/LevelHopping_02.png)

```matlab
total_time_in_seconds = toc
```

```text
total_time_in_seconds =
  456.596930
```

---

*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); prose and MATLAB code from the original example, copyright The University of Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*
