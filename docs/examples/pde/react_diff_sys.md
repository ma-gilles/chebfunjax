# Coupled Reaction-Diffusion System

**Source:** `pde/ReactDiffSys.m` — Nick Hale, October 2010
**Python:** `examples/pde/react_diff_sys.py`
**Original MATLAB:** https://github.com/chebfun/examples/blob/master/pde/ReactDiffSys.m

## Problem

Three-component reaction-diffusion system on `[-1, 1]` with Neumann BCs:

```
u_t = 0.1 u_xx - 100 uv
v_t = 0.2 v_xx - 100 uv
w_t = 0.001 w_xx + 200 uv
```

Chemicals `u` and `v` diffuse and react to produce product `w`.

## Initial conditions

```python
u0 = 1 - erf(10*(x + 0.7))    # concentrated near x = -1
v0 = 1 + erf(10*(x - 0.7))    # concentrated near x = +1
w0 = 0                          # no product initially
```

## Method

The reaction term `100 uv` is fast (stiff), so a stiff solver is needed.
Uses `scipy.integrate.solve_ivp` with `method='BDF'`:

```python
sol = solve_ivp(rhs, [0.0, T], uvw0, method='BDF',
                t_eval=t_snap, rtol=1e-5, atol=1e-7)
```

## Results

By `T = 2.0`:
- Reactants `u` and `v` have been consumed in the overlap region
- Product `w` has grown (max value ~1.2)
- Total reactant `u + v` decreases as reaction proceeds

## Plots

![React-diff sys](../../images/pde/react_diff_sys.png)

Left: initial conditions. Right: final state at T=2 showing depletion
of reactants and formation of product.
