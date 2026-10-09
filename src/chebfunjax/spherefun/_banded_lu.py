"""Scratch JAX band elimination for the source sphere Fourier systems.

Source equations: Chebfun7574c77 @spherefun/{poisson,helmholtz}.m.
This is explicit Gaussian elimination with row pivoting, not a claim of
identical MATLAB sparse-backslash ordering. All input bands are retained.
"""
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=('lower', 'upper'))
def solve_banded(ab, rhs, *, lower, upper):
    """Row storage ab[i, j-i+lower], with upper fill space preallocated.

    Input shape(n,2*lower+upper+1); final lower extra columns are zero fill
    storage initially. Return(solution, singular-pivot flag). No flooring,
    thresholding, or fallback. Vector or matrix RHS is supported.
    """
    ab = jnp.asarray(ab)
    rhs = jnp.asarray(rhs)
    dtype = jnp.result_type(ab, rhs, jnp.float64)
    ab, rhs = ab.astype(dtype), rhs.astype(dtype)
    n = ab.shape[0]
    if n < 1 or lower < 0 or upper < 0 or ab.shape != (n,2*lower+upper+1):
        raise ValueError('invalid band storage')
    if rhs.ndim not in (1,2) or rhs.shape[0] != n:
        raise ValueError('invalid RHS shape')
    vector = rhs.ndim == 1
    b = rhs[:,None] if vector else rhs
    reach = lower+upper
    width = ab.shape[1]
    offsets = jnp.arange(reach+1)
    low = jnp.arange(1,lower+1)

    def forward(k, state):
        a,b,bad = state
        rows = k+jnp.arange(lower+1)
        candidates = a[jnp.minimum(rows,n-1), k-rows+lower]
        candidates = jnp.where(rows<n, jnp.abs(candidates), -jnp.inf)
        pivot = k+jnp.argmax(candidates)
        cols = k+offsets
        validcols = cols<n
        kp = jnp.clip(cols-pivot+lower,0,width-1)
        kk = cols-k+lower
        rowk = a[k,kk]
        rowp = a[pivot,kp]
        a = a.at[jnp.where(validcols,k,n),kk].set(rowp,mode='drop')
        a = a.at[jnp.where(validcols,pivot,n),kp].set(rowk,mode='drop')
        bk,bp = b[k],b[pivot]
        b = b.at[k].set(bp).at[pivot].set(bk)
        diagonal = a[k,lower]
        bad = bad | (diagonal == 0)
        rows = k+low
        safe_rows = jnp.minimum(rows,n-1)
        factors = jnp.where(rows<n,a[safe_rows,lower-low]/diagonal,0)
        slots = cols[None,:]-rows[:,None]+lower
        safe_slots = jnp.clip(slots,0,width-1)
        values = a[safe_rows[:,None],safe_slots]-factors[:,None]*a[k,kk][None,:]
        values = values.at[:,0].set(0)
        valid = (rows[:,None]<n)&validcols[None,:]&(slots>=0)&(slots<width)
        a = a.at[jnp.where(valid,rows[:,None],n),safe_slots].set(values,mode='drop')
        b = b.at[jnp.where(rows<n,rows,n)].set(
            b[safe_rows]-factors[:,None]*b[k],mode='drop')
        return a,b,bad

    a,b,bad = jax.lax.fori_loop(0,n,forward,(ab,b,jnp.asarray(False)))
    steps = jnp.arange(1,reach+1)
    def backward(t,x):
        k = n-1-t
        cols = k+steps
        terms = a[k,lower+steps,None]*x[jnp.minimum(cols,n-1)]
        total = jnp.sum(jnp.where((cols<n)[:,None],terms,0),axis=0)
        return x.at[k].set((b[k]-total)/a[k,lower])
    x = jax.lax.fori_loop(0,n,backward,jnp.zeros_like(b))
    return (x[:,0] if vector else x),bad
