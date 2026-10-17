"""Compensated real Airy phase, evaluated entirely with JAX arrays.

TwoSum and the split-product error-free transformation follow the mathematics
of Ogita, Rump and Oishi, SIAM J. Sci. Comput. 26(6), 1955--1988 (2005),
doi:10.1137/030601818. This independently written instruction schedule keeps
one arithmetic operation per dynamic loop iteration: XLA must not contract
the multiply/subtract pairs used to recover rounding residuals.

The square-root correction uses (t-r*r)/(2*r), retaining the product residual;
the division-by-three correction similarly retains the quotient residual.
These produce an approximate doubleword phase, not exact transcendental
arithmetic. The caller supplies positive finite t and preserves its existing
nonfinite-output behavior. Mathematical accuracy is distinct from MATLAB
builtin bit parity, especially for large negative Airy arguments.
"""
import jax
import jax.numpy as jnp

# Binary64 pi/4 and its rounded residual, not fitted function values.
PI4_HI = float.fromhex('0x1.921fb54442d18p-1')
PI4_LO = float.fromhex('0x1.1a62633145c07p-55')

class _Program:

    def __init__(self):
        self.steps = []
        self.count = 8

    def op(self, k, a, b=1):
        i = self.count
        self.count += 1
        self.steps.append((k, i, a, b))
        return i

    def add(self, a, b):
        return self.op(1, a, b)

    def sub(self, a, b):
        return self.op(2, a, b)

    def mul(self, a, b):
        return self.op(0, a, b)

    def div(self, a, b):
        return self.op(3, a, b)

    def sum(self, a, b):
        s = self.add(a, b)
        bb = self.sub(s, a)
        e = self.add(self.sub(a, self.sub(s, bb)), self.sub(b, bb))
        return (s, e)

    def split(self, a):
        # Split the normalized mantissa to avoid splitter overflow.
        m = self.op(5, a)
        e = self.op(6, a)
        c = self.mul(5, m)
        big = self.sub(c, m)
        hi = self.sub(c, big)
        lo = self.sub(m, hi)
        return (self.op(7, hi, e), self.op(7, lo, e))

    def prod(self, a, b):
        p = self.mul(a, b)
        (ah, al) = self.split(a)
        (bh, bl) = self.split(b)
        e1 = self.sub(p, self.mul(ah, bh))
        e2 = self.sub(e1, self.mul(al, bh))
        e3 = self.sub(e2, self.mul(ah, bl))
        return (p, self.sub(self.mul(al, bl), e3))
# Registers 0..7: t, zero, one, two, three, splitter, pi/4 high, low.
P = _Program()
r = P.op(4, 0)
(sq, sqe) = P.prod(r, r)
rl = P.div(P.sub(P.sub(0, sq), sqe), P.mul(3, r))
(p, pe) = P.prod(0, r)
pl = P.add(pe, P.mul(0, rl))
(p, pl) = P.sum(p, pl)
q = P.div(p, 4)
(q3, q3e) = P.prod(q, 4)
ql = P.div(P.add(P.sub(P.sub(p, q3), q3e), pl), 4)
(q, ql) = P.sum(q, ql)
zh = P.mul(3, q)
zl = P.mul(3, ql)
(h, e) = P.sum(zh, P.sub(1, 6))
l = P.add(e, P.sub(zl, 7))
(h, l) = P.sum(h, l)
OUTPUTS = (h, l, zh, zl, r, rl, p, pl)

@jax.jit
def _phase_words(t):
    t = jnp.asarray(t, jnp.float64)
    shape = t.shape
    z = jnp.zeros_like(t)
    constants = [t, z, jnp.ones_like(t), jnp.full_like(t, 2.0), jnp.full_like(t, 3.0), jnp.full_like(t, 134217729.0), jnp.full_like(t, PI4_HI), jnp.full_like(t, PI4_LO)]
    regs = jnp.concatenate((jnp.stack(constants), jnp.zeros((P.count - 8,) + shape)), axis=0)

    def instruction(reg, code):
        (opcode, dst, a, b) = code
        (x, y) = (reg[a], reg[b])
        value = jax.lax.switch(opcode, (lambda : x * y, lambda : x + y, lambda : x - y, lambda : x / y, lambda : jnp.sqrt(x), lambda : jnp.frexp(x)[0], lambda : jnp.frexp(x)[1].astype(jnp.float64), lambda : jnp.ldexp(x, y.astype(jnp.int32))))
        return (reg.at[dst].set(value), None)
    (regs, _) = jax.lax.scan(instruction, regs, jnp.asarray(P.steps, jnp.int32), unroll=False)
    return tuple((regs[i] for i in OUTPUTS))
