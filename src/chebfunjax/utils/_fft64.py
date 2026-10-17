"""Ordered inverse Fourier butterflies for a real single-column length 64 input.

The caller supplies an even real sequence. C2R preprocessing uses its first
33 entries. The inverse 32 kernel uses real/imaginary pairs and an unnormalized
positive Fourier exponent; division by 64 recovers the inverse 64 transform.

Provenance: Chebfun 7574c77 @chebtech2/vals2coeffs.m calls MATLAB IFFT.
The single-column CPU plan in MATLAB R2025b is C2R DIF2 followed by
n1bv_32_avx. Its separate add/subtract/multiply ordering was established
from executed-plan/entry observations and inspected machine arithmetic;
this module expresses that arithmetic as a JAX butterfly. Independent pairs
are evaluated in 22 groups across nine dependency levels. Per-pair vmap
preserves each ordered-binary decision and every dependency edge.
No native binary, captured operands, or reference outputs are loaded.

The seven butterfly constants below are algebraic half-angle expressions,
evaluated at 80 decimal digits and rounded once to binary64. An exact audit
in Q[z]/(z**16+1) proves the ideal 32 transform for every complex input.
Independent native comparisons qualify the real single-column 64 route.
Native batches of four use a different direct 64 plan, outside this scope.
"""

import math
from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp

from chebfunjax.utils._ordered_binary import binary

with localcontext() as ctx:
    ctx.prec = 80
    one = Decimal(1)
    two = Decimal(2)
    c4 = two.sqrt() / two
    c8 = ((one + c4) / two).sqrt()
    s8 = ((one - c4) / two).sqrt()
    c16 = ((one + c8) / two).sqrt()
    s16 = ((one - c8) / two).sqrt()
    c3 = ((one + s8) / two).sqrt()
    s3 = ((one - s8) / two).sqrt()
    CONSTANTS = tuple(map(float, (c4, s8, c8, c3, s3, c16, s16)))


@jax.jit
def inverse32_pairs(x):
    "Unnormalized inverse DFT of 32 complex values stored as real pairs."
    (C4, S8, C8, C3_16, S3_16, C16, S16) = (jnp.asarray(z, jnp.float64) for z in CONSTANTS)
    stage0 = jax.vmap(lambda a, b: binary(a, b, 1))(
        jnp.stack(
            (
                jnp.stack((x[0, 0], x[0, 1])),
                jnp.stack((x[8, 0], x[8, 1])),
                jnp.stack((x[4, 0], x[4, 1])),
                jnp.stack((x[12, 0], x[12, 1])),
                jnp.stack((x[19, 0], x[19, 1])),
                jnp.stack((x[15, 0], x[15, 1])),
                jnp.stack((x[11, 0], x[11, 1])),
                jnp.stack((x[23, 0], x[23, 1])),
                jnp.stack((x[21, 0], x[21, 1])),
                jnp.stack((x[17, 0], x[17, 1])),
                jnp.stack((x[13, 0], x[13, 1])),
                jnp.stack((x[25, 0], x[25, 1])),
                jnp.stack((x[18, 0], x[18, 1])),
                jnp.stack((x[14, 0], x[14, 1])),
                jnp.stack((x[26, 0], x[26, 1])),
                jnp.stack((x[22, 0], x[22, 1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((x[16, 0], x[16, 1])),
                jnp.stack((x[24, 0], x[24, 1])),
                jnp.stack((x[20, 0], x[20, 1])),
                jnp.stack((x[28, 0], x[28, 1])),
                jnp.stack((x[3, 0], x[3, 1])),
                jnp.stack((x[31, 0], x[31, 1])),
                jnp.stack((x[27, 0], x[27, 1])),
                jnp.stack((x[7, 0], x[7, 1])),
                jnp.stack((x[5, 0], x[5, 1])),
                jnp.stack((x[1, 0], x[1, 1])),
                jnp.stack((x[29, 0], x[29, 1])),
                jnp.stack((x[9, 0], x[9, 1])),
                jnp.stack((x[2, 0], x[2, 1])),
                jnp.stack((x[30, 0], x[30, 1])),
                jnp.stack((x[10, 0], x[10, 1])),
                jnp.stack((x[6, 0], x[6, 1])),
            )
        ),
    )
    t1 = stage0[0]
    t3 = stage0[1]
    t7 = stage0[2]
    t9 = stage0[3]
    t21 = stage0[4]
    t23 = stage0[5]
    t25 = stage0[6]
    t27 = stage0[7]
    t48 = stage0[8]
    t50 = stage0[9]
    t52 = stage0[10]
    t56 = stage0[11]
    t75 = stage0[12]
    t77 = stage0[13]
    t79 = stage0[14]
    t84 = stage0[15]
    stage1 = jax.vmap(lambda a, b: binary(a, b, 2))(
        jnp.stack(
            (
                jnp.stack((x[0, 0], x[0, 1])),
                jnp.stack((x[8, 0], x[8, 1])),
                jnp.stack((x[4, 0], x[4, 1])),
                jnp.stack((x[28, 0], x[28, 1])),
                jnp.stack((x[3, 0], x[3, 1])),
                jnp.stack((x[31, 0], x[31, 1])),
                jnp.stack((x[27, 0], x[27, 1])),
                jnp.stack((x[7, 0], x[7, 1])),
                jnp.stack((x[5, 0], x[5, 1])),
                jnp.stack((x[1, 0], x[1, 1])),
                jnp.stack((x[29, 0], x[29, 1])),
                jnp.stack((x[9, 0], x[9, 1])),
                jnp.stack((x[2, 0], x[2, 1])),
                jnp.stack((x[30, 0], x[30, 1])),
                jnp.stack((x[10, 0], x[10, 1])),
                jnp.stack((x[6, 0], x[6, 1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((x[16, 0], x[16, 1])),
                jnp.stack((x[24, 0], x[24, 1])),
                jnp.stack((x[20, 0], x[20, 1])),
                jnp.stack((x[12, 0], x[12, 1])),
                jnp.stack((x[19, 0], x[19, 1])),
                jnp.stack((x[15, 0], x[15, 1])),
                jnp.stack((x[11, 0], x[11, 1])),
                jnp.stack((x[23, 0], x[23, 1])),
                jnp.stack((x[21, 0], x[21, 1])),
                jnp.stack((x[17, 0], x[17, 1])),
                jnp.stack((x[13, 0], x[13, 1])),
                jnp.stack((x[25, 0], x[25, 1])),
                jnp.stack((x[18, 0], x[18, 1])),
                jnp.stack((x[14, 0], x[14, 1])),
                jnp.stack((x[26, 0], x[26, 1])),
                jnp.stack((x[22, 0], x[22, 1])),
            )
        ),
    )
    t0 = stage1[0]
    t2 = stage1[1]
    t6 = stage1[2]
    t8 = stage1[3]
    t20 = stage1[4]
    t22 = stage1[5]
    t24 = stage1[6]
    t26 = stage1[7]
    t47 = stage1[8]
    t49 = stage1[9]
    t51 = stage1[10]
    t55 = stage1[11]
    t74 = stage1[12]
    t76 = stage1[13]
    t78 = stage1[14]
    t83 = stage1[15]
    stage2 = jax.vmap(lambda a, b: binary(a, b, 0))(
        jnp.stack(
            (
                jnp.stack((t74[0], t74[1])),
                jnp.stack((t76[0], t76[1])),
                jnp.stack((t74[0], t74[1])),
                jnp.stack((t76[0], t76[1])),
                jnp.stack((t78[0], t78[1])),
                jnp.stack((t78[0], t78[1])),
                jnp.stack((t83[0], t83[1])),
                jnp.stack((t83[0], t83[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((S8, S8)),
                jnp.stack((S8, S8)),
                jnp.stack((C8, C8)),
                jnp.stack((C8, C8)),
                jnp.stack((C8, C8)),
                jnp.stack((S8, S8)),
                jnp.stack((C8, C8)),
                jnp.stack((S8, S8)),
            )
        ),
    )
    t82 = stage2[0]
    t85 = stage2[1]
    t86 = stage2[2]
    t87 = stage2[3]
    t93 = stage2[4]
    t95 = stage2[5]
    t100 = stage2[6]
    t101 = stage2[7]
    stage3 = jax.vmap(lambda a, b: binary(a, b, 1))(
        jnp.stack(
            (
                jnp.stack((t1[0], t1[1])),
                jnp.stack((t6[0], t6[1])),
                jnp.stack((t7[0], t7[1])),
                jnp.stack((t20[0], t20[1])),
                jnp.stack((t23[0], t23[1])),
                jnp.stack((t21[0], t21[1])),
                jnp.stack((t48[0], t48[1])),
                jnp.stack((t47[0], t47[1])),
                jnp.stack((t50[0], t50[1])),
                jnp.stack((t75[0], t75[1])),
                jnp.stack((t77[0], t77[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((t3[0], t3[1])),
                jnp.stack((t8[0], t8[1])),
                jnp.stack((t9[0], t9[1])),
                jnp.stack((t24[0], t24[1])),
                jnp.stack((t27[0], t27[1])),
                jnp.stack((t25[0], t25[1])),
                jnp.stack((t52[0], t52[1])),
                jnp.stack((t51[0], t51[1])),
                jnp.stack((t56[0], t56[1])),
                jnp.stack((t79[0], t79[1])),
                jnp.stack((t84[0], t84[1])),
            )
        ),
    )
    t5 = stage3[0]
    t11 = stage3[1]
    t15 = stage3[2]
    t29 = stage3[3]
    t32 = stage3[4]
    t38 = stage3[5]
    t53 = stage3[6]
    t59 = stage3[7]
    t60 = stage3[8]
    t80 = stage3[9]
    t88 = stage3[10]
    stage4 = jax.vmap(lambda a, b: binary(a, b, 2))(
        jnp.stack(
            (
                jnp.stack((t1[0], t1[1])),
                jnp.stack((t6[0], t6[1])),
                jnp.stack((t7[0], t7[1])),
                jnp.stack((t20[0], t20[1])),
                jnp.stack((t23[0], t23[1])),
                jnp.stack((t21[0], t21[1])),
                jnp.stack((t48[0], t48[1])),
                jnp.stack((t47[0], t47[1])),
                jnp.stack((t50[0], t50[1])),
                jnp.stack((t75[0], t75[1])),
                jnp.stack((t77[0], t77[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((t3[0], t3[1])),
                jnp.stack((t8[0], t8[1])),
                jnp.stack((t9[0], t9[1])),
                jnp.stack((t24[0], t24[1])),
                jnp.stack((t27[0], t27[1])),
                jnp.stack((t25[0], t25[1])),
                jnp.stack((t52[0], t52[1])),
                jnp.stack((t51[0], t51[1])),
                jnp.stack((t56[0], t56[1])),
                jnp.stack((t79[0], t79[1])),
                jnp.stack((t84[0], t84[1])),
            )
        ),
    )
    t4 = stage4[0]
    t10 = stage4[1]
    t14 = stage4[2]
    t28 = stage4[3]
    t33 = stage4[4]
    t39 = stage4[5]
    t54 = stage4[6]
    t57 = stage4[7]
    t61 = stage4[8]
    t81 = stage4[9]
    t89 = stage4[10]
    stage5 = jax.vmap(lambda a, b: binary(a, b, 0))(
        jnp.stack(
            (
                jnp.stack((t10[0], t10[1])),
                jnp.stack((t11[0], t11[1])),
                jnp.stack((t28[0], t28[1])),
                jnp.stack((t29[0], t29[1])),
                jnp.stack((t33[0], t33[1])),
                jnp.stack((t39[0], t39[1])),
                jnp.stack((t39[0], t39[1])),
                jnp.stack((t33[0], t33[1])),
                jnp.stack((t57[0], t57[1])),
                jnp.stack((t54[0], t54[1])),
                jnp.stack((t61[0], t61[1])),
                jnp.stack((t54[0], t54[1])),
                jnp.stack((t61[0], t61[1])),
                jnp.stack((t59[0], t59[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((C4, C4)),
                jnp.stack((C4, C4)),
                jnp.stack((C4, C4)),
                jnp.stack((C4, C4)),
                jnp.stack((C8, C8)),
                jnp.stack((S8, S8)),
                jnp.stack((C8, C8)),
                jnp.stack((S8, S8)),
                jnp.stack((C4, C4)),
                jnp.stack((S8, S8)),
                jnp.stack((C8, C8)),
                jnp.stack((C8, C8)),
                jnp.stack((S8, S8)),
                jnp.stack((C4, C4)),
            )
        ),
    )
    t12 = stage5[0]
    t13 = stage5[1]
    t30 = stage5[2]
    t31 = stage5[3]
    t41 = stage5[4]
    t42 = stage5[5]
    t43 = stage5[6]
    t44 = stage5[7]
    t58 = stage5[8]
    t65 = stage5[9]
    t66 = stage5[10]
    t67 = stage5[11]
    t68 = stage5[12]
    t69 = stage5[13]
    stage6 = jax.vmap(lambda a, b: binary(a, b, 1))(
        jnp.stack(
            (
                jnp.stack((t81[0], t81[1])),
                jnp.stack((t82[0], t82[1])),
                jnp.stack((t87[0], t87[1])),
                jnp.stack((t60[0], t60[1])),
                jnp.stack((t80[0], t80[1])),
                jnp.stack((t15[0], t15[1])),
                jnp.stack((t38[0], t38[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((t89[0], t89[1])),
                jnp.stack((t93[0], t93[1])),
                jnp.stack((t101[0], t101[1])),
                jnp.stack((t53[0], t53[1])),
                jnp.stack((t88[0], t88[1])),
                jnp.stack((t5[0], t5[1])),
                jnp.stack((t32[0], t32[1])),
            )
        ),
    )
    t94 = stage6[0]
    t99 = stage6[1]
    t104 = stage6[2]
    t120 = stage6[3]
    t124 = stage6[4]
    t135 = stage6[5]
    t138 = stage6[6]
    stage7 = jax.vmap(lambda a, b: binary(a, b, 2))(
        jnp.stack(
            (
                jnp.stack((t32[0], t32[1])),
                jnp.stack((t60[0], t60[1])),
                jnp.stack((t81[0], t81[1])),
                jnp.stack((t80[0], t80[1])),
                jnp.stack((t86[0], t86[1])),
                jnp.stack((t100[0], t100[1])),
                jnp.stack((t5[0], t5[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((t38[0], t38[1])),
                jnp.stack((t53[0], t53[1])),
                jnp.stack((t89[0], t89[1])),
                jnp.stack((t88[0], t88[1])),
                jnp.stack((t95[0], t95[1])),
                jnp.stack((t85[0], t85[1])),
                jnp.stack((t15[0], t15[1])),
            )
        ),
    )
    t40 = stage7[0]
    t64 = stage7[1]
    t92 = stage7[2]
    t96 = stage7[3]
    t102 = stage7[4]
    t103 = stage7[5]
    t105 = stage7[6]
    stage8 = jax.vmap(lambda a, b: binary(a, b, 0))(
        jnp.stack(
            (
                jnp.stack((t92[0], t92[1])),
                jnp.stack((t94[0], t94[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((C4, C4)),
                jnp.stack((C4, C4)),
            )
        ),
    )
    t97 = stage8[0]
    t98 = stage8[1]
    stage9 = jax.vmap(lambda a, b: binary(a, b, 1))(
        jnp.stack(
            (
                jnp.stack((t2[0], t2[1])),
                jnp.stack((t0[0], t0[1])),
                jnp.stack((t26[0], t26[1])),
                jnp.stack((t22[0], t22[1])),
                jnp.stack((t41[0], t41[1])),
                jnp.stack((t55[0], t55[1])),
                jnp.stack((t68[0], t68[1])),
                jnp.stack((t49[0], t49[1])),
                jnp.stack((t99[0], t99[1])),
                jnp.stack((t102[0], t102[1])),
                jnp.stack((t64[0], t64[1])),
                jnp.stack((t135[0], t135[1])),
                jnp.stack((t120[0], t120[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((t12[0], t12[1])),
                jnp.stack((t13[0], t13[1])),
                jnp.stack((t30[0], t30[1])),
                jnp.stack((t31[0], t31[1])),
                jnp.stack((t42[0], t42[1])),
                jnp.stack((t58[0], t58[1])),
                jnp.stack((t67[0], t67[1])),
                jnp.stack((t69[0], t69[1])),
                jnp.stack((t103[0], t103[1])),
                jnp.stack((t104[0], t104[1])),
                jnp.stack((t40[0], t40[1])),
                jnp.stack((t124[0], t124[1])),
                jnp.stack((t138[0], t138[1])),
            )
        ),
    )
    t17 = stage9[0]
    t18 = stage9[1]
    t35 = stage9[2]
    t37 = stage9[3]
    t45 = stage9[4]
    t63 = stage9[5]
    t71 = stage9[6]
    t73 = stage9[7]
    t107 = stage9[8]
    t109 = stage9[9]
    t112 = stage9[10]
    t137 = stage9[11]
    t140 = stage9[12]
    stage10 = jax.vmap(lambda a, b: binary(a, b, 2))(
        jnp.stack(
            (
                jnp.stack((t12[0], t12[1])),
                jnp.stack((t0[0], t0[1])),
                jnp.stack((t30[0], t30[1])),
                jnp.stack((t22[0], t22[1])),
                jnp.stack((t43[0], t43[1])),
                jnp.stack((t58[0], t58[1])),
                jnp.stack((t66[0], t66[1])),
                jnp.stack((t49[0], t49[1])),
                jnp.stack((t64[0], t64[1])),
                jnp.stack((t99[0], t99[1])),
                jnp.stack((t102[0], t102[1])),
                jnp.stack((t135[0], t135[1])),
                jnp.stack((t120[0], t120[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((t2[0], t2[1])),
                jnp.stack((t13[0], t13[1])),
                jnp.stack((t26[0], t26[1])),
                jnp.stack((t31[0], t31[1])),
                jnp.stack((t44[0], t44[1])),
                jnp.stack((t55[0], t55[1])),
                jnp.stack((t65[0], t65[1])),
                jnp.stack((t69[0], t69[1])),
                jnp.stack((t40[0], t40[1])),
                jnp.stack((t103[0], t103[1])),
                jnp.stack((t104[0], t104[1])),
                jnp.stack((t124[0], t124[1])),
                jnp.stack((t138[0], t138[1])),
            )
        ),
    )
    t16 = stage10[0]
    t19 = stage10[1]
    t34 = stage10[2]
    t36 = stage10[3]
    t46 = stage10[4]
    t62 = stage10[5]
    t70 = stage10[6]
    t72 = stage10[7]
    t90 = stage10[8]
    t106 = stage10[9]
    t108 = stage10[10]
    t136 = stage10[11]
    t139 = stage10[12]
    stage11 = jax.vmap(lambda a, b: binary(a, b, 0))(
        jnp.stack(
            (
                jnp.stack((t90[0], t90[1])),
                jnp.stack((t112[0], t112[1])),
                jnp.stack((t63[0], t63[1])),
                jnp.stack((t35[0], t35[1])),
                jnp.stack((C16, C16)),
                jnp.stack((t63[0], t63[1])),
                jnp.stack((t37[0], t37[1])),
                jnp.stack((t73[0], t73[1])),
                jnp.stack((t35[0], t35[1])),
                jnp.stack((t37[0], t37[1])),
                jnp.stack((t72[0], t72[1])),
                jnp.stack((t62[0], t62[1])),
                jnp.stack((t34[0], t34[1])),
                jnp.stack((t36[0], t36[1])),
                jnp.stack((t36[0], t36[1])),
                jnp.stack((t62[0], t62[1])),
                jnp.stack((t72[0], t72[1])),
                jnp.stack((t34[0], t34[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((C4, C4)),
                jnp.stack((C4, C4)),
                jnp.stack((C16, C16)),
                jnp.stack((S16, S16)),
                jnp.stack((t73[0], t73[1])),
                jnp.stack((S16, S16)),
                jnp.stack((C16, C16)),
                jnp.stack((S16, S16)),
                jnp.stack((C16, C16)),
                jnp.stack((S16, S16)),
                jnp.stack((C3_16, C3_16)),
                jnp.stack((S3_16, S3_16)),
                jnp.stack((S3_16, S3_16)),
                jnp.stack((C3_16, C3_16)),
                jnp.stack((S3_16, S3_16)),
                jnp.stack((C3_16, C3_16)),
                jnp.stack((S3_16, S3_16)),
                jnp.stack((C3_16, C3_16)),
            )
        ),
    )
    t91 = stage11[0]
    t113 = stage11[1]
    t147 = stage11[2]
    t161 = stage11[3]
    t162 = stage11[4]
    t163 = stage11[5]
    t165 = stage11[6]
    t169 = stage11[7]
    t171 = stage11[8]
    t172 = stage11[9]
    t180 = stage11[10]
    t196 = stage11[11]
    t199 = stage11[12]
    t200 = stage11[13]
    t201 = stage11[14]
    t205 = stage11[15]
    t206 = stage11[16]
    t208 = stage11[17]
    stage12 = jax.vmap(lambda a, b: binary(a, b, 1))(
        jnp.stack(
            (
                jnp.stack((t97[0], t97[1])),
                jnp.stack((t136[0], t136[1])),
                jnp.stack((t137[0], t137[1])),
                jnp.stack((t98[0], t98[1])),
                jnp.stack((t45[0], t45[1])),
                jnp.stack((t46[0], t46[1])),
                jnp.stack((t107[0], t107[1])),
                jnp.stack((t109[0], t109[1])),
                jnp.stack((t108[0], t108[1])),
                jnp.stack((t19[0], t19[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((t14[0], t14[1])),
                jnp.stack((-t139[1], t139[0])),
                jnp.stack((t140[0], t140[1])),
                jnp.stack((t4[0], t4[1])),
                jnp.stack((t70[0], t70[1])),
                jnp.stack((t71[0], t71[1])),
                jnp.stack((t17[0], t17[1])),
                jnp.stack((t18[0], t18[1])),
                jnp.stack((t16[0], t16[1])),
                jnp.stack((t106[0], t106[1])),
            )
        ),
    )
    t125 = stage12[0]
    t142 = stage12[1]
    t143 = stage12[2]
    t145 = stage12[3]
    t146 = stage12[4]
    t150 = stage12[5]
    t159 = stage12[6]
    t160 = stage12[7]
    t195 = stage12[8]
    t197 = stage12[9]
    stage13 = jax.vmap(lambda a, b: binary(a, b, 2))(
        jnp.stack(
            (
                jnp.stack((t70[0], t70[1])),
                jnp.stack((t71[0], t71[1])),
                jnp.stack((t97[0], t97[1])),
                jnp.stack((t4[0], t4[1])),
                jnp.stack((t136[0], t136[1])),
                jnp.stack((t137[0], t137[1])),
                jnp.stack((t18[0], t18[1])),
                jnp.stack((t107[0], t107[1])),
                jnp.stack((t19[0], t19[1])),
                jnp.stack((t108[0], t108[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((t45[0], t45[1])),
                jnp.stack((t46[0], t46[1])),
                jnp.stack((t14[0], t14[1])),
                jnp.stack((t98[0], t98[1])),
                jnp.stack((-t139[1], t139[0])),
                jnp.stack((t140[0], t140[1])),
                jnp.stack((t109[0], t109[1])),
                jnp.stack((t17[0], t17[1])),
                jnp.stack((t106[0], t106[1])),
                jnp.stack((t16[0], t16[1])),
            )
        ),
    )
    t121 = stage13[0]
    t122 = stage13[1]
    t123 = stage13[2]
    t128 = stage13[3]
    t141 = stage13[4]
    t144 = stage13[5]
    t157 = stage13[6]
    t158 = stage13[7]
    t193 = stage13[8]
    t194 = stage13[9]
    stage14 = jax.vmap(lambda a, b: binary(a, b, 1))(
        jnp.stack(
            (
                jnp.stack((t91[0], t91[1])),
                jnp.stack((t105[0], t105[1])),
                jnp.stack((t121[0], t121[1])),
                jnp.stack((t128[0], t128[1])),
                jnp.stack((t145[0], t145[1])),
                jnp.stack((t125[0], t125[1])),
                jnp.stack((t161[0], t161[1])),
                jnp.stack((t147[0], t147[1])),
                jnp.stack((t199[0], t199[1])),
                jnp.stack((t205[0], t205[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((t96[0], t96[1])),
                jnp.stack((t113[0], t113[1])),
                jnp.stack((t123[0], t123[1])),
                jnp.stack((t122[0], t122[1])),
                jnp.stack((t146[0], t146[1])),
                jnp.stack((t150[0], t150[1])),
                jnp.stack((t165[0], t165[1])),
                jnp.stack((t169[0], t169[1])),
                jnp.stack((t200[0], t200[1])),
                jnp.stack((t206[0], t206[1])),
            )
        ),
    )
    t111 = stage14[0]
    t115 = stage14[1]
    t127 = stage14[2]
    t130 = stage14[3]
    t148 = stage14[4]
    t151 = stage14[5]
    t166 = stage14[6]
    t170 = stage14[7]
    t202 = stage14[8]
    t207 = stage14[9]
    stage15 = jax.vmap(lambda a, b: binary(a, b, 2))(
        jnp.stack(
            (
                jnp.stack((t91[0], t91[1])),
                jnp.stack((t105[0], t105[1])),
                jnp.stack((t121[0], t121[1])),
                jnp.stack((t128[0], t128[1])),
                jnp.stack((t145[0], t145[1])),
                jnp.stack((t150[0], t150[1])),
                jnp.stack((t162[0], t162[1])),
                jnp.stack((t171[0], t171[1])),
                jnp.stack((t180[0], t180[1])),
                jnp.stack((t208[0], t208[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((t96[0], t96[1])),
                jnp.stack((t113[0], t113[1])),
                jnp.stack((t123[0], t123[1])),
                jnp.stack((t122[0], t122[1])),
                jnp.stack((t146[0], t146[1])),
                jnp.stack((t125[0], t125[1])),
                jnp.stack((t163[0], t163[1])),
                jnp.stack((t172[0], t172[1])),
                jnp.stack((t196[0], t196[1])),
                jnp.stack((t201[0], t201[1])),
            )
        ),
    )
    t110 = stage15[0]
    t114 = stage15[1]
    t126 = stage15[2]
    t129 = stage15[3]
    t149 = stage15[4]
    t152 = stage15[5]
    t164 = stage15[6]
    t173 = stage15[7]
    t198 = stage15[8]
    t209 = stage15[9]
    stage16 = jax.vmap(lambda a, b: binary(a, b, 1))(
        jnp.stack(
            (
                jnp.stack((t114[0], t114[1])),
                jnp.stack((t115[0], t115[1])),
                jnp.stack((t129[0], t129[1])),
                jnp.stack((t130[0], t130[1])),
                jnp.stack((t148[0], t148[1])),
                jnp.stack((t149[0], t149[1])),
                jnp.stack((t164[0], t164[1])),
                jnp.stack((t170[0], t170[1])),
                jnp.stack((t198[0], t198[1])),
                jnp.stack((t207[0], t207[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((-t110[1], t110[0])),
                jnp.stack((-t111[1], t111[0])),
                jnp.stack((-t126[1], t126[0])),
                jnp.stack((-t127[1], t127[0])),
                jnp.stack((-t151[1], t151[0])),
                jnp.stack((-t152[1], t152[0])),
                jnp.stack((t166[0], t166[1])),
                jnp.stack((t173[0], t173[1])),
                jnp.stack((t202[0], t202[1])),
                jnp.stack((t209[0], t209[1])),
            )
        ),
    )
    t116 = stage16[0]
    t119 = stage16[1]
    t131 = stage16[2]
    t134 = stage16[3]
    t154 = stage16[4]
    t155 = stage16[5]
    t168 = stage16[6]
    t175 = stage16[7]
    t204 = stage16[8]
    t211 = stage16[9]
    stage17 = jax.vmap(lambda a, b: binary(a, b, 2))(
        jnp.stack(
            (
                jnp.stack((t114[0], t114[1])),
                jnp.stack((t115[0], t115[1])),
                jnp.stack((t129[0], t129[1])),
                jnp.stack((t130[0], t130[1])),
                jnp.stack((t148[0], t148[1])),
                jnp.stack((t149[0], t149[1])),
                jnp.stack((t164[0], t164[1])),
                jnp.stack((t170[0], t170[1])),
                jnp.stack((t198[0], t198[1])),
                jnp.stack((t207[0], t207[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((-t110[1], t110[0])),
                jnp.stack((-t111[1], t111[0])),
                jnp.stack((-t126[1], t126[0])),
                jnp.stack((-t127[1], t127[0])),
                jnp.stack((-t151[1], t151[0])),
                jnp.stack((-t152[1], t152[0])),
                jnp.stack((t166[0], t166[1])),
                jnp.stack((t173[0], t173[1])),
                jnp.stack((t202[0], t202[1])),
                jnp.stack((t209[0], t209[1])),
            )
        ),
    )
    t117 = stage17[0]
    t118 = stage17[1]
    t132 = stage17[2]
    t133 = stage17[3]
    t153 = stage17[4]
    t156 = stage17[5]
    t167 = stage17[6]
    t174 = stage17[7]
    t203 = stage17[8]
    t210 = stage17[9]
    stage18 = jax.vmap(lambda a, b: binary(a, b, 1))(
        jnp.stack(
            (
                jnp.stack((t158[0], t158[1])),
                jnp.stack((t157[0], t157[1])),
                jnp.stack((t159[0], t159[1])),
                jnp.stack((t160[0], t160[1])),
                jnp.stack((t194[0], t194[1])),
                jnp.stack((t193[0], t193[1])),
                jnp.stack((t197[0], t197[1])),
                jnp.stack((t195[0], t195[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((t167[0], t167[1])),
                jnp.stack((t174[0], t174[1])),
                jnp.stack((t175[0], t175[1])),
                jnp.stack((t168[0], t168[1])),
                jnp.stack((t203[0], t203[1])),
                jnp.stack((t210[0], t210[1])),
                jnp.stack((t204[0], t204[1])),
                jnp.stack((t211[0], t211[1])),
            )
        ),
    )
    t176 = stage18[0]
    t178 = stage18[1]
    t183 = stage18[2]
    t187 = stage18[3]
    t212 = stage18[4]
    t214 = stage18[5]
    t218 = stage18[6]
    t221 = stage18[7]
    stage19 = jax.vmap(lambda a, b: binary(a, b, 2))(
        jnp.stack(
            (
                jnp.stack((t167[0], t167[1])),
                jnp.stack((t157[0], t157[1])),
                jnp.stack((t175[0], t175[1])),
                jnp.stack((t160[0], t160[1])),
                jnp.stack((t203[0], t203[1])),
                jnp.stack((t193[0], t193[1])),
                jnp.stack((t211[0], t211[1])),
                jnp.stack((t197[0], t197[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((t158[0], t158[1])),
                jnp.stack((t174[0], t174[1])),
                jnp.stack((t159[0], t159[1])),
                jnp.stack((t168[0], t168[1])),
                jnp.stack((t194[0], t194[1])),
                jnp.stack((t210[0], t210[1])),
                jnp.stack((t195[0], t195[1])),
                jnp.stack((t204[0], t204[1])),
            )
        ),
    )
    t177 = stage19[0]
    t179 = stage19[1]
    t185 = stage19[2]
    t188 = stage19[3]
    t213 = stage19[4]
    t215 = stage19[5]
    t222 = stage19[6]
    t225 = stage19[7]
    stage20 = jax.vmap(lambda a, b: binary(a, b, 1))(
        jnp.stack(
            (
                jnp.stack((t178[0], t178[1])),
                jnp.stack((t179[0], t179[1])),
                jnp.stack((t187[0], t187[1])),
                jnp.stack((t188[0], t188[1])),
                jnp.stack((t214[0], t214[1])),
                jnp.stack((t215[0], t215[1])),
                jnp.stack((t218[0], t218[1])),
                jnp.stack((t225[0], t225[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((-t176[1], t176[0])),
                jnp.stack((-t177[1], t177[0])),
                jnp.stack((-t183[1], t183[0])),
                jnp.stack((-t185[1], t185[0])),
                jnp.stack((-t212[1], t212[0])),
                jnp.stack((-t213[1], t213[0])),
                jnp.stack((-t221[1], t221[0])),
                jnp.stack((-t222[1], t222[0])),
            )
        ),
    )
    t182 = stage20[0]
    t186 = stage20[1]
    t189 = stage20[2]
    t191 = stage20[3]
    t217 = stage20[4]
    t220 = stage20[5]
    t223 = stage20[6]
    t226 = stage20[7]
    stage21 = jax.vmap(lambda a, b: binary(a, b, 2))(
        jnp.stack(
            (
                jnp.stack((t178[0], t178[1])),
                jnp.stack((t179[0], t179[1])),
                jnp.stack((t187[0], t187[1])),
                jnp.stack((t188[0], t188[1])),
                jnp.stack((t214[0], t214[1])),
                jnp.stack((t215[0], t215[1])),
                jnp.stack((t218[0], t218[1])),
                jnp.stack((t225[0], t225[1])),
            )
        ),
        jnp.stack(
            (
                jnp.stack((-t176[1], t176[0])),
                jnp.stack((-t177[1], t177[0])),
                jnp.stack((-t183[1], t183[0])),
                jnp.stack((-t185[1], t185[0])),
                jnp.stack((-t212[1], t212[0])),
                jnp.stack((-t213[1], t213[0])),
                jnp.stack((-t221[1], t221[0])),
                jnp.stack((-t222[1], t222[0])),
            )
        ),
    )
    t181 = stage21[0]
    t184 = stage21[1]
    t190 = stage21[2]
    t192 = stage21[3]
    t216 = stage21[4]
    t219 = stage21[5]
    t224 = stage21[6]
    t227 = stage21[7]
    return jnp.stack(
        (
            jnp.stack((t143[0], t143[1])),
            jnp.stack((t189[0], t189[1])),
            jnp.stack((t154[0], t154[1])),
            jnp.stack((t223[0], t223[1])),
            jnp.stack((t119[0], t119[1])),
            jnp.stack((t217[0], t217[1])),
            jnp.stack((t134[0], t134[1])),
            jnp.stack((t182[0], t182[1])),
            jnp.stack((t142[0], t142[1])),
            jnp.stack((t186[0], t186[1])),
            jnp.stack((t131[0], t131[1])),
            jnp.stack((t220[0], t220[1])),
            jnp.stack((t116[0], t116[1])),
            jnp.stack((t226[0], t226[1])),
            jnp.stack((t155[0], t155[1])),
            jnp.stack((t191[0], t191[1])),
            jnp.stack((t144[0], t144[1])),
            jnp.stack((t192[0], t192[1])),
            jnp.stack((t156[0], t156[1])),
            jnp.stack((t227[0], t227[1])),
            jnp.stack((t117[0], t117[1])),
            jnp.stack((t219[0], t219[1])),
            jnp.stack((t132[0], t132[1])),
            jnp.stack((t184[0], t184[1])),
            jnp.stack((t141[0], t141[1])),
            jnp.stack((t181[0], t181[1])),
            jnp.stack((t133[0], t133[1])),
            jnp.stack((t216[0], t216[1])),
            jnp.stack((t118[0], t118[1])),
            jnp.stack((t224[0], t224[1])),
            jnp.stack((t153[0], t153[1])),
            jnp.stack((t190[0], t190[1])),
        )
    )


@jax.jit
def real_even_inverse64(values):
    """Inverse transform of an even real 64-vector, preserving its shape."""
    h = values.reshape(64)[:33]
    a = h[1:16]
    b = h[31:16:-1]
    sr = binary(a, b, 1)
    dr = binary(a, b, 2)
    angles = [math.pi * min(k, 16 - k) / 32 for k in range(1, 16)]
    sn = jnp.asarray(
        [math.sin(angles[k - 1]) if k <= 8 else math.cos(angles[k - 1]) for k in range(1, 16)]
    )
    cs = jnp.asarray(
        [math.cos(angles[k - 1]) if k <= 8 else math.sin(angles[k - 1]) for k in range(1, 16)]
    )
    sd = binary(sn, dr, 0)
    cd = binary(cs, dr, 0)
    gr = jnp.zeros(32, jnp.float64)
    gi = jnp.zeros_like(gr)
    gr = gr.at[0].set(binary(h[0], h[32], 1))
    gi = gi.at[0].set(binary(h[0], h[32], 2))
    gr = gr.at[16].set(binary(jnp.asarray(2.0, jnp.float64), h[16], 0))
    gi = gi.at[16].set(-0.0)
    gr = gr.at[1:16].set(binary(sr, sd, 2))
    gi = gi.at[1:16].set(cd)
    gr = gr.at[17:32].set(binary(sr, sd, 1)[::-1])
    gi = gi.at[17:32].set(cd[::-1])
    raw = inverse32_pairs(jnp.stack([gr, gi], axis=1)).reshape(64)
    return binary(raw, jnp.asarray(64.0, jnp.float64), 3).reshape(values.shape)
