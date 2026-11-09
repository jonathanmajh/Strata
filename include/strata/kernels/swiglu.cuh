// include/strata/kernels/swiglu.cuh - the SwiGLU expressions, ONE copy each.
//
// The fusion work (V100) moves `silu(gate) * up` into the epilogue of the projection that computes `up`.
// For that to be bit-exact the fused epilogue must compute EXACTLY the expression the standalone kernel
// computed - same operations, same precision, same order.  A second copy of the expression is a second
// thing to drift, so every caller (the standalone kernels and the fused epilogues) includes this file.
//
// THREE VARIANTS, because the engine really has three:
//
//   `swilu_legacy`  the `ref/moe.py` transcription: silu in DOUBLE then cast, multiply in float
//                   (`shared_expert.cu`'s `swiglu_kernel`, and the S2 expert path used to use this shape).
//   `swilu_fast`    the S2 expert path's float expression with `__expf`
//                   (`s2_expert_grouped.cu`'s `swiglu_kernel`).
//   `swilu_native`  the pinned CUDA contract: `__fdividef` and `__expf`
//                   (`shared_expert.cu`'s `native_swiglu_kernel`).
//
// Bit-exactness across translation units: both call sites compile this with the same CUDA flags (no
// `--use_fast_math` on the files that use the legacy and fast variants), the expressions are straight-line
// IEEE ops around one libdevice call, and the parity tests compare the fused and unfused pipelines bitwise.
#pragma once

#include <cuda_runtime.h>

namespace strata::kernels {

/// `out = (float)((double)g / (1.0 + exp(-(double)g))) * u` - `shared_expert.cu`'s `swiglu_kernel`.
__device__ __forceinline__ float swilu_legacy(float g, float u) {
    const double x = (double) g;
    return (float) (x / (1.0 + ::exp(-x))) * u;
}

/// `out = (g / (1.0f + __expf(-g))) * u` - `s2_expert_grouped.cu`'s `swiglu_kernel`.
///
/// HIP has no CUDA `__expf` by that name: libdevice (`__ocml_exp_f32`, the fast precision family) is the
/// device twin, and for `__fdividef` the equivalent fast division is the `/` operator compiled with fast
/// math (the OCKL bitcode has no `__fdividef`/`__div` fast entry; the plain division in a gfx906/900 TU
/// lowers to the same approximate quotient the CUDA fast path uses).  Both keep the "no
/// `--use_fast_math`" contract of the pinned paths: the call sites for `swilu_native` are the CUDA builds,
/// whose TU flags match `__fdividef` exactly.

__device__ __forceinline__ float swilu_fast(float g, float u) {
    return (g / (1.0f + ::expf(-g))) * u;
}

/// `out = __fdividef(g, 1.0f + __expf(-g)) * u` - `shared_expert.cu`'s `native_swiglu_kernel`.
__device__ __forceinline__ float swilu_native(float g, float u) {
    return (g / (1.0f + ::expf(-g))) * u;
}

/// `out = __fdividef(g, 1.0f + __expf(-g)) * u` - the CUDA-pinned twin of `swilu_native`: on CUDA this
/// is `__fdividef` (the pinned contract); on the HIP compat layer the same expression goes through the
/// fast approximate division the device libraries lower `/` to under fast math.
__device__ __forceinline__ float swilu_native_pinned(float g, float u) {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
    return __fdividef(g, 1.0f + __expf(-g)) * u;
#else
    return (g / (1.0f + ::expf(-g))) * u;
#endif
}

}  // namespace strata::kernels
