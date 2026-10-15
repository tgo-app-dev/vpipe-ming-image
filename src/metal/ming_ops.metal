// The Metal kernels Ming-Image's DiT dispatches that are NOT in the host's
// kernel contract (apple-silicon/metal-compute/kernel-contract.h).
//
// WHY THEY ARE HERE. libvpipe's own metallibs are reachable by name, but
// only the entries the contract lists are promised to keep their name,
// argument order and meaning; everything else in llm_elementwise is the
// host's private business and may change under a built plugin. The DiT
// was vendored from the in-tree Z-Image transformer, which reached for
// six such entries, so each is COPIED here verbatim from vpipe's
// gpu-kernels/metal/ops/llm_elementwise.metal (same entry name, same
// buffer layout) and this file is compiled into the plugin's own
// metallib. Keeping the names identical keeps the copies diffable
// against their source; the LIBRARY name ("ming_ops_bf16") is what tells
// them apart from the host's.
//
//   gated_residual_tanh_f16 / _v4_f16   h += tanh(gate) * sub   (the block's
//                                        two gated residuals)
//   swiglu_v4_f16                        the feed-forward's gate, 4-wide
//   residual_add_f16 / _v4_f16           the context refiner's plain
//                                        residuals (it is unmodulated)
//   copy_rect_f16                        the caption's two streams into
//                                        the joint sequence
//
// The contracted siblings (swiglu_f16, adaln_modulate*, rms_norm_fast,
// ...) are NOT copied: the contract keeps them stable, so the DiT loads
// them from the host.
//
// VPIPE_ELT is set per twin (bfloat for the DiT, half for the f16 twin);
// math is f32 throughout, exactly as in the source.

#include <metal_stdlib>
using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

// Defined BEFORE any vec4 kernel. In the source file this macro appears
// partway down, and a v4 kernel placed above it fails to compile -- which
// is not a build error there but an invalid library at load.
#define VPIPE_ELT4 vec<VPIPE_ELT, 4>

kernel void residual_add_f16(
    const device VPIPE_ELT* a   [[buffer(0)]],
    const device VPIPE_ELT* b   [[buffer(1)]],
    device VPIPE_ELT*       out [[buffer(2)]],
    constant int&      n   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  out[gid] = VPIPE_ELT(float(a[gid]) + float(b[gid]));
}

// Gated residual with a TANH-squashed gate (Lumina / NextDiT "RMSNormZero",
// the Boogu DiT): h[m,n] += tanh(gate[n]) * sub[m,n], gate [N] broadcast over
// the M rows (total = M*N). In place on h. The tanh is part of the model, not
// a numerical guard -- the gate Linear is zero-initialised and trained through
// tanh, so dropping it changes the result.
kernel void gated_residual_tanh_f16(
    device VPIPE_ELT*       h    [[buffer(0)]],
    const device VPIPE_ELT* gate [[buffer(1)]],
    const device VPIPE_ELT* sub  [[buffer(2)]],
    constant int&      N     [[buffer(3)]],
    constant int&      total [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)total) { return; }
  const uint n = gid % (uint)N;
  h[gid] = VPIPE_ELT(float(h[gid]) +
                     precise::tanh(float(gate[n])) * float(sub[gid]));
}

// Row-major vec4 twin: a 2-D grid (N/4 columns, M rows), so the row index is
// free and every access is an 8-byte vector. Requires N % 4 == 0; the caller
// keeps the scalar kernel for odd widths.
kernel void gated_residual_tanh_v4_f16(
    device VPIPE_ELT*       h    [[buffer(0)]],
    const device VPIPE_ELT* gate [[buffer(1)]],
    const device VPIPE_ELT* sub  [[buffer(2)]],
    constant int&      N4   [[buffer(3)]],
    constant int&      M    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= (uint)N4 || gid.y >= (uint)M) { return; }
  const uint o = gid.y * (uint)N4 + gid.x;
  const float4 gr = float4(
      reinterpret_cast<const device VPIPE_ELT4*>(gate)[gid.x]);
  const float4 g = float4(precise::tanh(gr.x), precise::tanh(gr.y),
                          precise::tanh(gr.z), precise::tanh(gr.w));
  const float4 s = float4(reinterpret_cast<const device VPIPE_ELT4*>(sub)[o]);
  device VPIPE_ELT4* hv = reinterpret_cast<device VPIPE_ELT4*>(h);
  hv[o] = VPIPE_ELT4(float4(hv[o]) + g * s);
}

// SwiGLU over FOUR elements per thread, contiguous, n % 4 == 0.
//   0:gate 1:up 2:out 3:n4.  grid (n/4).
kernel void swiglu_v4_f16(
    const device VPIPE_ELT* gate [[buffer(0)]],
    const device VPIPE_ELT* up   [[buffer(1)]],
    device VPIPE_ELT*       out  [[buffer(2)]],
    constant int&      n4   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n4) { return; }
  const float4 g =
      float4(reinterpret_cast<const device VPIPE_ELT4*>(gate)[gid]);
  const float4 u = float4(reinterpret_cast<const device VPIPE_ELT4*>(up)[gid]);
  // Plain exp(), matching the host's scalar swiglu_f16: the point is a
  // wider thread, not a different function.
  const float4 sg = g / (1.0f + exp(-g));
  reinterpret_cast<device VPIPE_ELT4*>(out)[gid] = VPIPE_ELT4(sg * u);
}

// out = a + b over 4 elements per thread, 1-D grid of n/4 threads (n % 4 == 0).
kernel void residual_add_v4_f16(
    const device VPIPE_ELT* a   [[buffer(0)]],
    const device VPIPE_ELT* b   [[buffer(1)]],
    device VPIPE_ELT*       out [[buffer(2)]],
    constant int&      n4  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n4) { return; }
  const float4 x = float4(reinterpret_cast<const device VPIPE_ELT4*>(a)[gid]);
  const float4 y = float4(reinterpret_cast<const device VPIPE_ELT4*>(b)[gid]);
  reinterpret_cast<device VPIPE_ELT4*>(out)[gid] = VPIPE_ELT4(x + y);
}

// Rect copy between two row-major planes with independent row strides:
// out[dst_off + r*dst_stride + c] = src[src_off + r*src_stride + c].
// `row_elems` counts ELEMENTS. One-dimensional grid over rows*row_elems.
// 0:src 1:out 2:src_off 3:dst_off 4:rows 5:row_elems 6:src_stride
// 7:dst_stride
kernel void copy_rect_f16(
    const device VPIPE_ELT* src        [[buffer(0)]],
    device VPIPE_ELT*       out        [[buffer(1)]],
    constant int&           src_off    [[buffer(2)]],
    constant int&           dst_off    [[buffer(3)]],
    constant int&           rows       [[buffer(4)]],
    constant int&           row_elems  [[buffer(5)]],
    constant int&           src_stride [[buffer(6)]],
    constant int&           dst_stride [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
  const uint n = (uint)rows * (uint)row_elems;
  if (gid >= n) { return; }
  const int r = (int)(gid / (uint)row_elems);
  const int c = (int)(gid % (uint)row_elems);
  out[dst_off + r * dst_stride + c] = src[src_off + r * src_stride + c];
}
