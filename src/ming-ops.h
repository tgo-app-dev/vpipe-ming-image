#ifndef VPIPE_MING_OPS_H
#define VPIPE_MING_OPS_H

// The op vocabulary Ming-Image's conditioning encoder is written in:
// the BailingMoeV2 MLLM (ming-llm.h) and the Qwen2 connector
// (ming-connector.h) are nothing but calls into this.
//
// Two sources, resolved and validated at init:
//
//   * the HOST's kernel contract (kernel-contract.h): dense GEMMs with
//     bias and the fused SwiGLU, the steel flash attention, the bf16
//     RMSNorm. Resolved by name from libvpipe's metallibs.
//   * this plugin's own "ming_llm" metallib (src/metal/ming_llm.metal):
//     the f32 residual ops, the per-head q/k norm + table RoPE, and the
//     grouped MoE GEMMs. Whoever hosts the code registers it first --
//     the plugin at load (ming-plugin.cc), a test in its main().
//
// Every entry point is checked at init: an unresolved ComputeFunction
// dispatches nothing (or re-runs the previous pipeline over this op's
// buffers), and both read as a plausible encoder with wrong numbers.
//
// Everything is bf16 except the residual stream, the router inputs and
// the expert outputs, which are f32 (ming_llm.metal says why).

#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ming {

using vpipe::metal_compute::ComputeEncoder;
using vpipe::metal_compute::SharedBuffer;

// The WeightSet part every transformed weight the encoder keeps is
// attributed to, and released from when its model is destroyed.
//
// WHY. A set nobody borrows is KEPT by the model manager (parked, to
// spare the next launch a re-open), and parking only reaches cached
// source tensors: a derived() product has no source to re-read, so it is
// never parked and would stay resident -- here ~30 GB of stacked expert
// slabs and 2.6 GB of narrowed connector, after the encoder the stage
// released precisely so the DiT could have that memory. Releasing the
// part drops the cache's hold; a live peer's aliases keep their buffers.
inline constexpr const char* kEncoderPart = "ming-encoder";

// The library name this plugin's encoder kernels are registered under.
inline constexpr const char* kLlmMetalLib = "ming_llm";
// Register the embedded "ming_llm" metallib with `mc`, for a host with no
// plugin context (a test). Defined in ming-llm-kernels.cc, which is
// linked only where the metallib's bytes are -- the plugin and the
// encoder test -- so the shared model objects carry no reference to it.
bool register_llm_kernels(vpipe::metal_compute::MetalCompute* mc);

std::uint16_t f32_to_bf16(float f);
float bf16_to_f32(std::uint16_t b);

// One attention geometry's steel param block + specialised function.
// Planned per shape and kept by its caller: the param buffer is read
// when the GPU RUNS the dispatch, so two shapes in flight need two.
struct AttnPlan {
  SharedBuffer params;
  vpipe::metal_compute::ComputeFunction fn;
  int heads = 0;
  int bq = 32;
  int tokens = 0;
  bool ok = false;
};

class LlmOps {
public:
  bool init(vpipe::metal_compute::MetalCompute* mc, std::string* err);
  vpipe::metal_compute::MetalCompute* mc() const noexcept { return _mc; }

  SharedBuffer alloc(std::size_t bytes) const;
  // bf16 upload of host floats (for the f32 tensors the reference casts).
  SharedBuffer upload_bf16(const float* v, std::size_t n) const;

  // y[ye..][M][N] = x[xe..][M][K] @ W[N][K]^T (+ bias[N] when given).
  // W is PyTorch's [out][in] as the checkpoint stores it. Offsets are in
  // ELEMENTS (bf16).
  void linear(ComputeEncoder& enc, const SharedBuffer& x, std::size_t xe,
              const SharedBuffer& w, const SharedBuffer* bias,
              const SharedBuffer& y, std::size_t ye, int M, int N,
              int K) const;
  // y[M][N2/2] = silu(x W_gate^T) * (x W_up^T), with W the INTERLEAVED
  // [gate_0, up_0, gate_1, up_1, ...] rows (N2 of them).
  void swiglu_linear(ComputeEncoder& enc, const SharedBuffer& x,
                     const SharedBuffer& w, const SharedBuffer& y, int M,
                     int N2, int K) const;

  // Plan a head-dim-128 attention over token-major rows: q is read at
  // [row * q_ld + head * 128], k / v at [row * kv_ld + kvhead * 128],
  // the output written [row * o_ld + head * 128] -- so the fused qkv
  // rows are attended where the projection left them.
  bool plan_attention(AttnPlan* plan, int heads, int kv_heads, int tokens,
                      int q_ld, int kv_ld, int o_ld, bool causal) const;
  void attention(ComputeEncoder& enc, const AttnPlan& plan,
                 const SharedBuffer& qkv, std::size_t q_e, std::size_t k_e,
                 std::size_t v_e, const SharedBuffer& out) const;

  // bf16 -> bf16 RMSNorm (the host's), rows of `dim`.
  void rms_bf16(ComputeEncoder& enc, const SharedBuffer& x,
                const SharedBuffer& w, const SharedBuffer& y, int rows,
                int dim, float eps) const;
  // f32 residual -> bf16 normed (and its f32 reading, when y32 given).
  void rms_f32(ComputeEncoder& enc, const SharedBuffer& x,
               const SharedBuffer& w, const SharedBuffer& y,
               const SharedBuffer* y32, int rows, int dim,
               float eps) const;
  void add_f32(ComputeEncoder& enc, const SharedBuffer& h,
               const SharedBuffer& y, std::size_t n) const;
  void qk_norm_rope(ComputeEncoder& enc, const SharedBuffer& qkv,
                    const SharedBuffer* q_w, const SharedBuffer* k_w,
                    const SharedBuffer& cos, const SharedBuffer& sin,
                    int rows, int row_stride, int n_q, int n_k,
                    int rot_half, float eps) const;
  void gather_rows(ComputeEncoder& enc, const SharedBuffer& src,
                   const SharedBuffer& dst, const SharedBuffer& idx,
                   int rows, int dim) const;
  void moe_gate_up(ComputeEncoder& enc, const SharedBuffer& x,
                   const SharedBuffer& wg, const SharedBuffer& wu,
                   const SharedBuffer& y, const SharedBuffer& tiles,
                   int n_tiles, int K, int N) const;
  void moe_down(ComputeEncoder& enc, const SharedBuffer& x,
                const SharedBuffer& wd, const SharedBuffer& y,
                const SharedBuffer& tiles, int n_tiles, int K,
                int N) const;
  void moe_combine(ComputeEncoder& enc, const SharedBuffer& h,
                   const SharedBuffer& out, const SharedBuffer& pair,
                   const SharedBuffer& wt, int rows, int H,
                   int top_k) const;
  // h(f32)[M][N] += x[M][K] @ w[N][K]^T. `tiles` covers the M rows in
  // kMoeTileRows runs of expert 0 (dense_tiles writes one).
  void gemm_acc_f32(ComputeEncoder& enc, const SharedBuffer& x,
                    const SharedBuffer& w, const SharedBuffer& h,
                    const SharedBuffer& tiles, int M, int N, int K) const;
  // The tile table gemm_acc_f32 walks for M rows; returns the count.
  static int dense_tiles(const SharedBuffer& tiles, int M);
  // dst[r * dst_ld + off + c] = src[(r + r0) * src_ld + c], c < cols.
  void copy_rows_f32(ComputeEncoder& enc, const SharedBuffer& src,
                     std::size_t r0, int src_ld, const SharedBuffer& dst,
                     int dst_ld, int off, int rows, int cols) const;
  void copy_rows_bf16(ComputeEncoder& enc, const SharedBuffer& src,
                      std::size_t r0, int src_ld, const SharedBuffer& dst,
                      int dst_ld, int off, int rows, int cols) const;

  // The grouped GEMMs (moe_*, gemm_acc_f32) need N % 64 == 0 and
  // K % 64 == 0, and their tile height is this: the host cuts each
  // expert's rows into runs of at most this many.
  static constexpr int kMoeTileRows = 32;

private:
  vpipe::metal_compute::MetalCompute* _mc = nullptr;
  vpipe::metal_compute::ComputeLibrary _lib_gemm, _lib_rms, _lib_attn,
      _lib_own;
  vpipe::metal_compute::ComputeFunction _fn_gemm, _fn_gemm_wide,
      _fn_swiglu, _fn_rms_bf16, _fn_rms_f32, _fn_add_f32, _fn_qk,
      _fn_gather, _fn_gate_up, _fn_down, _fn_combine, _fn_copy_f32,
      _fn_copy_bf16, _fn_gemm_acc;
};

}  // namespace ming

#endif
