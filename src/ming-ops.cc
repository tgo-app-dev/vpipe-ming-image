#include "ming-ops.h"

#include "apple-silicon/metal-compute/kernel-contract.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ming {

namespace {

using vpipe::metal_compute::ComputeFunction;
namespace contract = vpipe::metal_compute::contract;

}  // namespace

std::uint16_t
f32_to_bf16(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

float
bf16_to_f32(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

bool
LlmOps::init(vpipe::metal_compute::MetalCompute* mc, std::string* err)
{
  if (mc == nullptr || !mc->valid()) {
    if (err != nullptr) { *err = "no metal-compute backend"; }
    return false;
  }
  _mc = mc;
  // The contract's libraries. bf16 throughout: the checkpoint is bf16,
  // and the f16 twin would read its bits as garbage, not as an error.
  _lib_gemm = mc->load_library("dense_gemm_bf16");
  _lib_rms  = mc->load_library("rms_norm_bf16");
  _lib_attn = mc->load_library("attn_steel");
  _lib_own  = mc->load_library(kLlmMetalLib);
  if (!_lib_own.valid()) {
    if (err != nullptr) {
      *err = std::string("the plugin's '") + kLlmMetalLib +
             "' metallib is not registered (or failed to load)";
    }
    return false;
  }
  _fn_gemm      = _lib_gemm.function("dense_gemm_t_f16");
  _fn_gemm_wide = _lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  _fn_swiglu    = _lib_gemm.function("dense_gemm_swiglu_bm64_f16");
  _fn_rms_bf16  = _lib_rms.function("rms_norm_fast_f16");
  _fn_rms_f32   = _lib_own.function("ming_rms_f32");
  _fn_add_f32   = _lib_own.function("ming_add_f32");
  _fn_qk        = _lib_own.function("ming_qk_norm_rope");
  _fn_gather    = _lib_own.function("ming_gather_rows");
  _fn_gate_up   = _lib_own.function("ming_moe_gate_up");
  _fn_down      = _lib_own.function("ming_moe_down");
  _fn_combine   = _lib_own.function("ming_moe_combine");
  _fn_gemm_acc  = _lib_own.function("ming_gemm_acc_f32");
  _fn_copy_f32  = _lib_own.function("ming_copy_rows_f32_bf16");
  _fn_copy_bf16 = _lib_own.function("ming_copy_rows_bf16");
  struct Req { const ComputeFunction* fn; const char* name; };
  const Req required[] = {
      {&_fn_gemm, "dense_gemm_t_f16"},
      {&_fn_gemm_wide, "dense_gemm_t_bm64bn64_f16"},
      {&_fn_swiglu, "dense_gemm_swiglu_bm64_f16"},
      {&_fn_rms_bf16, "rms_norm_fast_f16"},
      {&_fn_rms_f32, "ming_rms_f32"},
      {&_fn_add_f32, "ming_add_f32"},
      {&_fn_qk, "ming_qk_norm_rope"},
      {&_fn_gather, "ming_gather_rows"},
      {&_fn_gate_up, "ming_moe_gate_up"},
      {&_fn_down, "ming_moe_down"},
      {&_fn_combine, "ming_moe_combine"},
      {&_fn_gemm_acc, "ming_gemm_acc_f32"},
      {&_fn_copy_f32, "ming_copy_rows_f32_bf16"},
      {&_fn_copy_bf16, "ming_copy_rows_bf16"},
  };
  for (const Req& r : required) {
    if (!r.fn->valid()) {
      if (err != nullptr) {
        *err = std::string("kernel '") + r.name + "' did not resolve";
      }
      return false;
    }
  }
  if (!_lib_attn.valid()) {
    if (err != nullptr) { *err = "the host's attn_steel library is missing"; }
    return false;
  }
  return true;
}

SharedBuffer
LlmOps::alloc(std::size_t bytes) const
{
  return _mc->make_shared_buffer(bytes == 0 ? 16 : bytes);
}

SharedBuffer
LlmOps::upload_bf16(const float* v, std::size_t n) const
{
  SharedBuffer b = alloc(n * 2);
  if (b.empty()) { return b; }
  auto* d = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < n; ++i) { d[i] = f32_to_bf16(v[i]); }
  return b;
}

void
LlmOps::linear(ComputeEncoder& enc, const SharedBuffer& x, std::size_t xe,
               const SharedBuffer& w, const SharedBuffer* bias,
               const SharedBuffer& y, std::size_t ye, int M, int N,
               int K) const
{
  // The 64x64 tile wherever there are 64 rows to fill it: it re-reads
  // each weight tile M/64 times instead of M/32. Below that the base
  // 32x32 tile wastes less.
  const bool wide = M >= 64;
  const int bm = wide ? 64 : 32;
  const int bn = wide ? 64 : 32;
  enc.set_function(wide ? _fn_gemm_wide : _fn_gemm);
  enc.set_buffer(0, x, xe * 2);
  enc.set_buffer(1, w);
  enc.set_buffer(2, bias != nullptr ? *bias : w);
  enc.set_buffer(3, y, ye * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, bias != nullptr ? 1 : 0);
  enc.dispatch({(unsigned)(((N + bn - 1) / bn) * 32),
                (unsigned)(((M + bm - 1) / bm) * 2), 2}, {32, 2, 2});
}

void
LlmOps::swiglu_linear(ComputeEncoder& enc, const SharedBuffer& x,
                      const SharedBuffer& w, const SharedBuffer& y, int M,
                      int N2, int K) const
{
  enc.set_function(_fn_swiglu);
  enc.set_buffer(0, x);
  enc.set_buffer(1, w);
  enc.set_buffer(2, y);
  enc.set_constant(3, K);
  enc.set_constant(4, N2);
  enc.set_constant(5, M);
  enc.set_constant(6, 0);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((N2 + 31) / 32) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
}

bool
LlmOps::plan_attention(AttnPlan* plan, int heads, int kv_heads,
                       int tokens, int q_ld, int kv_ld, int o_ld,
                       bool causal) const
{
  plan->ok = false;
  if (heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 || tokens <= 0) {
    return false;
  }
  if (plan->params.empty()) {
    plan->params = _mc->make_shared_buffer(sizeof(contract::SteelAttnParams));
    if (plan->params.empty()) { return false; }
  }
  // The steel entry's own tile: BQ 32, BK 16 at head_dim 128.
  const int bq = 32, bk = 16, d = 128;
  auto* p = static_cast<contract::SteelAttnParams*>(plan->params.contents());
  p->B = 1;
  p->H = heads;
  p->D = d;
  p->qL = tokens;
  p->kL = tokens;
  p->gqa_factor = heads / kv_heads;
  p->scale = 1.0f / std::sqrt((float)d);
  p->NQ = (tokens + bq - 1) / bq;
  p->NK = (tokens + bk - 1) / bk;
  p->NQ_aligned = tokens / bq;
  p->NK_aligned = tokens / bk;
  p->qL_rem = tokens - p->NQ_aligned * bq;
  p->kL_rem = tokens - p->NK_aligned * bk;
  // Self-attention over the whole prompt: query i is key i, so the
  // causal diagonal needs no offset.
  p->qL_off = 0;
  // (batch, head, row) strides in elements; the head dim is contiguous.
  p->Q_strides[0] = (std::int64_t)tokens * q_ld;
  p->Q_strides[1] = d;
  p->Q_strides[2] = q_ld;
  p->K_strides[0] = (std::int64_t)tokens * kv_ld;
  p->K_strides[1] = d;
  p->K_strides[2] = kv_ld;
  p->V_strides[0] = p->K_strides[0];
  p->V_strides[1] = d;
  p->V_strides[2] = kv_ld;
  p->O_strides[0] = (std::int64_t)tokens * o_ld;
  p->O_strides[1] = d;
  p->O_strides[2] = o_ld;
  vpipe::metal_compute::FunctionConstants fc;
  fc.set_bool(contract::kAttnAlignQ, (tokens % bq) == 0)
      .set_bool(contract::kAttnAlignK, (tokens % bk) == 0)
      .set_bool(contract::kAttnHasMask, false)
      .set_bool(contract::kAttnCausal, causal)
      .set_bool(contract::kAttnSinks, false)
      .set_bool(contract::kAttnQkInt8, false);
  plan->fn = _lib_attn.function("attn_steel_h_bd128_bf16", fc);
  plan->heads = heads;
  plan->bq = bq;
  plan->tokens = tokens;
  plan->ok = plan->fn.valid();
  return plan->ok;
}

void
LlmOps::attention(ComputeEncoder& enc, const AttnPlan& plan,
                  const SharedBuffer& qkv, std::size_t q_e, std::size_t k_e,
                  std::size_t v_e, const SharedBuffer& out) const
{
  enc.set_function(plan.fn);
  enc.set_buffer(0, qkv, q_e * 2);
  enc.set_buffer(1, qkv, k_e * 2);
  enc.set_buffer(2, qkv, v_e * 2);
  enc.set_buffer(3, out);
  enc.set_buffer(4, plan.params);
  const unsigned nqb = (unsigned)((plan.tokens + plan.bq - 1) / plan.bq);
  enc.dispatch({32 * nqb, 4 * (unsigned)plan.heads, 1}, {32, 4, 1});
}

void
LlmOps::rms_bf16(ComputeEncoder& enc, const SharedBuffer& x,
                 const SharedBuffer& w, const SharedBuffer& y, int rows,
                 int dim, float eps) const
{
  enc.set_function(_fn_rms_bf16);
  enc.set_buffer(0, x);
  enc.set_buffer(1, w);
  enc.set_buffer(2, y);
  enc.set_constant(3, dim);
  enc.set_constant(4, eps);
  enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
}

void
LlmOps::rms_f32(ComputeEncoder& enc, const SharedBuffer& x,
                const SharedBuffer& w, const SharedBuffer& y,
                const SharedBuffer* y32, int rows, int dim, float eps) const
{
  enc.set_function(_fn_rms_f32);
  enc.set_buffer(0, x);
  enc.set_buffer(1, w);
  enc.set_buffer(2, y);
  // Bound even when unused: an unbound argument is a validation error
  // on a debug device whether or not the kernel reads it.
  enc.set_buffer(3, y32 != nullptr ? *y32 : x);
  enc.set_constant(4, dim);
  enc.set_constant(5, eps);
  enc.set_constant(6, y32 != nullptr ? 1 : 0);
  enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
}

void
LlmOps::add_f32(ComputeEncoder& enc, const SharedBuffer& h,
                const SharedBuffer& y, std::size_t n) const
{
  const unsigned n4 = (unsigned)(n / 4);
  enc.set_function(_fn_add_f32);
  enc.set_buffer(0, h);
  enc.set_buffer(1, y);
  enc.set_constant(2, n4);
  enc.dispatch({n4, 1, 1}, {256, 1, 1});
}

void
LlmOps::qk_norm_rope(ComputeEncoder& enc, const SharedBuffer& qkv,
                     const SharedBuffer* q_w, const SharedBuffer* k_w,
                     const SharedBuffer& cos, const SharedBuffer& sin,
                     int rows, int row_stride, int n_q, int n_k,
                     int rot_half, float eps) const
{
  const bool norm = q_w != nullptr && k_w != nullptr;
  enc.set_function(_fn_qk);
  enc.set_buffer(0, qkv);
  enc.set_buffer(1, norm ? *q_w : qkv);
  enc.set_buffer(2, norm ? *k_w : qkv);
  enc.set_buffer(3, cos);
  enc.set_buffer(4, sin);
  enc.set_constant(5, row_stride);
  enc.set_constant(6, n_q);
  enc.set_constant(7, n_k);
  enc.set_constant(8, rot_half);
  enc.set_constant(9, norm ? 1 : 0);
  enc.set_constant(10, eps);
  enc.dispatch({32u * (unsigned)(n_q + n_k), (unsigned)rows, 1},
               {32, 1, 1});
}

void
LlmOps::gather_rows(ComputeEncoder& enc, const SharedBuffer& src,
                    const SharedBuffer& dst, const SharedBuffer& idx,
                    int rows, int dim) const
{
  const int d8 = dim / 8;
  enc.set_function(_fn_gather);
  enc.set_buffer(0, src);
  enc.set_buffer(1, dst);
  enc.set_buffer(2, idx);
  enc.set_constant(3, d8);
  enc.dispatch({(unsigned)d8, (unsigned)rows, 1}, {256, 1, 1});
}

void
LlmOps::moe_gate_up(ComputeEncoder& enc, const SharedBuffer& x,
                    const SharedBuffer& wg, const SharedBuffer& wu,
                    const SharedBuffer& y, const SharedBuffer& tiles,
                    int n_tiles, int K, int N) const
{
  enc.set_function(_fn_gate_up);
  enc.set_buffer(0, x);
  enc.set_buffer(1, wg);
  enc.set_buffer(2, wu);
  enc.set_buffer(3, y);
  enc.set_buffer(4, tiles);
  enc.set_constant(5, K);
  enc.set_constant(6, N);
  // 32 gate|up column pairs per threadgroup (ming_llm.metal).
  enc.dispatch({(unsigned)(N / 32) * 128u, (unsigned)n_tiles, 1},
               {128, 1, 1});
}

void
LlmOps::moe_down(ComputeEncoder& enc, const SharedBuffer& x,
                 const SharedBuffer& wd, const SharedBuffer& y,
                 const SharedBuffer& tiles, int n_tiles, int K, int N) const
{
  enc.set_function(_fn_down);
  enc.set_buffer(0, x);
  enc.set_buffer(1, wd);
  enc.set_buffer(2, y);
  enc.set_buffer(3, tiles);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.dispatch({(unsigned)(N / 64) * 128u, (unsigned)n_tiles, 1},
               {128, 1, 1});
}

void
LlmOps::moe_combine(ComputeEncoder& enc, const SharedBuffer& h,
                    const SharedBuffer& out, const SharedBuffer& pair,
                    const SharedBuffer& wt, int rows, int H, int top_k) const
{
  const int h4 = H / 4;
  enc.set_function(_fn_combine);
  enc.set_buffer(0, h);
  enc.set_buffer(1, out);
  enc.set_buffer(2, pair);
  enc.set_buffer(3, wt);
  enc.set_constant(4, h4);
  enc.set_constant(5, top_k);
  enc.dispatch({(unsigned)h4, (unsigned)rows, 1}, {256, 1, 1});
}

int
LlmOps::dense_tiles(const SharedBuffer& tiles, int M)
{
  auto* t = static_cast<std::int32_t*>(tiles.contents());
  int n = 0;
  for (int r0 = 0; r0 < M; r0 += kMoeTileRows, ++n) {
    t[4 * n + 0] = 0;
    t[4 * n + 1] = r0;
    t[4 * n + 2] = std::min(kMoeTileRows, M - r0);
    t[4 * n + 3] = 0;
  }
  return n;
}

void
LlmOps::gemm_acc_f32(ComputeEncoder& enc, const SharedBuffer& x,
                     const SharedBuffer& w, const SharedBuffer& h,
                     const SharedBuffer& tiles, int M, int N, int K) const
{
  const int n_tiles = (M + kMoeTileRows - 1) / kMoeTileRows;
  enc.set_function(_fn_gemm_acc);
  enc.set_buffer(0, x);
  enc.set_buffer(1, w);
  enc.set_buffer(2, h);
  enc.set_buffer(3, tiles);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.dispatch({(unsigned)(N / 64) * 128u, (unsigned)n_tiles, 1},
               {128, 1, 1});
}

void
LlmOps::copy_rows_f32(ComputeEncoder& enc, const SharedBuffer& src,
                      std::size_t r0, int src_ld, const SharedBuffer& dst,
                      int dst_ld, int off, int rows, int cols) const
{
  enc.set_function(_fn_copy_f32);
  enc.set_buffer(0, src, r0 * (std::size_t)src_ld * 4);
  enc.set_buffer(1, dst);
  enc.set_constant(2, src_ld);
  enc.set_constant(3, dst_ld);
  enc.set_constant(4, off);
  enc.set_constant(5, cols);
  enc.dispatch({(unsigned)cols, (unsigned)rows, 1}, {256, 1, 1});
}

void
LlmOps::copy_rows_bf16(ComputeEncoder& enc, const SharedBuffer& src,
                       std::size_t r0, int src_ld, const SharedBuffer& dst,
                       int dst_ld, int off, int rows, int cols) const
{
  enc.set_function(_fn_copy_bf16);
  enc.set_buffer(0, src, r0 * (std::size_t)src_ld * 2);
  enc.set_buffer(1, dst);
  enc.set_constant(2, src_ld);
  enc.set_constant(3, dst_ld);
  enc.set_constant(4, off);
  enc.set_constant(5, cols);
  enc.dispatch({(unsigned)cols, (unsigned)rows, 1}, {256, 1, 1});
}

}  // namespace ming
