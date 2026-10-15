// The kernels Ming-Image's conditioning encoder (the BailingMoeV2 MLLM
// and the Qwen2 connector) needs beyond the host's kernel contract.
//
// Everything the contract has is taken from the host: the dense GEMMs
// (with bias and the fused SwiGLU), the steel flash attention, the bf16
// RMSNorm. What is here is either new arithmetic or a shape the
// contract does not carry:
//
//   ming_rms_f32         RMSNorm of the f32 residual stream into bf16
//                        (plus, for a router, the same values in f32)
//   ming_add_f32         the residual add: h(f32) += y(bf16)
//   ming_qk_norm_rope    per-head q/k RMSNorm and rotate-half RoPE from
//                        a per-ROW angle table, in place on the fused
//                        qkv rows -- the 3-axis "video_rope" is a table,
//                        so the kernel needs no notion of axes
//   ming_gather_rows     MoE dispatch: rows into expert order
//   ming_moe_gate_up     grouped GEMM over every expert at once, with
//   ming_moe_down        the SwiGLU fused into the first
//   ming_moe_combine     sum_k w_k * expert_k, into the residual
//   ming_gemm_acc_f32    a projection ADDED into the f32 residual
//   ming_copy_rows_*     the stream-B concatenation of three taps
//
// WHY THE RESIDUAL IS F32. The reference keeps it bf16 and rounds it
// after every add. Holding it in f32 is strictly closer to the fp32
// reference the goldens are judged against, costs one f32 [L, 2048]
// buffer, and keeps the MoE routers -- which read the normed residual
// and pick among near-ties -- on the more accurate side.
//
// Element type: bfloat only. The checkpoint is bf16 and so is every
// activation; there is no f16 twin to build.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// ---- RMSNorm of the f32 residual -------------------------------------
//
// y = bf16(x * rsqrt(mean(x^2) + eps) * w), one threadgroup per row.
// `y32` receives float(y) when `write_f32` -- the router's input is the
// bf16 normed row read as f32 (`hidden.float()`), and writing it here
// saves the host a conversion pass per MoE layer.
kernel void ming_rms_f32(
    const device float*  x         [[buffer(0)]],
    const device bfloat* w         [[buffer(1)]],
    device bfloat*       y         [[buffer(2)]],
    device float*        y32       [[buffer(3)]],
    constant int&        D         [[buffer(4)]],
    constant float&      eps       [[buffer(5)]],
    constant int&        write_f32 [[buffer(6)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 ltid     [[thread_position_in_threadgroup]],
    uint3 tptg     [[threads_per_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]])
{
  const uint row = tid.y;
  const uint lid = ltid.x;
  const uint tg = tptg.x;
  const uint nsg = tg / 32;
  const device float* xr = x + (ulong)row * (uint)D;
  float acc = 0.0f;
  for (uint i = lid; i < (uint)D; i += tg) {
    const float v = xr[i];
    acc += v * v;
  }
  acc = simd_sum(acc);
  threadgroup float partial[32];
  if (simd_lid == 0) { partial[simd_gid] = acc; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float p = (simd_lid < nsg) ? partial[simd_lid] : 0.0f;
    p = simd_sum(p);
    if (simd_lid == 0) { partial[0] = p; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(partial[0] / float(D) + eps);
  device bfloat* yr = y + (ulong)row * (uint)D;
  device float* yr32 = y32 + (ulong)row * (uint)D;
  for (uint i = lid; i < (uint)D; i += tg) {
    const bfloat o = bfloat(xr[i] * inv * float(w[i]));
    yr[i] = o;
    if (write_f32 != 0) { yr32[i] = float(o); }
  }
}

// ---- residual add -------------------------------------------------------
// h[i] += y[i], four at a time; n4 = n / 4 (every width here is a
// multiple of 4).
kernel void ming_add_f32(
    device float4*        h   [[buffer(0)]],
    const device bfloat4* y   [[buffer(1)]],
    constant uint&        n4  [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= n4) { return; }
  h[gid] += float4(y[gid]);
}

// ---- q/k norm + RoPE, in place -------------------------------------------
//
// One simdgroup per (head, row) of the fused [rows, row_stride] qkv
// buffer: heads [0, n_q) are queries at column 0, heads [n_q, n_q+n_k)
// keys right after them; values are not touched. head_dim is 128 and
// lane l holds dims l, l+32, l+64, l+96, which puts BOTH rotate-half
// pairings in one lane:
//
//   rot_half 32  (BailingMoeV2: partial rotary on dims [0, 64))
//                pairs (l, l+32); dims 64..127 pass through
//   rot_half 64  (Qwen2: the whole head)
//                pairs (l, l+64) and (l+32, l+96)
//   rot_half 0   no rotation
//
// cos/sin are [rows, rot_half] f32: angle k of row r. Building the
// table per ROW is what lets one kernel run both the LLM's 3-axis
// video_rope (each pair reads its own axis's position) and the
// connector's plain 1-D positions.
//
// has_norm applies the per-head RMSNorm (eps) with q_w / k_w BEFORE the
// rotation, as the reference does; the result is rounded to bf16 once.
kernel void ming_qk_norm_rope(
    device bfloat*        qkv        [[buffer(0)]],
    const device bfloat*  q_w        [[buffer(1)]],
    const device bfloat*  k_w        [[buffer(2)]],
    const device float*   cs         [[buffer(3)]],
    const device float*   sn         [[buffer(4)]],
    constant int&         row_stride [[buffer(5)]],
    constant int&         n_q        [[buffer(6)]],
    constant int&         n_k        [[buffer(7)]],
    constant int&         rot_half   [[buffer(8)]],
    constant int&         has_norm   [[buffer(9)]],
    constant float&       eps        [[buffer(10)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int head = (int)tid.x;
  const uint row = tid.y;
  if (head >= n_q + n_k) { return; }
  device bfloat* p = qkv + (ulong)row * (uint)row_stride + (uint)head * 128u;
  float x0 = float(p[lane]);
  float x1 = float(p[lane + 32]);
  float x2 = float(p[lane + 64]);
  float x3 = float(p[lane + 96]);
  if (has_norm != 0) {
    const float ss = simd_sum(x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3);
    const float inv = rsqrt(ss / 128.0f + eps);
    const device bfloat* w = head < n_q ? q_w : k_w;
    x0 *= inv * float(w[lane]);
    x1 *= inv * float(w[lane + 32]);
    x2 *= inv * float(w[lane + 64]);
    x3 *= inv * float(w[lane + 96]);
  }
  if (rot_half == 32) {
    const float c = cs[row * 32u + lane];
    const float s = sn[row * 32u + lane];
    const float y0 = x0 * c - x1 * s;
    const float y1 = x1 * c + x0 * s;
    x0 = y0;
    x1 = y1;
  } else if (rot_half == 64) {
    const float c0 = cs[row * 64u + lane];
    const float s0 = sn[row * 64u + lane];
    const float c1 = cs[row * 64u + lane + 32];
    const float s1 = sn[row * 64u + lane + 32];
    const float y0 = x0 * c0 - x2 * s0;
    const float y2 = x2 * c0 + x0 * s0;
    const float y1 = x1 * c1 - x3 * s1;
    const float y3 = x3 * c1 + x1 * s1;
    x0 = y0;
    x1 = y1;
    x2 = y2;
    x3 = y3;
  }
  p[lane] = bfloat(x0);
  p[lane + 32] = bfloat(x1);
  p[lane + 64] = bfloat(x2);
  p[lane + 96] = bfloat(x3);
}

// ---- MoE dispatch ----------------------------------------------------------
// dst[p] = src[idx[p]], D bf16 wide, 8 per thread (D % 8 == 0).
kernel void ming_gather_rows(
    const device uint4* src [[buffer(0)]],
    device uint4*       dst [[buffer(1)]],
    const device int*   idx [[buffer(2)]],
    constant int&       d8  [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
  if ((int)gid.x >= d8) { return; }
  dst[(ulong)gid.y * (uint)d8 + gid.x] =
      src[(ulong)idx[gid.y] * (uint)d8 + gid.x];
}

// ---- grouped expert GEMM ---------------------------------------------------
//
// Y[p, n] = sum_k X[p, k] * W[e][n, k] for every row p of every expert e
// in ONE dispatch. Rows arrive sorted by expert (ming_gather_rows), and
// the host's tile table cuts each expert's run into <= 32-row tiles:
// tiles[t] = (expert, first row, row count, 0). N % 64 == 0 and
// K % 64 == 0. Grid: threadgroups (N / 64, tiles) -- (N / 32, tiles)
// for the SwiGLU -- of 128 threads = 4 simdgroups, each owning a 16 x 32
// block of the 32 x 64 output tile.
//
// Why one grouped dispatch and not a GEMM per expert: a prefill routes
// ~8L rows over nearly all 256 experts, and 256 serial dispatches of a
// few dozen rows each leave the GPU idle between them. Here every
// expert's tile is in flight at once and the whole layer is one pass
// over its expert weights.
//
// Staging is bf16 in 16-byte vectors, SOFTWARE-PIPELINED: the next K
// step's global loads are issued into registers before the current
// step's MMAs, so their latency hides under the math. The multiply is
// bf16 x bf16 into f32 simdgroup accumulators, so the sum is f32
// throughout, as the steel GEMMs'.
//
// SWIGLU does NOT carry a second accumulator set -- that doubled the
// registers and the kernel spilled (MEASURED on an M4 Pro: 27 GB/s for
// gate+up against 183 for down at P1's routing). Instead its 64-row
// weight tile is VIRTUAL: rows [0, 32) are gate rows n0..n0+31 and rows
// [32, 64) the SAME columns of up, so the MMA loop is exactly the plain
// kernel's and the epilogue pairs column c with column c + 32:
// act = silu(gate) * up, 32 columns per threadgroup.
//
// The result goes out through threadgroup memory (aliasing the weight
// staging) so the store can mask rows past the tile's count; ACC adds
// it to Y instead of writing.
constant constexpr int kBM = 32;        // = LlmOps::kMoeTileRows
constant constexpr int kBN = 64;        // weight rows per tile
constant constexpr int kBK = 64;
constant constexpr int kLDB = kBK + 8;  // padded bf16 row (144 bytes)
constant constexpr int kLDO = kBN + 4;  // padded f32 output row
// Threadgroup bf16 elements: the X tile and the weight tile. The f32
// output staging (kBM * kLDO floats) reuses the weight's region once
// the loop is done.
constant constexpr int kXs = kBM * kLDB;
constant constexpr int kWs = kBN * kLDB;
static_assert(kWs * 2 >= kBM * kLDO * 4, "output staging must fit in W");

template <bool SWIGLU, typename OutT, bool ACC>
METAL_FUNC void
moe_gemm_impl(
    const device bfloat* X,
    const device bfloat* Wa,
    const device bfloat* Wb,
    device OutT*         Y,
    const device int4*   tiles,
    int K, int N,
    threadgroup bfloat* Xs, threadgroup bfloat* Ws,
    uint3 tid, uint lid, uint sg)
{
  const int4 t = tiles[tid.y];
  const int e = t.x, row0 = t.y, rows = t.z;
  // Output columns this tile writes: 64, or 32 gate|up pairs.
  const int n0 = (int)tid.x * (SWIGLU ? kBN / 2 : kBN);
  const ulong ebase = (ulong)e * (ulong)N * (ulong)K;
  const device bfloat* x = X + (ulong)row0 * (ulong)K;

  // A row of a K step is 8 segments of 8 bf16 (one uint4). Thread lid
  // moves segments lid + 128 h: 2 of X's 256, 4 of the weight's 512.
  const int sm = ((int)sg / 2) * 16;
  const int sn = ((int)sg % 2) * 32;
  uint4 xr[2], wr[4];
  auto fetch = [&](int k0) {
    for (int h = 0; h < 2; ++h) {
      const int s = (int)lid + 128 * h, r = s / 8, c = (s % 8) * 8;
      xr[h] = r < rows
          ? *(const device uint4*)(x + (ulong)r * (ulong)K + k0 + c)
          : uint4(0);
    }
    for (int h = 0; h < 4; ++h) {
      const int s = (int)lid + 128 * h, r = s / 8, c = (s % 8) * 8;
      const device bfloat* w =
          SWIGLU ? (r < kBN / 2 ? Wa : Wb) : Wa;
      const int wrow = n0 + (SWIGLU ? (r % (kBN / 2)) : r);
      wr[h] = *(const device uint4*)(w + ebase + (ulong)wrow * (ulong)K +
                                     (ulong)(k0 + c));
    }
  };

  simdgroup_float8x8 acc[2][4];
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) { acc[i][j] = simdgroup_float8x8(0.0f); }
  }

  fetch(0);
  for (int k0 = 0; k0 < K; k0 += kBK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int h = 0; h < 2; ++h) {
      const int s = (int)lid + 128 * h;
      *(threadgroup uint4*)(Xs + (s / 8) * kLDB + (s % 8) * 8) = xr[h];
    }
    for (int h = 0; h < 4; ++h) {
      const int s = (int)lid + 128 * h;
      *(threadgroup uint4*)(Ws + (s / 8) * kLDB + (s % 8) * 8) = wr[h];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 + kBK < K) { fetch(k0 + kBK); }
    for (int kk = 0; kk < kBK; kk += 8) {
      simdgroup_bfloat8x8 a[2], b[4];
      for (int i = 0; i < 2; ++i) {
        simdgroup_load(a[i], Xs + (sm + i * 8) * kLDB + kk, kLDB);
      }
      for (int j = 0; j < 4; ++j) {
        // B(k, n) = W[n][k]: a transposed load of the staged W rows.
        simdgroup_load(b[j], Ws + (sn + j * 8) * kLDB + kk, kLDB,
                       ulong2(0, 0), true);
      }
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
          simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
      }
    }
  }

  threadgroup float* Os = (threadgroup float*)Ws;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      simdgroup_store(acc[i][j], Os + (sm + i * 8) * kLDO + sn + j * 8,
                      kLDO);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int lr = (int)lid / 4;
  if (lr >= rows) { return; }
  if (SWIGLU) {
    // 32 pairs per row: thread lid takes 8 of row lid/4.
    const int c0 = ((int)lid % 4) * 8;
    device OutT* y = Y + (ulong)(row0 + lr) * (ulong)(N) + n0 + c0;
    const threadgroup float* g = Os + lr * kLDO + c0;
    const threadgroup float* u = g + kBN / 2;
    for (int q = 0; q < 8; ++q) {
      y[q] = OutT(g[q] / (1.0f + exp(-g[q])) * u[q]);
    }
    return;
  }
  const int c0 = ((int)lid % 4) * 16;
  device OutT* y = Y + (ulong)(row0 + lr) * (ulong)N + n0 + c0;
  const threadgroup float* o = Os + lr * kLDO + c0;
  if (ACC) {
    for (int q = 0; q < 16; ++q) { y[q] += OutT(o[q]); }
  } else {
    for (int q = 0; q < 16; ++q) { y[q] = OutT(o[q]); }
  }
}

// act[p, n] = silu(x_p . Wg[e][n]) * (x_p . Wu[e][n]); Wg / Wu are the
// stacked per-expert [E, N, K] projections, N = moe_intermediate.
kernel void ming_moe_gate_up(
    const device bfloat* X     [[buffer(0)]],
    const device bfloat* Wg    [[buffer(1)]],
    const device bfloat* Wu    [[buffer(2)]],
    device bfloat*       Y     [[buffer(3)]],
    const device int4*   tiles [[buffer(4)]],
    constant int&        K     [[buffer(5)]],
    constant int&        N     [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint  lid [[thread_index_in_threadgroup]],
    uint  sg  [[simdgroup_index_in_threadgroup]])
{
  threadgroup bfloat Xs[kXs];
  threadgroup bfloat Ws[kWs];
  moe_gemm_impl<true, bfloat, false>(X, Wg, Wu, Y, tiles, K, N, Xs, Ws, tid,
                                     lid, sg);
}

// out[p, n] = act_p . Wd[e][n], f32 so the weighted combine sums
// unrounded expert outputs.
kernel void ming_moe_down(
    const device bfloat* X     [[buffer(0)]],
    const device bfloat* Wd    [[buffer(1)]],
    device float*        Y     [[buffer(2)]],
    const device int4*   tiles [[buffer(3)]],
    constant int&        K     [[buffer(4)]],
    constant int&        N     [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint  lid [[thread_index_in_threadgroup]],
    uint  sg  [[simdgroup_index_in_threadgroup]])
{
  threadgroup bfloat Xs[kXs];
  threadgroup bfloat Ws[kWs];
  moe_gemm_impl<false, float, false>(X, Wd, Wd, Y, tiles, K, N, Xs, Ws, tid,
                                     lid, sg);
}

// h(f32) += X @ W^T with W ONE [N, K] matrix: the projections that feed
// the residual stream (attention out, the dense and shared MLPs' down).
// The host's dense GEMMs store bf16, and the residual is f32 BECAUSE a
// few rows carry massive activations: this model writes ~ -14800 into
// one channel of one text row at layer 1, where a bf16 store is good to
// +-32. Same tile walk as the experts, with one "expert" covering every
// row (tiles = (0, r0, rows)), and the store ADDS: each output element
// has exactly one writer, so the accumulate needs no atomics.
kernel void ming_gemm_acc_f32(
    const device bfloat* X     [[buffer(0)]],
    const device bfloat* W     [[buffer(1)]],
    device float*        Y     [[buffer(2)]],
    const device int4*   tiles [[buffer(3)]],
    constant int&        K     [[buffer(4)]],
    constant int&        N     [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint  lid [[thread_index_in_threadgroup]],
    uint  sg  [[simdgroup_index_in_threadgroup]])
{
  threadgroup bfloat Xs[kXs];
  threadgroup bfloat Ws[kWs];
  moe_gemm_impl<false, float, true>(X, W, W, Y, tiles, K, N, Xs, Ws, tid,
                                    lid, sg);
}

// h[r] += sum_k w[r, k] * out[pair[r, k]], f32 throughout: the
// reference sums the weighted experts in f32 too. (The un-gated shared
// expert is accumulated into h by ming_gemm_acc_f32.)
kernel void ming_moe_combine(
    device float4*        h      [[buffer(0)]],
    const device float4*  out    [[buffer(1)]],
    const device int*     pair   [[buffer(2)]],
    const device float*   wt     [[buffer(3)]],
    constant int&         h4     [[buffer(4)]],
    constant int&         top_k  [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  if ((int)gid.x >= h4) { return; }
  const uint r = gid.y;
  float4 acc = float4(0.0f);
  for (int k = 0; k < top_k; ++k) {
    const int p = pair[r * (uint)top_k + (uint)k];
    acc += wt[r * (uint)top_k + (uint)k] *
           out[(ulong)p * (uint)h4 + gid.x];
  }
  h[(ulong)r * (uint)h4 + gid.x] += acc;
}

// ---- row copies -----------------------------------------------------------
// dst[r * dst_ld + off + c] = src[r * src_ld + c] for c < cols: the
// three taps into stream B's [N, 6144] concatenation.
kernel void ming_copy_rows_f32_bf16(
    const device float* src    [[buffer(0)]],
    device bfloat*      dst    [[buffer(1)]],
    constant int&       src_ld [[buffer(2)]],
    constant int&       dst_ld [[buffer(3)]],
    constant int&       off    [[buffer(4)]],
    constant int&       cols   [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  if ((int)gid.x >= cols) { return; }
  dst[(ulong)gid.y * (uint)dst_ld + (uint)off + gid.x] =
      bfloat(src[(ulong)gid.y * (uint)src_ld + gid.x]);
}

kernel void ming_copy_rows_bf16(
    const device bfloat* src    [[buffer(0)]],
    device bfloat*       dst    [[buffer(1)]],
    constant int&        src_ld [[buffer(2)]],
    constant int&        dst_ld [[buffer(3)]],
    constant int&        off    [[buffer(4)]],
    constant int&        cols   [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  if ((int)gid.x >= cols) { return; }
  dst[(ulong)gid.y * (uint)dst_ld + (uint)off + gid.x] =
      src[(ulong)gid.y * (uint)src_ld + gid.x];
}
