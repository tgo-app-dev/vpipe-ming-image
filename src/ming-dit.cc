// Ming-Image's DiT on metal-compute, VENDORED from vpipe's in-tree
// Z-Image transformer (generative-models/z-image/metal-z-image-
// transformer.{h,cc} at 8ab6c6cb). libvpipe does not export that class,
// so this plugin carries its own copy. The block math, the rotary
// tables, the timestep embedding, the final layer, every GEMM and
// attention route, block streaming and the i8 / Sage / Sol tiers are the
// in-tree code with its reasoning kept; what follows is only what
// differs.
//
// STRIPPED:
//
//   * the q/k/v ANE tier (ane_qkv). The feed-forward tier is here, on the
//     installed ane::Tier rather than the in-tree AneFeedForward; see "THE
//     ANE FEED-FORWARD" below for why q/k/v did not come with it.
//
// RUNTIME LoRA is the in-tree code (shared/runtime-lora.h, lora-names.h,
// now in the SDK's toolkit): the same seven projections per block, the
// same Applier after whichever base GEMM ran (tiled, qmm or matmul2d),
// two live slots. See "RUNTIME LoRA" near the end for the module list
// and what is and is not verified about it. Two things are this file's:
// the ANE feed-forward takes the adapter INTO its module (Tier's
// `<p>.lora<i>` bindings), and a strength change drops the caption
// cache, whose rows the context refiner's adapted projections built.
//
// CHANGED:
//
//   * SOL RUNS WITHOUT MATRIX CORES. The in-tree file gated it on the
//     NAX attention (`_sol && nax && ...`), which on an M4 meant never --
//     yet MetalSolAttention carries an ALU arm whose exact half is the
//     same steel kernel this DiT runs dense. The gate is the geometry
//     alone now; the toolkit picks the arm. With every key block kept it
//     is BIT-IDENTICAL to dense on that arm (ming-dit-test).
//
//     WHAT IT BUYS IS A PROPERTY OF THE OUTPUT SIZE: the attention is
//     16% of a 1024^2 forward and 43% of a 2048^2 one, running at the
//     GEMMs' rate (~6.2 against ~6.9 TFLOP/s), so the lever is doing less
//     of it. MEASURED on an M4 Pro (fan), one held forward, arms
//     interleaved: 2048^2 49.4 s dense -> 39.9 s at tau 0 (1.24x, 49.5%
//     of key blocks kept), 33.1 s at tau 1 (1.49x, 19%); 1024^2 only
//     1.08x / 1.14x. It is APPROXIMATE: tau 0 at 2048^2 kept every word
//     of a text-heavy prompt; tau 1 repeated the prompt's one button six
//     times. Judge it on pictures.
//   * Sol's main-stack calls keep the CAPTION exact (a KV sink over its
//     rows): this model renders the text in its prompt, and a centroid
//     over prompt tokens is a prompt half-read. VPIPE_MING_SOL_NO_SINK=1
//     is the A/B; at 512^2 the two are within each other's noise in
//     rel-L2 (0.554 / 0.530 from dense at tau 0), so the sink is kept on
//     the argument, not on a measurement.
//   * Sol borrows the feed-forward planes, which are dead during the
//     attention, instead of allocating its scratch.
//
// KERNELS. Every host kernel dispatched here is in the kernel contract
// (apple-silicon/metal-compute/kernel-contract.h), resolved by name in
// the library the contract names. The six the in-tree file used that
// the contract does NOT list -- gated_residual_tanh(_v4), swiglu_v4,
// residual_add(_v4), copy_rect -- are copied into the plugin's own
// metallib (src/metal/ming_ops.metal, library kMetalLibBf16). Every
// function is validated at load, and an invalid one REFUSES the load:
// an unbound dispatch re-runs the previous kernel over this one's
// buffers, which is a picture rather than an error.
//
// THE MING DELTAS (docs/MING-IMAGE.md 2.1-2.4):
//
//   1. "zero_masked" padding. The checkpoint has no x_pad_token /
//      cap_pad_token: its pad rows are zero and excluded as attention
//      KEYS everywhere, and no real row ever reads a pad query's output.
//      So this port runs the REAL rows only -- [image | caption], no pads
//      -- which removes every mask from the model. The positions still
//      count the pads (ming-layout.h): the image sits at axis-0 position
//      roundup32(cap_len) + 1. Every attention therefore runs at lengths
//      that are NOT multiples of 32, and the steel kernel's remainder
//      path (align_Q / align_K false, qL_rem / kL_rem) is the one that
//      handles it; the in-tree sequence was always padded, so on a box
//      without matrix cores that path was never taken there.
//   2. TWO caption streams in one tensor (ming-family.h): rows
//      [0, query_rows) are stream A in columns [0, cap_feat_dim) and go
//      through cap_embedder (RMSNorm + Linear); rows [query_rows,
//      cap_len) are stream B, already DiT-wide, copied as they are.
//   3. No caption-length cap: the axis-0 rotary table bounds it.
//   4. The config is ming::DitConfig ("DiffusionTransformer"), not the
//      in-tree read_dims.
//   5. Unchanged from the in-tree: the DiT is fed t_scale * (1 - sigma)
//      and returns the raw model output (the generator negates it).
//
// ADDED: the caption path (cap_embedder + context_refiner) does not
// depend on the timestep, so its output is CACHED per caption -- keyed
// on the caption's exact bytes, two entries (a CFG run alternates the
// positive and the negative), served by a copy into the joint sequence.
// A hit is bit-identical to recomputing, by construction: the same
// kernels over the same bytes.

#include "ming-dit.h"

#include "apple-silicon/metal-compute/kernel-contract.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/accel-keys.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/accel-settings.h"
#include "generative-models/shared/ane-tier.h"
#include "generative-models/shared/block-residency.h"
#include "generative-models/shared/block-slots.h"
#include "generative-models/shared/dit-flops.h"
#include "generative-models/shared/dit-gpu-progress.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/lora-names.h"
#include "generative-models/shared/metal-sage-attention.h"
#include "generative-models/shared/metal-sol-attention.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/shared/sage-attention.h"
#include "generative-models/shared/sol-attention.h"
#include "generative-models/shared/wired-pool.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-memory.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace ming {

namespace {

namespace fs = std::filesystem;
namespace accel = vpipe::genai::accel;
namespace ane = vpipe::genai::ane;
namespace lora = vpipe::genai::lora;
namespace sage = vpipe::genai::sage;
namespace sol = vpipe::genai::sol;
namespace contract = vpipe::metal_compute::contract;

using vpipe::fmt;
using vpipe::FlexData;
using vpipe::SessionContextIntf;
using vpipe::metal_compute::CommandStream;
using vpipe::metal_compute::ComputeEncoder;
using vpipe::metal_compute::ComputeFunction;
using vpipe::metal_compute::ComputeLibrary;
using vpipe::metal_compute::FunctionConstants;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;
using vpipe::genai::BlockResidency;
using vpipe::genai::BlockSlots;
using vpipe::genai::DitBlockProgressFn;
using vpipe::genai::GenerativeModelManager;
using vpipe::genai::GpuFlopsTally;
using vpipe::genai::I8GemmContext;
using vpipe::genai::MetalLlamaWeights;
using vpipe::genai::MetalSageAttention;
using vpipe::genai::MetalSolAttention;
using vpipe::genai::Placement;
using vpipe::genai::SolFlops;
using vpipe::genai::WeightSet;
using vpipe::genai::WiredPool;

// The steel flash-attention parameter block, as the CONTRACT states it
// (the host checks its layout field for field against the kernel's).
using SteelAttnParams = contract::SteelAttnParams;

inline std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t r = (u >> 16) & 1u;
  u += 0x7fffu + r;
  return (std::uint16_t)(u >> 16);
}

// Narrow a block of checkpoint words to bf16 in place of the caller's
// buffer. Shared by the cached loader and by the streaming refill's
// fill_unservable, which is the whole reason it takes raw pointers.
bool
narrow_to_bf16_(const void* src, const std::string& dtype, std::size_t n,
                std::uint16_t* dst)
{
  if (dtype == "BF16") {
    std::memcpy(dst, src, n * 2);
    return true;
  }
  if (dtype == "F32") {
    const auto* s = static_cast<const float*>(src);
    for (std::size_t i = 0; i < n; ++i) { dst[i] = f32_to_bf16_(s[i]); }
    return true;
  }
  if (dtype == "F16") {
    const auto* s = static_cast<const _Float16*>(src);
    for (std::size_t i = 0; i < n; ++i) {
      dst[i] = f32_to_bf16_((float)s[i]);
    }
    return true;
  }
  return false;
}

// Load a checkpoint tensor as bf16 (BF16 copied; F16/F32 converted).
SharedBuffer
to_bf16_(const MetalLlamaWeights& wts, MetalCompute* mc,
         const std::string& nm)
{
  const auto* info = wts.info(nm);
  if (info == nullptr || info->shape.empty()) { return {}; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer raw = wts.load(nm, mc);
  if (raw.empty()) { return {}; }
  SharedBuffer out = mc->make_shared_buffer(n * 2);
  if (out.empty()) { return {}; }
  if (!narrow_to_bf16_(raw.contents(), info->dtype, n,
                       static_cast<std::uint16_t*>(out.contents()))) {
    return {};
  }
  return out;
}

// Derived-cache namespace. A WeightSet is shared by everything reading
// one checkpoint, so a key names the TRANSFORM, not just the tensor.
constexpr const char* kKey = "ming-dit/";

// The streamed stack is ONE stack spelled under TWO prefixes.
constexpr const char* kStreamed[] = {"noise_refiner.", "layers."};

bool
clone_buf_(MetalCompute* mc, const SharedBuffer& s, SharedBuffer& d,
           bool copy)
{
  if (s.empty()) { d = {}; return true; }
  d = mc->make_shared_buffer(s.byte_size());
  if (d.empty()) { return false; }
  if (copy) { std::memcpy(d.contents(), s.contents(), s.byte_size()); }
  return true;
}

bool
env_on_(const char* name)
{
  const char* e = std::getenv(name);
  return e != nullptr && *e != '\0' && std::atoi(e) != 0;
}

// ---- the internal config ----------------------------------------------
//
// The in-tree Config, minus the ANE fields, filled from ming::DitConfig
// rather than read off config.json a second time.
struct Cfg {
  int hidden      = 3840;    // `dim`
  int n_heads     = 30;
  int head_dim    = 128;     // dim / n_heads; also sum(axes)
  int n_layers    = 30;      // the MAIN stack
  int n_refiner   = 2;       // each of context_refiner and noise_refiner
  int in_channels = 16;
  int patch       = 2;
  int f_patch     = 1;
  int ffn_inner   = 10240;   // int(dim / 3 * 8)
  int cap_dim     = 2560;    // cap_feat_dim: stream A's width
  int adaln_dim   = 256;     // ADALN_EMBED_DIM = min(dim, 256)
  int time_mid    = 1024;    // TimestepEmbedder mid_size (fixed)
  int time_freq   = 256;     // frequency_embedding_size
  float norm_eps  = 1e-5f;   // the RMSNorms and the qk norms
  float rope_theta = 256.0f;
  float t_scale   = 1000.0f;
  int axes[3]     = {32, 48, 48};
  int axes_lens[3] = {20480, 512, 512};

  bool i8_gemm = false;
  sage::Config sage;
  sol::Config  sol;
  // The ANE feed-forward: on (and granted), the ANE's share of the rows
  // (0 balances from measured rates) and a cap on the main-stack blocks
  // that use it (0: every one).
  bool  ane_ffn    = false;
  float ane_rows   = 0.0f;
  int   ane_layers = 0;

  int patch_dim() const { return f_patch * patch * patch * in_channels; }
  int n_mod_blocks() const { return n_refiner + n_layers; }
  int n_blocks() const { return n_refiner + n_mod_blocks(); }
};

// Everything this port does not run is REFUSED here by name, rather
// than run over weights it would read wrongly.
bool
cfg_from_(const DitConfig& d, Cfg* c, std::string* why)
{
  auto no = [&](const std::string& m) {
    if (why != nullptr) { *why = m; }
    return false;
  };
  if (d.n_heads <= 0 || d.dim <= 0 || d.dim % d.n_heads != 0) {
    return no("dim is not a whole number of heads");
  }
  // GQA would read past the k/v projections; no published checkpoint
  // uses it.
  if (d.n_kv_heads != d.n_heads) {
    return no("n_kv_heads != n_heads (GQA) is not supported");
  }
  if (d.axes_dims.size() != 3 || d.axes_lens.size() != 3) {
    return no("axes_dims / axes_lens must have three entries");
  }
  // THIS PORT DROPS THE PAD ROWS, which is exact only because a
  // zero_masked pad attends to nothing and is attended by nothing. A
  // "learned" checkpoint's pads are real tokens; running it here would
  // be a plausible picture of the wrong model.
  if (d.padding_mode != "zero_masked") {
    return no("alignment_padding_mode '" + d.padding_mode +
              "' -- this DiT runs zero_masked checkpoints only");
  }
  if (d.multi_frame_output) {
    return no("multi_frame_output -- this DiT runs one frame");
  }
  if (!d.qk_norm) {
    return no("qk_norm false is not supported (the checkpoint's own "
              "config has it true)");
  }
  if (d.f_patch != 1 || d.patch <= 0) {
    return no("only all_f_patch_size [1] and a positive patch run here");
  }
  c->hidden = d.dim;
  c->n_heads = d.n_heads;
  c->head_dim = d.dim / d.n_heads;
  c->n_layers = d.n_layers;
  c->n_refiner = d.n_refiner_layers;
  c->in_channels = d.in_channels;
  c->patch = d.patch;
  c->f_patch = d.f_patch;
  // Not in config.json at all: the reference computes int(dim / 3 * 8).
  c->ffn_inner = (int)((double)d.dim / 3.0 * 8.0);
  c->cap_dim = d.cap_feat_dim;
  c->adaln_dim = std::min(d.dim, 256);
  c->norm_eps = d.norm_eps;
  c->rope_theta = d.rope_theta;
  c->t_scale = d.t_scale;
  for (int i = 0; i < 3; ++i) {
    c->axes[i] = d.axes_dims[(std::size_t)i];
    c->axes_lens[i] = d.axes_lens[(std::size_t)i];
  }
  // sum(axes_dims) == head_dim is the reference's own assert.
  if (c->axes[0] + c->axes[1] + c->axes[2] != c->head_dim) {
    return no("sum(axes_dims) != head_dim");
  }
  // The flash kernels are instantiated per head dim, and this model's is
  // the 128-wide one; the transpose / rope kernels take any even width.
  if (c->head_dim != 128) {
    return no("head_dim " + std::to_string(c->head_dim) +
              " -- the attention here is the 128-wide flash kernel");
  }
  if (c->n_layers <= 0 || c->n_refiner <= 0 || c->cap_dim <= 0 ||
      c->in_channels <= 0) {
    return no("a stack or a width in the config is not positive");
  }
  return true;
}

// The patch key the embedder and the final layer are stored under:
// f"{patch_size}-{f_patch_size}", a ModuleDict key in the tensor name.
std::string
patch_key_(const Cfg& c)
{
  return std::to_string(c.patch) + "-" + std::to_string(c.f_patch);
}

// ---- the model ----------------------------------------------------------

class MingDitMetal final : public MingDit {
 public:
  ~MingDitMetal() override;

  static std::unique_ptr<MingDit>
  load_(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
        const DitConfig& dcfg, const FlexData* accel, bool stream_blocks,
        const SessionContextIntf* session, std::string* err,
        const LoadOptions& opts);

  SharedBuffer forward(const Request& req, std::string* err) override;

  void set_lora_scale(int slot, float s) override;
  float lora_scale(int slot) const override
  {
    return slot >= 0 && slot < kMaxLoraSlots ? _lora_scale[slot] : 0.0f;
  }
  int lora_modules(int slot) const override
  {
    return slot >= 0 && slot < kMaxLoraSlots ? _lora_bound[slot] : 0;
  }
  int lora_skipped(int slot) const override
  {
    return slot >= 0 && slot < kMaxLoraSlots ? _lora_skipped[slot] : 0;
  }

  void set_stream_stop(std::function<bool()> stop) override
  {
    _stream_stop = std::move(stop);
  }
  void set_block_progress(DitBlockProgressFn fn) override
  {
    _block_progress = std::move(fn);
  }
  void set_residency_reserve(std::size_t bytes) override
  {
    _resid.set_reserve(bytes);
  }
  void set_residency_schedule(int steps) override;
  std::size_t release_resident_blocks(std::size_t bytes) override;
  std::uint64_t resident_bytes() const override;
  bool streaming() const override { return _stream_blocks; }
  const DitConfig& config() const override { return _dcfg; }

  bool uses_mma2() const override { return _use_mma2; }
  bool uses_attn_nax() const override { return _use_attn_nax; }
  int gemm_tile() const override { return _gemm_tile; }
  bool uses_i8_gemm() const override { return (bool)_i8; }
  bool uses_sage_attn() const override { return (bool)_sage; }
  bool uses_sol_attn() const override { return (bool)_sol; }
  bool ane_requested() const override { return _cfg.ane_ffn; }
  bool ane_armed() const override { return (bool)_ane; }
  bool ane_gpu_only() const override;
  int quant_bits() const override { return _quant_bits; }
  int resident_block_count() const override { return _resid.count(); }
  DitEngagement last_engagement() const override { return _eng; }
  void set_caption_cache(bool on) override
  {
    _cap_cache_on = on;
    if (!on) {
      for (CapEntry& e : _cap_cache) { e = CapEntry{}; }
    }
  }

 private:
  MingDitMetal() = default;

  enum class Retain { Cached, Streamed };

  struct QWeight {
    SharedBuffer w;                    // dense [N,K]
    SharedBuffer codes, scales, qbias; // affine quant
    bool quantized = false;
    int  bits = 0;
    bool empty() const { return quantized ? codes.empty() : w.empty(); }
  };

  // Everything a block owns. The attention and feed-forward projections
  // are biasless; adaLN is the one Linear in a block that has a bias,
  // and the context_refiner blocks do not have it at all.
  struct Block {
    QWeight qw, kw, vw, ow;              // attention (to_q/k/v/to_out.0)
    SharedBuffer nq, nk;                 // q/k RMSNorm [head_dim]
    QWeight w1, w3, w2;                  // SwiGLU: gate, up, down
    SharedBuffer an1, an2;               // attention_norm1 / norm2
    SharedBuffer fn1, fn2;               // ffn_norm1 / ffn_norm2
    QWeight ada;                         // adaLN_modulation.0 [4H, 256]
    SharedBuffer ada_b;                  // its bias [4H]
  };

  // One slot's factors for one block. The adaLN projection is
  // deliberately NOT adapted: no publisher in this lineage trains it, and
  // binding it would take a file's silence for a zero.
  struct BlockLora {
    lora::Factors q, k, v, o, w1, w3, w2;
  };

  // One cached caption: the exact bytes it was built from, and the
  // context-refined rows [len, hidden] they produce.
  struct CapEntry {
    bool valid = false;
    std::vector<std::uint8_t> key;
    int q = 0, len = 0, stride = 0;
    SharedBuffer out;
    std::uint64_t tick = 0;
  };

  void log_(const vpipe::VpipeFormat& m) const
  {
    if (_session != nullptr) { _session->log_normal(m); }
  }
  void warn_(const vpipe::VpipeFormat& m) const
  {
    if (_session != nullptr) { _session->warn(m); }
  }

  std::string mod_prefix_(int i) const;
  std::string ctx_prefix_(int i) const;

  QWeight load_qw_(WeightSet& ws, const std::string& name, Retain r);
  SharedBuffer to_elt_(WeightSet& ws, const std::string& name, Retain r);
  bool load_block_(WeightSet& ws, const std::string& pre, bool with_ada,
                   Block& b, Retain r);
  bool load_fixed_(WeightSet& ws);
  bool check_shapes_(std::string* why) const;

  void each_block_tensor_(int L, Block& b,
                          const BlockSlots<Block>::TensorFn& fn) const;
  bool clone_block_(const Block& src, Block& dst, bool copy) const;
  SharedBuffer rebuild_one_(const std::string& nm, Placement how);
  bool fill_narrow_(const std::string& nm, const SharedBuffer& dst);
  void configure_slots_();
  static std::size_t qw_bytes_(const QWeight& w);
  static std::size_t block_bytes_(const Block& b);

  std::vector<float> time_proj_(float sigma) const;
  std::size_t checkpoint_block_bytes_() const;

  bool gemm_mma_(ComputeEncoder& enc, const SharedBuffer& xin,
                 std::size_t xe, const QWeight& w, const SharedBuffer& y,
                 std::size_t ye, int M, int N, int K) const;

  std::size_t wire_block_(Block& b, bool on);
  std::size_t wire_fixed_(bool on);
  std::size_t wire_retry_slack_() const;
  std::size_t evict_tail_block_();
  void resident_pages_(std::size_t* examined, std::size_t* incore,
                       std::size_t* paged_out) const;
  std::size_t _wire_block_hint = 0;

  // cos/sin tables for the JOINT sequence, f32, [joint, head_dim] each.
  // The refiners take a prefix (the image) and a suffix (the caption).
  void build_rope_(const Layout& lay, SharedBuffer* cos_out,
                   SharedBuffer* sin_out) const;

  // ---- the ANE feed-forward (`ane_ffn`) ----------------------------------
  bool ane_setup_(int seq);
  bool ane_eligible_(int L, const Block& b) const;
  bool ane_stage_(int L, const Block& b);
  std::unique_ptr<ane::Tier> _ane;
  // The longest sequence a create was tried at: a forward too short for
  // one chunk declines, and a longer one later is asked again.
  int  _ane_tried_seq = 0;
  bool _ane_warned = false;
  // What the scratch ledger was last told the module holds, and under
  // which label: the host's (LoadOptions::coreml_label) when it booked
  // the module itself, else kAneClaimLabel.
  std::size_t _ane_booked = SIZE_MAX;
  std::string _ane_label;
  // Whether the block staged last carried adapter factors for the ANE.
  bool _ane_stage_lora = false;
  // A TEST'S NEGATIVE CONTROL (VPIPE_MING_TEST_ANE_DROP_LORA, read at
  // load): stage the bare weights, so the ANE's rows run the base model.
  // It exists to show the adapter-on-the-ANE bar can fail; it is a wrong
  // picture by construction and never belongs in a graph.
  bool _ane_drop_lora = false;

  // ---- runtime LoRA ---------------------------------------------------
  //
  // Indexed [slot][block] over the SAME index space mod_prefix_ uses (the
  // streamed stack), with the context refiner's blocks kept separately
  // because they are not in it. Bound once at load and RESIDENT in both
  // modes: a streamed block's base weights come from a slot, its factors
  // from here. Only the strength is live.
  lora::Applier _lora;
  std::vector<BlockLora> _lora_mod[kMaxLoraSlots];
  std::vector<BlockLora> _lora_ctx[kMaxLoraSlots];
  float _lora_scale[kMaxLoraSlots] = {};
  int _lora_bound[kMaxLoraSlots] = {};
  int _lora_skipped[kMaxLoraSlots] = {};
  int _lora_rank_prefix[kMaxLoraSlots] = {};
  int _lora_rank_total = 0;
  int _lora_slots = 0;
  std::size_t _lora_bytes = 0;
  // Rows the [rows, rank] intermediate is sized for THIS forward, so a
  // slot's base offset is rows * its rank prefix.
  std::size_t _lora_scratch_rows = 0;
  bool bind_loras_(const std::vector<LoraSpec>& loras);
  // The per-projection stack for one block, empty when nothing is
  // attached (or every attached strength is 0). `ctx` selects the
  // context refiner's index space.
  lora::Stack lora_for_(bool ctx, int L,
                        lora::Factors BlockLora::* which) const;

  MetalCompute* _mc = nullptr;
  const SessionContextIntf* _session = nullptr;
  DitConfig _dcfg;
  Cfg _cfg;
  int _quant_bits = 0;
  int _quant_group = 64;
  bool _src_f32 = false;

  // Model-level weights. No pad tokens: a zero_masked checkpoint has
  // none, and this port runs no pad rows.
  QWeight _x_in;                            // all_x_embedder [H, patch_dim]
  SharedBuffer _x_in_b;
  SharedBuffer _cap_norm;                   // cap_embedder.0 [cap_dim]
  QWeight _cap_in;                          // cap_embedder.1 [H, cap_dim]
  SharedBuffer _cap_in_b;
  QWeight _time_1, _time_2;                 // t_embedder.mlp.0 / .2
  SharedBuffer _time_1b, _time_2b;
  QWeight _final_ada;                       // final adaLN [H, 256]
  SharedBuffer _final_ada_b;
  QWeight _final_lin;                       // [patch_dim, H]
  SharedBuffer _final_lin_b;

  std::vector<Block> _ctx;                  // context_refiner, ALWAYS held
  std::vector<Block> _blocks;               // the modulated stack
  bool _stream_blocks = false;
  BlockSlots<Block> _slots;
  BlockResidency _resid;
  WiredPool _wire;
  std::shared_ptr<WeightSet> _ws;
  std::function<bool()> _stream_stop;
  DitBlockProgressFn _block_progress;
  std::vector<std::uint8_t> _narrow_scratch;

  ComputeLibrary _lib_gemm, _lib_elt, _lib_own, _lib_rms, _lib_rope,
      _lib_attn, _lib_attn_nax, _lib_mma, _lib_dequant, _lib_qmm;
  ComputeFunction _fn_ln_plain, _fn_adaln, _fn_adaln4, _fn_gated_tanh,
      _fn_gated_tanh4, _fn_rms, _fn_swiglu, _fn_swiglu4, _fn_tr_rope,
      _fn_residual, _fn_residual4, _fn_gemm_t, _fn_qmm4, _fn_qmm8,
      _fn_dequant4, _fn_dequant8, _fn_gemm_mma, _fn_gemm_mma_deep,
      _fn_gemm_mma_tn2, _fn_copy_rect, _fn_gemm_bm64, _fn_gemm_bm64bn64,
      _fn_silu, _fn_transpose, _fn_bias_add;
  // The steel entry, specialized per (q-aligned, k-aligned). Memoized
  // because building a pipeline state is not free and a sampler runs
  // the same three sequence lengths on every one of its steps.
  std::map<unsigned, ComputeFunction> _attn_fn;
  bool _use_mma2 = false;
  bool _use_attn_nax = false;
  bool _steel_attn_ok = false;
  int  _mma_min_m = 64;
  int  _gemm_tile = 2;
  // Fold (1 + scale) into the norm's gain instead of modulating the
  // normed tensor. Exact algebra; the flag exists for the A/B.
  bool _fold_mod = true;
  static constexpr int kSageMinSegment = 512;
  static constexpr int kSolMinSegment = 1024;
  // How many blocks a NON-streamed stack may encode before it drains,
  // so a stop that arrives while the GPU is busy is seen at all. Only
  // paid when a stop callback is installed.
  static constexpr int kStopDrainBlocks = 4;
  mutable SharedBuffer _w_deq;
  std::unique_ptr<MetalSageAttention> _sage;
  std::unique_ptr<MetalSolAttention> _sol;
  std::unique_ptr<I8GemmContext> _i8;
  // The host's FLOP/s meter (perf-flops): GEMMs and attention counted as
  // they are encoded, credited to the GPU as each block's buffer
  // retires; Sol's exact half at the keep ratio it realizes. A host
  // without the meter takes none of it (the toolkit checks).
  GpuFlopsTally _flops;
  SolFlops _sol_flops;

  bool _cap_cache_on = true;
  CapEntry _cap_cache[2];
  std::uint64_t _cap_tick = 0;

  mutable DitEngagement _eng;
};

MingDitMetal::~MingDitMetal()
{
  // GIVE THE POOL BACK. Freeing a wired buffer unwires it in the kernel,
  // but the pool's own counter would not follow, and a DiT destroyed
  // and reloaded per prompt would leak its share of the budget each time.
  _slots.join();
  // Nothing the ANE worker may still be reading outlives this: `_ane` is
  // declared before the blocks, so it would otherwise die after them.
  if (_ane) { _ane->join(); }
  if (_wire.on()) {
    wire_fixed_(false);
    for (Block& b : _blocks) { wire_block_(b, false); }
    for (Block& b : _ctx) { wire_block_(b, false); }
  }
}

// ---- block names ------------------------------------------------------
//
// The streaming stack is noise_refiner THEN layers, which is not the
// order the reference runs them in -- the context refiner runs first
// here. What makes that safe is that the two refiners touch disjoint
// rows and never see each other's output.

std::string
MingDitMetal::mod_prefix_(int i) const
{
  if (i < _cfg.n_refiner) {
    return "noise_refiner." + std::to_string(i) + ".";
  }
  return "layers." + std::to_string(i - _cfg.n_refiner) + ".";
}

std::string
MingDitMetal::ctx_prefix_(int i) const
{
  return "context_refiner." + std::to_string(i) + ".";
}

// ---- residency + the wired pool ---------------------------------------

std::size_t
MingDitMetal::wire_block_(Block& b, bool on)
{
  std::size_t changed = 0;
  bool stop = false;
  auto one = [&](SharedBuffer& p) {
    if (stop) { return; }
    const std::size_t n = _wire.wire_one(_mc, p, on);
    // STOP AT THE FIRST REFUSAL: the pool is full, and the remaining
    // asks would each cost a syscall to be told so.
    if (on && n == 0 && p.byte_size() > 0 && !p.is_wired() &&
        _wire.refused(_mc, p)) {
      stop = true;
      return;
    }
    changed += n;
  };
  auto qw = [&](QWeight& w) {
    one(w.w); one(w.codes); one(w.scales); one(w.qbias);
  };
  QWeight* q[] = {&b.qw, &b.kw, &b.vw, &b.ow, &b.w1, &b.w3, &b.w2, &b.ada};
  for (QWeight* w : q) { qw(*w); }
  one(b.nq); one(b.nk);
  one(b.an1); one(b.an2); one(b.fn1); one(b.fn2);
  one(b.ada_b);
  return changed;
}

std::size_t
MingDitMetal::wire_fixed_(bool on)
{
  if (!_ws) { return 0; }
  std::size_t changed = 0;
  _ws->for_each_weight([&](SharedBuffer& b) {
    changed += _wire.wire_one(_mc, b, on);
  });
  return changed;
}

std::size_t
MingDitMetal::wire_retry_slack_() const
{
  if (_wire_block_hint > 0) { return _wire_block_hint; }
  if (_resid.count() > 0) {
    return _resid.bytes() / (std::size_t)_resid.count();
  }
  return 0;
}

std::size_t
MingDitMetal::evict_tail_block_()
{
  // From the END. The next forward starts at block 0, so the deepest
  // resident block is the one whose re-read is furthest away.
  for (int i = (int)_blocks.size() - 1; i >= 0; --i) {
    Block& b = _blocks[(std::size_t)i];
    const std::size_t n = block_bytes_(b);
    if (n == 0) { continue; }
    _wire.note_unwired(wire_block_(b, false));
    b = Block{};
    return n;
  }
  return 0;
}

void
MingDitMetal::resident_pages_(std::size_t* examined, std::size_t* incore,
                              std::size_t* paged_out) const
{
  *examined = 0;
  *incore = 0;
  if (paged_out != nullptr) { *paged_out = 0; }
  auto add = [&](const SharedBuffer& p) {
    if (p.byte_size() == 0) { return; }
    // A WIRED BUFFER CANNOT HAVE LEFT RAM.
    if (p.is_wired()) { return; }
    if (!GenerativeModelManager::pool_wirable(p)) { return; }
    const auto r = p.page_residency(64);
    if (!r.valid) { return; }
    *examined += r.examined;
    *incore += r.incore;
    if (paged_out != nullptr) { *paged_out += r.paged_out; }
  };
  auto qw = [&](const QWeight& w) {
    add(w.w); add(w.codes); add(w.scales); add(w.qbias);
  };
  for (const Block& b : _blocks) {
    if (block_bytes_(b) == 0) { continue; }
    qw(b.qw); qw(b.kw); qw(b.vw); qw(b.ow);
    qw(b.w1); qw(b.w3); qw(b.w2); qw(b.ada);
    add(b.nq); add(b.nk); add(b.an1); add(b.an2); add(b.fn1); add(b.fn2);
  }
}

// ---- weights ----------------------------------------------------------

SharedBuffer
MingDitMetal::to_elt_(WeightSet& ws, const std::string& name, Retain r)
{
  auto build = [&]() { return to_bf16_(ws.src(), _mc, name); };
  if (r == Retain::Streamed) { return ws.stream_derived(build); }
  return ws.derived(std::string(kKey) + "elt|" + name, build);
}

MingDitMetal::QWeight
MingDitMetal::load_qw_(WeightSet& ws, const std::string& name, Retain r)
{
  QWeight qw;
  const MetalLlamaWeights& wts = ws.src();
  const auto* si = wts.info(name + ".scales");
  const auto* ci = wts.info(name + ".weight");
  if (_quant_bits > 0 && si != nullptr && ci != nullptr &&
      si->shape.size() == 2 && ci->shape.size() == 2) {
    const long gcols = ci->shape[1];              // K*bits/32
    const long scols = si->shape[1];              // K/group
    const long K = scols * (long)_quant_group;
    const int bits = K > 0 ? (int)(gcols * 32 / K) : 0;
    qw.bits = (bits == 8) ? 8 : 4;
    qw.codes = r == Retain::Streamed
                   ? ws.stream_tensor(name + ".weight", _mc,
                                      WeightSet::Residency::Copied)
                   : ws.tensor(name + ".weight", _mc,
                               WeightSet::Residency::Copied);
    qw.scales = to_elt_(ws, name + ".scales", r);
    qw.qbias = to_elt_(ws, name + ".biases", r);
    if (!qw.codes.empty() && !qw.scales.empty() && !qw.qbias.empty()) {
      qw.quantized = true;
      return qw;
    }
    qw.codes = {}; qw.scales = {}; qw.qbias = {};
  }
  qw.w = to_elt_(ws, name + ".weight", r);
  return qw;
}

bool
MingDitMetal::load_block_(WeightSet& ws, const std::string& p,
                          bool with_ada, Block& b, Retain r)
{
  b.qw = load_qw_(ws, p + "attention.to_q", r);
  b.kw = load_qw_(ws, p + "attention.to_k", r);
  b.vw = load_qw_(ws, p + "attention.to_v", r);
  // to_out is a ModuleList, so the projection is ".0" -- not a typo.
  b.ow = load_qw_(ws, p + "attention.to_out.0", r);
  b.nq = to_elt_(ws, p + "attention.norm_q.weight", r);
  b.nk = to_elt_(ws, p + "attention.norm_k.weight", r);
  // SwiGLU as Llama names it: w1 gates, w3 is the up projection and w2
  // the down one. Reading w1 and w3 the other way round still denoises.
  b.w1 = load_qw_(ws, p + "feed_forward.w1", r);
  b.w3 = load_qw_(ws, p + "feed_forward.w3", r);
  b.w2 = load_qw_(ws, p + "feed_forward.w2", r);
  b.an1 = to_elt_(ws, p + "attention_norm1.weight", r);
  b.an2 = to_elt_(ws, p + "attention_norm2.weight", r);
  b.fn1 = to_elt_(ws, p + "ffn_norm1.weight", r);
  b.fn2 = to_elt_(ws, p + "ffn_norm2.weight", r);
  bool ok = !b.qw.empty() && !b.kw.empty() && !b.vw.empty() &&
            !b.ow.empty() && !b.nq.empty() && !b.nk.empty() &&
            !b.w1.empty() && !b.w3.empty() && !b.w2.empty() &&
            !b.an1.empty() && !b.an2.empty() && !b.fn1.empty() &&
            !b.fn2.empty();
  if (with_ada) {
    // `.0` -- a bare Linear with NO SiLU in front of it. The final
    // layer's adaLN is `.1` because that one does have one.
    b.ada = load_qw_(ws, p + "adaLN_modulation.0", r);
    b.ada_b = to_elt_(ws, p + "adaLN_modulation.0.bias", r);
    ok = ok && !b.ada.empty() && !b.ada_b.empty();
  }
  return ok;
}

bool
MingDitMetal::load_fixed_(WeightSet& ws)
{
  const Retain r = Retain::Cached;
  const std::string pk = patch_key_(_cfg);
  _x_in = load_qw_(ws, "all_x_embedder." + pk, r);
  _x_in_b = to_elt_(ws, "all_x_embedder." + pk + ".bias", r);
  _cap_norm = to_elt_(ws, "cap_embedder.0.weight", r);
  _cap_in = load_qw_(ws, "cap_embedder.1", r);
  _cap_in_b = to_elt_(ws, "cap_embedder.1.bias", r);
  _time_1 = load_qw_(ws, "t_embedder.mlp.0", r);
  _time_1b = to_elt_(ws, "t_embedder.mlp.0.bias", r);
  _time_2 = load_qw_(ws, "t_embedder.mlp.2", r);
  _time_2b = to_elt_(ws, "t_embedder.mlp.2.bias", r);
  _final_ada = load_qw_(ws, "all_final_layer." + pk + ".adaLN_modulation.1",
                        r);
  _final_ada_b = to_elt_(
      ws, "all_final_layer." + pk + ".adaLN_modulation.1.bias", r);
  _final_lin = load_qw_(ws, "all_final_layer." + pk + ".linear", r);
  _final_lin_b = to_elt_(ws, "all_final_layer." + pk + ".linear.bias", r);
  return !_x_in.empty() && !_x_in_b.empty() && !_cap_norm.empty() &&
         !_cap_in.empty() && !_cap_in_b.empty() && !_time_1.empty() &&
         !_time_1b.empty() && !_time_2.empty() && !_time_2b.empty() &&
         !_final_ada.empty() && !_final_ada_b.empty() &&
         !_final_lin.empty() && !_final_lin_b.empty();
}

// The config's widths against the tensors' own shapes, for a DENSE
// checkpoint (a pack's codes are not [N, K]). Several of these widths
// are not in config.json at all -- the feed-forward is int(dim/3*8) and
// the timestep MLP's middle is a constant -- so a checkpoint that
// disagrees fails HERE, by name, rather than at the first GEMM as a
// read past the end of a weight.
bool
MingDitMetal::check_shapes_(std::string* why) const
{
  if (_quant_bits > 0 || !_ws) { return true; }
  const MetalLlamaWeights& w = _ws->src();
  const Cfg& c = _cfg;
  const std::string pk = patch_key_(c);
  struct Want { std::string name; long rows, cols; };
  const std::string b0 = mod_prefix_(0);
  const Want want[] = {
      {"all_x_embedder." + pk + ".weight", c.hidden, c.patch_dim()},
      {"cap_embedder.1.weight", c.hidden, c.cap_dim},
      {"t_embedder.mlp.0.weight", c.time_mid, c.time_freq},
      {"t_embedder.mlp.2.weight", c.adaln_dim, c.time_mid},
      {"all_final_layer." + pk + ".linear.weight", c.patch_dim(), c.hidden},
      {b0 + "attention.to_q.weight", c.hidden, c.hidden},
      {b0 + "feed_forward.w1.weight", c.ffn_inner, c.hidden},
      {b0 + "feed_forward.w2.weight", c.hidden, c.ffn_inner},
      {b0 + "adaLN_modulation.0.weight", 4L * c.hidden, c.adaln_dim},
  };
  for (const Want& x : want) {
    const auto* ti = w.info(x.name);
    if (ti == nullptr || ti->shape.size() != 2 ||
        ti->shape[0] != x.rows || ti->shape[1] != x.cols) {
      if (why != nullptr) {
        *why = x.name + (ti == nullptr ? " is missing"
                                       : " does not have the shape [" +
                                             std::to_string(x.rows) + ", " +
                                             std::to_string(x.cols) +
                                             "] the config implies");
      }
      return false;
    }
  }
  return true;
}

std::size_t
MingDitMetal::qw_bytes_(const QWeight& w)
{
  if (w.quantized) {
    return w.codes.byte_size() + w.scales.byte_size() + w.qbias.byte_size();
  }
  return w.w.byte_size();
}

std::size_t
MingDitMetal::block_bytes_(const Block& b)
{
  return qw_bytes_(b.qw) + qw_bytes_(b.kw) + qw_bytes_(b.vw) +
         qw_bytes_(b.ow) + b.nq.byte_size() + b.nk.byte_size() +
         qw_bytes_(b.w1) + qw_bytes_(b.w3) + qw_bytes_(b.w2) +
         b.an1.byte_size() + b.an2.byte_size() + b.fn1.byte_size() +
         b.fn2.byte_size() + qw_bytes_(b.ada) + b.ada_b.byte_size();
}

void
MingDitMetal::each_block_tensor_(int L, Block& b,
                                 const BlockSlots<Block>::TensorFn& fn) const
{
  const std::string p = mod_prefix_(L);
  auto qw = [&](QWeight& w, const char* nm) {
    if (w.quantized) {
      fn(p + nm + ".weight", w.codes, Placement::kRaw);
      fn(p + nm + ".scales", w.scales, Placement::kBf16);
      fn(p + nm + ".biases", w.qbias, Placement::kBf16);
    } else {
      fn(p + nm + ".weight", w.w, Placement::kBf16);
    }
  };
  qw(b.qw, "attention.to_q");
  qw(b.kw, "attention.to_k");
  qw(b.vw, "attention.to_v");
  qw(b.ow, "attention.to_out.0");
  fn(p + "attention.norm_q.weight", b.nq, Placement::kBf16);
  fn(p + "attention.norm_k.weight", b.nk, Placement::kBf16);
  qw(b.w1, "feed_forward.w1");
  qw(b.w3, "feed_forward.w3");
  qw(b.w2, "feed_forward.w2");
  fn(p + "attention_norm1.weight", b.an1, Placement::kBf16);
  fn(p + "attention_norm2.weight", b.an2, Placement::kBf16);
  fn(p + "ffn_norm1.weight", b.fn1, Placement::kBf16);
  fn(p + "ffn_norm2.weight", b.fn2, Placement::kBf16);
  qw(b.ada, "adaLN_modulation.0");
  fn(p + "adaLN_modulation.0.bias", b.ada_b, Placement::kBf16);
}

bool
MingDitMetal::clone_block_(const Block& src, Block& dst, bool copy) const
{
  bool ok = true;
  auto cq = [&](const QWeight& s, QWeight& d) {
    d.quantized = s.quantized;
    d.bits = s.bits;
    ok = clone_buf_(_mc, s.w, d.w, copy) && ok;
    ok = clone_buf_(_mc, s.codes, d.codes, copy) && ok;
    ok = clone_buf_(_mc, s.scales, d.scales, copy) && ok;
    ok = clone_buf_(_mc, s.qbias, d.qbias, copy) && ok;
  };
  cq(src.qw, dst.qw); cq(src.kw, dst.kw); cq(src.vw, dst.vw);
  cq(src.ow, dst.ow);
  ok = clone_buf_(_mc, src.nq, dst.nq, copy) && ok;
  ok = clone_buf_(_mc, src.nk, dst.nk, copy) && ok;
  cq(src.w1, dst.w1); cq(src.w3, dst.w3); cq(src.w2, dst.w2);
  ok = clone_buf_(_mc, src.an1, dst.an1, copy) && ok;
  ok = clone_buf_(_mc, src.an2, dst.an2, copy) && ok;
  ok = clone_buf_(_mc, src.fn1, dst.fn1, copy) && ok;
  ok = clone_buf_(_mc, src.fn2, dst.fn2, copy) && ok;
  cq(src.ada, dst.ada);
  ok = clone_buf_(_mc, src.ada_b, dst.ada_b, copy) && ok;
  return ok;
}

SharedBuffer
MingDitMetal::rebuild_one_(const std::string& nm, Placement how)
{
  if (_ws == nullptr) { return {}; }
  if (how == Placement::kRaw) {
    return _ws->stream_tensor(nm, _mc, WeightSet::Residency::Copied);
  }
  return _ws->stream_derived(
      [&]() { return to_bf16_(_ws->src(), _mc, nm); });
}

bool
MingDitMetal::fill_narrow_(const std::string& nm, const SharedBuffer& dst)
{
  // AN F32 CHECKPOINT has bytes twice the width of every buffer this
  // model reads, so a raw read has nowhere to put them -- which is what
  // BlockSlots' fill_unservable exists for. One scratch for the run,
  // grown to the largest tensor asked for. (The published checkpoint is
  // bf16 and never reaches this; an f32 test checkpoint does.)
  if (_ws == nullptr || dst.empty()) { return false; }
  const auto* info = _ws->src().info(nm);
  if (info == nullptr || info->shape.empty()) { return false; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  if (dst.byte_size() < n * 2) { return false; }
  const std::size_t need = (std::size_t)info->nbytes;
  if (_narrow_scratch.size() < need) { _narrow_scratch.resize(need); }
  if (!_ws->stream_into(nm, _narrow_scratch.data(), need)) { return false; }
  return narrow_to_bf16_(_narrow_scratch.data(), info->dtype, n,
                         static_cast<std::uint16_t*>(dst.contents()));
}

void
MingDitMetal::configure_slots_()
{
  BlockSlots<Block>::Ops ops;
  ops.each = [this](int L, Block& b,
                    const BlockSlots<Block>::TensorFn& fn) {
    each_block_tensor_(L, b, fn);
  };
  ops.rebuild_one = [this](const std::string& nm, Placement how) {
    return rebuild_one_(nm, how);
  };
  ops.fill_unservable = [this](const std::string& nm,
                               const SharedBuffer& d) {
    return fill_narrow_(nm, d);
  };
  ops.build = [this](int L, Block& b) {
    return _ws != nullptr &&
           load_block_(*_ws, mod_prefix_(L), true, b, Retain::Streamed);
  };
  ops.clone = [this](const Block& s, Block& d, bool copy) {
    return clone_block_(s, d, copy);
  };
  ops.bytes = [](const Block& b) { return block_bytes_(b); };
  ops.empty = [](const Block& b) { return b.qw.empty(); };
  _slots.set_weight_set(_ws.get());
  _slots.configure(_mc, std::move(ops), "MingDit", "VPIPE_MING_NO_SLOTS");
}

// ---- load -------------------------------------------------------------

std::unique_ptr<MingDit>
MingDitMetal::load_(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                    const DitConfig& dcfg, const FlexData* accel_bag,
                    bool stream_blocks, const SessionContextIntf* session,
                    std::string* err, const LoadOptions& opts)
{
  auto refuse = [&](const std::string& why) -> std::unique_ptr<MingDit> {
    if (err != nullptr) { *err = why; }
    return nullptr;
  };
  if (ws == nullptr || mc == nullptr) {
    return refuse("no weight set or no metal-compute backend");
  }
  std::unique_ptr<MingDitMetal> m(new MingDitMetal());
  m->_mc = mc;
  m->_session = session != nullptr ? session : mc->session();
  m->_dcfg = dcfg;
  {
    std::string why;
    if (!cfg_from_(dcfg, &m->_cfg, &why)) { return refuse(why); }
  }
  m->_ws = std::move(ws);
  m->_stream_blocks = stream_blocks;

  // The acceleration bag. Null reads as every default.
  m->_cfg.i8_gemm = accel::flag(accel_bag, accel::kI8Gemm);
  m->_cfg.sage = sage::config_from_flex(accel_bag);
  m->_cfg.sol = sol::config_from_flex(accel_bag);
  m->_cfg.ane_ffn = accel::flag(accel_bag, accel::kAneFfn);
  m->_cfg.ane_rows = (float)accel::real(accel_bag, accel::kAneRows, 0.0);
  m->_cfg.ane_layers = (int)std::max<long long>(
      0, accel::integer(accel_bag, accel::kAneLayers, 0));
  for (std::string_view t : accel::tiers_on(accel_bag)) {
    if (t == accel::kAneQkv || t == accel::kMotionCache) {
      m->warn_(fmt("ming-image DiT: {} was requested and this DiT does "
                   "not implement it; it runs without", t));
    }
  }

  // Quantization is read off the checkpoint, not configured: a pack
  // carries `.scales` beside its codes and a dense one does not.
  {
    const MetalLlamaWeights& w = m->_ws->src();
    const std::string p0 = m->mod_prefix_(0);
    const auto* si = w.info(p0 + "attention.to_q.scales");
    const auto* ci = w.info(p0 + "attention.to_q.weight");
    if (ci == nullptr) {
      return refuse("'" + p0 + "attention.to_q.weight' is not in the "
                    "checkpoint -- not this model's transformer");
    }
    if (si != nullptr && si->shape.size() == 2 && ci->shape.size() == 2 &&
        si->shape[1] > 0) {
      m->_quant_group = (int)(m->_cfg.hidden / si->shape[1]);
      if (m->_quant_group <= 0) { m->_quant_group = 64; }
      const long K = si->shape[1] * (long)m->_quant_group;
      m->_quant_bits = K > 0 ? (int)(ci->shape[1] * 32 / K) : 0;
      if (m->_quant_bits != 4 && m->_quant_bits != 8) { m->_quant_bits = 0; }
    }
    if (m->_quant_bits == 0) { m->_src_f32 = ci->dtype == "F32"; }
  }
  {
    std::string why;
    if (!m->check_shapes_(&why)) { return refuse(why); }
  }

  // THE LIBRARY PICKS THE ELEMENT TYPE; the entry names below are the
  // MSL declarations, looked up in the .metal sources rather than
  // inferred (dense_gemm_t_f16 is the same entry in both twins).
  m->_lib_gemm = mc->load_library("dense_gemm_bf16");
  m->_lib_elt = mc->load_library("llm_elementwise_bf16");
  m->_lib_own = mc->load_library(kMetalLibBf16);
  m->_lib_rms = mc->load_library("rms_norm_bf16");
  m->_lib_rope = mc->load_library("rope_bf16");
  m->_lib_attn = mc->load_library("attn_steel");
  m->_lib_mma = mc->load_library("dense_gemm_mma_bf16");
  m->_lib_dequant = mc->load_library("affine_dequant_bf16");
  m->_lib_qmm = mc->load_library("affine_qmm_steel_bf16");
  if (mc->supports_matrix_cores()) {
    m->_lib_attn_nax = mc->load_library("attn_steel_nax");
  }
  if (!m->_lib_own.valid()) {
    return refuse(std::string("the plugin's own kernels ('") +
                  kMetalLibBf16 + "') are not registered -- the plugin "
                  "registers them at load; a test must do it itself");
  }

  m->_fn_gemm_t = m->_lib_gemm.function("dense_gemm_t_f16");
  // THE WIDER TILES, which are what a box WITHOUT matrix cores runs for
  // every GEMM in the model.
  m->_fn_gemm_bm64 = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
  m->_fn_gemm_bm64bn64 = m->_lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  m->_fn_ln_plain = m->_lib_elt.function("layer_norm_plain_f16");
  m->_fn_adaln = m->_lib_elt.function("adaln_modulate_f16");
  m->_fn_adaln4 = m->_lib_elt.function("adaln_modulate_v4_f16");
  m->_fn_swiglu = m->_lib_elt.function("swiglu_f16");
  m->_fn_bias_add = m->_lib_elt.function("bias_add_rows_f16");
  m->_fn_silu = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  // The plugin's own copies (src/metal/ming_ops.metal).
  m->_fn_gated_tanh = m->_lib_own.function("gated_residual_tanh_f16");
  m->_fn_gated_tanh4 = m->_lib_own.function("gated_residual_tanh_v4_f16");
  m->_fn_swiglu4 = m->_lib_own.function("swiglu_v4_f16");
  m->_fn_residual = m->_lib_own.function("residual_add_f16");
  m->_fn_residual4 = m->_lib_own.function("residual_add_v4_f16");
  m->_fn_copy_rect = m->_lib_own.function("copy_rect_f16");
  m->_fn_rms = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_tr_rope = m->_lib_rope.function("transpose_rope_pair_ftab_f16");
  // THE QUANTIZED ENTRY POINTS ARE NAMED BY THEIR GROUP -- not by a
  // `_f16` suffix. The contract carries groups 32 and 64.
  if (m->_quant_bits > 0) {
    const std::string g = "g" + std::to_string(m->_quant_group);
    m->_fn_dequant4 = m->_lib_dequant.function("affine_dequant_w4" + g);
    m->_fn_dequant8 = m->_lib_dequant.function("affine_dequant_w8" + g);
    m->_fn_qmm4 = m->_lib_qmm.function("affine_qmm_steel_w4" + g);
    m->_fn_qmm8 = m->_lib_qmm.function("affine_qmm_steel_w8" + g);
  }
  m->_fn_gemm_mma = m->_lib_mma.function("dense_gemm_mma_t_n128_f16");
  m->_fn_gemm_mma_deep =
      m->_lib_mma.function("dense_gemm_mma_t_n128x256_f16");
  m->_fn_gemm_mma_tn2 =
      m->_lib_mma.function("dense_gemm_mma_t_n128x256_tn2_f16");

  // VALIDATE, and name what failed. An unvalidated ComputeFunction
  // encodes nothing and poisons its dispatch. The vec4 twins and the
  // plain residual are REQUIRED here (the in-tree tested them per call
  // and fell back) because they are the plugin's own: a missing one is
  // a broken plugin build, not a host that lacks an optional entry.
  {
    const std::pair<const char*, const ComputeFunction*> req[] = {
        {"dense_gemm_t_f16", &m->_fn_gemm_t},
        {"layer_norm_plain_f16", &m->_fn_ln_plain},
        {"adaln_modulate_f16", &m->_fn_adaln},
        {"swiglu_f16", &m->_fn_swiglu},
        {"bias_add_rows_f16", &m->_fn_bias_add},
        {"mul_sigmoid_f16", &m->_fn_silu},
        {"transpose_abd_f16", &m->_fn_transpose},
        {"rms_norm_fast_f16", &m->_fn_rms},
        {"transpose_rope_pair_ftab_f16", &m->_fn_tr_rope},
        {"gated_residual_tanh_f16 (plugin)", &m->_fn_gated_tanh},
        {"gated_residual_tanh_v4_f16 (plugin)", &m->_fn_gated_tanh4},
        {"swiglu_v4_f16 (plugin)", &m->_fn_swiglu4},
        {"residual_add_f16 (plugin)", &m->_fn_residual},
        {"residual_add_v4_f16 (plugin)", &m->_fn_residual4},
        {"copy_rect_f16 (plugin)", &m->_fn_copy_rect},
    };
    for (const auto& r : req) {
      if (!r.second->valid()) {
        return refuse(std::string("a required kernel did not resolve: ") +
                      r.first);
      }
    }
  }
  // A QUANTIZED CHECKPOINT WHOSE GEMM DID NOT LOAD IS A REFUSAL, not a
  // degraded run: the dispatch would run the previous kernel over codes.
  if (m->_quant_bits > 0 &&
      (!m->_fn_qmm4.valid() || !m->_fn_qmm8.valid())) {
    return refuse("no quantized GEMM for group " +
                  std::to_string(m->_quant_group) +
                  " (the contract carries groups 32 and 64)");
  }
  m->_steel_attn_ok = m->_lib_attn.valid();
  if (!m->_steel_attn_ok) {
    return refuse("the steel attention library did not load");
  }
  m->_use_attn_nax = m->_lib_attn_nax.valid() &&
                     mc->supports_matrix_cores() &&
                     std::getenv("VPIPE_MING_NO_NAX_ATTN") == nullptr;
  m->_use_mma2 = mc->supports_matrix_cores() && m->_fn_gemm_mma.valid() &&
                 std::getenv("VPIPE_MING_NO_MMA2") == nullptr;
  {
    auto i8 = std::make_unique<I8GemmContext>(mc, m->_cfg.i8_gemm,
                                              /*bf16=*/true);
    if (i8->enabled()) { m->_i8 = std::move(i8); }
  }
  if (const char* e = std::getenv("VPIPE_MING_SOL")) {
    m->_cfg.sol.enabled = std::atoi(e) != 0;
  }
  // HIGHER tau keeps FEWER blocks: it is a threshold, not a budget.
  if (const char* e = std::getenv("VPIPE_MING_SOL_TAU")) {
    m->_cfg.sol.tau = (float)std::atof(e);
  }
  if (m->_cfg.sol.enabled) {
    std::string serr;
    m->_sol = MetalSolAttention::load(mc, /*bf16=*/true, &serr);
    if (m->_sol == nullptr) {
      m->log_(fmt("ming-image DiT: sol_attn unavailable -- {}",
                  serr.empty() ? "kernels did not load" : serr));
    }
  }
  // ---- the ANE feed-forward: the PLAN's answer -----------------------
  //
  // The family books the module unconditionally (its declare_resources
  // sees neither the graph's accel bag nor the stage's id). A CURRENT
  // HOST keeps that booking only when the graph asked for the tier,
  // relabels it to its own stage, reads the grant itself, turns ane_ffn
  // OFF in the bag when it was refused, and hands the label over -- so
  // with a label the bag is the answer. Without one (an older host, a
  // tool, a test) the booking is still under kAneClaimLabel and the
  // grant is read here. Either way a graph that gets no module WITHDRAWS
  // the booking, so peers sizing after this load stop counting a module
  // nothing will build.
  if (const char* e = std::getenv("VPIPE_MING_ANE")) {
    m->_cfg.ane_ffn = std::atoi(e) != 0;
  }
  if (const char* e = std::getenv("VPIPE_MING_ANE_ROWS")) {
    m->_cfg.ane_rows = (float)std::atof(e);
  }
  m->_ane_drop_lora = env_on_("VPIPE_MING_TEST_ANE_DROP_LORA");
  {
    const SessionContextIntf* ss = m->_session;
    auto* gm = ss != nullptr ? ss->services()->generative_model_manager()
                             : nullptr;
    const bool host_label = !opts.coreml_label.empty();
    m->_ane_label = host_label ? opts.coreml_label
                               : std::string(kAneClaimLabel);
    const std::size_t ab = MingDit::ane_runtime_bytes(dcfg);
    if (m->_cfg.ane_ffn) {
      int granted = 0;
      if (ab > 0 && host_label) {
        granted = 1;   // the host read it, and the bag says yes
      } else if (ab > 0 && ss != nullptr) {
        granted =
            vpipe::model_memory::coreml_grant(ss, kAneClaimLabel, ab, 1);
      }
      if (granted <= 0) {
        m->_cfg.ane_ffn = false;
        if (ss != nullptr) {
          ss->info(fmt("ming-image DiT: ane_ffn was requested but {}; "
                       "keeping the GPU feed-forward",
                       ab == 0 ? "this host cannot build the module"
                               : fmt("the plan left no room for its "
                                     "module ({} MB)", ab >> 20)()));
        }
      }
    }
    if (!m->_cfg.ane_ffn && gm != nullptr) {
      // revise_scratch refuses to CREATE, so this is a no-op when the
      // plan booked nothing.
      gm->revise_scratch(std::string("coreml:") + m->_ane_label, 0);
    }
  }
  {
    if (const char* e = std::getenv("VPIPE_MING_SAGE")) {
      m->_cfg.sage.enabled = std::atoi(e) != 0;
    }
    bool sage_fatal = false;
    m->_sage = MetalSageAttention::load_for_model(
        mc, /*bf16=*/true, m->_cfg.sage, "MingDit", &sage_fatal);
    if (sage_fatal) { return refuse("sage_attn failed to load"); }
    if (m->_sage && !m->_use_attn_nax) {
      m->log_(fmt("ming-image DiT: sage_attn is off -- the matrix-core "
                  "flash entry is unavailable here"));
      m->_sage.reset();
    }
  }
  if (const char* e = std::getenv("VPIPE_MING_NO_FOLD")) {
    m->_fold_mod = std::atoi(e) == 0;
  }
  if (const char* t = std::getenv("VPIPE_MING_GEMM_TILE")) {
    m->_gemm_tile = std::atoi(t);
  }
  if (m->_gemm_tile == 2 && !m->_fn_gemm_bm64bn64.valid()) {
    m->_gemm_tile = 1;
  }
  if (m->_gemm_tile == 1 && !m->_fn_gemm_bm64.valid()) { m->_gemm_tile = 0; }
  if (env_on_("VPIPE_MING_NO_CAP_CACHE")) { m->_cap_cache_on = false; }

  if (!m->load_fixed_(*m->_ws)) {
    return refuse("the model-level weights are incomplete");
  }

  // THE CONTEXT REFINER IS ALWAYS HELD, in both modes: two blocks, the
  // same as a slot pair, so streaming it would cost the same resident
  // bytes AND re-read them every forward. It also carries no adaLN, so
  // the streamed stack stays one uniform shape.
  m->_ctx.resize((std::size_t)m->_cfg.n_refiner);
  for (int i = 0; i < m->_cfg.n_refiner; ++i) {
    if (!m->load_block_(*m->_ws, m->ctx_prefix_(i), /*with_ada=*/false,
                        m->_ctx[(std::size_t)i], Retain::Cached)) {
      return refuse("context_refiner block " + std::to_string(i) +
                    " is incomplete");
    }
  }

  m->_wire.open(mc);
  const int NM = m->_cfg.n_mod_blocks();
  // Sized to the stack in BOTH modes: an unfilled entry reads as empty,
  // which is how the forward tells a resident block from a streamed one.
  m->_blocks.resize((std::size_t)NM);
  if (stream_blocks) {
    m->configure_slots_();
  } else {
    for (int L = 0; L < NM; ++L) {
      if (!m->load_block_(*m->_ws, m->mod_prefix_(L), /*with_ada=*/true,
                          m->_blocks[(std::size_t)L], Retain::Cached)) {
        return refuse("block " + m->mod_prefix_(L) + " is incomplete");
      }
    }
  }
  // The adapters, AFTER the blocks and the matrix-core decision: the
  // Applier picks its tiles by that, and the factors bind by NAME and
  // shape from the config, so they bind the same streamed or held.
  if (!opts.loras.empty() && !m->bind_loras_(opts.loras)) {
    m->warn_(fmt("ming-image DiT: the LoRA applier did not load; NO "
                 "adapter is applied and the base model runs"));
  }
  // Every slot that bound something, by its config key.
  std::string bound;
  for (int i = 0; i < m->_lora_slots; ++i) {
    if (m->_lora_bound[i] == 0) { continue; }
    bound += fmt("{}{} {}", bound.empty() ? "" : ", ",
                 i == 0 ? std::string("lora")
                        : "lora" + std::to_string(i + 1),
                 m->_lora_bound[i])();
  }
  m->log_(fmt(
      "ming-image DiT: {} blocks ({} ctx + {} noise + {} main), hidden {}, "
      "{} heads x {}, ffn {}, {}{}{}{}{}{}{}",
      m->_cfg.n_blocks(), m->_cfg.n_refiner, m->_cfg.n_refiner,
      m->_cfg.n_layers, m->_cfg.hidden, m->_cfg.n_heads, m->_cfg.head_dim,
      m->_cfg.ffn_inner,
      m->_quant_bits > 0
          ? "w" + std::to_string(m->_quant_bits) + " g" +
                std::to_string(m->_quant_group)
          : std::string(m->_src_f32 ? "f32 -> bf16" : "bf16"),
      stream_blocks ? ", streaming" : "",
      m->_use_mma2 ? ", matmul2d" : "",
      // WHETHER THE TIER TOOK, not whether it was asked for.
      m->_i8              ? ", i8"
      : m->_cfg.i8_gemm   ? ", i8 REQUESTED BUT UNAVAILABLE"
                          : "",
      m->_sol ? fmt(", sol tau {} on the {} arm", m->_cfg.sol.tau,
                    m->_sol->uses_matrix_cores() ? "NAX" : "ALU steel")()
              : std::string(),
      m->_cfg.ane_ffn ? ", ane_ffn (granted; arms at the first forward)"
                      : "",
      bound.empty() ? std::string()
                    : fmt(", runtime LoRA modules bound: {}", bound)()));
  return m;
}

// ---- the timestep -------------------------------------------------------
//
// COS occupies the first half and sin the second. The argument is
// (1 - sigma) * t_scale: the pipeline hands the model 1 - sigma, so a
// model at the noisy end sees an input near 0.
std::vector<float>
MingDitMetal::time_proj_(float sigma) const
{
  const int D = _cfg.time_freq;
  const int half = D / 2;
  std::vector<float> out((std::size_t)D, 0.0f);
  const double t = (double)_cfg.t_scale * (1.0 - (double)sigma);
  for (int i = 0; i < half; ++i) {
    const double f =
        std::exp(-std::log(10000.0) * (double)i / (double)half);
    const double a = t * f;
    out[(std::size_t)i] = (float)std::cos(a);
    out[(std::size_t)(half + i)] = (float)std::sin(a);
  }
  return out;
}

// ONE joint table for the whole sequence, read by the refiners as a
// prefix (the image) and a suffix (the caption). f32, because bf16
// table rounding compounds over 34 blocks and over denoise steps; angles
// in f64 to match the reference's float64 freqs.
void
MingDitMetal::build_rope_(const Layout& lay, SharedBuffer* cos_out,
                          SharedBuffer* sin_out) const
{
  const int HD = _cfg.head_dim;
  const int pairs = HD / 2;
  const int seq = lay.joint_len;
  *cos_out = _mc->make_shared_buffer((std::size_t)seq * HD * sizeof(float));
  *sin_out = _mc->make_shared_buffer((std::size_t)seq * HD * sizeof(float));
  if (cos_out->empty() || sin_out->empty()) { return; }
  auto* c = static_cast<float*>(cos_out->contents());
  auto* s = static_cast<float*>(sin_out->contents());
  const double theta = (double)_cfg.rope_theta;
  std::vector<double> pair_freq((std::size_t)pairs);
  std::vector<int> pair_axis((std::size_t)pairs);
  {
    int pair = 0;
    for (int a = 0; a < 3; ++a) {
      const int adim = _cfg.axes[a];
      for (int j = 0; j < adim / 2 && pair < pairs; ++j, ++pair) {
        pair_freq[(std::size_t)pair] =
            1.0 / std::pow(theta, (double)(2 * j) / (double)adim);
        pair_axis[(std::size_t)pair] = a;
      }
    }
  }
  for (int t = 0; t < seq; ++t) {
    const double pos[3] = {(double)lay.pos_t[(std::size_t)t],
                           (double)lay.pos_h[(std::size_t)t],
                           (double)lay.pos_w[(std::size_t)t]};
    for (int pair = 0; pair < pairs; ++pair) {
      const double ang =
          pos[pair_axis[(std::size_t)pair]] * pair_freq[(std::size_t)pair];
      const float cb = (float)std::cos(ang);
      const float sb = (float)std::sin(ang);
      // Written TWICE per pair: the rotation is over INTERLEAVED complex
      // pairs (x[2k], x[2k+1]) and the kernel reads one entry per element.
      c[(std::size_t)t * HD + 2 * pair] = cb;
      c[(std::size_t)t * HD + 2 * pair + 1] = cb;
      s[(std::size_t)t * HD + 2 * pair] = sb;
      s[(std::size_t)t * HD + 2 * pair + 1] = sb;
    }
  }
}

// ---- forward ------------------------------------------------------------

SharedBuffer
MingDitMetal::forward(const Request& req, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return SharedBuffer{};
  };
  _eng = DitEngagement{};
  if (req.layout == nullptr || req.latents == nullptr || req.cap == nullptr) {
    return fail("ming DiT forward: missing input");
  }
  const Layout& lay = *req.layout;
  const int H = _cfg.hidden, NH = _cfg.n_heads, Hd = _cfg.head_dim;
  const int FFI = _cfg.ffn_inner, PD = _cfg.patch_dim();
  const int JT = lay.joint_len, IMG = lay.img_tokens;
  const int CAP = lay.cap_len, Q = lay.query_rows;
  const int CS = req.cap_stride;
  const int NM = _cfg.n_mod_blocks();
  // THE LAYOUT IS CHECKED, not trusted: a joint length that does not
  // add up is a read past a buffer, not a wrong picture.
  if (IMG <= 0 || CAP <= 0 || JT != IMG + CAP ||
      IMG != lay.grid_h * lay.grid_w || Q < 0 || Q > CAP ||
      Q + lay.text_rows != CAP || (int)lay.pos_t.size() != JT ||
      (int)lay.pos_h.size() != JT || (int)lay.pos_w.size() != JT) {
    return fail("ming DiT forward: the layout does not add up");
  }
  if ((Q > 0 && CS < _cfg.cap_dim) || (CAP > Q && CS < H)) {
    return fail("ming DiT forward: the caption's row stride " +
                std::to_string(CS) + " is narrower than a row it carries");
  }
  if (req.cap->byte_size() < (std::size_t)CAP * CS * 2 ||
      req.latents->byte_size() < (std::size_t)IMG * PD * 2) {
    return fail("ming DiT forward: an input buffer is smaller than the "
                "layout says");
  }
  // THE ROTARY TABLES ARE BOUNDED. The reference indexes a precomputed
  // table of `axes_lens` rows per axis, so a position past the end is an
  // index error there and would be a silently plausible angle here.
  for (int t = 0; t < JT; ++t) {
    const std::size_t i = (std::size_t)t;
    if (lay.pos_t[i] < 0 || lay.pos_h[i] < 0 || lay.pos_w[i] < 0 ||
        lay.pos_t[i] >= _cfg.axes_lens[0] ||
        lay.pos_h[i] >= _cfg.axes_lens[1] ||
        lay.pos_w[i] >= _cfg.axes_lens[2]) {
      return fail("ming DiT forward: a rotary position is past the "
                  "checkpoint's axes_lens -- the picture or the caption "
                  "is larger than this model was built for");
    }
  }

  MetalCompute* mc = _mc;
  const float eps = _cfg.norm_eps;

  // ---- the caption cache ---------------------------------------------
  //
  // Keyed on the caption's EXACT bytes and shape, so a hit is the same
  // computation's result and not a guess. The context refiner's rotary
  // positions (1 + i, 0, 0) depend on the caption alone, never on the
  // image, so one entry serves every picture size.
  const std::size_t cap_bytes = (std::size_t)CAP * CS * 2;
  CapEntry* hit = nullptr;
  if (_cap_cache_on) {
    for (CapEntry& e : _cap_cache) {
      if (e.valid && e.q == Q && e.len == CAP && e.stride == CS &&
          e.key.size() == cap_bytes &&
          std::memcmp(e.key.data(), req.cap->contents(), cap_bytes) == 0) {
        hit = &e;
        break;
      }
    }
  }
  // The entry a miss will fill: an empty one, else the least recently
  // used. Marked valid only once the forward has COMPLETED, so a stopped
  // or failed pass never leaves half-written rows behind a valid key.
  CapEntry* fill = nullptr;
  if (_cap_cache_on && hit == nullptr) {
    fill = &_cap_cache[0];
    for (CapEntry& e : _cap_cache) {
      if (!e.valid) { fill = &e; break; }
      if (e.tick < fill->tick) { fill = &e; }
    }
    fill->valid = false;
    if (fill->out.byte_size() < (std::size_t)CAP * H * 2) {
      fill->out = mc->make_shared_buffer((std::size_t)CAP * H * 2);
    }
    if (fill->out.empty()) { fill = nullptr; }
  }
  _eng.cap_cache_hit = hit != nullptr;

  auto buf = [&](std::size_t elems) {
    return mc->make_shared_buffer(elems * 2);
  };
  // Peak scratch is ~10 [JT,H] planes plus 2 [JT,FFI] ones.
  SharedBuffer joint = buf((std::size_t)JT * H);
  SharedBuffer nrm = buf((std::size_t)JT * H);
  SharedBuffer ob = buf((std::size_t)JT * H);
  SharedBuffer qb = buf((std::size_t)JT * H);
  SharedBuffer kb = buf((std::size_t)JT * H);
  SharedBuffer vb = buf((std::size_t)JT * H);
  SharedBuffer qh = buf((std::size_t)JT * H);
  SharedBuffer kh = buf((std::size_t)JT * H);
  SharedBuffer vh = buf((std::size_t)JT * H);
  SharedBuffer ao = buf((std::size_t)JT * H);
  SharedBuffer ffg = buf((std::size_t)JT * FFI);
  SharedBuffer ffu = buf((std::size_t)JT * FFI);
  // Stream A, compacted to its own width: rms_norm and the GEMM read
  // rows at a stride of their own width, not the caption's.
  SharedBuffer capn = buf((std::size_t)std::max(Q, 1) * _cfg.cap_dim);
  SharedBuffer te1 = buf((std::size_t)_cfg.time_mid);
  SharedBuffer temb = buf((std::size_t)_cfg.adaln_dim);
  SharedBuffer tsilu = buf((std::size_t)_cfg.adaln_dim);
  SharedBuffer mods = buf((std::size_t)4 * H);
  // THE FOLDED NORM GAINS: `rms(x)*w*(1+scale)` is `rms(x)*(w*(1+scale))`,
  // exact algebra that turns two full-width passes into one.
  SharedBuffer nw1 = buf((std::size_t)H);
  SharedBuffer nw3 = buf((std::size_t)H);
  SharedBuffer outmod = buf((std::size_t)H);
  SharedBuffer tin = buf((std::size_t)_cfg.time_freq);
  SharedBuffer zero = buf((std::size_t)H);
  SharedBuffer vel = buf((std::size_t)IMG * PD);
  SharedBuffer rcos, rsin;
  build_rope_(lay, &rcos, &rsin);
  if (joint.empty() || nrm.empty() || ob.empty() || qb.empty() ||
      kb.empty() || vb.empty() || qh.empty() || kh.empty() || vh.empty() ||
      ao.empty() || ffg.empty() || ffu.empty() || capn.empty() ||
      te1.empty() || temb.empty() || tsilu.empty() || mods.empty() ||
      nw1.empty() || nw3.empty() || outmod.empty() || tin.empty() ||
      zero.empty() || vel.empty() || rcos.empty() || rsin.empty()) {
    return fail("ming DiT forward: scratch allocation failed");
  }
  std::memset(zero.contents(), 0, zero.byte_size());
  // SOL BORROWS THE FEED-FORWARD PLANES. Everything it holds lives for one
  // call (summaries, routing, CSR, both partial softmaxes), and the
  // attention runs between one block's feed-forward and the next, while
  // the gate/up planes are dead -- 2 x [joint, ffn_inner], more than the
  // call wants at any size here. Taken back when the forward ends, so the
  // windows never keep a finished forward's planes alive.
  struct SolLoan {
    MetalSolAttention* s;
    ~SolLoan()
    {
      if (s != nullptr) { s->set_arena(SharedBuffer{}, SharedBuffer{}); }
    }
  } sol_loan{_sol.get()};
  if (_sol) {
    _sol->set_arena(ffg, ffu);
    _sol->reset_counts();
    _sol_flops.begin();
  }
  {
    const std::vector<float> tp = time_proj_(req.sigma);
    auto* d = static_cast<std::uint16_t*>(tin.contents());
    for (int i = 0; i < _cfg.time_freq; ++i) {
      d[i] = f32_to_bf16_(tp[(std::size_t)i]);
    }
  }

  // The int8 split's width, for the shapes an earlier forward recorded.
  // HERE because it runs its own command streams (no encoder open).
  if (_i8) { _i8->tune_pending(mc); }

  CommandStream stream = mc->make_command_stream();
  ComputeEncoder enc = stream.begin_compute();
  if (!enc.valid()) { return fail("ming DiT forward: no compute encoder"); }
  // What an aborted forward counted and never flushed is dropped.
  _flops.begin();
  const std::uint64_t flops0 = _flops.flushed();

  // Per-stage TRACE, for locating a numeric gap against the reference.
  // It COMMITS, so a traced run is not a timing run.
  const char* trace_dir = std::getenv("VPIPE_MING_DIT_TRACE");
  bool trace_ok = true;
  auto trace = [&](const std::string& tag, const SharedBuffer& b,
                   std::size_t e0, std::size_t n) {
    if (trace_dir == nullptr || *trace_dir == '\0' || !trace_ok) { return; }
    enc.end();
    _flops.flush(stream);
    std::string ge;
    if (!stream.commit().wait_ok(&ge)) { trace_ok = false; return; }
    std::error_code ec;
    fs::create_directories(trace_dir, ec);
    if (b.byte_size() >= (e0 + n) * 2) {
      const auto* sp = static_cast<const std::uint16_t*>(b.contents()) + e0;
      std::vector<float> v(n);
      for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t u = (std::uint32_t)sp[i] << 16;
        std::memcpy(&v[i], &u, 4);
      }
      std::ofstream o(fs::path(trace_dir) / (tag + ".f32"),
                      std::ios::binary);
      if (o) {
        o.write(reinterpret_cast<const char*>(v.data()),
                (std::streamsize)(v.size() * sizeof(float)));
      }
    }
    stream = mc->make_command_stream();
    enc = stream.begin_compute();
  };

  // Per-CATEGORY profile. READ IT WITH ITS MARK COUNT: each mark is a
  // commit AND a wait, so a bucket with many marks and little work is
  // mostly measuring the profiler -- and every mark defeats the
  // prefetch, so read the disk from VPIPE_MING_STREAM_PROFILE instead.
  const bool prof = env_on_("VPIPE_MING_DIT_PROFILE");
  struct Bucket { const char* name; double ms; long marks; };
  std::vector<Bucket> pbuckets;
  long pmarks = 0;
  auto pclock = std::chrono::steady_clock::now();
  auto mark = [&](const char* bucket) {
    if (!prof) { return; }
    enc.end();
    _flops.flush(stream);
    std::string ge;
    (void)stream.commit().wait_ok(&ge);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - pclock).count();
    ++pmarks;
    bool found = false;
    for (auto& b : pbuckets) {
      if (std::strcmp(b.name, bucket) == 0) {
        b.ms += ms; ++b.marks; found = true;
      }
    }
    if (!found) { pbuckets.push_back(Bucket{bucket, ms, 1}); }
    stream = mc->make_command_stream();
    enc = stream.begin_compute();
    pclock = std::chrono::steady_clock::now();
  };

  // ---- dispatch helpers -----------------------------------------------
  // The BASE projection, on whichever route this box and shape take.
  auto lin_base = [&](const SharedBuffer& x, std::size_t xe,
                      const QWeight& w, const SharedBuffer& y,
                      std::size_t ye, int M, int N, int K) {
    _flops.gemm(M, N, K);
    if (gemm_mma_(enc, x, xe, w, y, ye, M, N, K)) { return; }
    if (!w.quantized) {
      int bm = 32, bn = 32;
      const ComputeFunction* fn = &_fn_gemm_t;
      int* count = &_eng.gemm_bm32;
      if (M >= 128 && _gemm_tile == 1) {
        fn = &_fn_gemm_bm64; bm = 64; count = &_eng.gemm_bm64;
      } else if (M >= 128 && _gemm_tile == 2) {
        fn = &_fn_gemm_bm64bn64; bm = 64; bn = 64;
        count = &_eng.gemm_bm64bn64;
      }
      ++*count;
      enc.set_function(*fn);
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, w.w);
      // Bound twice: the kernel takes the weight as both operands of
      // its biasless form.
      enc.set_buffer(2, w.w); enc.set_buffer(3, y, ye * 2);
      enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
      enc.set_constant(7, 0);
      // THREADGROUP-indexed: a thread-indexed grid here computes a
      // fraction of the output and leaves the rest as it found it.
      enc.dispatch({(unsigned)(((N + bn - 1) / bn) * 32),
                    (unsigned)(((M + bm - 1) / bm) * 2), 2}, {32, 2, 2});
      return;
    }
    ++_eng.gemm_qmm;
    enc.set_function(w.bits == 8 ? _fn_qmm8 : _fn_qmm4);
    enc.set_buffer(0, w.codes); enc.set_buffer(1, w.scales);
    enc.set_buffer(2, w.qbias); enc.set_buffer(3, x, xe * 2);
    enc.set_buffer(4, y, ye * 2);
    enc.set_constant(5, K); enc.set_constant(6, N); enc.set_constant(7, M);
    enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                  (unsigned)(((M + 31) / 32) * 2), 2u}, {32, 2, 2});
  };
  // A projection: the base, then every live adapter on it. The adapter
  // ACCUMULATES onto the base result, so it runs after whichever route
  // lin_base took -- matmul2d, a dense tile or the quantized GEMM -- and
  // is the same for all of them.
  auto lin = [&](const SharedBuffer& x, std::size_t xe, const QWeight& w,
                 const SharedBuffer& y, std::size_t ye, int M, int N,
                 int K, const lora::Stack& lf = lora::Stack{}) {
    lin_base(x, xe, w, y, ye, M, N, K);
    if (!lf.empty()) {
      ++_eng.lora_proj;
      _lora.apply(enc, x, xe, lf, y, ye, M, N, K, _mma_min_m);
    }
  };
  auto bias = [&](const SharedBuffer& b, const SharedBuffer& y,
                  std::size_t ye, int M, int N) {
    if (b.empty()) { return; }
    enc.set_function(_fn_bias_add);
    enc.set_buffer(0, y, ye * 2); enc.set_buffer(1, b);
    enc.set_constant(2, N); enc.set_constant(3, M * N);
    enc.dispatch({(unsigned)((std::size_t)M * N), 1, 1}, {256, 1, 1});
  };
  auto rms = [&](const SharedBuffer& x, std::size_t xe, const SharedBuffer& w,
                 const SharedBuffer& y, std::size_t ye, int R, int D) {
    enc.set_function(_fn_rms);
    enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, w);
    enc.set_buffer(2, y, ye * 2);
    enc.set_constant(3, D); enc.set_constant(4, eps);
    enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
  };
  auto silu = [&](const SharedBuffer& x, const SharedBuffer& y, int n) {
    enc.set_function(_fn_silu);
    enc.set_buffer(0, x); enc.set_buffer(1, x); enc.set_buffer(2, y);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };
  // out = (1 + scale) * x, the model's modulation. The kernel's shift
  // slot takes the zero row: this model emits no shift at all.
  auto modulate = [&](const SharedBuffer& x, std::size_t xe,
                      const SharedBuffer& sc, std::size_t sce,
                      const SharedBuffer& y, std::size_t ye, int R) {
    if ((H % 4) == 0 && _fn_adaln4.valid()) {
      enc.set_function(_fn_adaln4);
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, sc, sce * 2);
      enc.set_buffer(2, zero); enc.set_buffer(3, y, ye * 2);
      enc.set_constant(4, H / 4); enc.set_constant(5, R);
      enc.dispatch({(unsigned)(H / 4), (unsigned)R, 1}, {256, 1, 1});
      return;
    }
    enc.set_function(_fn_adaln);
    enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, sc, sce * 2);
    enc.set_buffer(2, zero); enc.set_buffer(3, y, ye * 2);
    enc.set_constant(4, H);
    enc.set_constant(5, (int)((std::size_t)R * H));
    enc.dispatch({(unsigned)((std::size_t)R * H), 1, 1}, {256, 1, 1});
  };
  auto gated = [&](const SharedBuffer& h, std::size_t he,
                   const SharedBuffer& gate, std::size_t ge,
                   const SharedBuffer& sub, std::size_t se, int R) {
    if ((H % 4) == 0) {
      enc.set_function(_fn_gated_tanh4);
      enc.set_buffer(0, h, he * 2); enc.set_buffer(1, gate, ge * 2);
      enc.set_buffer(2, sub, se * 2);
      enc.set_constant(3, H / 4); enc.set_constant(4, R);
      enc.dispatch({(unsigned)(H / 4), (unsigned)R, 1}, {256, 1, 1});
      return;
    }
    enc.set_function(_fn_gated_tanh);
    enc.set_buffer(0, h, he * 2); enc.set_buffer(1, gate, ge * 2);
    enc.set_buffer(2, sub, se * 2);
    enc.set_constant(3, H);
    enc.set_constant(4, (int)((std::size_t)R * H));
    enc.dispatch({(unsigned)((std::size_t)R * H), 1, 1}, {256, 1, 1});
  };
  auto resid = [&](const SharedBuffer& a, std::size_t ae,
                   const SharedBuffer& b, std::size_t be, int n) {
    if ((n % 4) == 0) {
      enc.set_function(_fn_residual4);
      enc.set_buffer(0, a, ae * 2); enc.set_buffer(1, b, be * 2);
      enc.set_buffer(2, a, ae * 2); enc.set_constant(3, n / 4);
      enc.dispatch({(unsigned)(n / 4), 1, 1}, {256, 1, 1});
      return;
    }
    enc.set_function(_fn_residual);
    enc.set_buffer(0, a, ae * 2); enc.set_buffer(1, b, be * 2);
    enc.set_buffer(2, a, ae * 2); enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };
  auto swiglu = [&](const SharedBuffer& g, const SharedBuffer& u,
                    const SharedBuffer& o, int n) {
    if ((n % 4) == 0) {
      enc.set_function(_fn_swiglu4);
      enc.set_buffer(0, g); enc.set_buffer(1, u); enc.set_buffer(2, o);
      enc.set_constant(3, n / 4);
      enc.dispatch({(unsigned)(n / 4), 1, 1}, {256, 1, 1});
      return;
    }
    enc.set_function(_fn_swiglu);
    enc.set_buffer(0, g); enc.set_buffer(1, u); enc.set_buffer(2, o);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };
  // A rectangle of rows between two planes of different row strides --
  // the caption's two streams into the joint sequence, and the cache.
  // ELEMENT offsets, as int: the largest is (img + cap) * hidden, ~64M
  // at 2048^2, well inside the kernel's int arithmetic.
  auto copy_rect = [&](const SharedBuffer& src, std::size_t so,
                       const SharedBuffer& dst, std::size_t do_, int rows,
                       int row_elems, int sstride, int dstride) {
    if (rows <= 0 || row_elems <= 0) { return; }
    enc.set_function(_fn_copy_rect);
    enc.set_buffer(0, src); enc.set_buffer(1, dst);
    enc.set_constant(2, (int)so); enc.set_constant(3, (int)do_);
    enc.set_constant(4, rows); enc.set_constant(5, row_elems);
    enc.set_constant(6, sstride); enc.set_constant(7, dstride);
    enc.dispatch({(unsigned)((std::size_t)rows * row_elems), 1, 1},
                 {256, 1, 1});
  };
  auto ln_plain = [&](const SharedBuffer& x, std::size_t xe,
                      const SharedBuffer& y, std::size_t ye, int R) {
    enc.set_function(_fn_ln_plain);
    enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, y, ye * 2);
    enc.set_constant(2, H);
    // eps 1e-6 -- the FINAL layer's norm, not the model's norm_eps.
    enc.set_constant(3, 1e-6f);
    enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
  };
  auto tr_rope = [&](const SharedBuffer& in, const SharedBuffer& out,
                     int rows, int row0) {
    const std::size_t tb = (std::size_t)row0 * Hd * sizeof(float);
    enc.set_function(_fn_tr_rope);
    enc.set_buffer(0, in); enc.set_buffer(1, out);
    enc.set_buffer(2, rcos, tb); enc.set_buffer(3, rsin, tb);
    enc.set_constant(4, NH); enc.set_constant(5, rows);
    enc.set_constant(6, Hd);
    enc.dispatch({(unsigned)(Hd / 2), (unsigned)rows, (unsigned)NH},
                 {(unsigned)(Hd / 2), 1, 1});
  };
  auto transpose = [&](const SharedBuffer& in, const SharedBuffer& out,
                       int A, int B) {
    enc.set_function(_fn_transpose);
    enc.set_buffer(0, in); enc.set_buffer(1, out);
    enc.set_constant(2, A); enc.set_constant(3, B); enc.set_constant(4, Hd);
    enc.dispatch({(unsigned)Hd, (unsigned)B, (unsigned)A},
                 {(unsigned)Hd, 1, 1});
  };

  // ---- attention --------------------------------------------------------
  //
  // DENSE AND BIDIRECTIONAL, one dispatch per block, and -- with the pad
  // rows dropped -- MASKLESS. The three stages run at three sequence
  // lengths (image, caption, joint), none of them a multiple of 32 in
  // general, so a params block and a specialized function are memoized
  // per length. The remainder is the kernel's: align_Q / align_K FALSE
  // and qL_rem / kL_rem set, which loads the last block "safe" (zero
  // fill) and masks the last key block's missing columns out of the
  // softmax -- exactly what excluding a pad as a key means.
  const bool nax = _use_attn_nax;
  const int A_BQ = nax ? 64 : 32;
  const int A_BK = nax ? 32 : 16;
  const float ascale = 1.0f / std::sqrt((float)Hd);
  std::vector<std::pair<int, SharedBuffer>> aparams;
  auto params_for = [&](int R) -> const SharedBuffer* {
    for (auto& p : aparams) {
      if (p.first == R) { return &p.second; }
    }
    SharedBuffer b = mc->make_shared_buffer(sizeof(SteelAttnParams));
    if (b.empty()) { return nullptr; }
    auto* p = static_cast<SteelAttnParams*>(b.contents());
    p->B = 1; p->H = NH; p->D = Hd; p->qL = R; p->kL = R;
    p->gqa_factor = 1; p->scale = ascale;
    p->NQ = (R + A_BQ - 1) / A_BQ; p->NK = (R + A_BK - 1) / A_BK;
    p->NQ_aligned = R / A_BQ; p->NK_aligned = R / A_BK;
    p->qL_rem = R - p->NQ_aligned * A_BQ;
    p->kL_rem = R - p->NK_aligned * A_BK;
    p->qL_off = 0;
    const std::int64_t rs = (std::int64_t)R * Hd;
    p->Q_strides[0] = (std::int64_t)NH * rs;
    p->Q_strides[1] = rs; p->Q_strides[2] = Hd;
    p->K_strides[0] = p->Q_strides[0];
    p->K_strides[1] = rs; p->K_strides[2] = Hd;
    p->V_strides[0] = p->Q_strides[0];
    p->V_strides[1] = rs; p->V_strides[2] = Hd;
    p->O_strides[0] = p->Q_strides[0];
    p->O_strides[1] = rs; p->O_strides[2] = Hd;
    aparams.emplace_back(R, std::move(b));
    return &aparams.back().second;
  };
  auto attn_fn = [&](int R, bool i8) -> const ComputeFunction* {
    const unsigned key = (nax ? 8u : 0u) | (i8 ? 4u : 0u) |
                         (((R % A_BQ) == 0) ? 2u : 0u) |
                         (((R % A_BK) == 0) ? 1u : 0u);
    auto it = _attn_fn.find(key);
    if (it != _attn_fn.end()) {
      return it->second.valid() ? &it->second : nullptr;
    }
    FunctionConstants fc;
    fc.set_bool(contract::kAttnAlignQ, (R % A_BQ) == 0)
        .set_bool(contract::kAttnAlignK, (R % A_BK) == 0)
        .set_bool(contract::kAttnHasMask, false)
        .set_bool(contract::kAttnCausal, false)
        .set_bool(contract::kAttnSinks, false)
        .set_bool(sage::kQkInt8Constant, i8);
    ComputeFunction fn =
        nax ? _lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc)
            : _lib_attn.function("attn_steel_h_bd128_bf16", fc);
    auto ins = _attn_fn.emplace(key, std::move(fn));
    return ins.first->second.valid() ? &ins.first->second : nullptr;
  };
  bool attn_ok = true;
  // The (query block, key block) pairs Sol chose from this forward; what
  // it KEPT of them is its own GPU counter, read after the last wait.
  long long sol_total = 0;
  const bool sol_sink = !env_on_("VPIPE_MING_SOL_NO_SINK");
  // `routable`: a MODULATED block. The context refiner runs once per
  // caption (its output is cached), so approximating it saves nothing --
  // and its block index is not the modulated stack's, which is what
  // `dense_layers` counts.
  auto run_attn = [&](int L, int rows, bool routable) {
    if (!attn_ok) { return; }
    // SOL, where the sequence is long enough to have something to skip.
    // Every attention in this model is square and non-causal. NOT GATED
    // ON MATRIX CORES: the toolkit runs its exact half on whichever flash
    // kernel this box has -- steel here without them, as the dense path.
    if (_sol && routable && _cfg.sol.enabled &&
        L >= _cfg.sol.dense_layers && rows >= kSolMinSegment) {
      const int blk = _cfg.sol.key_block > 0 ? _cfg.sol.key_block
                                             : sol::kBlock;
      if ((rows + blk - 1) / blk >= 16) {
        MetalSolAttention::Band band{rows, rows, rows, rows, 0, 0};
        // THE CAPTION IS ATTENDED EXACTLY on the main stack, where the
        // sequence is [image | caption]: it carries the text this model
        // is asked to render, and a centroid over prompt tokens is a
        // prompt half-read. A few hundred rows against tens of thousands
        // of image keys, so it costs little. The noise refiner is image
        // rows alone and has no caption to keep.
        sol::Config sc = _cfg.sol;
        if (sol_sink && rows == JT && CAP > 0) {
          sc.sink_start = IMG;
          sc.sink_tokens = CAP;
        }
        std::string serr;
        if (_sol->encode(enc, qh, kh, vh, ao, NH, NH, band, Hd, ascale, sc,
                         &serr)) {
          ++_eng.attn_sol;
          sol_total += _sol->total_blocks();
          _flops.add(_sol_flops.routed(
              vpipe::genai::attention_flops(NH, (long long)rows * rows, Hd),
              _sol->total_blocks()));
          return;
        }
        if (_session != nullptr) {
          _session->log_debug(fmt("ming DiT: sol declined this stage ({}); "
                                  "dense",
                                  serr.empty() ? "no reason" : serr));
        }
      }
    }
    bool i8_ok = false;
    if (_sage && nax && _cfg.sage.enabled && L >= _cfg.sage.dense_layers &&
        rows >= kSageMinSegment) {
      const MetalSageAttention::Operand qo{&qh, 0, Hd, rows * Hd};
      const MetalSageAttention::Operand ko{&kh, 0, Hd, rows * Hd};
      std::string gerr;
      i8_ok = _sage->prepare(enc, qo, ko, NH, NH, rows, rows, Hd, A_BQ,
                             A_BK, _cfg.sage, &gerr);
    }
    const SharedBuffer* pp = params_for(rows);
    const ComputeFunction* fn = attn_fn(rows, i8_ok);
    if (pp == nullptr || fn == nullptr) { attn_ok = false; return; }
    if (nax) { ++_eng.attn_nax; } else { ++_eng.attn_steel; }
    if (i8_ok) { ++_eng.attn_sage; }
    if ((rows % A_BQ) != 0) { ++_eng.attn_unaligned_q; }
    if ((rows % A_BK) != 0) { ++_eng.attn_unaligned_k; }
    enc.set_function(*fn);
    enc.set_buffer(0, qh); enc.set_buffer(1, kh); enc.set_buffer(2, vh);
    enc.set_buffer(3, ao); enc.set_buffer(4, *pp);
    if (i8_ok) { _sage->bind(enc); }
    enc.dispatch({(unsigned)(32 * ((rows + A_BQ - 1) / A_BQ)),
                  (unsigned)(4 * NH), 1}, {32, 4, 1});
    _flops.attention(NH, (long long)rows * rows, Hd);
  };

  // ---- one block --------------------------------------------------------
  //
  // The sandwich: norm1 BEFORE the sublayer and norm2 on its OUTPUT,
  // inside the residual and under the gate. `mods` holds the four
  // chunks (scale_msa | gate_msa | scale_mlp | gate_mlp).
  //
  // `xe` is the block's first ELEMENT in `joint`, `row0` its first row
  // in the rotary table -- the table is built over the joint sequence in
  // the joint's own row order, so the caption's rows start at IMG there.
  // A failure inside a block -- a GPU error at an ANE split point, or rows
  // the ANE lost with no range to recompute -- read after the block.
  std::string blk_err;
  using AnePlan = ane::Tier::Plan;
  const bool ane_verify =
      _cfg.ane_ffn && env_on_("VPIPE_MING_ANE_VERIFY");
  SharedBuffer ane_ver;
  auto ms_since = [](std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t).count();
  };
  auto restart = [&]() {
    stream = mc->make_command_stream();
    enc = stream.begin_compute();
  };
  auto run_block = [&](const Block& b, std::size_t xe, int rows, int row0,
                       bool modulated, int L,
                       AnePlan ane_plan = AnePlan::kGpu) {
    // The adapter's factors for this block's projections. Every
    // UNMODULATED block here is a context-refiner block, which is the
    // adapter's other index space.
    auto LA = [&](lora::Factors BlockLora::* w) {
      return lora_for_(/*ctx=*/!modulated, L, w);
    };
    const bool fold = modulated && _fold_mod;
    rms(joint, xe, fold ? nw1 : b.an1, nrm, 0, rows, H);
    if (modulated && !fold) { modulate(nrm, 0, mods, 0, nrm, 0, rows); }
    mark("elt");
    lin(nrm, 0, b.qw, qb, 0, rows, H, H, LA(&BlockLora::q));
    lin(nrm, 0, b.kw, kb, 0, rows, H, H, LA(&BlockLora::k));
    lin(nrm, 0, b.vw, vb, 0, rows, H, H, LA(&BlockLora::v));
    mark("gemm.qkv");
    // Per-head RMSNorm over the last Hd of a token-major [rows, NH, Hd].
    rms(qb, 0, b.nq, qb, 0, rows * NH, Hd);
    rms(kb, 0, b.nk, kb, 0, rows * NH, Hd);
    tr_rope(qb, qh, rows, row0);
    tr_rope(kb, kh, rows, row0);
    // V is transposed but NOT rotated.
    transpose(vb, vh, rows, NH);
    mark("rope");
    run_attn(L, rows, modulated);
    mark("attn");
    transpose(ao, nrm, NH, rows);
    mark("transpose");
    lin(nrm, 0, b.ow, ob, 0, rows, H, H, LA(&BlockLora::o));
    mark("gemm.out");
    rms(ob, 0, b.an2, ob, 0, rows, H);
    if (modulated) {
      gated(joint, xe, mods, (std::size_t)H, ob, 0, rows);
    } else {
      resid(joint, xe, ob, 0, rows * H);
    }

    rms(joint, xe, fold ? nw3 : b.fn1, nrm, 0, rows, H);
    if (modulated && !fold) {
      modulate(nrm, 0, mods, (std::size_t)2 * H, nrm, 0, rows);
    }
    mark("elt");
    // ---- the ANE split point ------------------------------------------
    //
    // `nrm` now holds the feed-forward input for every row. THE ORDER
    // HERE IS LOAD-BEARING: the attention is committed and drained, the
    // ANE STARTS on the tail rows, and only then are the GPU's rows
    // encoded. Starting it after the GPU's w1/w3 would leave only the
    // down projection to hide behind. (The weights were staged before
    // this block was encoded, so their conversion ran under the
    // attention.) A PROBE block takes the same commits with every row
    // on the GPU: that is the balancer's on/off measurement.
    const bool ane_split = ane_plan == AnePlan::kSplit;
    const bool ane_probe = ane_plan == AnePlan::kProbe;
    int a_rows = 0;
    double drain_ms = 0.0;
    if (ane_split || ane_probe) {
      enc.end();
      _flops.flush(stream);
      const auto t_d0 = std::chrono::steady_clock::now();
      std::string ge;
      if (!stream.commit().wait_ok(&ge)) {
        _ane->join();     // the staging job may still read the weights
        blk_err = ge.empty() ? std::string("GPU error") : ge;
        restart();
        return;
      }
      drain_ms = ms_since(t_d0);
      if (ane_split && _ane->join_stage(L)) {
        FlexData args = FlexData::make_object();
        accel::set_integer(&args, ane::kSeq, rows);
        const ane::Binding io[] = {{ane::kIoIn, &nrm}, {ane::kIoOut, &ob}};
        a_rows = _ane->begin(args, io);
      } else if (ane_split && !_ane_warned) {
        _ane_warned = true;
        const FlexData inf = _ane->info();
        warn_(fmt("ming-image DiT: staging block {}'s feed-forward for the "
                  "ANE failed ({}); it keeps the GPU feed-forward", L,
                  accel::text(&inf, ane::kInfoError)));
      }
      restart();
    }
    // The GPU's feed-forward over rows [r0, r0 + n), `nrm` into `ob`: its
    // share of a split, every row of an unsplit block, and rows the ANE
    // lost. The gate/up scratch is used from its first row whatever r0.
    auto gpu_ff = [&](int r0, int n, const SharedBuffer& dst,
                      std::size_t de) {
      if (n <= 0) { return; }
      const std::size_t e0 = (std::size_t)r0 * H;
      lin(nrm, e0, b.w1, ffg, 0, n, FFI, H, LA(&BlockLora::w1));
      lin(nrm, e0, b.w3, ffu, 0, n, FFI, H, LA(&BlockLora::w3));
      // In place on the gate: same index in, same index out.
      swiglu(ffg, ffu, ffg, n * FFI);
      lin(ffg, 0, b.w2, dst, de, n, H, FFI, LA(&BlockLora::w2));
    };
    const auto t_g0 = std::chrono::steady_clock::now();
    gpu_ff(0, rows - a_rows, ob, 0);
    mark("gemm.ff");
    if (ane_split || ane_probe) {
      // The GPU's rows must be DONE before the norm below reads every
      // row, so this commit is the join -- and its time is the GPU's
      // share, which is what the balancer weighs the ANE against.
      enc.end();
      _flops.flush(stream);
      std::string ge;
      const bool gpu_ok = stream.commit().wait_ok(&ge);
      FlexData timing = FlexData::make_object();
      accel::set_real(&timing, ane::kGpuMs, ms_since(t_g0));
      accel::set_real(&timing, ane::kDrainMs, drain_ms);
      // JOINED BEFORE ANY RETURN: the worker is still writing `ob`.
      int lost = -1;
      if (a_rows > 0 && !_ane->finish(L, timing)) {
        // THE GPU FALLBACK. The host names the first row the ANE did not
        // write; the rows before it are the ANE's and kept. `nrm` was
        // only the tier's input, so the rest are recomputed from it. From
        // here on the tier answers kGpu (latched). A host that names no
        // row leaves nothing to trust.
        const FlexData inf = _ane->info();
        lost = (int)accel::integer(&inf, ane::kInfoLostRow0, -1);
        ++_eng.ane_lost;
        if (lost < rows - a_rows || lost >= rows) {
          blk_err = fmt("block {}: the ANE feed-forward lost rows and "
                        "named no range to recompute", L)();
        }
      } else if (a_rows == 0 && ane_probe) {
        _ane->note_probe(timing);
      }
      if (!gpu_ok && blk_err.empty()) {
        blk_err = ge.empty() ? std::string("GPU error") : ge;
      }
      restart();
      if (!blk_err.empty()) { return; }
      if (lost >= 0) {
        gpu_ff(lost, rows - lost, ob, (std::size_t)lost * H);
      }
      if (a_rows > 0) {
        ++_eng.ane_split;
        _eng.ane_rows += (lost >= 0 ? lost : rows) - (rows - a_rows);
        if (_ane_stage_lora) { ++_eng.ane_lora; }
      }
      if (ane_probe) { ++_eng.ane_probe; }
      // DIAGNOSTIC (VPIPE_MING_ANE_VERIFY): the ANE's rows again on the
      // GPU, compared row for row -- the tier's own error per block,
      // before 30 blocks compound it. A commit and a CPU pass per block.
      if (ane_verify && a_rows > 0 && lost < 0) {
        const int g0 = rows - a_rows;
        if (ane_ver.byte_size() < (std::size_t)a_rows * H * 2) {
          ane_ver = mc->make_shared_buffer((std::size_t)a_rows * H * 2);
        }
        if (!ane_ver.empty()) {
          gpu_ff(g0, a_rows, ane_ver, 0);
          enc.end();
          _flops.flush(stream);
          std::string ge2;
          if (stream.commit().wait_ok(&ge2)) {
            const auto* pa = static_cast<const std::uint16_t*>(
                                 ob.contents()) + (std::size_t)g0 * H;
            const auto* pg =
                static_cast<const std::uint16_t*>(ane_ver.contents());
            auto f = [](std::uint16_t h) {
              const std::uint32_t u = (std::uint32_t)h << 16;
              float v;
              std::memcpy(&v, &u, 4);
              return v;
            };
            double num = 0.0, den = 0.0, mx = 0.0, gmx = 0.0;
            // The SIGNED error along each value's own sign, over the sum
            // of magnitudes: 0 for an unbiased rounding, negative for a
            // conversion that truncates toward zero.
            double bias = 0.0, mag = 0.0;
            const std::size_t n = (std::size_t)a_rows * H;
            for (std::size_t i = 0; i < n; ++i) {
              const double a = f(pa[i]), g = f(pg[i]);
              num += (a - g) * (a - g);
              den += g * g;
              mx = std::max(mx, std::fabs(a - g));
              gmx = std::max(gmx, std::fabs(g));
              bias += g < 0.0 ? (g - a) : (a - g);
              mag += std::fabs(g);
            }
            log_(fmt("ming-image DiT: ANE L{} verify: {} rows, rel-L2 {:.4e}"
                     " vs the GPU's own, bias {:+.3e} of |y|, max |err| "
                     "{:.3g} of max |y| {:.3g}", L, a_rows,
                     den > 0.0 ? std::sqrt(num / den) : 0.0,
                     mag > 0.0 ? bias / mag : 0.0, mx, gmx));
          }
          restart();
        }
      }
    }
    rms(ob, 0, b.fn2, ob, 0, rows, H);
    if (modulated) {
      gated(joint, xe, mods, (std::size_t)3 * H, ob, 0, rows);
    } else {
      resid(joint, xe, ob, 0, rows * H);
    }
    mark("elt");
  };
  // mods = adaLN_modulation(temb) -- a BARE Linear, no activation in
  // front of it. The final layer's is SiLU + Linear.
  auto fill_mod = [&](const Block& b) {
    lin(temb, 0, b.ada, mods, 0, 1, 4 * H, _cfg.adaln_dim);
    bias(b.ada_b, mods, 0, 1, 4 * H);
    if (_fold_mod) {
      modulate(b.an1, 0, mods, 0, nw1, 0, 1);
      modulate(b.fn1, 0, mods, (std::size_t)2 * H, nw3, 0, 1);
    }
  };

  // ---- timestep ---------------------------------------------------------
  lin(tin, 0, _time_1, te1, 0, 1, _cfg.time_mid, _cfg.time_freq);
  bias(_time_1b, te1, 0, 1, _cfg.time_mid);
  silu(te1, te1, _cfg.time_mid);
  lin(te1, 0, _time_2, temb, 0, 1, _cfg.adaln_dim, _cfg.time_mid);
  bias(_time_2b, temb, 0, 1, _cfg.adaln_dim);
  silu(temb, tsilu, _cfg.adaln_dim);

  // ---- the joint stream: [ image | caption ], no pads -----------------
  const std::size_t cap0 = (std::size_t)IMG * H;   // caption's first elem
  lin(*req.latents, 0, _x_in, joint, 0, IMG, H, PD);
  bias(_x_in_b, joint, 0, IMG, H);
  if (hit != nullptr) {
    // The whole caption path, served: context-refined rows straight in.
    copy_rect(hit->out, 0, joint, cap0, CAP, H, H, H);
  } else {
    // Stream A -> cap_embedder: RMSNorm(cap_feat_dim) then the Linear,
    // on the query rows only. Compacted first, because both kernels
    // read rows at a stride of cap_feat_dim, not of the caption.
    if (Q > 0) {
      copy_rect(*req.cap, 0, capn, 0, Q, _cfg.cap_dim, CS, _cfg.cap_dim);
      rms(capn, 0, _cap_norm, capn, 0, Q, _cfg.cap_dim);
      lin(capn, 0, _cap_in, joint, cap0, Q, H, _cfg.cap_dim);
      bias(_cap_in_b, joint, cap0, Q, H);
    }
    // Stream B: already DiT-wide, BYPASSES cap_embedder entirely.
    copy_rect(*req.cap, (std::size_t)Q * CS, joint,
              cap0 + (std::size_t)Q * H, CAP - Q, H, CS, H);
  }
  trace("embed", joint, 0, (std::size_t)JT * H);
  mark("embed");

  // ---- blocks -------------------------------------------------------------
  //
  // Freeing or reading a slot while its reader thread is still writing
  // into it is a use-after-free; a scope guard survives a later return.
  struct SlotJoin {
    BlockSlots<Block>* s;
    bool on;
    ~SlotJoin() { if (on) { s->join(); } }
  } slot_join{&_slots, _stream_blocks};
  if (_stream_blocks) { _slots.begin_forward(); }

  // ---- how much of the stack this box can KEEP ---------------------------
  if (_stream_blocks) {
    if (_wire.on()) {
      if (_wire.retry(_mc, wire_retry_slack_())) {
        _resid.note_landscape_changed();
      }
      // The trunk takes its place BEFORE this forward's admissions start
      // asking: the blocks are the shed-able half.
      wire_fixed_(true);
      for (Block& b : _ctx) {
        _wire.note_wired(_mc, wire_block_(b, true), 0);
      }
    }
    const auto mbudget = mc->memory_budget();
    _resid.begin_forward(mbudget, [this]() -> std::size_t {
      return evict_tail_block_();
    });
    bool shortfall = false;
    if (_resid.count() > 0 &&
        _resid.self_compression_grew(mbudget.self_compressed)) {
      std::size_t examined = 0, incore = 0, paged_out = 0;
      resident_pages_(&examined, &incore, &paged_out);
      if (examined > 0 && incore < examined) {
        shortfall = true;
        const std::size_t freed = _resid.note_weight_residency(
            examined, incore,
            [this]() -> std::size_t { return evict_tail_block_(); });
        log_(fmt("ming-image DiT: resident weights are only {}% in RAM "
                 "({} of {} sampled pages paged out, {} MB wired) -- "
                 "released {} MB, now {} blocks resident",
                 (int)(100.0 * (double)incore / (double)examined),
                 paged_out, examined, _wire.wired_bytes() >> 20,
                 freed >> 20, _resid.count()));
      }
    }
    if (!shortfall) { _resid.note_healthy_forward(); }
  }

  // ---- env-gated streaming split (VPIPE_MING_STREAM_PROFILE) -------------
  const bool sprof = std::getenv("VPIPE_MING_STREAM_PROFILE") != nullptr;
  double sp_read_ms = 0.0, sp_gpu_ms = 0.0;
  std::size_t sp_read_bytes = 0;
  int sp_blocks = 0;
  auto sp_now = [&]() {
    return sprof ? std::chrono::steady_clock::now()
                 : std::chrono::steady_clock::time_point{};
  };
  auto sp_add = [&](double& sink, std::chrono::steady_clock::time_point t0) {
    if (sprof) {
      sink += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
    }
  };

  // THE ADAPTER'S [rows, rank] INTERMEDIATE, sized ONCE for the widest
  // projection this pass takes -- the joint sequence. Without it apply()
  // returns early and the adapter silently does nothing, which looks
  // exactly like an adapter that had no effect. Each slot gets its own
  // rows * rank region (lora_for_).
  if (_lora_rank_total > 0) {
    _lora_scratch_rows = (std::size_t)JT;
    if (!_lora.ensure_scratch(_lora_scratch_rows *
                              (std::size_t)_lora_rank_total)) {
      return fail("ming DiT forward: the LoRA scratch did not allocate");
    }
  }

  const int TOTAL_BLOCKS = _cfg.n_blocks();
  int done = 0;

  // ---- context_refiner, over the caption rows alone ----------------------
  //
  // The reference runs the NOISE refiner first. The two touch disjoint
  // rows, so the order is free, and the caption first leaves the 32
  // modulated blocks contiguous for streaming. UNMODULATED: no timestep
  // enters the caption stream -- which is what makes it cacheable.
  for (int i = 0; i < _cfg.n_refiner; ++i) {
    if (_stream_stop && _stream_stop()) { return SharedBuffer{}; }
    // The blocks before this one, credited when the buffer now open
    // retires -- the checkpoint the progress report rides.
    _flops.flush(stream);
    if (_block_progress) {
      vpipe::genai::report_block(stream, _block_progress, done,
                                 TOTAL_BLOCKS);
    }
    if (hit == nullptr) {
      run_block(_ctx[(std::size_t)i], cap0, CAP, IMG,
                /*modulated=*/false, i);
    }
    ++done;
  }
  if (fill != nullptr) {
    // Encoded BEFORE the main stack, which rewrites these rows in place.
    copy_rect(joint, cap0, fill->out, 0, CAP, H, H, H);
  }
  trace("ctx", joint, cap0, (std::size_t)CAP * H);

  // ---- noise_refiner then layers -----------------------------------------
  for (int L = 0; L < NM; ++L) {
    // COOPERATIVE STOP, checked every block. EMPTY WITHOUT AN ERROR: a
    // stop is not a failure.
    if (_stream_stop && _stream_stop()) { return SharedBuffer{}; }
    // RESIDENT once residency has promoted it (or always, preloaded),
    // read into a slot until then.
    const bool held = L < (int)_blocks.size() &&
                      !_blocks[(std::size_t)L].qw.empty();
    const bool streaming = _stream_blocks && !held;
    const Block* b = nullptr;
    if (!streaming) {
      b = &_blocks[(std::size_t)L];
      ++_eng.blocks_held;
    } else {
      const auto rd0 = sp_now();
      b = _slots.acquire(L);
      if (b == nullptr) {
        return fail("ming DiT forward: a streamed block did not load");
      }
      ++_eng.blocks_streamed;
      sp_add(sp_read_ms, rd0);
      if (sprof) {
        sp_read_bytes += _slots.last_bytes();
        ++sp_blocks;
      }
      // THE READ IS NOT ELEMENTWISE WORK: give it its own bucket.
      mark("stream");
    }
    // The noise refiner runs over the IMAGE rows; the main stack over
    // the whole joint sequence.
    const bool is_refiner = L < _cfg.n_refiner;
    const int rows = is_refiner ? IMG : JT;

    // THE ANE'S PLAN FOR THIS BLOCK, decided before anything is encoded,
    // because the staging has to ride under the attention. The MAIN stack
    // only: the noise refiner runs the image rows alone, a shorter
    // sequence than the module's share was set from.
    AnePlan ane_plan = AnePlan::kGpu;
    if (_cfg.ane_ffn && !is_refiner && ane_setup_(JT) &&
        ane_eligible_(L, *b)) {
      ane_plan = _ane->plan_block();
    }
    if (ane_plan != AnePlan::kGpu &&
        (L == _cfg.n_refiner || _ane->needs_barrier())) {
      // An unsplit block commits nothing mid-block, so the split point's
      // drain would wait for every block queued before it and hand the
      // balancer their time as this block's. Drain them FIRST.
      enc.end();
      _flops.flush(stream);
      std::string ge;
      if (!stream.commit().wait_ok(&ge)) {
        return fail("ming DiT forward: " + ge);
      }
      restart();
    }
    if (ane_plan == AnePlan::kSplit) { (void)ane_stage_(L, *b); }

    _flops.flush(stream);
    if (_block_progress) {
      // `done` of `total`: the blocks finished BEFORE this one -- a
      // report that lags the truth and never leads it.
      vpipe::genai::report_block(stream, _block_progress, done,
                                 TOTAL_BLOCKS);
    }

    fill_mod(*b);
    run_block(*b, 0, rows, 0, /*modulated=*/true, L, ane_plan);
    if (!blk_err.empty()) {
      return fail("ming DiT forward: " + blk_err);
    }
    ++done;

    // ---- the streaming boundary ------------------------------------------
    //
    // A STREAMED block must commit: the two slots are reused, so anything
    // encoded but not yet run still points at bytes the next read would
    // overwrite. The commit is a correctness boundary, and it buys the
    // window the prefetch needs.
    if (streaming) {
      const auto gp0 = sp_now();
      enc.end();
      _flops.flush(stream);
      std::string ge;
      {
        CommandStream::Fence bf = stream.commit();
        // BETWEEN THE COMMIT AND THE WAIT the next block's read goes into
        // the OTHER slot -- the only destination this block is not using.
        // Skip what residency already kept.
        int nxt = -1;
        for (int n = L + 1; n < NM; ++n) {
          const bool h = n < (int)_blocks.size() &&
                         !_blocks[(std::size_t)n].qw.empty();
          if (!h) { nxt = n; break; }
        }
        _slots.prefetch(nxt);
        if (!bf.wait_ok(&ge)) { return fail("ming DiT forward: " + ge); }
      }
      // ---- keep this block, if the box can hold it -----------------------
      //
      // AFTER the wait, so nothing encoded still points at the slot; the
      // promotion CLONES so the slot stays usable for the read in flight.
      const std::size_t nb = _slots.last_bytes();
      if (nb > 0 && _wire.wirable(nb) && _resid.admit(_mc, nb) &&
          _slots.promote_into(_blocks[(std::size_t)L])) {
        // Wired LAST, after every write this block will ever get.
        _wire.note_wired(_mc, wire_block_(_blocks[(std::size_t)L], true),
                         nb);
        _resid.note_admitted(nb);
      }
      sp_add(sp_gpu_ms, gp0);
      stream = mc->make_command_stream();
      enc = stream.begin_compute();
      if (!enc.valid()) { return fail("ming DiT forward: no encoder"); }
      mark("stream");
    } else if (_stream_stop && kStopDrainBlocks > 0 &&
               ((L + 1) % kStopDrainBlocks) == 0 && L + 1 < NM) {
      // A HELD BLOCK COMMITS NOTHING, so without this the whole forward
      // runs inside one wait and a stop requested mid-forward is never
      // seen. Only when a stop is wired.
      enc.end();
      _flops.flush(stream);
      std::string ge;
      if (!stream.commit().wait_ok(&ge)) {
        return fail("ming DiT forward: " + ge);
      }
      stream = mc->make_command_stream();
      enc = stream.begin_compute();
      if (!enc.valid()) { return fail("ming DiT forward: no encoder"); }
    }
    trace("blk" + std::to_string(L), joint, 0, (std::size_t)JT * H);
  }
  _eng.prefetch_hits = _slots.prefetch_hits();
  if (!attn_ok) {
    return fail("ming DiT forward: the steel attention kernel did not "
                "specialize for this geometry");
  }

  // ---- the final layer ----------------------------------------------------
  //
  // LN(x) * (1 + linear(SiLU(temb))) then a projection back to patch
  // space -- THIS adaLN has the SiLU in front of it. Only the IMAGE rows
  // are projected: they are the front of the joint sequence.
  lin(tsilu, 0, _final_ada, outmod, 0, 1, H, _cfg.adaln_dim);
  bias(_final_ada_b, outmod, 0, 1, H);
  ln_plain(joint, 0, nrm, 0, IMG);
  modulate(nrm, 0, outmod, 0, nrm, 0, IMG);
  lin(nrm, 0, _final_lin, vel, 0, IMG, PD, H);
  bias(_final_lin_b, vel, 0, IMG, PD);
  enc.end();
  _flops.flush(stream);
  {
    std::string ge;
    if (!stream.commit().wait_ok(&ge)) {
      return fail("ming DiT forward: " + (ge.empty() ? "GPU failed" : ge));
    }
  }
  // Sol's counter is the GPU's, so it is read only now, past the last
  // wait -- an earlier read would report a number the device had not
  // finished producing.
  if (_sol && _eng.attn_sol > 0) {
    _eng.sol_kept = _sol->exact_blocks();
    _eng.sol_total = sol_total;
  }
  if (_sol) { _sol_flops.finish(_sol->exact_blocks()); }
  _eng.gpu_flops = _flops.flushed() - flops0;
  // THE MODULE'S BOOKING, once its fate is known. The plan granted one
  // unit; the tier arms (or declines -- too few rows for a chunk) at the
  // first forward, so the scratch ledger is set to what is HELD: the
  // module when it armed, 0 when it did not -- under the label it was
  // booked under (the host's own, when it relabelled the claim). Both
  // ways and every time it changes, since a short forward that declined
  // may be followed by a long one that arms. revise_scratch refuses to
  // create, so this is a no-op where the plan booked nothing (a test, a
  // tool).
  if (_cfg.ane_ffn && _ane_tried_seq > 0 && _session != nullptr) {
    const std::size_t held = _ane ? MingDit::ane_runtime_bytes(_dcfg) : 0;
    if (held != _ane_booked) {
      if (auto* gm = _session->services()->generative_model_manager()) {
        gm->revise_scratch("coreml:" + _ane_label, held);
      }
      _ane_booked = held;
    }
  }
  // The pass COMPLETED: only now may the caption it computed be served.
  if (fill != nullptr) {
    const auto* cp = static_cast<const std::uint8_t*>(req.cap->contents());
    fill->key.assign(cp, cp + cap_bytes);
    fill->q = Q;
    fill->len = CAP;
    fill->stride = CS;
    fill->valid = true;
    fill->tick = ++_cap_tick;
  } else if (hit != nullptr) {
    hit->tick = ++_cap_tick;
  }
  if (prof && _session != nullptr) {
    double tot = 0.0;
    for (const auto& b : pbuckets) { tot += b.ms; }
    std::string line;
    for (const auto& b : pbuckets) {
      line += " | " + std::string(b.name) + " " +
              std::to_string((long)b.ms) + " ms (" +
              std::to_string(tot > 0.0 ? (int)(100.0 * b.ms / tot) : 0) +
              "%, " + std::to_string(b.marks) + " marks)";
    }
    _session->log_normal(fmt(
        "ming-image DiT profile (joint={} img={} cap={}, {} blocks, {}, {} "
        "marks -- each is a commit AND a wait): total {} ms{}",
        JT, IMG, CAP, TOTAL_BLOCKS,
        _use_mma2 ? "matmul2d"
                  : (_gemm_tile == 2 ? "bm64bn64"
                                     : (_gemm_tile == 1 ? "bm64" : "bm32")),
        pmarks, (long)tot, line));
  }
  if (sprof && sp_blocks > 0 && _session != nullptr) {
    const double tot = sp_read_ms + sp_gpu_ms;
    const double gbs = sp_read_ms > 0.0
        ? (double)sp_read_bytes / (sp_read_ms / 1000.0) / 1e9 : 0.0;
    _session->log_normal(fmt(
        "ming-image DiT: streamed {} of {} blocks -- read {} ms ({} MB at "
        "{:.2f} GB/s), gpu {} ms; a perfect prefetch hides at most {:.1f}% "
        "of this pass",
        sp_blocks, NM, (long)sp_read_ms, sp_read_bytes >> 20, gbs,
        (long)sp_gpu_ms, tot > 0.0 ? 100.0 * sp_read_ms / tot : 0.0));
  }
  return vel;
}

// ---- the matrix-core GEMM ---------------------------------------------
//
// Returns false with NOTHING encoded when this box or this shape is not
// for it, and the caller keeps its tiled path -- a decline is a
// fallback, never a hole.
bool
MingDitMetal::gemm_mma_(ComputeEncoder& enc, const SharedBuffer& xin,
                        std::size_t xe, const QWeight& w,
                        const SharedBuffer& y, std::size_t ye, int M, int N,
                        int K) const
{
  // Only where the hardware has matrix cores, where M amortizes the
  // 128-row tile, and where N is not degenerate. The timestep, the
  // modulation and the final adaLN are M=1 and stay on steel.
  if (!_use_mma2 || M < _mma_min_m || N < 16) { return false; }
  // Past 2^31 BYTES on any operand the mma tiles stop storing, SILENTLY.
  if (M > vpipe::genai::mma_row_band(N, K)) { return false; }

  const SharedBuffer* wdense = nullptr;
  if (w.quantized) {
    const ComputeFunction& dq = (w.bits == 8) ? _fn_dequant8 : _fn_dequant4;
    if (!dq.valid()) { return false; }
    const std::size_t need = (std::size_t)N * K * 2;   // bf16
    if (_w_deq.empty() || _w_deq.byte_size() < need) {
      _w_deq = _mc->make_shared_buffer(need);
      if (_w_deq.empty()) { return false; }
    }
    enc.set_function(dq);
    enc.set_buffer(0, w.codes); enc.set_buffer(1, w.scales);
    enc.set_buffer(2, w.qbias); enc.set_buffer(3, _w_deq);
    enc.set_constant(4, K); enc.set_constant(5, N);
    const unsigned words = (unsigned)(w.bits == 8 ? (K / 4) : (K / 8));
    enc.dispatch({words, (unsigned)N, 1}, {64, 1, 1});
    wdense = &_w_deq;
  } else {
    wdense = &w.w;
  }

  // Accelerated mode, from the SAME dense operand the tiles would read.
  if (_i8 && _i8->gemm(enc, xin, xe, *wdense, y, ye, M, N, K)) {
    ++_eng.gemm_i8;
    return true;
  }

  // Tile by REDUCTION DEPTH, which decides whether the deeper N-region
  // pays for itself.
  int RN = 256;                                  // TN*BN, the N per group
  const ComputeFunction* fn = &_fn_gemm_mma_deep;
  if (K < 6144) {
    fn = &_fn_gemm_mma; RN = 128;
  } else if (K < 12288 && _fn_gemm_mma_tn2.valid()) {
    fn = &_fn_gemm_mma_tn2; RN = 512;
  }
  if (!fn->valid()) { return false; }
  ++_eng.gemm_mma2;
  enc.set_function(*fn);
  enc.set_buffer(0, xin, xe * 2);
  enc.set_buffer(1, *wdense); enc.set_buffer(2, *wdense);
  enc.set_buffer(3, y, ye * 2);
  enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
  return true;
}

// ---- the ANE feed-forward (`ane_ffn`) ----------------------------------
//
// THE ANE FEED-FORWARD. A SwiGLU feed-forward is row-independent, so the
// ANE takes the LAST rows of a main-stack block's feed-forward and the
// GPU the head rows, concurrently: the two compute the same function over
// disjoint rows of the same tensors, so there is no seam to blend. The
// ANE's rows are fp16 (more mantissa than bf16, less range -- the host
// rescales the up projection when a chunk overflows) and come back
// bf16, so the bar is a rounding difference, not bit-identity.
//
// ON THE INSTALLED ane::Tier, not the in-tree AneFeedForward (whose
// header the SDK does not ship): the same module, balancer, staging,
// lost-row fallback and latch, described by FlexData keys. What Tier's
// vocabulary does NOT carry today is the in-tree q/k/v tier's shape --
// three source matrices stacked into one weight slot (AneFfnSource::
// slot_row) and output columns scattered into three planes (OutSeg) --
// so q/k/v would need a fused [3H, H] weight copied per block and a
// [joint, 3H] plane split back out on the GPU. Not done.
//
// WHERE IT IS FOR: the M4 series, which has no GPU matrix cores, so its
// GEMMs run the tiled path and there is proportionally more for the ANE
// to take. A fanless box shares one power budget between the engines and
// does not truly overlap them -- do not judge the tier there.
//
// MEASURED on an M4 Pro (fan), one held forward, arms interleaved: 1024^2
// 8.81 -> 7.20 s (1.22x; the ANE takes one 2048-row chunk of 4386 and
// finishes ~35 ms before the GPU's half every block -- the chunk is the
// granularity), 2048^2 49.4 -> 40.7 s (1.21x; five chunks of 16674).
// With Sol at tau 0 beside it, 1.32x / 1.59x. ACCURACY: each split
// block's ANE rows sit ~0.3% (rel-L2) from the GPU's own, about half of
// it a bias toward zero -- the host converts the ANE's f32 rows to bf16
// by TRUNCATION -- yet this model's output moves ~0.04 for one tensor
// rounded differently per block, so the whole forward lands 0.05-0.18
// from the GPU's. Against the reference at 512^2 it is 0.112 (GPU 0.053,
// the reference's own bf16 0.062). The up projection overflows fp16 on
// real prompts (L2, L14, L16 at 2048^2) and the host's headroom retry
// takes it to 1/256, losslessly.

namespace {

FlexData
ane_spec_(int hidden, int inner, int seq, float rows)
{
  FlexData s = FlexData::make_object();
  accel::set_text(&s, ane::kKind, std::string(ane::kKindFfn));
  accel::set_text(&s, ane::kActivation, std::string(ane::kActSwiglu));
  accel::set_integer(&s, ane::kHidden, hidden);
  accel::set_integer(&s, ane::kInner, inner);
  accel::set_integer(&s, ane::kSeq, seq);
  accel::set_real(&s, ane::kRows, (double)rows);
  if (const char* c = std::getenv("VPIPE_MING_ANE_CHUNK")) {
    accel::set_integer(&s, ane::kChunk, std::atoi(c));
  }
  accel::set_flag(&s, ane::kProfile,
                  std::getenv("VPIPE_MING_ANE_PROFILE") != nullptr);
  // The log prefix, and what VPIPE_ANE_FAIL_TAG selects.
  accel::set_text(&s, ane::kTag, "ming-image");
  return s;
}

}  // namespace

bool
MingDitMetal::ane_setup_(int seq)
{
  if (_ane) { return true; }
  // Created at the first forward, when the row count is finally known --
  // and asked again only by a LONGER one: a picture too small for one
  // 2048-row chunk declines, and asking at every forward would re-log it.
  if (!_cfg.ane_ffn || _session == nullptr || seq <= _ane_tried_seq) {
    return false;
  }
  _ane_tried_seq = seq;
  std::string why;
  _ane = ane::create(_session, _mc,
                     ane_spec_(_cfg.hidden, _cfg.ffn_inner, seq,
                               _cfg.ane_rows),
                     &why);
  if (!_ane) {
    _session->info(fmt("ming-image DiT: the ANE feed-forward is not used "
                       "at {} rows: {}", seq, why));
  }
  return (bool)_ane;
}

bool
MingDitMetal::ane_eligible_(int L, const Block& b) const
{
  if (!_ane) { return false; }
  // Counted over the MAIN stack: L indexes the modulated stack, whose
  // first n_refiner entries are the noise refiner.
  const int main_idx = L - _cfg.n_refiner;
  if (main_idx < 0) { return false; }
  const int cap = _cfg.ane_layers > 0
                      ? std::min(_cfg.ane_layers, _cfg.n_layers)
                      : _cfg.n_layers;
  if (main_idx >= cap) { return false; }
  auto ok = [](const QWeight& q) {
    if (q.empty()) { return false; }
    if (!q.quantized) { return true; }
    return (q.bits == 4 || q.bits == 8) && !q.scales.empty() &&
           !q.qbias.empty();
  };
  return ok(b.w1) && ok(b.w3) && ok(b.w2);
}

bool
MingDitMetal::ane_stage_(int L, const Block& b)
{
  _ane_stage_lora = false;
  FlexData params = FlexData::make_object();
  accel::set_integer(&params, ane::kGroup, _quant_group);
  // A Binding VIEWS its name, so the names live here, in a deque: a push
  // never moves the strings already in it, however many adapters a
  // projection carries (eight slots of two factors each, beside up to
  // three weight tensors). A reserved vector was right only while the
  // count was small and fixed.
  std::deque<std::string> names;
  std::vector<ane::Binding> binds;
  bool any_lora = false;
  auto add = [&](const std::string& name, const SharedBuffer& buf) {
    if (buf.empty()) { return; }
    names.push_back(name);
    binds.push_back({names.back(), &buf});
  };
  auto bind = [&](std::string_view p, const QWeight& q,
                  lora::Factors BlockLora::* which) {
    if (q.quantized) {
      add(ane::key(p, "codes"), q.codes);
      add(ane::key(p, "scales"), q.scales);
      add(ane::key(p, "qbias"), q.qbias);
      accel::set_integer(&params, ane::key(p, "bits"), q.bits);
    } else {
      add(ane::key(p, "w"), q.w);
    }
    // THE ADAPTER GOES INTO THE MODULE. The ANE is handed each block's
    // weights, so an adapter left out of what is staged runs the BASE
    // model on the ANE's rows and the adapted one on the GPU's -- a seam
    // that reads as the adapter at a fraction of its strength. The same
    // stack the GPU's GEMMs take (live strengths, zero-strength slots
    // already dropped), as `<p>.lora<i>`: A [rank, in], B [out, rank],
    // the file's alpha already folded into A, the strength as `.scale`.
    const lora::Stack st = _ane_drop_lora
                               ? lora::Stack{}
                               : lora_for_(/*ctx=*/false, L, which);
    for (int i = 0; i < st.n; ++i) {
      const lora::Factors& f = *st.s[i].f;
      const std::string lp = ane::key(p, "lora" + std::to_string(i));
      add(lp + ".a", f.a);
      add(lp + ".b", f.b);
      accel::set_integer(&params, lp + ".rank", f.b_rank());
      accel::set_real(&params, lp + ".scale", (double)st.s[i].scale);
      any_lora = true;
    }
  };
  // SwiGLU as Llama names it: w1 gates, w3 is up, w2 is down. Separate
  // matrices here, so each is read whole (stride 1, offset 0).
  bind(ane::kProjGate, b.w1, &BlockLora::w1);
  bind(ane::kProjUp, b.w3, &BlockLora::w3);
  bind(ane::kProjDown, b.w2, &BlockLora::w2);
  if (_ane->stage(L, params, binds)) {
    _ane_stage_lora = any_lora;
    return true;
  }
  if (!_ane_warned) {
    // The split then finds nothing staged and keeps the GPU -- adapter
    // included, since its rows then all run there; say why once.
    _ane_warned = true;
    const FlexData inf = _ane->info();
    warn_(fmt("ming-image DiT: the ANE refused block {}'s feed-forward: {}",
              L, accel::text(&inf, ane::kInfoError)));
  }
  return false;
}

bool
MingDitMetal::ane_gpu_only() const
{
  if (!_ane) { return false; }
  const FlexData inf = _ane->info();
  return accel::flag(&inf, ane::kInfoGpuOnly);
}

// ---- runtime LoRA -------------------------------------------------------
//
// THE MODULE NAMES ARE THE MODEL'S OWN -- the convention a diffusers or
// peft export of this architecture carries (the reference's DiT is a
// diffusers PeftAdapterMixin, so `save_lora_adapter` writes exactly
// these): the base tensor's path without `.weight`, per block
//
//   attention.to_q / to_k / to_v / to_out.0
//   feed_forward.w1 / w3 / w2
//
// over context_refiner.<i>, noise_refiner.<i> and layers.<i>. The shared
// reader also finds them under `transformer.` / `diffusion_model.`
// prefixes, peft's `.lora_A.<adapter>.` infix and kohya's
// `lora_unet_<flattened>.lora_down/lora_up`, with the alpha a file
// states (per module, or once in its metadata) folded in. No Ming-Image
// adapter has been published, so this list is a READING of those
// conventions, verified against synthetic adapters in them (ming-dit-
// test, "runtime LoRA"); a real one that binds 0 modules is a rename row
// in the toolkit's lora-names.h, not a change here.
//
// NOT adapted: adaLN_modulation (nothing in this lineage trains it, and
// binding it would read a file's silence as a zero), and the model-level
// embedders, timestep MLP and final layer, which the in-tree Z-Image
// does not adapt either.

bool
MingDitMetal::bind_loras_(const std::vector<LoraSpec>& loras)
{
  if (loras.empty()) { return true; }
  if (!_lora.init(_mc, _use_mma2) || !_lora.valid()) { return false; }
  const int nmod = _cfg.n_mod_blocks(), nctx = _cfg.n_refiner;
  const int H = _cfg.hidden, FFI = _cfg.ffn_inner;
  // POSITIONAL: spec i is slot i, and an empty path leaves its slot
  // unbound rather than letting the next adapter move into it -- so the
  // caller's `lora2_scale` always reaches the adapter named by `lora2`,
  // whether or not `lora` named one.
  for (int slot = 0; slot < (int)loras.size(); ++slot) {
    const LoraSpec& spec = loras[(std::size_t)slot];
    if (slot >= kMaxLoraSlots) {
      if (!spec.path.empty()) {
        warn_(fmt("ming-image DiT: only {} adapters run at once; '{}' is "
                  "NOT applied", kMaxLoraSlots, spec.path));
      }
      continue;
    }
    _lora_slots = slot + 1;
    _lora_scale[slot] = spec.scale;
    if (spec.path.empty()) { continue; }
    std::string err;
    auto ad = lora::Adapter::open(spec.path, _mc, &err,
                                  lora::remap_ai_toolkit_module);
    if (ad == nullptr) {
      warn_(fmt("ming-image DiT: lora '{}' did not open ({}); it is NOT "
                "applied", spec.path, err.empty() ? "unreadable" : err));
      continue;
    }
    // resize, not assign: Factors holds SharedBuffers.
    _lora_mod[slot].clear();
    _lora_mod[slot].resize((std::size_t)nmod);
    _lora_ctx[slot].clear();
    _lora_ctx[slot].resize((std::size_t)nctx);
    auto one = [&](const std::string& pre, BlockLora& d) {
      ad->bind(pre + "attention.to_q", H, H, &d.q);
      ad->bind(pre + "attention.to_k", H, H, &d.k);
      ad->bind(pre + "attention.to_v", H, H, &d.v);
      ad->bind(pre + "attention.to_out.0", H, H, &d.o);
      ad->bind(pre + "feed_forward.w1", FFI, H, &d.w1);
      ad->bind(pre + "feed_forward.w3", FFI, H, &d.w3);
      ad->bind(pre + "feed_forward.w2", H, FFI, &d.w2);
    };
    for (int L = 0; L < nmod; ++L) {
      one(mod_prefix_(L), _lora_mod[slot][(std::size_t)L]);
    }
    for (int i = 0; i < nctx; ++i) {
      one(ctx_prefix_(i), _lora_ctx[slot][(std::size_t)i]);
    }
    _lora_skipped[slot] = ad->skipped();
    log_(fmt("ming-image DiT: {}", ad->summary(spec.path, spec.scale)));
    if (ad->skipped() > 0) {
      // A NAME THAT FITS AT THE WRONG SHAPE means the file was trained
      // against a different model of this lineage, and the modules that
      // happen to fit are half of an adapter, not a weaker one: the whole
      // slot is refused rather than applied in part.
      for (BlockLora& b : _lora_mod[slot]) { b = BlockLora{}; }
      for (BlockLora& b : _lora_ctx[slot]) { b = BlockLora{}; }
      warn_(fmt("ming-image DiT: lora '{}' is NOT applied: {} module(s) "
                "matched by name at the wrong shape ({} fit) -- it was "
                "trained against a different model; the base model runs",
                spec.path, ad->skipped(), ad->modules()));
      continue;
    }
    _lora_bound[slot] = ad->modules();
    for (const auto* v : {&_lora_mod[slot], &_lora_ctx[slot]}) {
      for (const BlockLora& b : *v) {
        for (const lora::Factors* f :
             {&b.q, &b.k, &b.v, &b.o, &b.w1, &b.w3, &b.w2}) {
          _lora_bytes += f->a.byte_size() + f->b.byte_size();
        }
      }
    }
    _lora_rank_prefix[slot] = _lora_rank_total;
    _lora_rank_total += ad->max_rank();
    if (ad->modules() == 0) {
      // LOUD. An adapter for another model binds nothing and would
      // otherwise report success by saying nothing at all.
      warn_(fmt("ming-image DiT: lora '{}' bound NO modules -- it is for "
                "a different model, or its names are in a convention "
                "this family does not map; the base model runs",
                spec.path));
    }
  }
  return true;
}

lora::Stack
MingDitMetal::lora_for_(bool ctx, int L,
                        lora::Factors BlockLora::* which) const
{
  lora::Stack st;
  for (int i = 0; i < _lora_slots; ++i) {
    const std::vector<BlockLora>& v = ctx ? _lora_ctx[i] : _lora_mod[i];
    if (L < 0 || L >= (int)v.size()) { continue; }
    // Each slot gets its OWN region of the [rows, rank] scratch: sharing
    // one would make the last writer's intermediate the thing every B
    // factor multiplies. add() drops an unbound module and a 0 strength.
    st.add(v[(std::size_t)L].*which, _lora_scale[i],
           _lora_scratch_rows * (std::size_t)_lora_rank_prefix[i]);
  }
  return st;
}

void
MingDitMetal::set_lora_scale(int slot, float s)
{
  if (slot < 0 || slot >= kMaxLoraSlots || _lora_scale[slot] == s) {
    return;
  }
  _lora_scale[slot] = s;
  // THE CAPTION CACHE HOLDS CONTEXT-REFINED ROWS, and the context
  // refiner's projections carry the adapter: served after a strength
  // change, they would be the OLD strength's caption under the new one's
  // image -- a picture, not an error. Keyed on the caption's bytes alone,
  // so it is dropped here rather than keyed on the strengths too.
  if (_lora_bound[slot] > 0) {
    for (CapEntry& e : _cap_cache) { e.valid = false; }
  }
}

// ---- residency ----------------------------------------------------------

std::size_t
MingDitMetal::checkpoint_block_bytes_() const
{
  if (_ws == nullptr) { return 0; }
  const MetalLlamaWeights& w = _ws->src();
  // Block 0 of the streaming stack (noise_refiner.0), the same shape as
  // every block in the stack.
  const std::string pre = mod_prefix_(0);
  std::size_t n = 0;
  for (const std::string& nm : w.tensor_names()) {
    if (nm.rfind(pre, 0) != 0) { continue; }
    const auto* ti = w.info(nm);
    if (ti != nullptr) { n += (std::size_t)ti->nbytes; }
  }
  // WHAT A BLOCK COSTS RESIDENT, not on disk: an F32 checkpoint is
  // narrowed to bf16 on the way in.
  if (_src_f32) { n /= 2; }
  return n;
}

void
MingDitMetal::set_residency_schedule(int steps)
{
  const std::size_t blk = checkpoint_block_bytes_();
  _wire_block_hint = blk;
  _wire.new_run();
  _resid.set_schedule(steps, _cfg.n_mod_blocks(), blk, _wire.on(),
                      _mc != nullptr ? _mc->memory_budget()
                                     : MetalCompute::MemoryBudget{});
}

std::size_t
MingDitMetal::release_resident_blocks(std::size_t bytes)
{
  const std::size_t freed = _resid.release(bytes, [this]() -> std::size_t {
    return evict_tail_block_();
  });
  if (freed > 0 && _session != nullptr) {
    _session->log_debug(fmt("ming-image DiT: released {} MB of resident "
                            "blocks ({} left)", freed >> 20, _resid.count()));
  }
  return freed;
}

std::uint64_t
MingDitMetal::resident_bytes() const
{
  std::uint64_t n = qw_bytes_(_x_in) + _x_in_b.byte_size() +
                    _cap_norm.byte_size() + qw_bytes_(_cap_in) +
                    _cap_in_b.byte_size() + qw_bytes_(_time_1) +
                    _time_1b.byte_size() + qw_bytes_(_time_2) +
                    _time_2b.byte_size() + qw_bytes_(_final_ada) +
                    _final_ada_b.byte_size() + qw_bytes_(_final_lin) +
                    _final_lin_b.byte_size();
  // The context refiner is held in BOTH modes: trunk, not residency.
  for (const Block& b : _ctx) { n += block_bytes_(b); }
  // `_blocks` IS the answer in both modes: the whole stack preloaded,
  // exactly what residency promoted when streaming.
  for (const Block& b : _blocks) { n += block_bytes_(b); }
  for (const CapEntry& e : _cap_cache) { n += e.out.byte_size(); }
  // The adapters' factors: resident in both modes, whatever streams.
  n += _lora_bytes;
  return n;
}

}  // namespace

// ---- the public face ------------------------------------------------------

std::unique_ptr<MingDit>
MingDit::load(std::shared_ptr<vpipe::genai::WeightSet> ws,
              vpipe::metal_compute::MetalCompute* mc, const DitConfig& cfg,
              const vpipe::FlexData* accel, bool stream_blocks,
              const vpipe::SessionContextIntf* session, std::string* err,
              const LoadOptions& opts)
{
  return MingDitMetal::load_(std::move(ws), mc, cfg, accel, stream_blocks,
                             session, err, opts);
}

int
MingDit::lora_module_capacity(const DitConfig& cfg)
{
  // Seven projections per block: q, k, v, out, w1, w3, w2.
  return 7 * (2 * cfg.n_refiner_layers + cfg.n_layers);
}

// THE STREAMING FLOOR: trunk + ONE slot pair, in the bytes this model
// actually HOLDS. The streamed stack is ONE stack spelled under TWO
// prefixes (`noise_refiner.` then `layers.`) sharing one slot pair,
// while `context_refiner.` is held in both modes and belongs to the
// trunk -- so the shared per-stem helper, which books a pair per STEM,
// is the wrong tool. An F32 checkpoint lands as bf16, so its floor is
// half its on-disk bytes. 0 when the directory cannot be read.
std::size_t
MingDit::streaming_floor_bytes(const std::string& dir)
{
  if (dir.empty()) { return 0; }
  auto wts = MetalLlamaWeights::open_model(dir);
  if (!wts.has_value()) { return 0; }
  std::size_t trunk = 0;
  // Keyed by the STREAM index, so noise_refiner.0 and layers.0 are
  // different blocks of one stack.
  std::vector<std::size_t> blocks;
  bool any_f32 = false;
  for (const std::string& nm : wts->tensor_names()) {
    const auto* ti = wts->info(nm);
    const std::size_t nb = ti != nullptr ? (std::size_t)ti->nbytes : 0;
    if (ti != nullptr && ti->dtype == "F32") { any_f32 = true; }
    int which = -1;
    for (int i = 0; i < 2; ++i) {
      if (nm.rfind(kStreamed[i], 0) == 0) { which = i; break; }
    }
    if (which < 0) { trunk += nb; continue; }
    const std::size_t i0 = std::strlen(kStreamed[which]);
    const std::size_t dot = nm.find('.', i0);
    if (dot == std::string::npos) { trunk += nb; continue; }
    const std::size_t idx =
        (std::size_t)(which * 1000 +
                      std::atol(nm.substr(i0, dot - i0).c_str()));
    if (blocks.size() <= idx) { blocks.resize(idx + 1, 0); }
    blocks[idx] += nb;
  }
  std::size_t widest = 0;
  for (std::size_t b : blocks) { widest = std::max(widest, b); }
  if (widest == 0) { return 0; }
  std::size_t floor = trunk + 2 * widest;
  if (any_f32) { floor /= 2; }
  return floor;
}

std::size_t
MingDit::ane_runtime_bytes(const DitConfig& cfg)
{
  Cfg c;
  if (!cfg_from_(cfg, &c, nullptr)) { return 0; }
  // The module's cost does not depend on the sequence (its host rows are
  // one chunk), so any positive seq answers for every picture size.
  return vpipe::genai::ane::runtime_bytes(
      ane_spec_(c.hidden, c.ffn_inner, 1, 0.0f));
}

}  // namespace ming
