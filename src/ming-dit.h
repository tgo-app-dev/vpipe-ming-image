#ifndef VPIPE_MING_DIT_H
#define VPIPE_MING_DIT_H

// Ming-Image's DiT: diffusers' ZImageTransformer2DModel at Z-Image 6B's
// dimensions, with three deltas (docs/MING-IMAGE.md section 2):
//
//   1. "zero_masked" padding -- no learned pad tokens; this port runs
//      the real rows only (see ming-layout.h for why that is exact);
//   2. a SECOND caption stream: the caption is [cap_embedder(stream A) ;
//      stream B], where stream B arrives already DiT-wide;
//   3. no caption-length cap (the axis-0 rotary table is 20480 long).
//
// Vendored from vpipe's in-tree MetalZImageTransformer, which libvpipe
// does not export. See the head of ming-dit.cc for what was kept, what
// was stripped and why.

#include "ming-config.h"
#include "ming-layout.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/dit-block-progress.h"
#include "generative-models/shared/runtime-lora.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
class FlexData;
class SessionContextIntf;
namespace genai { class WeightSet; }
}  // namespace vpipe

namespace ming {

// The plugin's own kernels (src/metal/ming_ops.metal): the entries the
// DiT dispatches that the host's kernel contract does not promise. The
// plugin registers both twins at load under these names; the DiT runs
// the bf16 one and refuses to load without it.
inline constexpr const char* kMetalLibBf16 = "ming_ops_bf16";
inline constexpr const char* kMetalLibF16  = "ming_ops_f16";

// The ANE feed-forward module's CoreML claim (`ane_ffn`). The family
// books it under this label (it cannot see the stage's id or the
// graph's accel bag -- see MingDit::ane_runtime_bytes). A current host
// RELABELS it to its own stage and hands that label back at load
// (MingDit::LoadOptions::coreml_label); an older one keeps this label,
// and the DiT then reads the grant back under it itself.
inline constexpr const char* kAneClaimLabel = "ming-image/ane-ffn";

// What ONE forward actually dispatched, counted at the dispatch site.
// A tier that is declared, configured and never dispatched passes every
// numeric bar -- so a test asserts these before it asserts accuracy.
struct DitEngagement {
  // GEMM routes. Every Linear in the model lands in exactly one.
  int gemm_mma2 = 0;       // matmul2d (M5 matrix cores)
  int gemm_i8 = 0;         // int8 activations on matmul2d
  int gemm_bm32 = 0;       // dense tiled, 32x32 (M < 128, or the fallback)
  int gemm_bm64 = 0;       // dense tiled, 64x32
  int gemm_bm64bn64 = 0;   // dense tiled, 64x64
  int gemm_qmm = 0;        // affine quantized steel
  // Attention, one dispatch per block.
  int attn_steel = 0;      // flash, simdgroup MMA (no matrix cores)
  int attn_nax = 0;        // flash, matrix cores
  int attn_sage = 0;       // of those, int8 QK (SageAttention)
  int attn_sol = 0;        // Sol-Attn routed
  // What Sol's routing KEPT exact, of the (query block, key block) pairs
  // it chose from, summed over every routed dispatch of the forward --
  // read back after the forward's last wait. A property of the data.
  long long sol_kept = 0;
  long long sol_total = 0;
  // The ANE feed-forward (`ane_ffn`), main-stack blocks only.
  int ane_split = 0;       // blocks whose feed-forward rows were split
  int ane_probe = 0;       // blocks the balancer ran GPU-only to measure
  long long ane_rows = 0;  // feed-forward rows the ANE computed
  int ane_lost = 0;        // splits that lost rows, recomputed on the GPU
  // Dense attentions whose length is NOT a multiple of the kernel's
  // query / key tile -- the remainder path the padded in-tree sequence
  // never took on a box without matrix cores.
  int attn_unaligned_q = 0;
  int attn_unaligned_k = 0;
  // Blocks of the modulated stack run from a streaming slot vs held.
  int blocks_streamed = 0;
  int blocks_held = 0;
  int prefetch_hits = 0;
  // Runtime LoRA, counted where the adapter GEMMs are ENCODED: the
  // projections that carried at least one live adapter (after whichever
  // base route ran), and the split blocks whose ANE staging carried the
  // adapter's factors for the ANE's rows.
  int lora_proj = 0;
  int ane_lora = 0;
  // The caption path (cap_embedder + context_refiner) was served from
  // the per-caption cache instead of being recomputed.
  bool cap_cache_hit = false;
  // The GPU arithmetic this forward handed the host's FLOP/s meter
  // (GEMMs at 2*M*N*K, attention at the pairs it attended), credited as
  // its buffers retire. 0 on a host without the meter.
  std::uint64_t gpu_flops = 0;
};

class MingDit {
public:
  virtual ~MingDit() = default;

  // ---- runtime LoRA ---------------------------------------------------
  struct LoraSpec {
    std::string path;          // one .safetensors (a resolved FILE)
    float       scale = 1.0f;  // 1.0 = as trained; 0 = exactly off
  };
  // Two adapters at once -- a distillation or identity adapter beside a
  // style one -- in separate slots, so each strength stays a live knob
  // rather than being folded into one merged set of factors.
  static constexpr int kMaxLoraSlots = vpipe::genai::lora::Stack::kMax;

  // What a load takes beyond the weights. Defaults are a plain load.
  struct LoadOptions {
    // Bound at load: the factors are the adapter's and stay RESIDENT
    // (a streamed block gets them from here, not from its slot); only
    // the STRENGTH is live (set_lora_scale).
    std::vector<LoraSpec> loras;
    // The label the HOST booked the ANE module's CoreML claim under
    // (ImageModelCreateArgs::borrowed_extra[create_args::kCoreMLLabel]).
    // Set: the host already read the grant and turned `ane_ffn` off in
    // the bag when it was refused, so the bag is TRUSTED and this label
    // is what the held bytes are revised under. Empty: an older host or
    // no stage at all -- this DiT reads the grant itself under
    // kAneClaimLabel, as the family booked it.
    std::string coreml_label;
  };

  // `accel` is the request's acceleration bag (i8_gemm / sage_attn /
  // sol_attn / ane_ffn ...), read at load; null reads as every default.
  // `ane_ffn` also needs `session` (the tier's worker is the session's)
  // and the plan's grant (see LoadOptions::coreml_label). A
  // block-streaming load holds the trunk and two slots.
  static std::unique_ptr<MingDit>
  load(std::shared_ptr<vpipe::genai::WeightSet> ws,
       vpipe::metal_compute::MetalCompute*      mc,
       const DitConfig&                         cfg,
       const vpipe::FlexData*                   accel,
       bool                                     stream_blocks,
       const vpipe::SessionContextIntf*         session,
       std::string*                             err,
       const LoadOptions&                       opts = LoadOptions());

  // LIVE. The strength rides the accumulating GEMM as a constant, so it
  // can be turned between forwards without rebuilding any factors;
  // exactly 0 encodes nothing at all, so "off" is off rather than a pair
  // of roundings that cancel. A change drops the caption cache: the
  // context refiner's projections carry the adapter too.
  virtual void set_lora_scale(int slot, float s) = 0;
  virtual float lora_scale(int slot) const = 0;
  // How many projections each slot actually BOUND. 0 is the answer a
  // caller has to refuse to call success: an adapter for another model
  // binds nothing and otherwise reports success by saying nothing.
  virtual int lora_modules(int slot) const = 0;
  // Modules the file carries at the right NAME but the wrong SHAPE --
  // an adapter trained against a different model of this lineage. Any at
  // all and the WHOLE slot is refused (lora_modules 0): the modules that
  // happen to fit are half an adapter, not a weaker one.
  virtual int lora_skipped(int slot) const = 0;
  // The projections an adapter may bind per block, and so the most a
  // full adapter binds: 7 x (context + noise refiner + main blocks).
  static int lora_module_capacity(const DitConfig& cfg);

  struct Request {
    // bf16 [layout.img_tokens, patch_dim] packed latent rows
    // (ming-layout.h pack_latents order).
    const vpipe::metal_compute::SharedBuffer* latents = nullptr;
    // bf16 [layout.cap_len, cap_stride], the conditioning contract
    // (ming-family.h): rows [0, query_rows) carry stream A in columns
    // [0, cap_feat_dim); rows [query_rows, cap_len) carry stream B,
    // `dim` wide.
    const vpipe::metal_compute::SharedBuffer* cap = nullptr;
    int cap_stride = 0;
    const Layout* layout = nullptr;
    // The SCHEDULER's sigma. The DiT is fed t_scale * (1 - sigma).
    float sigma = 0.0f;
  };

  // One forward. Returns bf16 [img_tokens, patch_dim], the model output
  // BEFORE the reference's negation. Empty on failure (with `err` set)
  // or on a cooperative stop (with `err` empty).
  virtual vpipe::metal_compute::SharedBuffer
  forward(const Request& req, std::string* err) = 0;

  // Cooperative stop, polled between blocks; true stops the forward.
  virtual void set_stream_stop(std::function<bool()> stop) = 0;
  virtual void set_block_progress(vpipe::genai::DitBlockProgressFn fn) = 0;

  // Residency for a streaming load (see the host's BlockResidency):
  // nothing is admitted until a reserve is DECLARED.
  virtual void set_residency_reserve(std::size_t bytes) = 0;
  virtual void set_residency_schedule(int steps) = 0;
  virtual std::size_t release_resident_blocks(std::size_t bytes) = 0;

  virtual std::uint64_t resident_bytes() const = 0;
  virtual bool streaming() const = 0;
  virtual const DitConfig& config() const = 0;

  // ---- engagement: what the LOAD armed, and what a forward RAN ------
  virtual bool uses_mma2() const = 0;       // matmul2d GEMMs armed
  virtual bool uses_attn_nax() const = 0;   // matrix-core attention armed
  virtual int  gemm_tile() const = 0;       // 0 bm32, 1 bm64, 2 bm64bn64
  virtual bool uses_i8_gemm() const = 0;
  virtual bool uses_sage_attn() const = 0;
  virtual bool uses_sol_attn() const = 0;
  // The ANE feed-forward: REQUESTED and granted at load; ARMED once the
  // first forward long enough for one chunk built the module; GPU-ONLY
  // once a split lost rows (the tier then keeps the GPU for its life).
  virtual bool ane_requested() const = 0;
  virtual bool ane_armed() const = 0;
  virtual bool ane_gpu_only() const = 0;
  virtual int  quant_bits() const = 0;      // 0 = dense checkpoint
  virtual int  resident_block_count() const = 0;
  virtual DitEngagement last_engagement() const = 0;
  // The per-caption cache (on by default; VPIPE_MING_NO_CAP_CACHE turns
  // it off at load). A test drives both arms in one process.
  virtual void set_caption_cache(bool on) = 0;

  // Trunk plus a slot pair: what a streaming load holds.
  static std::size_t streaming_floor_bytes(const std::string& dir);

  // What the ANE feed-forward module holds (CoreML's, invisible to every
  // other ledger): ONE runtime-weight module whose weights are staged per
  // block, so a fixed unit however many blocks use it and whatever the
  // sequence. 0 when this host cannot build the module.
  static std::size_t ane_runtime_bytes(const DitConfig& cfg);
};

}  // namespace ming

#endif
