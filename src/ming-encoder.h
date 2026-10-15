#ifndef VPIPE_MING_ENCODER_H
#define VPIPE_MING_ENCODER_H

// Ming-Image's text conditioning, prompt in, the DiT's caption out
// (docs/MING-IMAGE.md section 1):
//
//   prompt --(BailingMM2 template + Ling tokenizer)--> N ids
//          + [<image>] + 256 learned query rows + [</image>]
//   BailingMoeV2 (20 layers, causal, 3-axis "video_rope", MoE with a
//   text router and an image router)
//     stream B: hs[5] | hs[12] | hs[20] at the N text rows
//               -> RMSNorm(6144) -> Linear(6144 -> 3840)    [N, 3840]
//     stream A: hs[20] at the 256 query rows -> Linear(2048 -> 1536)
//               -> Qwen2-1.5B connector, BIDIRECTIONAL, final norm
//               -> Linear(1536 -> 2560)                     [256, 2560]
//
// and packs both into the conditioning contract ming-family.h states:
// bf16 [256 + N, 3840], stream A in columns [0, 2560) of its rows.
//
// No vision tower runs for text-to-image, the MLLM's lm_head is never
// read, and the prompt is never truncated.

#include "ming-config.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
class SessionContextIntf;
namespace metal_compute { class MetalCompute; }
}  // namespace vpipe

namespace ming {

class MingEncoder {
public:
  virtual ~MingEncoder() = default;

  struct Options {
    // Stream the MLLM's layers from disk instead of holding them (its
    // bf16 body is ~32 GB). A prompt is ONE prefill pass, so a streamed
    // load reads each layer once per prompt.
    bool stream_layers = false;
  };

  static std::unique_ptr<MingEncoder>
  load(const Checkpoint& ck, vpipe::metal_compute::MetalCompute* mc,
       const vpipe::SessionContextIntf* session, const Options& opt,
       std::string* err);

  struct Result {
    std::vector<std::uint16_t> cond;   // bf16 [rows, dim]
    int rows = 0;
    int dim = 0;
    int query_rows = 0;                // stream A
    int text_rows = 0;                 // stream B (= the template's N)
  };

  // The template, tokenization and every model above, for one prompt.
  // False with `err` on failure; never throws.
  virtual bool encode(const std::string& prompt, Result* out,
                      std::string* err) = 0;

  // The exact token ids the template + tokenizer produce for `prompt`
  // (the N text ids, without the appended generation tokens).
  virtual std::vector<std::int32_t>
  template_ids(const std::string& prompt) const = 0;

  virtual std::uint64_t resident_bytes() const = 0;

  // What a resident load holds, and the least a streaming one holds.
  //
  // Both are the MLLM's (mllm/) part -- the only component that
  // streams, and the figure a caller claims mllm/ at. The connector and
  // the heads are always held whole; connector_weight_bytes() is what
  // they hold, which is NOT their size on disk: both are f32 there and
  // bf16 here, so a claim taken from the directory over-states them by
  // ~2.4x. Every figure is the loader's own allocation arithmetic.
  static std::size_t resident_weight_bytes(const Checkpoint& ck);
  static std::size_t streaming_floor_bytes(const Checkpoint& ck);
  static std::size_t connector_weight_bytes(const Checkpoint& ck);

  // ---- test instrumentation ----------------------------------------
  //
  // What the reference goldens dump, as f32, alongside the ordinary
  // result. `tap_layers` names the HF hidden_states wanted (k < 20 is
  // the residual after layer k-1, 20 the final-normed output), every
  // row of each.
  struct Debug {
    std::vector<int> tap_layers{5, 12, 20};
    std::vector<std::int32_t> ids;                     // all L rows
    std::vector<std::vector<float>> taps;              // [L, 2048] each
    std::vector<std::vector<std::int32_t>> experts;    // [L * 8] / layer
    std::vector<float> cap_feats_2;     // [N, 3840]
    std::vector<float> connector_in;    // [256, 1536]
    std::vector<float> connector_out;   // [256, 1536]
    std::vector<float> cap_feats;       // [256, 2560]
    // Teacher forcing (see MingLlm::Debug): the reference's hidden
    // states as every layer's input, optionally its routing too, and
    // each layer's output back.
    const float* tf_hs = nullptr;           // [21, L, 2048]
    const std::int32_t* tf_idx = nullptr;   // [19, L, 8]
    const float* tf_w = nullptr;            // [19, L, 8]
    std::vector<std::vector<float>> layer_out;
    // Per-phase timing of the MLLM (see MingLlm::Debug::profile).
    bool profile = false;
    double gpu_attn_ms = 0, gpu_moe_ms = 0, cpu_route_ms = 0,
           cpu_embed_ms = 0, wait_slot_ms = 0, heads_ms = 0;
  };
  virtual bool encode_debug(const std::string& prompt, Result* out,
                            Debug* dbg, std::string* err) = 0;
};

}  // namespace ming

#endif
