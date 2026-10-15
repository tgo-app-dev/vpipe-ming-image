#ifndef VPIPE_MING_LLM_H
#define VPIPE_MING_LLM_H

// Ming-Image's conditioning MLLM: the BailingMoeV2 language model of
// BailingMM2, run as a ONE-PASS causal prefill encoder over the template
// rows and the 256 learned query rows (docs/MING-IMAGE.md 1.5-1.6). No
// decoding, no KV cache, no lm_head, no vision tower.
//
//   20 layers, hidden 2048, 16 q / 4 kv heads of 128, fused
//   query_key_value split by head [16 q | 4 k | 4 v], per-head q/k
//   RMSNorm ALWAYS, partial rotary (dims [0, 64), rotate-half) with the
//   3-axis "video_rope", causal attention over every row.
//   Layer 0: dense SwiGLU (5120). Layers 1-19: 256 experts top-8 with a
//   sigmoid router -- expert_bias for SELECTION only, group-limited (8
//   groups of 32, a group scores its top-2, the top 4 groups survive),
//   weights renormalised and scaled by 2.5 -- one router for text rows
//   (`gate`) and one for the query rows (`image_gate`), plus an
//   UN-GATED shared expert.
//
// HOW THE MoE RUNS. A prompt is ~300-1700 rows and 8 picks each, which
// touches nearly every expert of every layer, so the expert weights are
// read once per layer however they are scheduled. Routing runs on the
// CPU in f32 (Accelerate), exactly as the reference does it (its router
// is an f32 Linear over the bf16 normed row); the rows are then gathered
// into expert order and every expert's GEMMs run as ONE grouped
// dispatch each (gate+up with the SwiGLU fused, then down), followed by
// an f32 weighted combine into the residual. One CPU sync per MoE layer
// is the price of CPU routing: ~1-2 ms of a layer the GPU spends far
// longer on.
//
// WEIGHTS go through the WeightSet. What the model keeps as the
// checkpoint stores it (q/k/v, the output projection, norms, the
// embedding table) is `tensor(..., Copied)`; what it keeps TRANSFORMED
// (the 256 per-expert tensors stacked into one [E, N, K] slab per
// projection, the interleaved shared-expert gate|up, the f32 routers)
// is `derived()`, built from uncached reads.
//
// STREAMING (Options::stream_layers). Layer 0 and the trunk stay
// resident; MoE layers 1-19 -- identical in shape -- are pread into one
// of two slots a layer at a time, the next layer's read issued as soon
// as the GPU has finished with the slot it will land in. Same kernels,
// same bytes: a streamed forward is bit-identical to a resident one.

#include "ming-ops.h"

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace vpipe::genai { class WeightSet; }

namespace ming {

// mllm/config.json `llm_config`, plus the few facts the forward
// hard-codes and therefore REFUSES to run without (head_dim 128,
// partial rotary of half the head, video_rope).
struct LlmConfig {
  int   hidden = 2048;
  int   layers = 20;
  int   heads = 16;
  int   kv_heads = 4;
  int   head_dim = 128;
  int   inter = 5120;             // the dense layers' SwiGLU
  int   moe_inter = 512;
  int   experts = 256;
  int   top_k = 8;
  int   n_group = 8;
  int   topk_group = 4;
  int   first_dense = 1;          // first_k_dense_replace
  int   shared_experts = 1;
  int   vocab = 157184;
  float eps = 1e-6f;
  float rope_theta = 600000.0f;
  float routed_scale = 2.5f;
  int   image_patch = 157157;
  int   image_start = 157158;
  int   image_end = 157159;

  int rot_half() const { return head_dim / 4; }   // 64 rotary dims
  int qkv_rows() const { return (heads + 2 * kv_heads) * head_dim; }
  bool is_moe(int layer) const { return layer >= first_dense; }
};

// Read and validate; false with `err` naming what this port cannot run.
bool read_llm_config(const std::string& mllm_dir, LlmConfig* cfg,
                     std::string* err);

class MingLlm {
public:
  ~MingLlm();

  static std::unique_ptr<MingLlm>
  load(std::shared_ptr<vpipe::genai::WeightSet> ws, LlmOps* ops,
       const LlmConfig& cfg, bool stream_layers, std::string* err);

  struct Input {
    std::vector<std::int32_t> ids;  // all L rows, appended tokens included
    int n_text = 0;                 // N: the template's rows
    int query_row0 = 0;             // first <imagePatch> row (N + 1)
    int query_rows = 0;             // 256
    // bf16 [query_rows, hidden]: replaces those rows' embeddings.
    const SharedBuffer* query_embeds = nullptr;
    // HF hidden_states indices to concatenate for stream B, in order:
    // k < layers is the residual AFTER layer k-1 (un-normed), k ==
    // layers is the FINAL-NORMED output.
    std::vector<int> taps;
  };

  // Test instrumentation: f32 copies of every row of the requested
  // taps, and the experts each row was routed to per MoE layer.
  //
  // TEACHER FORCING, for a per-layer bar free of upstream drift: with
  // `tf_hs` set, layer i starts from tf_hs[i] (f32 [layers + 1, L,
  // hidden], the reference's hidden_states) instead of its predecessor's
  // output, and every layer's raw output is returned in `layer_out`.
  // With `tf_idx` / `tf_w` too ([moe layers, L, top_k]) the MoE routing
  // is FORCED to the reference's, which isolates the arithmetic from the
  // router's near-ties.
  struct Debug {
    std::vector<int> tap_layers;                 // HF indices wanted
    std::vector<std::vector<float>> taps;        // [L, hidden] each
    std::vector<std::vector<std::int32_t>> experts;  // [L * top_k] / layer
    const float* tf_hs = nullptr;
    const std::int32_t* tf_idx = nullptr;
    const float* tf_w = nullptr;
    std::vector<std::vector<float>> layer_out;   // [L, hidden] / layer
    // Profile: sync after every command buffer and split the time. The
    // extra syncs cost a little, so this is off unless asked for.
    bool profile = false;
    double gpu_attn_ms = 0, gpu_moe_ms = 0, cpu_route_ms = 0,
           cpu_embed_ms = 0, wait_slot_ms = 0;
  };

  struct Output {
    SharedBuffer final;    // bf16 [L, hidden], the final-normed rows
    SharedBuffer concat;   // bf16 [n_text, hidden * taps], stream B input
  };

  // The whole prefill. Waits for the GPU before returning.
  bool forward(const Input& in, Output* out, Debug* dbg, std::string* err);

  // What this instance holds in weights (resident: everything; streamed:
  // the trunk and the two slots).
  std::size_t weight_bytes() const;
  bool streaming() const { return _stream; }

  // What a resident load of `cfg` holds, and the least a streaming one
  // does -- the same arithmetic the loader allocates by.
  static std::size_t resident_bytes_for(const LlmConfig& cfg);
  static std::size_t streaming_floor_for(const LlmConfig& cfg);

  struct Layer;

private:
  MingLlm() = default;
  bool load_trunk_(std::string* err);
  bool fill_layer_(int i, Layer& l, bool streamed, std::string* err);
  bool alloc_layer_(Layer& l, bool moe);
  bool ensure_scratch_(int L, int n_text, int n_taps);
  void route_(int layer, int L, int query_row0, int query_rows,
              Debug* dbg);

  std::shared_ptr<vpipe::genai::WeightSet> _ws;
  LlmOps*   _ops = nullptr;
  LlmConfig _cfg;
  bool      _stream = false;

  SharedBuffer _embed;        // bf16 [vocab, hidden]
  SharedBuffer _final_norm;
  std::vector<std::unique_ptr<Layer>> _layers;   // resident (or layer 0)
  std::unique_ptr<Layer> _slot[2];                // streaming
  // f32 routers, host side: [experts, hidden] each, per layer.
  std::vector<std::vector<float>> _gate, _gate_bias, _img_gate, _img_bias;

  // Per-encode scratch, grown to the longest prompt seen.
  int _cap_rows = 0, _cap_text = 0, _cap_taps = 0;
  SharedBuffer _h, _xn, _xn32, _qkv, _attn, _sh_act, _dtiles, _d_act,
      _xs, _act, _eout, _perm, _pair, _wt, _tiles, _cos, _sin,
      _final, _concat;
  int _n_tiles = 0;
  std::vector<float> _logits;
  AttnPlan _plan;
};

}  // namespace ming

#endif
