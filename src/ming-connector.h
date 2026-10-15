#ifndef VPIPE_MING_CONNECTOR_H
#define VPIPE_MING_CONNECTOR_H

// The connector between Ming-Image's MLLM and its DiT: a Qwen2-1.5B
// stack (connector/, docs/MING-IMAGE.md 1.8) run over the 256 query
// rows as a BIDIRECTIONAL encoder -- the reference sets is_causal False
// on every layer and passes an all-ones float mask, which softmax
// cancels, so this is full attention with no mask at all.
//
//   28 layers, hidden 1536, 12 q / 2 kv heads of 128, q/k/v BIAS (no
//   o bias), no q/k norm, SwiGLU 8960, RMSNorm 1e-6, full-head
//   rotate-half RoPE (theta 1e6) at positions 0..S-1, final norm.
//
// Its checkpoint is F32; the reference loads it as bf16, so the weights
// are narrowed once at load into `derived()` products (a fused q|k|v
// with its fused bias, an interleaved gate|up) -- the f32 pieces are
// consumed and dropped. embed_tokens and the tied lm_head are never
// read.

#include "ming-ops.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace vpipe::genai { class WeightSet; }

namespace ming {

struct ConnectorConfig {
  int   hidden = 1536;
  int   layers = 28;
  int   heads = 12;
  int   kv_heads = 2;
  int   head_dim = 128;
  int   inter = 8960;
  float eps = 1e-6f;
  float rope_theta = 1000000.0f;

  int qkv_rows() const { return (heads + 2 * kv_heads) * head_dim; }
};

bool read_connector_config(const std::string& dir, ConnectorConfig* cfg,
                           std::string* err);

class MingConnector {
public:
  ~MingConnector();

  static std::unique_ptr<MingConnector>
  load(std::shared_ptr<vpipe::genai::WeightSet> ws, LlmOps* ops,
       const ConnectorConfig& cfg, std::string* err);

  // Encode the stack into `enc`: `in` is bf16 [S, hidden] (proj_in's
  // output), `out` bf16 [S, hidden] the final-normed rows. The caller
  // commits; nothing here waits.
  bool encode(ComputeEncoder& enc, const SharedBuffer& in,
              const SharedBuffer& out, int S, std::string* err);

  std::size_t weight_bytes() const;
  // What a load of `cfg` holds (bf16).
  static std::size_t bytes_for(const ConnectorConfig& cfg);

private:
  MingConnector() = default;
  bool ensure_scratch_(int S);

  struct Layer {
    SharedBuffer in_norm, post_norm, qkv, qkv_bias, o, gu, down;
  };

  std::shared_ptr<vpipe::genai::WeightSet> _ws;
  LlmOps*         _ops = nullptr;
  ConnectorConfig _cfg;
  std::vector<Layer> _layers;
  SharedBuffer    _norm;

  int _cap = 0;
  SharedBuffer _x, _xn, _qkv, _attn, _o, _act, _cos, _sin;
  AttnPlan _plan;
};

}  // namespace ming

#endif
