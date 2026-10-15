#include "ming-llm.h"

#include "common/flex-data.h"
#include "generative-models/weight-set.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace ming {

using vpipe::genai::WeightSet;
using Residency = vpipe::genai::WeightSet::Residency;

struct MingLlm::Layer {
  SharedBuffer in_norm, post_norm, q_norm, k_norm, qkv, dense;
  // Dense layers: the SwiGLU's gate|up INTERLEAVED (row 2g = gate g,
  // row 2g+1 = up g), as the host's fused SwiGLU GEMM reads it.
  SharedBuffer mlp_gu, mlp_down;
  // MoE layers: every expert's projection stacked into one slab,
  // [E, moe_inter, hidden] x2 and [E, hidden, moe_inter], plus the
  // shared expert (interleaved like a dense layer's).
  SharedBuffer wg, wu, wd, sh_gu, sh_down;
  bool moe = false;

  std::size_t bytes() const
  {
    std::size_t n = 0;
    for (const SharedBuffer* b :
         {&in_norm, &post_norm, &q_norm, &k_norm, &qkv, &dense, &mlp_gu,
          &mlp_down, &wg, &wu, &wd, &sh_gu, &sh_down}) {
      n += b->byte_size();
    }
    return n;
  }
};

namespace {

std::string
lname(int i, const char* rest)
{
  return "model.model.layers." + std::to_string(i) + "." + rest;
}

std::string
ename(int i, int e, const char* proj)
{
  return "model.model.layers." + std::to_string(i) + ".mlp.experts." +
         std::to_string(e) + "." + proj + ".weight";
}

int
get_int(const vpipe::FlexData& o, const char* k, int d)
{
  if (!o.is_object() || !o.as_object().contains(k)) { return d; }
  return (int)o.as_object().at(k).as_int(d);
}

float
get_real(const vpipe::FlexData& o, const char* k, float d)
{
  if (!o.is_object() || !o.as_object().contains(k)) { return d; }
  const vpipe::FlexData v = o.as_object().at(k);
  return v.is_null() ? d : (float)v.as_real(d);
}

// Rows of two [R, K] bf16 matrices interleaved into [2R, K].
void
interleave_rows(const void* gate, const void* up, void* dst, std::size_t R,
                std::size_t K)
{
  const std::size_t row = K * 2;
  auto* d = static_cast<char*>(dst);
  const auto* g = static_cast<const char*>(gate);
  const auto* u = static_cast<const char*>(up);
  for (std::size_t r = 0; r < R; ++r) {
    std::memcpy(d + (2 * r) * row, g + r * row, row);
    std::memcpy(d + (2 * r + 1) * row, u + r * row, row);
  }
}

// The video_rope axis of rotary pair k (docs/MING-IMAGE.md 1.5):
// mrope_section [8, 12, 12] becomes [24, 8, 24, 8] over the 64 rotary
// dims, where a 24-run alternates H (even) / W (odd) and an 8-run is T.
// Pair (k, k + 32) therefore reads ONE axis: 0 = T, 1 = H, 2 = W.
int
video_rope_axis(int k)
{
  if (k >= 24) { return 0; }
  return (k % 2 == 0) ? 1 : 2;
}

}  // namespace

bool
read_llm_config(const std::string& mllm_dir, LlmConfig* cfg,
                std::string* err)
{
  auto fail = [err](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  std::ifstream in(mllm_dir + "/config.json");
  if (!in) { return fail("no " + mllm_dir + "/config.json"); }
  vpipe::FlexData root;
  try {
    root = vpipe::FlexData::from_json(in);
  } catch (const std::exception& e) {
    return fail(std::string("mllm/config.json: ") + e.what());
  }
  if (!root.is_object() || !root.as_object().contains("llm_config")) {
    return fail("mllm/config.json has no llm_config");
  }
  const vpipe::FlexData c = root.as_object().at("llm_config");
  LlmConfig& g = *cfg;
  g.hidden       = get_int(c, "hidden_size", g.hidden);
  g.layers       = get_int(c, "num_hidden_layers", g.layers);
  g.heads        = get_int(c, "num_attention_heads", g.heads);
  g.kv_heads     = get_int(c, "num_key_value_heads", g.kv_heads);
  g.head_dim     = get_int(c, "head_dim", g.head_dim);
  g.inter        = get_int(c, "intermediate_size", g.inter);
  g.moe_inter    = get_int(c, "moe_intermediate_size", g.moe_inter);
  g.experts      = get_int(c, "num_experts", g.experts);
  g.top_k        = get_int(c, "num_experts_per_tok", g.top_k);
  g.n_group      = get_int(c, "n_group", g.n_group);
  g.topk_group   = get_int(c, "topk_group", g.topk_group);
  g.first_dense  = get_int(c, "first_k_dense_replace", g.first_dense);
  g.shared_experts = get_int(c, "num_shared_experts", g.shared_experts);
  g.vocab        = get_int(c, "vocab_size", g.vocab);
  g.eps          = get_real(c, "rms_norm_eps", g.eps);
  g.rope_theta   = get_real(c, "rope_theta", g.rope_theta);
  g.routed_scale = get_real(c, "routed_scaling_factor", g.routed_scale);
  g.image_patch  = get_int(c, "image_patch_token", g.image_patch);
  g.image_start  = get_int(c, "image_start_token", g.image_start);
  g.image_end    = get_int(c, "image_end_token", g.image_end);
  const float partial = get_real(c, "partial_rotary_factor", 1.0f);
  std::string rope_type;
  if (c.as_object().contains("rope_scaling")) {
    const vpipe::FlexData rs = c.as_object().at("rope_scaling");
    if (rs.is_object() && rs.as_object().contains("type")) {
      rope_type = std::string(rs.as_object().at("type").as_string(""));
    }
  }
  std::string router;
  if (c.as_object().contains("router_type")) {
    router = std::string(c.as_object().at("router_type").as_string(""));
  }
  // What the kernels and the routing below hard-code. A checkpoint that
  // differs is refused by name rather than run with the wrong math.
  if (g.head_dim != 128 || partial != 0.5f || rope_type != "video_rope") {
    return fail("this port runs head_dim 128 with half-head video_rope "
                "only (got head_dim " + std::to_string(g.head_dim) +
                ", rope '" + rope_type + "')");
  }
  if (router != "MultiRouter" || g.experts % g.n_group != 0 ||
      g.top_k > g.experts || g.topk_group > g.n_group ||
      g.experts / g.n_group < 2 || g.shared_experts != 1) {
    return fail("unexpected MoE routing configuration");
  }
  // The grouped GEMMs tile N and K by 64 (ming_llm.metal).
  if (g.hidden != g.heads * g.head_dim || g.heads % g.kv_heads != 0 ||
      g.hidden % 64 != 0 || g.moe_inter % 64 != 0 || g.inter % 64 != 0) {
    return fail("unexpected attention / MoE geometry");
  }
  return true;
}

MingLlm::~MingLlm()
{
  // See kEncoderPart: the stacked slabs must leave with the model.
  if (_ws) { _ws->release_part(kEncoderPart); }
}

std::size_t
MingLlm::resident_bytes_for(const LlmConfig& c)
{
  const std::size_t H = (std::size_t)c.hidden;
  const std::size_t attn =
      ((std::size_t)c.qkv_rows() * H + H * H + 2 * H +
       2 * (std::size_t)c.head_dim) * 2;
  const std::size_t dense = 3 * (std::size_t)c.inter * H * 2;
  const std::size_t moe_gpu =
      3 * (std::size_t)c.experts * c.moe_inter * H * 2 +
      3 * (std::size_t)c.moe_inter * c.shared_experts * H * 2;
  const std::size_t routers = 2 * ((std::size_t)c.experts * H + c.experts) * 4;
  std::size_t n = (std::size_t)c.vocab * H * 2 + H * 2;
  for (int i = 0; i < c.layers; ++i) {
    n += attn + (c.is_moe(i) ? moe_gpu + routers : dense);
  }
  return n;
}

std::size_t
MingLlm::streaming_floor_for(const LlmConfig& c)
{
  const std::size_t H = (std::size_t)c.hidden;
  const std::size_t attn =
      ((std::size_t)c.qkv_rows() * H + H * H + 2 * H +
       2 * (std::size_t)c.head_dim) * 2;
  const std::size_t dense = 3 * (std::size_t)c.inter * H * 2;
  const std::size_t moe_gpu =
      3 * (std::size_t)c.experts * c.moe_inter * H * 2 +
      3 * (std::size_t)c.moe_inter * c.shared_experts * H * 2;
  const std::size_t routers = 2 * ((std::size_t)c.experts * H + c.experts) * 4;
  // The trunk -- embeddings, final norm, the dense layers whole, every
  // MoE layer's f32 routers (the CPU reads them) -- and two slots.
  std::size_t n = (std::size_t)c.vocab * H * 2 + H * 2;
  for (int i = 0; i < c.layers; ++i) {
    n += c.is_moe(i) ? routers : attn + dense;
  }
  return n + 2 * (attn + moe_gpu);
}

std::unique_ptr<MingLlm>
MingLlm::load(std::shared_ptr<WeightSet> ws, LlmOps* ops,
              const LlmConfig& cfg, bool stream_layers, std::string* err)
{
  if (!ws || ops == nullptr) {
    if (err != nullptr) { *err = "no weight set / ops"; }
    return nullptr;
  }
  auto m = std::unique_ptr<MingLlm>(new MingLlm());
  m->_ws = std::move(ws);
  m->_ops = ops;
  m->_cfg = cfg;
  m->_stream = stream_layers;
  if (!m->load_trunk_(err)) { return nullptr; }
  return m;
}

bool
MingLlm::alloc_layer_(Layer& l, bool moe)
{
  const std::size_t H = (std::size_t)_cfg.hidden;
  const LlmOps& o = *_ops;
  l.moe = moe;
  l.in_norm = o.alloc(H * 2);
  l.post_norm = o.alloc(H * 2);
  l.q_norm = o.alloc((std::size_t)_cfg.head_dim * 2);
  l.k_norm = o.alloc((std::size_t)_cfg.head_dim * 2);
  l.qkv = o.alloc((std::size_t)_cfg.qkv_rows() * H * 2);
  l.dense = o.alloc(H * H * 2);
  if (moe) {
    const std::size_t slab = (std::size_t)_cfg.experts * _cfg.moe_inter * H * 2;
    const std::size_t sh = (std::size_t)_cfg.moe_inter * H * 2;
    l.wg = o.alloc(slab);
    l.wu = o.alloc(slab);
    l.wd = o.alloc(slab);
    l.sh_gu = o.alloc(2 * sh);
    l.sh_down = o.alloc(sh);
    return !l.wg.empty() && !l.wu.empty() && !l.wd.empty() &&
           !l.sh_gu.empty() && !l.sh_down.empty() && !l.qkv.empty();
  }
  return !l.qkv.empty() && !l.dense.empty();
}

bool
MingLlm::fill_layer_(int i, Layer& l, bool streamed, std::string* err)
{
  WeightSet& ws = *_ws;
  auto* mc = _ops->mc();
  const std::size_t H = (std::size_t)_cfg.hidden;
  const bool moe = _cfg.is_moe(i);
  auto fail = [err](const std::string& what) {
    if (err != nullptr) { *err = "MLLM weight '" + what + "' failed to load"; }
    return false;
  };
  // An as-is tensor: cached and parkable when resident; pread into the
  // slot's own buffer when streamed.
  auto as_is = [&](const std::string& name, SharedBuffer& dst) {
    if (streamed) {
      return ws.stream_into(name, dst.contents(), dst.byte_size());
    }
    dst = ws.tensor(name, mc, Residency::Copied);
    return !dst.empty();
  };
  l.moe = moe;
  const struct { const char* rest; SharedBuffer* dst; } plain[] = {
      {"input_layernorm.weight", &l.in_norm},
      {"post_attention_layernorm.weight", &l.post_norm},
      {"attention.q_norm.weight", &l.q_norm},
      {"attention.k_norm.weight", &l.k_norm},
      {"attention.query_key_value.weight", &l.qkv},
      {"attention.dense.weight", &l.dense},
  };
  for (const auto& p : plain) {
    if (!as_is(lname(i, p.rest), *p.dst)) { return fail(lname(i, p.rest)); }
  }

  // gate|up interleaved for the fused SwiGLU GEMM, from two uncached
  // reads (the pieces are consumed; only the product is kept).
  auto interleaved = [&](const std::string& gname, const std::string& uname,
                         std::size_t R, SharedBuffer& dst) -> bool {
    const std::size_t piece = R * H * 2;
    if (streamed) {
      std::vector<std::uint8_t> g(piece), u(piece);
      if (!ws.stream_into(gname, g.data(), piece) ||
          !ws.stream_into(uname, u.data(), piece)) {
        return false;
      }
      interleave_rows(g.data(), u.data(), dst.contents(), R, H);
      return true;
    }
    dst = ws.derived("ming-llm/" + gname + "+up/interleaved-bf16", [&] {
      SharedBuffer out = _ops->alloc(2 * piece);
      SharedBuffer g = ws.read(gname, mc, Residency::Copied);
      SharedBuffer u = ws.read(uname, mc, Residency::Copied);
      if (out.empty() || g.byte_size() != piece || u.byte_size() != piece) {
        return SharedBuffer{};
      }
      interleave_rows(g.contents(), u.contents(), out.contents(), R, H);
      return out;
    }, kEncoderPart);
    return !dst.empty();
  };

  if (!moe) {
    const std::size_t I = (std::size_t)_cfg.inter;
    if (!interleaved(lname(i, "mlp.gate_proj.weight"),
                     lname(i, "mlp.up_proj.weight"), I, l.mlp_gu)) {
      return fail(lname(i, "mlp.gate_proj/up_proj"));
    }
    if (!as_is(lname(i, "mlp.down_proj.weight"), l.mlp_down)) {
      return fail(lname(i, "mlp.down_proj.weight"));
    }
    return true;
  }

  const std::size_t I = (std::size_t)_cfg.moe_inter;
  if (!interleaved(lname(i, "mlp.shared_experts.gate_proj.weight"),
                   lname(i, "mlp.shared_experts.up_proj.weight"),
                   I * (std::size_t)_cfg.shared_experts, l.sh_gu)) {
    return fail(lname(i, "mlp.shared_experts.gate_proj/up_proj"));
  }
  if (!as_is(lname(i, "mlp.shared_experts.down_proj.weight"), l.sh_down)) {
    return fail(lname(i, "mlp.shared_experts.down_proj.weight"));
  }
  // The 256 experts' tensors stacked into one slab per projection. Each
  // expert's piece lands at its offset straight from the file (pread):
  // nothing is staged, and the slab is the only copy held.
  const std::size_t piece = I * H * 2;
  auto stack = [&](const char* proj, SharedBuffer& dst) -> bool {
    auto fill = [&](void* base) {
      for (int e = 0; e < _cfg.experts; ++e) {
        auto* p = static_cast<char*>(base) + (std::size_t)e * piece;
        const std::string n = ename(i, e, proj);
        const bool ok = streamed ? ws.stream_into(n, p, piece)
                                 : ws.src().pread_into(n, p, piece);
        if (!ok) { return false; }
      }
      return true;
    };
    if (streamed) { return fill(dst.contents()); }
    dst = ws.derived("ming-llm/" + lname(i, "mlp.experts.") + proj +
                         "/stack-bf16", [&] {
      SharedBuffer out = _ops->alloc(piece * (std::size_t)_cfg.experts);
      if (out.empty() || !fill(out.contents())) { return SharedBuffer{}; }
      return out;
    }, kEncoderPart);
    return !dst.empty();
  };
  if (!stack("gate_proj", l.wg) || !stack("up_proj", l.wu) ||
      !stack("down_proj", l.wd)) {
    return fail(lname(i, "mlp.experts.*"));
  }
  return true;
}

bool
MingLlm::load_trunk_(std::string* err)
{
  WeightSet& ws = *_ws;
  auto* mc = _ops->mc();
  const std::size_t H = (std::size_t)_cfg.hidden;
  _embed = ws.tensor("model.model.word_embeddings.weight", mc,
                     Residency::Copied);
  _final_norm = ws.tensor("model.model.norm.weight", mc, Residency::Copied);
  if (_embed.byte_size() != (std::size_t)_cfg.vocab * H * 2 ||
      _final_norm.empty()) {
    if (err != nullptr) { *err = "MLLM embeddings / final norm missing"; }
    return false;
  }
  // The routers, in f32 on the host: the reference's router is an f32
  // Linear (`F.linear(x.float(), W.float())`) and so is this one.
  const int E = _cfg.experts;
  _gate.resize((std::size_t)_cfg.layers);
  _gate_bias.resize((std::size_t)_cfg.layers);
  _img_gate.resize((std::size_t)_cfg.layers);
  _img_bias.resize((std::size_t)_cfg.layers);
  auto to_f32 = [&](const std::string& name, std::size_t n,
                    std::vector<float>* out) {
    SharedBuffer b = ws.read(name, mc, Residency::Copied);
    if (b.byte_size() != n * 2) { return false; }
    const auto* s = static_cast<const std::uint16_t*>(b.contents());
    out->resize(n);
    for (std::size_t k = 0; k < n; ++k) { (*out)[k] = bf16_to_f32(s[k]); }
    return true;
  };
  for (int i = 0; i < _cfg.layers; ++i) {
    if (!_cfg.is_moe(i)) { continue; }
    const std::size_t i_ = (std::size_t)i;
    if (!to_f32(lname(i, "mlp.gate.weight"), (std::size_t)E * H, &_gate[i_]) ||
        !to_f32(lname(i, "mlp.gate.expert_bias"), (std::size_t)E,
                &_gate_bias[i_]) ||
        !to_f32(lname(i, "mlp.image_gate.weight"), (std::size_t)E * H,
                &_img_gate[i_]) ||
        !to_f32(lname(i, "mlp.image_gate.expert_bias"), (std::size_t)E,
                &_img_bias[i_])) {
      if (err != nullptr) {
        *err = "MLLM router weights missing at layer " + std::to_string(i);
      }
      return false;
    }
  }
  _layers.resize((std::size_t)_cfg.layers);
  for (int i = 0; i < _cfg.layers; ++i) {
    if (_stream && _cfg.is_moe(i)) { continue; }
    auto l = std::make_unique<Layer>();
    if (!fill_layer_(i, *l, false, err)) { return false; }
    _layers[(std::size_t)i] = std::move(l);
  }
  if (_stream) {
    for (auto& s : _slot) {
      s = std::make_unique<Layer>();
      if (!alloc_layer_(*s, true)) {
        if (err != nullptr) { *err = "could not allocate a layer slot"; }
        return false;
      }
    }
  }
  return true;
}

std::size_t
MingLlm::weight_bytes() const
{
  std::size_t n = _embed.byte_size() + _final_norm.byte_size();
  for (const auto& l : _layers) {
    if (l) { n += l->bytes(); }
  }
  for (const auto& s : _slot) {
    if (s) { n += s->bytes(); }
  }
  for (const auto* v : {&_gate, &_gate_bias, &_img_gate, &_img_bias}) {
    for (const auto& x : *v) { n += x.size() * 4; }
  }
  return n;
}

bool
MingLlm::ensure_scratch_(int L, int n_text, int n_taps)
{
  if (L <= _cap_rows && n_text <= _cap_text && n_taps <= _cap_taps) {
    return true;
  }
  const std::size_t R = (std::size_t)std::max(L, _cap_rows);
  const std::size_t T = (std::size_t)std::max(n_text, _cap_text);
  const std::size_t NT = (std::size_t)std::max(n_taps, _cap_taps);
  const std::size_t H = (std::size_t)_cfg.hidden;
  const std::size_t K = (std::size_t)_cfg.top_k;
  const std::size_t P = R * K;
  const std::size_t I = (std::size_t)_cfg.moe_inter;
  const LlmOps& o = *_ops;
  _h = o.alloc(R * H * 4);
  _xn = o.alloc(R * H * 2);
  _xn32 = o.alloc(R * H * 4);
  _qkv = o.alloc(R * (std::size_t)_cfg.qkv_rows() * 2);
  _attn = o.alloc(R * H * 2);
  _sh_act = o.alloc(R * I * (std::size_t)_cfg.shared_experts * 2);
  _dtiles = o.alloc((R / LlmOps::kMoeTileRows + 1) * 16);
  _d_act = o.alloc(R * (std::size_t)_cfg.inter * 2);
  _xs = o.alloc(P * H * 2);
  _act = o.alloc(P * I * 2);
  _eout = o.alloc(P * H * 4);
  _perm = o.alloc(P * 4);
  _pair = o.alloc(P * 4);
  _wt = o.alloc(P * 4);
  // Each expert's run is cut into <= kMoeTileRows-row tiles.
  _tiles = o.alloc((P / LlmOps::kMoeTileRows + (std::size_t)_cfg.experts) *
                   16);
  _cos = o.alloc(R * (std::size_t)_cfg.rot_half() * 4);
  _sin = o.alloc(R * (std::size_t)_cfg.rot_half() * 4);
  _final = o.alloc(R * H * 2);
  _concat = o.alloc(std::max<std::size_t>(T, 1) * H * NT * 2);
  _logits.resize(R * (std::size_t)_cfg.experts);
  for (const SharedBuffer* b :
       {&_h, &_xn, &_xn32, &_qkv, &_attn, &_sh_act, &_dtiles, &_d_act,
        &_xs, &_act, &_eout, &_perm, &_pair, &_wt, &_tiles, &_cos, &_sin,
        &_final, &_concat}) {
    if (b->empty()) {
      _cap_rows = _cap_text = _cap_taps = 0;
      return false;
    }
  }
  _cap_rows = (int)R;
  _cap_text = (int)T;
  _cap_taps = (int)NT;
  return true;
}

void
MingLlm::route_(int layer, int L, int q0, int nq, Debug* dbg)
{
  const int E = _cfg.experts, Hd = _cfg.hidden, K = _cfg.top_k;
  const int G = _cfg.n_group, per = E / G;
  const auto li = (std::size_t)layer;
  const float* x = static_cast<const float*>(_xn32.contents());
  float* lg = _logits.data();
  // Logits in f32: the text router over every row, then the image
  // router over the query rows, which take ITS choice (MultiRouter).
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, L, E, Hd, 1.0f, x,
              Hd, _gate[li].data(), Hd, 0.0f, lg, E);
  if (nq > 0) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nq, E, Hd, 1.0f,
                x + (std::size_t)q0 * Hd, Hd, _img_gate[li].data(), Hd,
                0.0f, lg + (std::size_t)q0 * E, E);
  }
  std::vector<std::int32_t> idx((std::size_t)L * K);
  std::vector<float> w((std::size_t)L * K);
  std::vector<float> s((std::size_t)E), r((std::size_t)E);
  std::vector<int> order((std::size_t)E), gorder((std::size_t)G);
  std::vector<float> gs((std::size_t)G);
  std::vector<char> gkeep((std::size_t)G);
  const bool forced = dbg != nullptr && dbg->tf_idx != nullptr &&
                      dbg->tf_w != nullptr;
  const std::size_t moe_i = (std::size_t)(layer - _cfg.first_dense);
  for (int row = 0; row < L && forced; ++row) {
    for (int k = 0; k < K; ++k) {
      const std::size_t rk = (std::size_t)row * K + k;
      idx[rk] = dbg->tf_idx[moe_i * (std::size_t)L * K + rk];
      w[rk] = dbg->tf_w[moe_i * (std::size_t)L * K + rk];
    }
  }
  for (int row = 0; row < L && !forced; ++row) {
    const bool img = row >= q0 && row < q0 + nq;
    const float* bias = img ? _img_bias[li].data() : _gate_bias[li].data();
    const float* l = lg + (std::size_t)row * E;
    for (int e = 0; e < E; ++e) {
      s[(std::size_t)e] = 1.0f / (1.0f + std::exp(-l[e]));
      r[(std::size_t)e] = s[(std::size_t)e] + bias[e];
    }
    // A group scores the sum of its two best biased scores; the
    // topk_group best groups survive.
    for (int g = 0; g < G; ++g) {
      float m1 = -std::numeric_limits<float>::infinity(), m2 = m1;
      for (int e = g * per; e < (g + 1) * per; ++e) {
        const float v = r[(std::size_t)e];
        if (v > m1) { m2 = m1; m1 = v; }
        else if (v > m2) { m2 = v; }
      }
      gs[(std::size_t)g] = m1 + m2;
      gorder[(std::size_t)g] = g;
    }
    std::partial_sort(gorder.begin(), gorder.begin() + _cfg.topk_group,
                      gorder.end(), [&](int a, int b) {
                        return gs[(std::size_t)a] > gs[(std::size_t)b] ||
                               (gs[(std::size_t)a] == gs[(std::size_t)b] &&
                                a < b);
                      });
    std::fill(gkeep.begin(), gkeep.end(), 0);
    for (int k = 0; k < _cfg.topk_group; ++k) {
      gkeep[(std::size_t)gorder[(std::size_t)k]] = 1;
    }
    int n = 0;
    for (int e = 0; e < E; ++e) {
      if (gkeep[(std::size_t)(e / per)]) { order[(std::size_t)n++] = e; }
    }
    // Top-k by the BIASED score among the surviving groups ...
    std::partial_sort(order.begin(), order.begin() + K, order.begin() + n,
                      [&](int a, int b) {
                        return r[(std::size_t)a] > r[(std::size_t)b] ||
                               (r[(std::size_t)a] == r[(std::size_t)b] &&
                                a < b);
                      });
    // ... weighted by the UNBIASED sigmoid, renormalised, x 2.5.
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += s[(std::size_t)order[(std::size_t)k]];
    }
    for (int k = 0; k < K; ++k) {
      const int e = order[(std::size_t)k];
      idx[(std::size_t)row * K + k] = e;
      w[(std::size_t)row * K + k] =
          s[(std::size_t)e] / (sum + 1e-20f) * _cfg.routed_scale;
    }
  }
  if (dbg != nullptr) { dbg->experts.push_back(idx); }

  // Pairs in expert order (rows ascending within an expert): the gather
  // permutation, each (row, k)'s pair slot, the combine weights, and
  // the tile table the grouped GEMMs walk.
  std::vector<int> count((std::size_t)E, 0), start((std::size_t)E, 0);
  for (std::int32_t e : idx) { ++count[(std::size_t)e]; }
  for (int e = 1; e < E; ++e) {
    start[(std::size_t)e] =
        start[(std::size_t)e - 1] + count[(std::size_t)e - 1];
  }
  auto* perm = static_cast<std::int32_t*>(_perm.contents());
  auto* pair = static_cast<std::int32_t*>(_pair.contents());
  auto* wt = static_cast<float*>(_wt.contents());
  std::vector<int> fill = start;
  for (int row = 0; row < L; ++row) {
    for (int k = 0; k < K; ++k) {
      const std::size_t rk = (std::size_t)row * K + k;
      const int p = fill[(std::size_t)idx[rk]]++;
      perm[p] = row;
      pair[rk] = p;
      wt[rk] = w[rk];
    }
  }
  auto* tiles = static_cast<std::int32_t*>(_tiles.contents());
  int t = 0;
  for (int e = 0; e < E; ++e) {
    for (int r0 = 0; r0 < count[(std::size_t)e]; r0 += LlmOps::kMoeTileRows) {
      tiles[4 * t + 0] = e;
      tiles[4 * t + 1] = start[(std::size_t)e] + r0;
      tiles[4 * t + 2] =
          std::min(LlmOps::kMoeTileRows, count[(std::size_t)e] - r0);
      tiles[4 * t + 3] = 0;
      ++t;
    }
  }
  _n_tiles = t;
}

bool
MingLlm::forward(const Input& in, Output* out, Debug* dbg, std::string* err)
{
  auto fail = [err](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const LlmConfig& c = _cfg;
  const int L = (int)in.ids.size();
  const int N = in.n_text;
  const int H = c.hidden;
  const int ntap = (int)in.taps.size();
  if (L <= 0 || N <= 0 || N > L || in.query_row0 + in.query_rows > L ||
      (in.query_rows > 0 && in.query_embeds == nullptr)) {
    return fail("MLLM input rows are inconsistent");
  }
  if (!ensure_scratch_(L, N, ntap)) {
    return fail("could not allocate the MLLM's activations");
  }

  const bool prof = dbg != nullptr && dbg->profile;
  using clk = std::chrono::steady_clock;
  auto ms_since = [](clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
  };
  const auto t_embed = clk::now();
  // ---- embeddings into the f32 residual --------------------------------
  {
    auto* h = static_cast<float*>(_h.contents());
    const auto* emb = static_cast<const std::uint16_t*>(_embed.contents());
    const auto* q = in.query_rows > 0
        ? static_cast<const std::uint16_t*>(in.query_embeds->contents())
        : nullptr;
    for (int r = 0; r < L; ++r) {
      const int id = in.ids[(std::size_t)r];
      const bool query = r >= in.query_row0 &&
                         r < in.query_row0 + in.query_rows;
      if (!query && (id < 0 || id >= c.vocab)) {
        return fail("token id " + std::to_string(id) + " outside the vocab");
      }
      const std::uint16_t* src =
          query ? q + (std::size_t)(r - in.query_row0) * H
                : emb + (std::size_t)id * H;
      float* dst = h + (std::size_t)r * H;
      for (int d = 0; d < H; ++d) { dst[d] = bf16_to_f32(src[d]); }
    }
  }

  // ---- the video_rope angle table (docs/MING-IMAGE.md 1.5) --------------
  // Rows before the queries are text: (t, h, w) = (i, i, i). The query
  // grid is t=1, h=1, w=nq after the 2x2 merge; w is centred, then t's
  // offset (the text length) added to all three, so w goes NEGATIVE for
  // short prompts. Text after the grid resumes at max(t) + 1.
  {
    const int rh = c.rot_half();
    const int q0 = in.query_row0, nq = in.query_rows;
    std::vector<float> inv((std::size_t)rh);
    for (int k = 0; k < rh; ++k) {
      // transformers' default init, in f32 as torch computes it.
      inv[(std::size_t)k] = 1.0f / std::pow(c.rope_theta,
                                            (float)(2 * k) / (float)(2 * rh));
    }
    auto* cs = static_cast<float*>(_cos.contents());
    auto* sn = static_cast<float*>(_sin.contents());
    for (int r = 0; r < L; ++r) {
      int pos[3];
      if (r < q0 || nq == 0) {
        pos[0] = pos[1] = pos[2] = r;
      } else if (r < q0 + nq) {
        pos[0] = pos[1] = q0;
        pos[2] = q0 + (r - q0) - (nq - 1) / 2;
      } else {
        pos[0] = pos[1] = pos[2] = q0 + 1 + (r - q0 - nq);
      }
      for (int k = 0; k < rh; ++k) {
        const float a = inv[(std::size_t)k] * (float)pos[video_rope_axis(k)];
        cs[(std::size_t)r * rh + k] = (float)std::cos((double)a);
        sn[(std::size_t)r * rh + k] = (float)std::sin((double)a);
      }
    }
  }
  LlmOps::dense_tiles(_dtiles, L);
  if (prof) { dbg->cpu_embed_ms += ms_since(t_embed); }
  if (!_ops->plan_attention(&_plan, c.heads, c.kv_heads, L, c.qkv_rows(),
                            c.qkv_rows(), H, /*causal=*/true)) {
    return fail("the steel attention did not specialise");
  }

  const LlmOps& o = *_ops;
  auto stream = o.mc()->make_command_stream();
  ComputeEncoder enc = stream.begin_compute();
  double* gpu_bucket = nullptr;   // where the next sync's GPU time goes
  auto sync = [&](bool reopen = true) -> bool {
    enc.end();
    auto f = stream.commit();
    std::string why;
    if (f.valid() && !f.wait_ok(&why)) {
      if (err != nullptr) { *err = "MLLM GPU work failed: " + why; }
      return false;
    }
    if (prof && f.valid() && gpu_bucket != nullptr) {
      *gpu_bucket += 1e3 * f.gpu_times().gpu_s;
    }
    if (reopen) { enc = stream.begin_compute(); }
    return true;
  };
  // HF hidden_states index k (k < layers) is the residual after layer
  // k-1: copy it where stream B wants it, and to the test if asked.
  auto taps_after = [&](int k) -> bool {
    for (int t = 0; t < ntap; ++t) {
      if (in.taps[(std::size_t)t] == k && k < c.layers) {
        o.copy_rows_f32(enc, _h, 0, H, _concat, H * ntap, t * H, N, H);
      }
    }
    if (dbg != nullptr &&
        std::find(dbg->tap_layers.begin(), dbg->tap_layers.end(), k) !=
            dbg->tap_layers.end() && k < c.layers) {
      if (!sync()) { return false; }
      const auto* h = static_cast<const float*>(_h.contents());
      dbg->taps.emplace_back(h, h + (std::size_t)L * H);
    }
    return true;
  };

  // ---- streaming: the first MoE layer's read starts now ----------------
  std::future<std::string> pending;
  int moe_seen = 0;
  auto prefetch = [&](int layer, Layer* slot) {
    pending = std::async(std::launch::async, [this, layer, slot] {
      std::string e;
      return fill_layer_(layer, *slot, true, &e) ? std::string() : e;
    });
  };
  if (_stream) {
    for (int i = 0; i < c.layers; ++i) {
      if (c.is_moe(i)) { prefetch(i, _slot[0].get()); break; }
    }
  }

  if (!taps_after(0)) { return false; }
  const std::size_t LH = (std::size_t)L * H;
  for (int i = 0; i < c.layers; ++i) {
    const bool moe = c.is_moe(i);
    Layer* l = nullptr;
    if (_stream && moe) {
      l = _slot[moe_seen % 2].get();
      const auto t_slot = clk::now();
      const std::string e = pending.get();
      if (prof) { dbg->wait_slot_ms += ms_since(t_slot); }
      if (!e.empty()) { return fail(e); }
    } else {
      l = _layers[(std::size_t)i].get();
    }
    const bool tf = dbg != nullptr && dbg->tf_hs != nullptr;
    if (tf) {
      if (!sync()) { return false; }
      std::memcpy(_h.contents(), dbg->tf_hs + (std::size_t)i * LH, LH * 4);
    }
    // ---- attention ----
    o.rms_f32(enc, _h, l->in_norm, _xn, nullptr, L, H, c.eps);
    o.linear(enc, _xn, 0, l->qkv, nullptr, _qkv, 0, L, c.qkv_rows(), H);
    o.qk_norm_rope(enc, _qkv, &l->q_norm, &l->k_norm, _cos, _sin, L,
                   c.qkv_rows(), c.heads, c.kv_heads, c.rot_half(), c.eps);
    o.attention(enc, _plan, _qkv, 0, (std::size_t)c.heads * c.head_dim,
                (std::size_t)(c.heads + c.kv_heads) * c.head_dim, _attn);
    // Every projection that feeds the residual adds into it in f32
    // (ming_llm.metal, ming_gemm_acc_f32, says why).
    o.gemm_acc_f32(enc, _attn, l->dense, _h, _dtiles, L, H, H);
    if (!moe) {
      o.rms_f32(enc, _h, l->post_norm, _xn, nullptr, L, H, c.eps);
      o.swiglu_linear(enc, _xn, l->mlp_gu, _d_act, L, 2 * c.inter, H);
      o.gemm_acc_f32(enc, _d_act, l->mlp_down, _h, _dtiles, L, H, c.inter);
    } else {
      o.rms_f32(enc, _h, l->post_norm, _xn, &_xn32, L, H, c.eps);
      // The shared expert (un-gated) runs while the CPU routes.
      const int SI = c.moe_inter * c.shared_experts;
      o.swiglu_linear(enc, _xn, l->sh_gu, _sh_act, L, 2 * SI, H);
      o.gemm_acc_f32(enc, _sh_act, l->sh_down, _h, _dtiles, L, H, SI);
      if (prof) { gpu_bucket = &dbg->gpu_attn_ms; }
      if (!sync()) { return false; }
      // The GPU has now finished every earlier layer, so the OTHER slot
      // is free: the next MoE layer's read goes there, under this one.
      if (_stream) {
        for (int j = i + 1; j < c.layers; ++j) {
          if (c.is_moe(j)) {
            prefetch(j, _slot[(moe_seen + 1) % 2].get());
            break;
          }
        }
      }
      const auto t_route = clk::now();
      route_(i, L, in.query_row0, in.query_rows, dbg);
      if (prof) { dbg->cpu_route_ms += ms_since(t_route); }
      const int P = L * c.top_k;
      o.gather_rows(enc, _xn, _xs, _perm, P, H);
      o.moe_gate_up(enc, _xs, l->wg, l->wu, _act, _tiles, _n_tiles, H,
                    c.moe_inter);
      o.moe_down(enc, _act, l->wd, _eout, _tiles, _n_tiles, c.moe_inter, H);
      o.moe_combine(enc, _h, _eout, _pair, _wt, L, H, c.top_k);
      ++moe_seen;
      if (prof) {
        gpu_bucket = &dbg->gpu_moe_ms;
        if (!sync()) { return false; }
      }
    }
    if (tf) {
      if (!sync()) { return false; }
      const auto* h = static_cast<const float*>(_h.contents());
      dbg->layer_out.emplace_back(h, h + LH);
    }
    if (!taps_after(i + 1)) { return false; }
  }
  o.rms_f32(enc, _h, _final_norm, _final, nullptr, L, H, c.eps);
  for (int t = 0; t < ntap; ++t) {
    if (in.taps[(std::size_t)t] == c.layers) {
      o.copy_rows_bf16(enc, _final, 0, H, _concat, H * ntap, t * H, N, H);
    }
  }
  if (!sync(false)) { return false; }
  if (dbg != nullptr &&
      std::find(dbg->tap_layers.begin(), dbg->tap_layers.end(), c.layers) !=
          dbg->tap_layers.end()) {
    const auto* f = static_cast<const std::uint16_t*>(_final.contents());
    std::vector<float> v(LH);
    for (std::size_t k = 0; k < LH; ++k) { v[k] = bf16_to_f32(f[k]); }
    dbg->taps.push_back(std::move(v));
  }
  out->final = _final.subview(0, LH * 2);
  out->concat = _concat.subview(0, (std::size_t)N * H * ntap * 2);
  return true;
}

}  // namespace ming
