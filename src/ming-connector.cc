#include "ming-connector.h"

#include "common/flex-data.h"
#include "generative-models/weight-set.h"

#include <cmath>
#include <cstring>
#include <fstream>

namespace ming {

using vpipe::genai::WeightSet;
using Residency = vpipe::genai::WeightSet::Residency;

namespace {

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

std::string
cname(int i, const char* rest)
{
  return "model.layers." + std::to_string(i) + "." + rest;
}

}  // namespace

bool
read_connector_config(const std::string& dir, ConnectorConfig* cfg,
                      std::string* err)
{
  std::ifstream in(dir + "/config.json");
  if (!in) {
    if (err != nullptr) { *err = "no " + dir + "/config.json"; }
    return false;
  }
  vpipe::FlexData c;
  try {
    c = vpipe::FlexData::from_json(in);
  } catch (const std::exception& e) {
    if (err != nullptr) { *err = std::string("connector config: ") + e.what(); }
    return false;
  }
  ConnectorConfig& g = *cfg;
  g.hidden     = get_int(c, "hidden_size", g.hidden);
  g.layers     = get_int(c, "num_hidden_layers", g.layers);
  g.heads      = get_int(c, "num_attention_heads", g.heads);
  g.kv_heads   = get_int(c, "num_key_value_heads", g.kv_heads);
  g.inter      = get_int(c, "intermediate_size", g.inter);
  g.eps        = get_real(c, "rms_norm_eps", g.eps);
  g.rope_theta = get_real(c, "rope_theta", g.rope_theta);
  g.head_dim   = get_int(c, "head_dim", g.heads > 0 ? g.hidden / g.heads : 0);
  bool scaled = false;
  if (c.is_object() && c.as_object().contains("rope_scaling")) {
    scaled = !c.as_object().at("rope_scaling").is_null();
  }
  if (g.head_dim != 128 || g.hidden != g.heads * g.head_dim ||
      g.heads % g.kv_heads != 0 || scaled) {
    if (err != nullptr) {
      *err = "the connector is not the Qwen2 shape this port runs "
             "(head_dim 128, unscaled RoPE)";
    }
    return false;
  }
  return true;
}

MingConnector::~MingConnector()
{
  // See kEncoderPart.
  if (_ws) { _ws->release_part(kEncoderPart); }
}

std::size_t
MingConnector::bytes_for(const ConnectorConfig& c)
{
  const std::size_t H = (std::size_t)c.hidden;
  const std::size_t per =
      ((std::size_t)c.qkv_rows() * H + c.qkv_rows() + H * H +
       3 * (std::size_t)c.inter * H + 2 * H) * 2;
  return per * (std::size_t)c.layers + H * 2;
}

std::unique_ptr<MingConnector>
MingConnector::load(std::shared_ptr<WeightSet> ws, LlmOps* ops,
                    const ConnectorConfig& cfg, std::string* err)
{
  auto m = std::unique_ptr<MingConnector>(new MingConnector());
  m->_ws = std::move(ws);
  m->_ops = ops;
  m->_cfg = cfg;
  WeightSet& w = *m->_ws;
  auto* mc = ops->mc();
  const std::size_t H = (std::size_t)cfg.hidden;

  // f32 checkpoint tensors narrowed to bf16 and concatenated in order,
  // as one derived product. The key names the transform and every
  // piece, which is all the cache compares.
  auto narrow = [&](const std::vector<std::string>& names,
                    std::size_t elems) -> SharedBuffer {
    std::string key = "ming-connector/bf16";
    for (const auto& n : names) { key += "/" + n; }
    return w.derived(key, [&] {
      SharedBuffer out = ops->alloc(elems * 2);
      if (out.empty()) { return SharedBuffer{}; }
      auto* d = static_cast<std::uint16_t*>(out.contents());
      std::size_t at = 0;
      for (const auto& n : names) {
        SharedBuffer f = w.read(n, mc, Residency::Copied);
        const auto* info = w.src().info(n);
        if (f.empty() || info == nullptr || info->dtype != "F32" ||
            at + f.byte_size() / 4 > elems) {
          return SharedBuffer{};
        }
        const auto* s = static_cast<const float*>(f.contents());
        const std::size_t n_el = f.byte_size() / 4;
        for (std::size_t k = 0; k < n_el; ++k) {
          d[at + k] = f32_to_bf16(s[k]);
        }
        at += n_el;
      }
      return at == elems ? std::move(out) : SharedBuffer{};
    }, kEncoderPart);
  };
  // gate|up narrowed AND interleaved by row, for the fused SwiGLU GEMM.
  auto narrow_gu = [&](int i) -> SharedBuffer {
    const std::string g = cname(i, "mlp.gate_proj.weight");
    const std::string u = cname(i, "mlp.up_proj.weight");
    const std::size_t I = (std::size_t)cfg.inter;
    return w.derived("ming-connector/bf16-interleaved/" + g + "+" + u, [&] {
      SharedBuffer out = ops->alloc(2 * I * H * 2);
      SharedBuffer gf = w.read(g, mc, Residency::Copied);
      SharedBuffer uf = w.read(u, mc, Residency::Copied);
      if (out.empty() || gf.byte_size() != I * H * 4 ||
          uf.byte_size() != I * H * 4) {
        return SharedBuffer{};
      }
      auto* d = static_cast<std::uint16_t*>(out.contents());
      const auto* gs = static_cast<const float*>(gf.contents());
      const auto* us = static_cast<const float*>(uf.contents());
      for (std::size_t r = 0; r < I; ++r) {
        for (std::size_t k = 0; k < H; ++k) {
          d[(2 * r) * H + k] = f32_to_bf16(gs[r * H + k]);
          d[(2 * r + 1) * H + k] = f32_to_bf16(us[r * H + k]);
        }
      }
      return out;
    }, kEncoderPart);
  };

  const std::size_t QKV = (std::size_t)cfg.qkv_rows();
  m->_layers.resize((std::size_t)cfg.layers);
  for (int i = 0; i < cfg.layers; ++i) {
    Layer& l = m->_layers[(std::size_t)i];
    l.in_norm = narrow({cname(i, "input_layernorm.weight")}, H);
    l.post_norm = narrow({cname(i, "post_attention_layernorm.weight")}, H);
    l.qkv = narrow({cname(i, "self_attn.q_proj.weight"),
                    cname(i, "self_attn.k_proj.weight"),
                    cname(i, "self_attn.v_proj.weight")}, QKV * H);
    l.qkv_bias = narrow({cname(i, "self_attn.q_proj.bias"),
                         cname(i, "self_attn.k_proj.bias"),
                         cname(i, "self_attn.v_proj.bias")}, QKV);
    l.o = narrow({cname(i, "self_attn.o_proj.weight")}, H * H);
    l.gu = narrow_gu(i);
    l.down = narrow({cname(i, "mlp.down_proj.weight")},
                    H * (std::size_t)cfg.inter);
    if (l.in_norm.empty() || l.post_norm.empty() || l.qkv.empty() ||
        l.qkv_bias.empty() || l.o.empty() || l.gu.empty() || l.down.empty()) {
      if (err != nullptr) {
        *err = "connector layer " + std::to_string(i) + " failed to load";
      }
      return nullptr;
    }
  }
  m->_norm = narrow({"model.norm.weight"}, H);
  if (m->_norm.empty()) {
    if (err != nullptr) { *err = "connector final norm missing"; }
    return nullptr;
  }
  return m;
}

std::size_t
MingConnector::weight_bytes() const
{
  std::size_t n = _norm.byte_size();
  for (const Layer& l : _layers) {
    n += l.in_norm.byte_size() + l.post_norm.byte_size() +
         l.qkv.byte_size() + l.qkv_bias.byte_size() + l.o.byte_size() +
         l.gu.byte_size() + l.down.byte_size();
  }
  return n;
}

bool
MingConnector::ensure_scratch_(int S)
{
  if (S <= _cap) { return true; }
  const std::size_t R = (std::size_t)S;
  const std::size_t H = (std::size_t)_cfg.hidden;
  const LlmOps& o = *_ops;
  _x = o.alloc(R * H * 4);
  _xn = o.alloc(R * H * 2);
  _qkv = o.alloc(R * (std::size_t)_cfg.qkv_rows() * 2);
  _attn = o.alloc(R * H * 2);
  _o = o.alloc(R * H * 2);
  _act = o.alloc(R * (std::size_t)_cfg.inter * 2);
  _cos = o.alloc(R * 64 * 4);
  _sin = o.alloc(R * 64 * 4);
  for (const SharedBuffer* b : {&_x, &_xn, &_qkv, &_attn, &_o, &_act, &_cos,
                                &_sin}) {
    if (b->empty()) { _cap = 0; return false; }
  }
  // Positions 0..S-1 on every head dim (cache_position, no position_ids):
  // inv_freq as transformers' default init computes it, in f32.
  const int half = _cfg.head_dim / 2;
  auto* cs = static_cast<float*>(_cos.contents());
  auto* sn = static_cast<float*>(_sin.contents());
  for (int k = 0; k < half; ++k) {
    const float inv = 1.0f / std::pow(_cfg.rope_theta,
                                      (float)(2 * k) / (float)_cfg.head_dim);
    for (int r = 0; r < S; ++r) {
      const float a = inv * (float)r;
      cs[(std::size_t)r * half + k] = (float)std::cos((double)a);
      sn[(std::size_t)r * half + k] = (float)std::sin((double)a);
    }
  }
  _cap = S;
  return true;
}

bool
MingConnector::encode(ComputeEncoder& enc, const SharedBuffer& in,
                      const SharedBuffer& out, int S, std::string* err)
{
  // The scratch is sized once per S: a different S rebuilds the table
  // the plan reads, so only one S is ever planned at a time.
  if (S != _cap) { _cap = 0; }
  if (!ensure_scratch_(S)) {
    if (err != nullptr) { *err = "connector scratch allocation failed"; }
    return false;
  }
  const ConnectorConfig& c = _cfg;
  const int H = c.hidden;
  if (!_plan.ok || _plan.tokens != S) {
    if (!_ops->plan_attention(&_plan, c.heads, c.kv_heads, S, c.qkv_rows(),
                              c.qkv_rows(), H, /*causal=*/false)) {
      if (err != nullptr) { *err = "connector attention did not specialise"; }
      return false;
    }
  }
  const LlmOps& o = *_ops;
  const std::size_t SH = (std::size_t)S * H;
  // The f32 residual starts as proj_in's bf16 output: zero, then add.
  // Written by the CPU now, which is safe because every encode is
  // waited on before the next one is built.
  std::memset(_x.contents(), 0, SH * 4);
  o.add_f32(enc, _x, in, SH);
  for (const Layer& l : _layers) {
    o.rms_f32(enc, _x, l.in_norm, _xn, nullptr, S, H, c.eps);
    o.linear(enc, _xn, 0, l.qkv, &l.qkv_bias, _qkv, 0, S, c.qkv_rows(), H);
    o.qk_norm_rope(enc, _qkv, nullptr, nullptr, _cos, _sin, S, c.qkv_rows(),
                   c.heads, c.kv_heads, c.head_dim / 2, c.eps);
    o.attention(enc, _plan, _qkv, 0, (std::size_t)c.heads * c.head_dim,
                (std::size_t)(c.heads + c.kv_heads) * c.head_dim, _attn);
    o.linear(enc, _attn, 0, l.o, nullptr, _o, 0, S, H, H);
    o.add_f32(enc, _x, _o, SH);
    o.rms_f32(enc, _x, l.post_norm, _xn, nullptr, S, H, c.eps);
    o.swiglu_linear(enc, _xn, l.gu, _act, S, 2 * c.inter, H);
    o.linear(enc, _act, 0, l.down, nullptr, _o, 0, S, H, c.inter);
    o.add_f32(enc, _x, _o, SH);
  }
  o.rms_f32(enc, _x, _norm, out, nullptr, S, H, c.eps);
  return true;
}

}  // namespace ming
