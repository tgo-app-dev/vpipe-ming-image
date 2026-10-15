#include "ming-encoder.h"

#include "ming-connector.h"
#include "ming-llm.h"
#include "ming-ops.h"
#include "ming-tokenize.h"

#include "generative-models/weight-set.h"

#include <algorithm>
#include <cstring>

namespace ming {

namespace {

using vpipe::genai::WeightSet;
using Residency = vpipe::genai::WeightSet::Residency;

// mlp/: the two conditioning heads, the connector's in/out projections
// and the learned query rows. All F32 on disk, bf16 here as in the
// reference (`.to(dtype=torch_dtype)` at load).
struct Heads {
  SharedBuffer dvlm_norm;    // [hidden * taps]      proj_directvlm.0
  SharedBuffer dvlm_w;       // [inner, hidden*taps] proj_directvlm.1
  SharedBuffer dvlm_b;
  SharedBuffer in_w, in_b;   // [conn, hidden]       proj_in
  SharedBuffer out_w, out_b; // [c_input, conn]      proj_out
  SharedBuffer queries;      // [query_rows, hidden] query_tokens_dict

  std::size_t bytes() const
  {
    return dvlm_norm.byte_size() + dvlm_w.byte_size() + dvlm_b.byte_size() +
           in_w.byte_size() + in_b.byte_size() + out_w.byte_size() +
           out_b.byte_size() + queries.byte_size();
  }
};

std::size_t
heads_bytes(const CondConfig& q, const LlmConfig& l, const ConnectorConfig& c)
{
  const std::size_t D = (std::size_t)l.hidden * q.taps.size();
  return (D + (std::size_t)q.inner_dim * D + q.inner_dim +
          (std::size_t)c.hidden * l.hidden + c.hidden +
          (std::size_t)q.c_input_dim * c.hidden + q.c_input_dim +
          (std::size_t)q.query_rows() * l.hidden) * 2;
}

class MingEncoderImpl final : public MingEncoder {
public:
  ~MingEncoderImpl() override
  {
    // The heads' narrowed copies (kEncoderPart); the models release
    // their own when their members go.
    _heads = Heads{};
    if (_ws_mlp) { _ws_mlp->release_part(kEncoderPart); }
  }

  bool init(const Checkpoint& ck, vpipe::metal_compute::MetalCompute* mc,
            const vpipe::SessionContextIntf* session, const Options& opt,
            std::string* err);

  bool encode(const std::string& prompt, Result* out,
              std::string* err) override
  {
    return run_(prompt, out, nullptr, err);
  }
  bool encode_debug(const std::string& prompt, Result* out, Debug* dbg,
                    std::string* err) override
  {
    return run_(prompt, out, dbg, err);
  }
  std::vector<std::int32_t>
  template_ids(const std::string& prompt) const override
  {
    return _tok->encode(ming_t2i_template(prompt));
  }
  std::uint64_t resident_bytes() const override
  {
    return _llm->weight_bytes() + _conn->weight_bytes() + _heads.bytes();
  }

private:
  bool run_(const std::string& prompt, Result* out, Debug* dbg,
            std::string* err);
  bool load_heads_(std::string* err);

  Checkpoint _ck;
  LlmConfig _lc;
  ConnectorConfig _cc;
  LlmOps _ops;
  std::unique_ptr<LingTokenizer> _tok;
  std::shared_ptr<WeightSet> _ws_mlp;
  std::unique_ptr<MingLlm> _llm;
  std::unique_ptr<MingConnector> _conn;
  Heads _heads;
};

bool
MingEncoderImpl::load_heads_(std::string* err)
{
  WeightSet& w = *_ws_mlp;
  auto* mc = _ops.mc();
  const CondConfig& q = _ck.cond;
  // The f32 tensor narrowed to bf16 -- a transform the encoder KEEPS, so
  // a derived() product; the f32 read is consumed.
  auto narrow = [&](const std::string& name, std::size_t elems) {
    return w.derived("ming-mlp/bf16/" + name, [&] {
      SharedBuffer f = w.read(name, mc, Residency::Copied);
      const auto* info = w.src().info(name);
      if (f.byte_size() != elems * 4 || info == nullptr ||
          info->dtype != "F32") {
        return SharedBuffer{};
      }
      return _ops.upload_bf16(static_cast<const float*>(f.contents()),
                              elems);
    }, kEncoderPart);
  };
  const std::size_t H = (std::size_t)_lc.hidden;
  const std::size_t D = H * q.taps.size();
  const std::size_t C = (std::size_t)_cc.hidden;
  const int s = q.scales.back();
  const std::string qname =
      "query_tokens_dict." + std::to_string(s) + "x" + std::to_string(s);
  _heads.dvlm_norm = narrow("proj_directvlm.0.weight", D);
  _heads.dvlm_w =
      narrow("proj_directvlm.1.weight", (std::size_t)q.inner_dim * D);
  _heads.dvlm_b = narrow("proj_directvlm.1.bias", (std::size_t)q.inner_dim);
  _heads.in_w = narrow("proj_in.weight", C * H);
  _heads.in_b = narrow("proj_in.bias", C);
  _heads.out_w = narrow("proj_out.weight", (std::size_t)q.c_input_dim * C);
  _heads.out_b = narrow("proj_out.bias", (std::size_t)q.c_input_dim);
  _heads.queries = narrow(qname, (std::size_t)q.query_rows() * H);
  for (const SharedBuffer* b :
       {&_heads.dvlm_norm, &_heads.dvlm_w, &_heads.dvlm_b, &_heads.in_w,
        &_heads.in_b, &_heads.out_w, &_heads.out_b, &_heads.queries}) {
    if (b->empty()) {
      if (err != nullptr) {
        *err = "mlp/model.safetensors: a conditioning head is missing or "
               "not the shape the configs imply";
      }
      return false;
    }
  }
  return true;
}

bool
MingEncoderImpl::init(const Checkpoint& ck,
                      vpipe::metal_compute::MetalCompute* mc,
                      const vpipe::SessionContextIntf* session,
                      const Options& opt, std::string* err)
{
  auto fail = [err](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (!ck.valid) { return fail("not a Ming-Image checkpoint: " + ck.why); }
  _ck = ck;
  const CondConfig& q = ck.cond;
  // The conditioning contract (ming-family.h) is BOTH streams, raw: a
  // checkpoint that switches either off, or that normalizes them, is a
  // different contract and is refused by name rather than half-served.
  if (!q.directvlm || !q.learnable || q.connector_norm ||
      q.text_encoder_norm || !q.identity_mlp || q.scales.empty() ||
      q.taps.empty()) {
    return fail("mlp/config.json asks for a conditioning variant this "
                "encoder does not implement (it runs directvlm + "
                "learnable tokens, no normalization, identity mlp)");
  }
  if (!read_llm_config(ck.mllm_dir, &_lc, err) ||
      !read_connector_config(ck.connector_dir, &_cc, err)) {
    return false;
  }
  for (int t : q.taps) {
    if (t < 0 || t > _lc.layers) {
      return fail("selected_hidden_states_layers names a layer the MLLM "
                  "does not have");
    }
  }
  std::string terr;
  _tok = LingTokenizer::load(ck.mllm_dir + "/tokenizer.json", &terr);
  if (!_tok) { return fail("tokenizer: " + terr); }
  if (!_ops.init(mc, err)) { return false; }

  auto ws_llm = vpipe::genai::open_weight_set(ck.mllm_dir, session);
  auto ws_conn = vpipe::genai::open_weight_set(ck.connector_dir, session);
  _ws_mlp = vpipe::genai::open_weight_set(ck.mlp_dir, session);
  if (!ws_llm || !ws_conn || !_ws_mlp) {
    return fail("could not open the mllm / connector / mlp checkpoints");
  }
  if (!load_heads_(err)) { return false; }
  _conn = MingConnector::load(ws_conn, &_ops, _cc, err);
  if (!_conn) { return false; }
  _llm = MingLlm::load(ws_llm, &_ops, _lc, opt.stream_layers, err);
  return (bool)_llm;
}

bool
MingEncoderImpl::run_(const std::string& prompt, Result* out, Debug* dbg,
                      std::string* err)
{
  auto fail = [err](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (out == nullptr) { return fail("no result"); }
  const CondConfig& q = _ck.cond;
  const int H = _lc.hidden;
  const int nq = q.query_rows();
  const int C = _cc.hidden;
  const int ntap = (int)q.taps.size();
  const int D = H * ntap;

  // The template's N rows, then [<image>] + nq x [<imagePatch>] +
  // [</image>] (docs/MING-IMAGE.md 1.3).
  MingLlm::Input in;
  in.ids = _tok->encode(ming_t2i_template(prompt));
  const int N = (int)in.ids.size();
  in.n_text = N;
  in.ids.push_back(_lc.image_start);
  for (int j = 0; j < nq; ++j) { in.ids.push_back(_lc.image_patch); }
  in.ids.push_back(_lc.image_end);
  in.query_row0 = N + 1;
  in.query_rows = nq;
  in.query_embeds = &_heads.queries;
  in.taps = q.taps;

  MingLlm::Debug ldbg;
  if (dbg != nullptr) {
    ldbg.tap_layers = dbg->tap_layers;
    ldbg.tf_hs = dbg->tf_hs;
    ldbg.tf_idx = dbg->tf_idx;
    ldbg.tf_w = dbg->tf_w;
    ldbg.profile = dbg->profile;
    dbg->ids = in.ids;
  }
  MingLlm::Output lo;
  if (!_llm->forward(in, &lo, dbg != nullptr ? &ldbg : nullptr, err)) {
    return false;
  }

  // ---- the heads and the connector, one command buffer ---------------
  const std::size_t NB = (std::size_t)N;
  SharedBuffer bn = _ops.alloc(NB * D * 2);
  SharedBuffer b = _ops.alloc(NB * q.inner_dim * 2);
  SharedBuffer e = _ops.alloc((std::size_t)nq * C * 2);
  SharedBuffer cout = _ops.alloc((std::size_t)nq * C * 2);
  SharedBuffer a = _ops.alloc((std::size_t)nq * q.c_input_dim * 2);
  if (bn.empty() || b.empty() || e.empty() || cout.empty() || a.empty()) {
    return fail("could not allocate the conditioning heads' activations");
  }
  auto stream = _ops.mc()->make_command_stream();
  ComputeEncoder enc = stream.begin_compute();
  // Stream B: one RMSNorm over the whole [hs5 | hs12 | hs20] row (1e-5,
  // diffusers' RMSNorm), then the projection to the DiT's width.
  _ops.rms_bf16(enc, lo.concat, _heads.dvlm_norm, bn, N, D, 1e-5f);
  _ops.linear(enc, bn, 0, _heads.dvlm_w, &_heads.dvlm_b, b, 0, N,
              q.inner_dim, D);
  // Stream A: the final-normed query rows -> proj_in -> connector ->
  // proj_out. No L2 normalization (connector_norm is false).
  _ops.linear(enc, lo.final, (std::size_t)in.query_row0 * H, _heads.in_w,
              &_heads.in_b, e, 0, nq, C, H);
  if (!_conn->encode(enc, e, cout, nq, err)) { return false; }
  _ops.linear(enc, cout, 0, _heads.out_w, &_heads.out_b, a, 0, nq,
              q.c_input_dim, C);
  enc.end();
  auto f = stream.commit();
  std::string why;
  if (!f.wait_ok(&why)) { return fail("conditioning heads failed: " + why); }
  if (dbg != nullptr && dbg->profile) {
    dbg->heads_ms = 1e3 * f.gpu_times().gpu_s;
  }

  // ---- the conditioning contract: bf16 [nq + N, inner] ----------------
  const int dim = q.inner_dim;
  out->rows = nq + N;
  out->dim = dim;
  out->query_rows = nq;
  out->text_rows = N;
  out->cond.assign((std::size_t)out->rows * dim, 0);
  const auto* pa = static_cast<const std::uint16_t*>(a.contents());
  const auto* pb = static_cast<const std::uint16_t*>(b.contents());
  for (int r = 0; r < nq; ++r) {
    std::memcpy(&out->cond[(std::size_t)r * dim],
                pa + (std::size_t)r * q.c_input_dim,
                (std::size_t)q.c_input_dim * 2);
  }
  std::memcpy(&out->cond[(std::size_t)nq * dim], pb, NB * dim * 2);

  if (dbg != nullptr) {
    auto f32 = [](const SharedBuffer& s, std::size_t n) {
      std::vector<float> v(n);
      const auto* p = static_cast<const std::uint16_t*>(s.contents());
      for (std::size_t k = 0; k < n; ++k) { v[k] = bf16_to_f32(p[k]); }
      return v;
    };
    // MingLlm hands taps back in layer order; give them in the order
    // asked for.
    std::vector<int> sorted = dbg->tap_layers;
    std::sort(sorted.begin(), sorted.end());
    dbg->taps.clear();
    for (int k : dbg->tap_layers) {
      const auto it = std::find(sorted.begin(), sorted.end(), k);
      dbg->taps.push_back(ldbg.taps[(std::size_t)(it - sorted.begin())]);
    }
    dbg->experts = std::move(ldbg.experts);
    dbg->layer_out = std::move(ldbg.layer_out);
    dbg->gpu_attn_ms = ldbg.gpu_attn_ms;
    dbg->gpu_moe_ms = ldbg.gpu_moe_ms;
    dbg->cpu_route_ms = ldbg.cpu_route_ms;
    dbg->cpu_embed_ms = ldbg.cpu_embed_ms;
    dbg->wait_slot_ms = ldbg.wait_slot_ms;
    dbg->cap_feats_2 = f32(b, NB * dim);
    dbg->connector_in = f32(e, (std::size_t)nq * C);
    dbg->connector_out = f32(cout, (std::size_t)nq * C);
    dbg->cap_feats = f32(a, (std::size_t)nq * q.c_input_dim);
  }
  return true;
}

}  // namespace

std::unique_ptr<MingEncoder>
MingEncoder::load(const Checkpoint& ck,
                  vpipe::metal_compute::MetalCompute* mc,
                  const vpipe::SessionContextIntf* session,
                  const Options& opt, std::string* err)
{
  auto e = std::make_unique<MingEncoderImpl>();
  if (!e->init(ck, mc, session, opt, err)) { return nullptr; }
  return e;
}

std::size_t
MingEncoder::resident_weight_bytes(const Checkpoint& ck)
{
  LlmConfig lc;
  if (!ck.valid || !read_llm_config(ck.mllm_dir, &lc, nullptr)) { return 0; }
  return MingLlm::resident_bytes_for(lc);
}

std::size_t
MingEncoder::streaming_floor_bytes(const Checkpoint& ck)
{
  LlmConfig lc;
  if (!ck.valid || !read_llm_config(ck.mllm_dir, &lc, nullptr)) { return 0; }
  return MingLlm::streaming_floor_for(lc);
}

std::size_t
MingEncoder::connector_weight_bytes(const Checkpoint& ck)
{
  LlmConfig lc;
  ConnectorConfig cc;
  if (!ck.valid || !read_llm_config(ck.mllm_dir, &lc, nullptr) ||
      !read_connector_config(ck.connector_dir, &cc, nullptr)) {
    return 0;
  }
  return MingConnector::bytes_for(cc) + heads_bytes(ck.cond, lc, cc);
}

}  // namespace ming
