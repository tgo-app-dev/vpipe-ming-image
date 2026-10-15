#include "ming-config.h"

#include "common/flex-data.h"

#include <filesystem>
#include <fstream>
#include <string>

namespace ming {

namespace {

namespace fs = std::filesystem;

// A JSON file as FlexData; a missing or unreadable file is a default
// FlexData, which every caller tests with is_object().
vpipe::FlexData
read_json(const fs::path& p)
{
  std::error_code ec;
  if (!fs::exists(p, ec)) { return {}; }
  std::ifstream in(p);
  if (!in) { return {}; }
  return vpipe::FlexData::from_json(in);
}

int
get_int(const vpipe::FlexData& fd, const char* k, int d)
{
  if (!fd.is_object()) { return d; }
  auto o = fd.as_object();
  return o.contains(k) ? (int)o.at(k).as_int(d) : d;
}

float
get_real(const vpipe::FlexData& fd, const char* k, float d)
{
  if (!fd.is_object()) { return d; }
  auto o = fd.as_object();
  if (!o.contains(k)) { return d; }
  const vpipe::FlexData v = o.at(k);
  if (v.is_null()) { return d; }
  return (float)v.as_real(d);
}

bool
get_bool(const vpipe::FlexData& fd, const char* k, bool d)
{
  if (!fd.is_object()) { return d; }
  auto o = fd.as_object();
  return o.contains(k) ? o.at(k).as_bool(d) : d;
}

std::string
get_str(const vpipe::FlexData& fd, const char* k)
{
  if (!fd.is_object()) { return {}; }
  auto o = fd.as_object();
  return o.contains(k) ? std::string(o.at(k).as_string("")) : std::string();
}

// An integer array, REPLACED wholesale when present: its length is part
// of the meaning (three RoPE axes, three tapped layers).
std::vector<int>
get_ints(const vpipe::FlexData& fd, const char* k, std::vector<int> d)
{
  if (!fd.is_object()) { return d; }
  auto o = fd.as_object();
  if (!o.contains(k)) { return d; }
  const vpipe::FlexData a = o.at(k);
  if (!a.is_array()) { return d; }
  auto arr = a.as_array();
  std::vector<int> v;
  v.reserve(arr.size());
  for (std::size_t i = 0; i < arr.size(); ++i) {
    v.push_back((int)arr[i].as_int(0));
  }
  return v;
}

bool
has_key(const vpipe::FlexData& fd, const char* k)
{
  return fd.is_object() && fd.as_object().contains(k);
}

}  // namespace

bool
read_dit_config(const std::string& transformer_dir, DitConfig* out,
                std::string* why)
{
  auto fail = [why](std::string m) {
    if (why != nullptr) { *why = std::move(m); }
    return false;
  };
  if (out == nullptr) { return fail("no output"); }
  const vpipe::FlexData t =
      read_json(fs::path(transformer_dir) / "config.json");
  if (!t.is_object()) { return fail("no transformer/config.json"); }
  if (get_str(t, "_class_name") != "DiffusionTransformer") {
    return fail("transformer/config.json is not a DiffusionTransformer");
  }
  // The Ming checkpoint contract: BOTH keys, as the reference's loader
  // requires -- one without the other is a checkpoint it refuses too.
  if (!has_key(t, "alignment_padding_mode") ||
      !has_key(t, "multi_frame_output")) {
    return fail("transformer/config.json lacks the Ming checkpoint "
                "contract (alignment_padding_mode + multi_frame_output)");
  }
  DitConfig d;
  d.dim              = get_int(t, "dim", d.dim);
  d.n_layers         = get_int(t, "n_layers", d.n_layers);
  d.n_refiner_layers = get_int(t, "n_refiner_layers", d.n_refiner_layers);
  d.n_heads          = get_int(t, "n_heads", d.n_heads);
  d.n_kv_heads       = get_int(t, "n_kv_heads", d.n_kv_heads);
  d.in_channels      = get_int(t, "in_channels", d.in_channels);
  d.cap_feat_dim     = get_int(t, "cap_feat_dim", d.cap_feat_dim);
  d.norm_eps         = get_real(t, "norm_eps", d.norm_eps);
  d.rope_theta       = get_real(t, "rope_theta", d.rope_theta);
  d.t_scale          = get_real(t, "t_scale", d.t_scale);
  d.qk_norm          = get_bool(t, "qk_norm", d.qk_norm);
  d.axes_dims        = get_ints(t, "axes_dims", d.axes_dims);
  d.axes_lens        = get_ints(t, "axes_lens", d.axes_lens);
  const std::vector<int> ps = get_ints(t, "all_patch_size", {2});
  const std::vector<int> fs_ = get_ints(t, "all_f_patch_size", {1});
  d.patch   = ps.empty() ? 2 : ps.front();
  d.f_patch = fs_.empty() ? 1 : fs_.front();
  d.padding_mode       = get_str(t, "alignment_padding_mode");
  d.multi_frame_output = get_bool(t, "multi_frame_output", false);
  if (d.padding_mode != "zero_masked" && d.padding_mode != "learned") {
    return fail("unknown alignment_padding_mode '" + d.padding_mode + "'");
  }
  if (d.axes_dims.size() != 3 || d.axes_lens.size() != 3 ||
      d.n_heads <= 0 || d.dim % d.n_heads != 0) {
    return fail("transformer/config.json has an unexpected geometry");
  }
  *out = std::move(d);
  return true;
}

Checkpoint
read_checkpoint(const std::string& root)
{
  Checkpoint ck;
  ck.root = root;
  const fs::path r(root);
  auto fail = [&ck](std::string why) {
    ck.valid = false;
    ck.why = std::move(why);
    return ck;
  };

  // ---- the DiT -----------------------------------------------------
  const fs::path tdir = r / "transformer";
  {
    std::string why;
    if (!read_dit_config(tdir.string(), &ck.dit, &why)) {
      return fail(why);
    }
  }
  const DitConfig& d = ck.dit;
  ck.transformer_dir = tdir.string();

  // ---- the conditioning MLLM, connector and heads -------------------
  const vpipe::FlexData m = read_json(r / "mllm" / "config.json");
  if (get_str(m, "model_type").rfind("bailingmm", 0) != 0) {
    return fail("mllm/config.json is not a BailingMM2 model");
  }
  ck.mllm_dir = (r / "mllm").string();
  const vpipe::FlexData c = read_json(r / "connector" / "config.json");
  if (!c.is_object()) { return fail("no connector/config.json"); }
  ck.connector_dir = (r / "connector").string();
  const vpipe::FlexData p = read_json(r / "mlp" / "config.json");
  if (!p.is_object()) { return fail("no mlp/config.json"); }
  ck.mlp_dir = (r / "mlp").string();
  CondConfig& q = ck.cond;
  q.c_input_dim = get_int(p, "diffusion_c_input_dim", q.c_input_dim);
  q.inner_dim   = get_int(p, "diffusion_inner_dim", q.inner_dim);
  q.taps        = get_ints(p, "selected_hidden_states_layers", q.taps);
  q.scales      = get_ints(p, "img_gen_scales", q.scales);
  q.directvlm   = get_bool(p, "use_vlm_directvlm_condition", q.directvlm);
  q.learnable   = get_bool(p, "use_learnable_token_condition", q.learnable);
  q.identity_mlp = get_bool(p, "use_identity_mlp", q.identity_mlp);
  q.text_encoder_norm = get_bool(p, "text_encoder_norm", false);
  q.connector_norm    = get_bool(p, "connector_norm", false);
  // The two streams must land at the DiT's widths: stream A feeds its
  // cap_embedder, stream B its hidden width directly. A mismatch here is
  // a conditioner and a DiT from different checkpoints.
  if (q.c_input_dim != d.cap_feat_dim || q.inner_dim != d.dim) {
    return fail("mlp/config.json's stream widths (" +
                std::to_string(q.c_input_dim) + ", " +
                std::to_string(q.inner_dim) + ") do not match the DiT's (" +
                std::to_string(d.cap_feat_dim) + ", " +
                std::to_string(d.dim) + ")");
  }

  // ---- scheduler and VAE --------------------------------------------
  const vpipe::FlexData s =
      read_json(r / "scheduler" / "scheduler_config.json");
  ck.shift = get_real(s, "shift", ck.shift);
  const vpipe::FlexData v = read_json(r / "vae" / "config.json");
  if (get_str(v, "_class_name") != "AutoencoderKLQwenImage") {
    return fail("vae/config.json is not an AutoencoderKLQwenImage");
  }
  ck.vae_dir      = (r / "vae").string();
  ck.vae_channels = get_int(v, "input_channels", 3);
  ck.vae_scaling  = get_real(v, "scaling_factor", ck.vae_scaling);
  ck.vae_shift    = get_real(v, "shift_factor", ck.vae_shift);

  ck.valid = true;
  return ck;
}

bool
is_ming_image(const std::string& root)
{
  return read_checkpoint(root).valid;
}

}  // namespace ming
