#ifndef VPIPE_MING_CONFIG_H
#define VPIPE_MING_CONFIG_H

// What a Ming-Image checkpoint says about itself, read from its
// component configs -- and the one place that decides whether a
// directory IS one.
//
// The DiT's class name is "DiffusionTransformer", which says nothing: it
// is diffusers' generic word, and a future checkpoint of an unrelated
// family could carry it. So identity is STRUCTURAL and every part must
// agree -- the transformer declares the Ming checkpoint contract
// (`alignment_padding_mode` + `multi_frame_output`, both present, as the
// reference's own loader demands), the conditioning MLLM is a
// BailingMM2, and the connector and mlp components are there. A
// directory that matches on some of these and not others is NOT claimed;
// an identity check that can pass by default is how one plugin family
// shadows another's checkpoints.
//
// Two checkpoints share the structure: Ming-Image-0.1-Design (text to
// image; padding "zero_masked", single-frame output) and
// Ming-Image-0.1-Design-Layer (layer decomposition; "learned", multi
// frame). Both are CLAIMED -- so a Layer checkpoint is answered by this
// family rather than falling through to a host family that would load a
// foreign config over its weights -- and the load refuses the one this
// plugin does not run yet, by name.

#include <string>
#include <vector>

namespace ming {

// transformer/config.json. Defaults are the published checkpoint's, and
// every field the forward depends on is read rather than assumed.
struct DitConfig {
  int   dim = 3840;
  int   n_layers = 30;
  int   n_refiner_layers = 2;
  int   n_heads = 30;
  int   n_kv_heads = 30;
  int   in_channels = 16;
  int   patch = 2;              // all_patch_size[0]
  int   f_patch = 1;            // all_f_patch_size[0]
  int   cap_feat_dim = 2560;
  float norm_eps = 1e-5f;
  float rope_theta = 256.0f;
  float t_scale = 1000.0f;
  bool  qk_norm = true;
  std::vector<int> axes_dims{32, 48, 48};
  std::vector<int> axes_lens{20480, 512, 512};
  // The checkpoint contract. "zero_masked": no pad tokens, pad rows are
  // zero and excluded as attention keys. "learned": the Layer model's.
  std::string padding_mode;
  bool        multi_frame_output = false;

  int head_dim() const { return n_heads > 0 ? dim / n_heads : 0; }
};

// mlp/config.json: how the MLLM's hidden states become the DiT's
// caption. See docs/MING-IMAGE.md section 1.
struct CondConfig {
  int  c_input_dim = 2560;      // diffusion_c_input_dim: stream A width
  int  inner_dim = 3840;        // diffusion_inner_dim: stream B width
  std::vector<int> taps{5, 12, 20};   // selected_hidden_states_layers
  std::vector<int> scales{16};        // img_gen_scales: 16x16 queries
  bool directvlm = true;        // use_vlm_directvlm_condition (stream B)
  bool learnable = true;        // use_learnable_token_condition (A)
  bool identity_mlp = true;     // use_identity_mlp
  bool text_encoder_norm = false;
  bool connector_norm = false;

  // Rows of stream A: the last scale's query grid.
  int query_rows() const
  {
    const int s = scales.empty() ? 0 : scales.back();
    return s * s;
  }
};

struct Checkpoint {
  bool        valid = false;
  std::string why;              // when !valid: what was missing
  std::string root;
  std::string transformer_dir;
  std::string vae_dir;
  std::string mllm_dir;
  std::string connector_dir;
  std::string mlp_dir;
  DitConfig   dit;
  CondConfig  cond;
  // scheduler/scheduler_config.json: STATIC shift. The reference's
  // dynamic-shifting override is a no-op (docs/MING-IMAGE.md 3.1).
  float shift = 6.0f;
  // vae/config.json
  int   vae_channels = 4;       // input_channels: RGBA
  float vae_scaling = 8.0064f;  // latents_mean/std are null
  float vae_shift = 0.0f;

  // The text-to-image checkpoint (the one this plugin runs).
  bool text_to_image() const
  {
    return dit.padding_mode == "zero_masked" && !dit.multi_frame_output;
  }
};

// transformer/config.json alone, with the Ming contract checked -- for a
// DiT named on its own (generate-image's `dit_dir`, a quantized pack).
bool read_dit_config(const std::string& transformer_dir, DitConfig* out,
                     std::string* why);

// Every component config, read and cross-checked. Never throws; an
// invalid result carries `why`.
Checkpoint read_checkpoint(const std::string& root);

// The identity test: a root this family answers for. Cheap -- a handful
// of small JSON reads, no weights.
bool is_ming_image(const std::string& root);

// The latent stride (VAE /8) times the DiT patch (2).
constexpr int kVaeScale = 8;

}  // namespace ming

#endif
