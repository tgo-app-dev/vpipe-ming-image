#ifndef VPIPE_MING_FAMILY_H
#define VPIPE_MING_FAMILY_H

// Ming-Image as an out-of-tree IMAGE family: what `generate-image` calls
// once it has resolved a checkpoint this plugin claims.
//
// The seam is the GENERATION. generate-image owns the ports, the beats
// and the geometry bookkeeping; this family owns its schedule (static
// shift, 11 effective Euler steps for the reference's "12"), its
// guidance rule (zeros as the negative, as the reference does) and its
// whole denoise loop. It returns the DiT's latent; the host's
// `vae-decode` un-scales it (latent / scaling_factor) and decodes the
// RGBA picture, so this plugin registers no VAE.
//
// THE CONDITIONING CONTRACT with `ming-image-conditioner`, which is this
// plugin's own stage (ARCHITECTURE.md):
//
//   cond   bf16 [A + N, inner_dim]
//          rows [0, A)    stream A, the connector's caption features,
//                         in columns [0, c_input_dim) -- the DiT's
//                         cap_embedder runs on these, the rest is zero
//          rows [A, A+N)  stream B, the MLLM's text rows already
//                         projected to the DiT width
//   sideband  kSbQueryRows = A
//
// Both streams travel in ONE tensor because generate-image hands a
// family one conditioning tensor; the DiT's own cap_embedder stays with
// the DiT's weights rather than being exported to the conditioner.

#include "generative-models/image-model-registry.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ming {

// The sideband key naming stream A's row count.
inline constexpr const char* kSbQueryRows = "ming_query_rows";

class MingImageFamily final : public vpipe::genai::ImageModelFamily {
public:
  std::string_view tag() const noexcept override { return "ming-image"; }

  bool claims(const std::string& root,
              const std::string& model_type) const override;

  // The VAE's 8x stride times the DiT's patch of 2.
  void size_grid(const std::string& root, int* gh, int* gw) const override;
  int latent_scale(const std::string& root) const override;

  std::vector<vpipe::ResourceClaim>
  declare_resources(const std::string& root) const override;
  std::vector<vpipe::StageHolding>
  declare_holdings(const std::string& root) const override;
  std::size_t latent_bytes(const std::string& root, int width,
                           int height) const override;

  std::unique_ptr<vpipe::genai::ImageGenerator>
  load(const vpipe::genai::ImageModelCreateArgs& args) override;
};

}  // namespace ming

#endif
