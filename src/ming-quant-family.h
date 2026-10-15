#ifndef VPIPE_MING_QUANT_FAMILY_H
#define VPIPE_MING_QUANT_FAMILY_H

// Ming-Image as a QUANTIZABLE family, so `model-quantize` writes a w8 or
// w4 pack of the DiT the way it packages a built-in model: one
// self-contained directory holding the quantized transformer/ and every
// other component untouched (hard-linked on the same volume), usable as
// generate-image's and ming-image-conditioner's `hf_dir` as it stands.
//
// WHY THIS SEAM AND NOT A QUANTIZE PROFILE. The host has two:
//
//   * a quantize PROFILE (quantize-profile.h) is found by the DiT's
//     `_class_name`, matched EXACTLY. This DiT's class is diffusers'
//     generic "DiffusionTransformer", which is not ours to claim: a
//     profile keyed on it would quantize any future checkpoint that
//     carries the same word with this family's scope -- the "plausible
//     checkpoint that is wrong" the header warns about.
//   * a QuantizableFamily (this) is found by `claims(root)`, which reads
//     the checkpoint. It answers with the same structural identity test
//     the image family loads by (ming-config.h), so a directory is
//     quantized as Ming-Image exactly when generate-image would run it
//     as Ming-Image. It is also asked BEFORE the built-in chain.
//
// What the profile path has and this one does not: a leaf list with an
// 8-bit modulation opt-in (`quant_modulation`), per-layer `mixed`
// precision and AWQ. None is needed to mirror the in-tree Z-Image pack
// (below), and AWQ could not be had anyway -- its on-device collectors
// are in-tree code that drives a family's own encoder, and the stage
// refuses AWQ for every out-of-tree family without a `calib_dir`.
//
// WHAT IS QUANTIZED -- the in-tree Z-Image row of model-quantize, which
// is the same NextDiT. Every 2-D matrix of the three block stacks
// (context_refiner, noise_refiner, layers): attention to_q / to_k /
// to_v / to_out.0, feed-forward w1 / w3 / w2, and the per-block
// adaLN_modulation.0 (Z-Image's leaf "0" takes it along with to_out.0,
// so its packs carry it at the body's width too). The vectors -- the
// sandwich norms and the q/k norms -- are 1-D and never quantized.
//
// WHAT STAYS DENSE -- the model level, exactly Z-Image's exclude list:
// all_x_embedder (K = 64: one group, and every latent enters through
// it), cap_embedder (stream A's projection), t_embedder (feeds every
// modulation vector in the model) and all_final_layer (every velocity
// value leaves through its 64 rows). Small, and precision-sensitive.
//
// The vendored DiT reads the result with no configuration: it detects a
// pack by `.scales` beside a block's codes and derives each tensor's
// bits from its own shapes (ming-dit.cc load_qw_).

#include "generative-models/quantize-family-registry.h"

#include <string>
#include <string_view>
#include <vector>

namespace ming {

class MingQuantFamily final : public vpipe::genai::QuantizableFamily {
public:
  // "ming-image": the image family's tag, so the record model-quantize
  // registers for a pack carries the type the pickers filter on.
  std::string_view tag() const noexcept override;

  // The TEXT-TO-IMAGE checkpoint only. The Layer checkpoint
  // (Ming-Image-0.1-Design-Layer) passes the structural test too, and
  // its "learned" padding brings pad-token tensors the scope below has
  // never been checked against -- quantizing a pad row is a silent wrong
  // answer -- and this plugin does not run it anyway.
  bool claims(const std::string& root) const override;

  std::vector<vpipe::genai::QuantizableComponent>
  components() const override;
};

}  // namespace ming

#endif
