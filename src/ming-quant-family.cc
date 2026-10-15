#include "ming-quant-family.h"

#include "ming-config.h"
#include "ming-profiles.h"

namespace ming {

using vpipe::genai::QuantizableComponent;

std::string_view
MingQuantFamily::tag() const noexcept
{
  return kModelType;
}

bool
MingQuantFamily::claims(const std::string& root) const
{
  // The image family's own identity test: the transformer declares the
  // Ming checkpoint contract, the MLLM is a BailingMM2, and connector/,
  // mlp/ and the QwenImage VAE are there. A handful of small JSON reads.
  //
  // It holds for a pack too, which is what lets a second pass (or a
  // re-run with skip_existing) find its own output: model-quantize
  // rewrites transformer/config.json with every key kept and a
  // `quantization` block added, and copies the other components whole.
  const Checkpoint ck = read_checkpoint(root);
  return ck.valid && ck.text_to_image();
}

std::vector<QuantizableComponent>
MingQuantFamily::components() const
{
  QuantizableComponent dit;
  dit.target = "dit";
  dit.role   = "transformer";
  // A DIFFUSERS directory, not a Comfy-Org repack: there is no single
  // component file to pick, and an empty metadata key matches none of
  // the shards. The stage then takes `<root>/transformer` itself (its
  // config.json is there), which is the directory checkpoint wanted.
  dit.meta_key.clear();
  // The selection rule. `scope` is a SUBSTRING and every tensor name
  // has a dot, so "." puts the whole DiT in scope; `all_in_scope` takes
  // every 2-D float tensor whose leaf is not a norm or an embedding and
  // whose width the group divides. In this checkpoint that is exactly
  // the block stacks' matrices plus the four model-level weights named
  // in `exclude` -- the same set the in-tree Z-Image leaf list selects
  // (ming-quant-family.h). A single prefix cannot name the three stacks
  // (layers., noise_refiner., context_refiner.), which is why the rule
  // is spelled as "everything but the model level" rather than "the
  // stacks"; the pack test checks the outcome tensor by tensor.
  dit.scope        = ".";
  dit.all_in_scope = true;
  dit.exclude = "all_x_embedder,cap_embedder,t_embedder.,all_final_layer";
  // Stamped into the pack's transformer/config.json, so a DiT directory
  // that leaves this pipeline still says what it was.
  dit.component_tag = kDitComponentTag;
  return {dit};
}

}  // namespace ming
