#include "ming-family.h"

#include "ming-config.h"
#include "ming-dit.h"
#include "ming-encoder.h"
#include "ming-generator.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-memory.h"

#include <filesystem>


namespace ming {

using vpipe::fmt;

bool
MingImageFamily::claims(const std::string& root,
                        const std::string& model_type) const
{
  (void)model_type;
  // Structural only. The DiT's class name is diffusers' generic
  // "DiffusionTransformer", so a model-type hint or a class-name match
  // alone would claim directories that are not ours.
  return is_ming_image(root);
}

void
MingImageFamily::size_grid(const std::string& root, int* gh, int* gw) const
{
  const Checkpoint ck = read_checkpoint(root);
  const int g = kVaeScale * (ck.valid ? ck.dit.patch : 2);
  if (gh != nullptr) { *gh = g; }
  if (gw != nullptr) { *gw = g; }
}

int
MingImageFamily::latent_scale(const std::string& root) const
{
  (void)root;
  return kVaeScale;
}

std::vector<vpipe::ResourceClaim>
MingImageFamily::declare_resources(const std::string& root) const
{
  const Checkpoint ck = read_checkpoint(root);
  if (!ck.valid) { return {}; }
  // STREAMABLE: a graph is judged at this claim's floor, and judging a
  // 12 GB DiT that can run from its trunk and two slots at its full size
  // refuses boxes that could run it.
  // The DiT's OWN floor: its streamed stack includes the noise refiner
  // blocks, which a per-stem helper on "layers." would count as trunk.
  const std::size_t floor =
      MingDit::streaming_floor_bytes(ck.transformer_dir);
  namespace mm = vpipe::model_memory;
  std::vector<vpipe::ResourceClaim> out;
  if (floor == 0) {
    out = mm::weight_claims({ck.transformer_dir});
  } else {
    out.push_back(mm::weight_claim_streamable(ck.transformer_dir, floor));
  }
  // THE ENCODER, in the CONDITION phase -- the least it needs, since the
  // prompt is encoded before the first step. Claiming it here is what
  // tells generate-image this family answers for it: without that the
  // stage books the encoder directory for the whole run on the family's
  // behalf, and a 32 GB MLLM that `ming-image-conditioner` releases after
  // the prompt still pushes the DiT into streaming. The conditioner adds
  // its own every-phase claims when it keeps the encoder.
  vpipe::ResourceClaim llm = mm::weight_claim_streamable(
      ck.mllm_dir, MingEncoder::streaming_floor_bytes(ck));
  llm.phase = std::string(mm::kPhaseCondition);
  out.push_back(std::move(llm));
  for (auto& c : mm::weight_claims_in_phase({ck.connector_dir, ck.mlp_dir},
                                            mm::kPhaseCondition)) {
    out.push_back(std::move(c));
  }
  // THE ANE FEED-FORWARD MODULE (`ane_ffn`): ONE unit in the denoise
  // phase, in UNITS because CoreML holds those bytes and no other ledger
  // can see them. Booked UNCONDITIONALLY: this call sees neither the
  // graph's accel bag nor the stage's id. A current generate-image keeps
  // it only when the graph asked for the tier, RELABELS it to itself and
  // hands that label to load (create_args::kCoreMLLabel); an older one
  // passes it through as it is, and the DiT then reads the grant back
  // under kAneClaimLabel and WITHDRAWS the booking when the graph did not
  // ask (ming-dit.cc, "the PLAN's answer").
  for (auto& c : mm::coreml_claims(kAneClaimLabel,
                                   MingDit::ane_runtime_bytes(ck.dit), 1,
                                   mm::kPhaseDenoise)) {
    out.push_back(std::move(c));
  }
  return out;
}

std::vector<vpipe::StageHolding>
MingImageFamily::declare_holdings(const std::string& root) const
{
  // The SAME directory the claim names: the plan merges holdings by
  // source.
  const Checkpoint ck = read_checkpoint(root);
  if (!ck.valid) { return {}; }
  const std::size_t total =
      vpipe::model_memory::dir_weights_bytes(ck.transformer_dir);
  if (total == 0) { return {}; }
  vpipe::StageHolding h;
  h.source = ck.transformer_dir;
  h.preload = total;
  const std::size_t floor =
      MingDit::streaming_floor_bytes(ck.transformer_dir);
  h.floor = floor > 0 && floor < total ? floor : total;
  return {h};
}

std::size_t
MingImageFamily::latent_bytes(const std::string& root, int width,
                              int height) const
{
  const Checkpoint ck = read_checkpoint(root);
  if (!ck.valid || width <= 0 || height <= 0) { return 0; }
  // f32, which is what generate-image publishes and vae-decode reads.
  return (std::size_t)ck.dit.in_channels * (std::size_t)(height / kVaeScale) *
         (std::size_t)(width / kVaeScale) * 4;
}

std::unique_ptr<vpipe::genai::ImageGenerator>
MingImageFamily::load(const vpipe::genai::ImageModelCreateArgs& args)
{
  const vpipe::SessionContextIntf* session = args.session;
  auto warn = [&](const vpipe::VpipeFormat& m) {
    if (session != nullptr) { session->warn(m); }
  };
  Checkpoint ck = read_checkpoint(args.root);
  if (!ck.valid) {
    warn(fmt("ming-image: '{}' is not a readable Ming-Image checkpoint: {}",
             args.root, ck.why));
    return nullptr;
  }
  if (!ck.text_to_image()) {
    // CLAIMED, so a host family does not load a foreign config over these
    // weights -- and refused by name, rather than run as text-to-image.
    warn(fmt("ming-image: '{}' is a layer-decomposition checkpoint "
             "(alignment_padding_mode '{}', multi_frame_output {}); this "
             "plugin runs the text-to-image checkpoint only",
             args.root, ck.dit.padding_mode, ck.dit.multi_frame_output));
    return nullptr;
  }
  if (args.metal == nullptr) {
    warn(fmt("ming-image: no metal-compute backend; inert"));
    return nullptr;
  }
  // A DiT NAMED ON ITS OWN (generate-image's `dit_dir`): a quantized pack,
  // or a pack's transformer/. Everything else -- the encoder, the VAE,
  // the schedule -- stays the root's. Left unread, a `dit_dir` pointing
  // at a w8 pack silently ran the root's bf16 DiT.
  if (!args.dit_dir.empty()) {
    namespace fs = std::filesystem;
    fs::path d(args.dit_dir);
    std::error_code ec;
    if (fs::exists(d / "transformer" / "config.json", ec)) {
      d /= "transformer";
    }
    DitConfig alt;
    std::string why;
    if (!read_dit_config(d.string(), &alt, &why)) {
      warn(fmt("ming-image: dit_dir '{}' is not a Ming-Image DiT ({}); "
               "inert", args.dit_dir, why));
      return nullptr;
    }
    // The SAME model, possibly quantized: everything that shapes the
    // sequence and the conditioning contract must agree, or the root's
    // conditioner and VAE are talking to a different network.
    if (alt.dim != ck.dit.dim || alt.n_layers != ck.dit.n_layers ||
        alt.n_refiner_layers != ck.dit.n_refiner_layers ||
        alt.n_heads != ck.dit.n_heads ||
        alt.in_channels != ck.dit.in_channels ||
        alt.cap_feat_dim != ck.dit.cap_feat_dim ||
        alt.padding_mode != ck.dit.padding_mode ||
        alt.multi_frame_output != ck.dit.multi_frame_output) {
      warn(fmt("ming-image: dit_dir '{}' is not this checkpoint's DiT "
               "(its geometry or padding contract differs); inert",
               args.dit_dir));
      return nullptr;
    }
    ck.transformer_dir = d.string();
    ck.dit = alt;
  }
  return MingImageGenerator::load(ck, args);
}

}  // namespace ming
