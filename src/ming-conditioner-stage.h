#ifndef VPIPE_MING_CONDITIONER_STAGE_H
#define VPIPE_MING_CONDITIONER_STAGE_H

// `ming-image-conditioner`: a prompt becomes the conditioning
// `generate-image` hands the Ming-Image family (the contract in
// ming-family.h). A stage of this plugin's own because the host's
// `diffusion-conditioner` has no seam for a third-party encoder, and
// this one is not a text encoder in the host's sense: a 16B MoE MLLM
// with learned query tokens, a second (connector) transformer, and two
// output streams packed into one tensor.
//
// The encoder is the biggest thing a Ming graph loads (~32 GB of bf16
// MLLM) and is idle for the whole denoise, so what happens to it between
// prompts is the difference between fitting and thrashing -- see
// `unload_when_idle`. The conditioning depends on the prompt alone, not
// the seed or size, so the last prompt's is kept and a repeat costs
// nothing.

#include "ming-config.h"
#include "ming-encoder.h"

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage-spec.h"
#include "pipeline/typed-stage.h"
#include "stages/model-memory.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ming {

class MingConditionerStage final
  : public vpipe::TypedStage<MingConditionerStage> {
public:
  static constexpr const char* kTypeName = "ming-image-conditioner";

  MingConditionerStage(const vpipe::SessionContextIntf* session,
                       std::string                      id,
                       std::vector<vpipe::InEdge>       iports,
                       vpipe::FlexData                  config);
  ~MingConditionerStage() override;

  const vpipe::StageSpec& spec() const noexcept override;
  static const vpipe::StageSpec* stage_spec() noexcept;

  vpipe::Job process(vpipe::RuntimeContext& ctx) override;

  // A model-select constant, latched BEFORE the planning phase so the
  // claims below name the checkpoint this graph will run.
  void apply_constant(unsigned iport, const vpipe::FlexData& beat) override;

  std::vector<vpipe::ResourceClaim> declare_resources() const override;
  vpipe::StageMemory declare_memory() const override;
  void reset_run_state() override;

private:
  std::string model_root_() const;
  bool ensure_loaded_(const Checkpoint& ck);
  // Whether the encoder is let go after each prompt. Decided from the
  // config, or for "auto" from whether it and the DiT fit the box
  // together -- against TOTAL RAM, so every run of a graph answers the
  // same way.
  bool releases_(const Checkpoint& ck) const;
  bool streams_(const Checkpoint& ck) const;

  std::string _hf_dir;
  vpipe::model_memory::UnloadPolicy _unload_cfg =
      vpipe::model_memory::UnloadPolicy::kAuto;
  std::string _stream_cfg = "auto";

  std::unique_ptr<MingEncoder> _enc;
  std::string _loaded_root;
  // The last prompt's conditioning: a repeat is free.
  std::string _last_prompt, _last_root;
  MingEncoder::Result _last;
  std::uint64_t _emitted = 0;
};

}  // namespace ming

#endif
