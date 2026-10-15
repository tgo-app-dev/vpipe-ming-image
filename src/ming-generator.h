#ifndef VPIPE_MING_GENERATOR_H
#define VPIPE_MING_GENERATOR_H

// One loaded Ming-Image DiT and its denoise loop (docs/MING-IMAGE.md
// section 3):
//
//   * STATIC shift (6.0, from the checkpoint's scheduler_config). The
//     reference's dynamic-shifting override is a silent no-op.
//   * `steps` COUNTS DIT FORWARDS, as everywhere in vpipe. The
//     reference's "12 steps" is 11 forwards -- its last runs at sigma 0
//     with dt 0 -- so the default is 11 and the schedule is
//     shift(linspace(1, 1/S, S)) + [0].
//   * The DiT is fed 1 - sigma, its output is NEGATED, Euler in f32.
//   * Noise is torch's CPU randn(seed) in NCHW, bit for bit.
//   * CFG engages only above 1, as `pos + g * (pos - neg)`, with ZEROS
//     as the negative unless the graph wired one -- which is what the
//     reference does (it has no negative prompt).

#include "ming-config.h"
#include "ming-dit.h"

#include "common/vpipe-format.h"
#include "generative-models/image-model-registry.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace ming {

// The reference's 12 sampling steps, as DiT forwards.
inline constexpr int kDefaultSteps = 11;

// The sigma grid for `steps` forwards plus the terminal 0, as diffusers'
// FlowMatchEulerDiscreteScheduler computes it (double, then f32).
std::vector<float> schedule_sigmas(int steps, float shift);

class MingImageGenerator final : public vpipe::genai::ImageGenerator {
public:
  static std::unique_ptr<MingImageGenerator>
  load(const Checkpoint& ck, const vpipe::genai::ImageModelCreateArgs& args);

  int latent_channels() const override { return _ck.dit.in_channels; }
  int spatial_compression() const override { return kVaeScale; }

  bool generate(const vpipe::genai::ImageGenRequest& req,
                vpipe::genai::ImageGenResult* out) override;

  void release_idle() override;
  std::uint64_t resident_bytes() const override;

private:
  MingImageGenerator() = default;
  void warn_(const std::string& m) const;
  void warn_(const vpipe::VpipeFormat& m) const;
  // Once per distinct message for this generator: a graph that pins its
  // steps through scheduler-select would otherwise repeat the same note
  // every image.
  void warn_once_(const std::string& m);
  // The adapter keys of a model_config beat: a strength is applied LIVE,
  // a path that differs from the one loaded is reported and ignored.
  void apply_lora_keys_(const vpipe::FlexData* cfg);

  Checkpoint                          _ck;
  vpipe::metal_compute::MetalCompute* _mc = nullptr;
  const vpipe::SessionContextIntf*    _session = nullptr;
  std::unique_ptr<MingDit>            _dit;
  std::set<std::string>               _noted;
  // What the graph named per slot at load (`lora` .. `lora8`), as
  // written -- a registry key or a path -- and the strength now in force
  // (create() sets every slot's).
  std::string                         _lora_ref[MingDit::kMaxLoraSlots];
  float _lora_scale[MingDit::kMaxLoraSlots] = {};
};

}  // namespace ming

#endif
