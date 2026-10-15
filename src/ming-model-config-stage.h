#ifndef VPIPE_MING_MODEL_CONFIG_STAGE_H
#define VPIPE_MING_MODEL_CONFIG_STAGE_H

// `ming-image-model-config`: the Ming-Image family's own knobs, as one
// model_config beat for generate-image's model_config port. They do not
// go on generate-image itself: none of them means anything to another
// family, and a key that applies to one family of many is silently inert
// on the rest.
//
// The runtime LoRA keys live here too: generate-image's own `lora` keys
// reach only its built-in families, never a registered one, so this is
// the one place a Ming-Image graph can name an adapter.
//
// Why `shift` lives HERE and not on a scheduler-select beat: that stage
// always emits a shift -- 1.15 when nothing was set -- so its beat
// cannot say whether a shift was chosen. Taking it would run this
// checkpoint (static shift 6.0) at 1.15 whenever a graph pins its steps
// through scheduler-select. A shift here is only ever forwarded when it
// was actually set.

#include "stages/model-config-source.h"

#include <string>
#include <vector>

namespace ming {

// The keys, shared by the stage that writes them and the generator that
// reads them.
inline constexpr const char* kCfgShift   = "shift";
inline constexpr const char* kCfgTrunc   = "cfg_truncation";
inline constexpr const char* kCfgNorm    = "cfg_normalization";
// Runtime LoRA, spelled as the in-tree families spell it (generate-image,
// qwen-image-21-model-config, krea2-model-config): slot 0 is `lora`, slot
// i `lora<i+1>`, eight in all. The paths are LOAD-time, the strengths
// live.
inline constexpr const char* kCfgLora        = "lora";
inline constexpr const char* kCfgLoraScale   = "lora_scale";
inline constexpr const char* kCfgLora2       = "lora2";
inline constexpr const char* kCfgLora2Scale  = "lora2_scale";
// The slots this stage offers. The DiT runs as many as the host's
// lora::Stack holds (eight on a host with the ane-adapters feature, two
// before it); a reader takes the smaller, so a slot past the DiT's is
// reported and dropped rather than read off the end of a table.
inline constexpr int kCfgLoraSlots = 8;
// Indexed by slot, for the readers that loop over them.
inline constexpr const char* kCfgLoraKeys[kCfgLoraSlots][2] = {
    {kCfgLora, kCfgLoraScale},
    {kCfgLora2, kCfgLora2Scale},
    {"lora3", "lora3_scale"},
    {"lora4", "lora4_scale"},
    {"lora5", "lora5_scale"},
    {"lora6", "lora6_scale"},
    {"lora7", "lora7_scale"},
    {"lora8", "lora8_scale"},
};

class MingModelConfigStage final
  : public vpipe::ModelConfigSourceStage<MingModelConfigStage> {
public:
  static constexpr const char* kTypeName = "ming-image-model-config";

  static const vpipe::StageSpec& stage_spec() noexcept;

  MingModelConfigStage(const vpipe::SessionContextIntf* session,
                       std::string                      id,
                       std::vector<vpipe::InEdge>       iports,
                       vpipe::FlexData                  config);

  const vpipe::StageSpec& spec() const noexcept override;

  vpipe::FlexData resolved_config() const;
};

}  // namespace ming

#endif
