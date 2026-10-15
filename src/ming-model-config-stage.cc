#include "ming-model-config-stage.h"

#include "common/flex-data.h"
#include "stages/model-registry.h"

#include <string>
#include <utility>

namespace ming {

using vpipe::ConfigKey;
using vpipe::ConfigType;
using vpipe::FlexData;
using vpipe::FlexDataPayload;
using vpipe::PortSpec;
using vpipe::StageCategory;
using vpipe::StageSpec;
namespace model_config = vpipe::model_config;

namespace {

// Slots 3..8 read exactly as `lora2` does, so they share one doc.
constexpr const char* kMoreLoraDoc =
    "one more runtime LoRA slot, read exactly as `lora2` is: applied "
    "alongside the others, the same accepted forms, LOAD-time, with its "
    "own live strength in the matching `_scale` key. Up to eight "
    "adapters run at once on a host whose DiT holds eight; each costs "
    "one more pair of skinny GEMMs";
constexpr const char* kMoreLoraScaleDoc =
    "this slot's strength, per FORWARD and independent of every other "
    "slot's. 0 skips its two GEMMs";

const ConfigKey kAttrs[] = {
  {.key = kCfgShift, .type = ConfigType::Real, .required = false,
   .doc = "override the static time shift. Unset => the checkpoint's own "
          "(scheduler_config.json: 6.0). The reference runs this static "
          "shift at every size; a scheduler-select beat's shift is NOT "
          "taken, because that stage emits 1.15 when nothing was set"},
  {.key = kCfgTrunc, .type = ConfigType::Real, .required = false,
   .doc = "CFG truncation, the reference pipeline's: guidance is switched "
          "off once (1 - sigma) passes this value, which also skips the "
          "negative pass. Unset => 1.0, never. Only matters with "
          "generate-image guidance_scale > 1"},
  {.key = kCfgNorm, .type = ConfigType::Real, .required = false,
   .doc = "CFG renormalization, the reference pipeline's: the guided "
          "prediction's norm is clamped to this multiple of the conditional "
          "one's. Unset => off. Only matters with guidance_scale > 1"},
  {.key = kCfgLora, .type = ConfigType::String, .required = false,
   .doc = "a LoRA applied AT RUNTIME -- every adapted projection computes "
          "W x + scale * B (A x) rather than having the delta folded into "
          "the weights, which on a quantized DiT is the difference between "
          "keeping a small correction and rounding it away. A registered "
          "model, a directory holding one .safetensors, or a path to one, "
          "in diffusers / peft or kohya spelling; it adapts the attention "
          "(to_q/k/v/out) and feed-forward (w1/w3/w2) projections of every "
          "block. An adapter that binds NO module of this model is reported "
          "and the base model runs. LOAD-TIME: a beat that changes it once "
          "the DiT is up is reported and ignored. generate-image's own "
          "`lora` does not reach this family; this one does. Unset => no "
          "adapter",
   .suggest_db = vpipe::kModelRegistryDb,
   .suggest_db_type = "ming-image-lora",
   .is_path = true, .path_filter = "weights"},
  {.key = kCfgLoraScale, .type = ConfigType::Real, .required = false,
   .doc = "the adapter's strength, applied PER FORWARD. Live: it rides the "
          "GEMM as a constant, so it can be swept without a reload. 1.0 = as "
          "trained; 0 skips both adapter GEMMs, so off is exactly off",
   .def_real = 1.0},
  {.key = kCfgLora2, .type = ConfigType::String, .required = false,
   .doc = "a SECOND runtime LoRA beside the first -- a few-step adapter in "
          "`lora` and a style one here, say; either slot may hold either. "
          "Same accepted forms as `lora`, and LOAD-time in the same way",
   .suggest_db = vpipe::kModelRegistryDb,
   .suggest_db_type = "ming-image-lora",
   .is_path = true, .path_filter = "weights"},
  {.key = kCfgLora2Scale, .type = ConfigType::Real, .required = false,
   .doc = "`lora2`'s strength, per FORWARD and independent of `lora_scale`. "
          "0 skips its two GEMMs", .def_real = 1.0},
  // Slots 3..8, which read exactly as `lora2` does.
  {.key = kCfgLoraKeys[2][0], .type = ConfigType::String,
   .required = false, .doc = kMoreLoraDoc,
   .suggest_db = vpipe::kModelRegistryDb,
   .suggest_db_type = "ming-image-lora",
   .is_path = true, .path_filter = "weights"},
  {.key = kCfgLoraKeys[2][1], .type = ConfigType::Real, .required = false,
   .doc = kMoreLoraScaleDoc, .def_real = 1.0},
  {.key = kCfgLoraKeys[3][0], .type = ConfigType::String,
   .required = false, .doc = kMoreLoraDoc,
   .suggest_db = vpipe::kModelRegistryDb,
   .suggest_db_type = "ming-image-lora",
   .is_path = true, .path_filter = "weights"},
  {.key = kCfgLoraKeys[3][1], .type = ConfigType::Real, .required = false,
   .doc = kMoreLoraScaleDoc, .def_real = 1.0},
  {.key = kCfgLoraKeys[4][0], .type = ConfigType::String,
   .required = false, .doc = kMoreLoraDoc,
   .suggest_db = vpipe::kModelRegistryDb,
   .suggest_db_type = "ming-image-lora",
   .is_path = true, .path_filter = "weights"},
  {.key = kCfgLoraKeys[4][1], .type = ConfigType::Real, .required = false,
   .doc = kMoreLoraScaleDoc, .def_real = 1.0},
  {.key = kCfgLoraKeys[5][0], .type = ConfigType::String,
   .required = false, .doc = kMoreLoraDoc,
   .suggest_db = vpipe::kModelRegistryDb,
   .suggest_db_type = "ming-image-lora",
   .is_path = true, .path_filter = "weights"},
  {.key = kCfgLoraKeys[5][1], .type = ConfigType::Real, .required = false,
   .doc = kMoreLoraScaleDoc, .def_real = 1.0},
  {.key = kCfgLoraKeys[6][0], .type = ConfigType::String,
   .required = false, .doc = kMoreLoraDoc,
   .suggest_db = vpipe::kModelRegistryDb,
   .suggest_db_type = "ming-image-lora",
   .is_path = true, .path_filter = "weights"},
  {.key = kCfgLoraKeys[6][1], .type = ConfigType::Real, .required = false,
   .doc = kMoreLoraScaleDoc, .def_real = 1.0},
  {.key = kCfgLoraKeys[7][0], .type = ConfigType::String,
   .required = false, .doc = kMoreLoraDoc,
   .suggest_db = vpipe::kModelRegistryDb,
   .suggest_db_type = "ming-image-lora",
   .is_path = true, .path_filter = "weights"},
  {.key = kCfgLoraKeys[7][1], .type = ConfigType::Real, .required = false,
   .doc = kMoreLoraScaleDoc, .def_real = 1.0},
};
const PortSpec kIports[] = {
  {.name = "trigger",
   .doc = "OPTIONAL beat that gates re-emitting the config. Any payload -- "
          "receipt is the signal. Unwired, the stage emits once for the run",
   .type = nullptr, .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "model_config",
   .doc = "Ming-Image parameters as one FlexData object {model_family: "
          "ming-image, +shift, cfg_truncation, cfg_normalization, lora .. "
          "lora8 and their scales when set}. Wire it to generate-image's "
          "model_config port",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = MingModelConfigStage::kTypeName,
  .doc       = "Source: the Ming-Image-specific parameters -- the static "
               "time shift, the reference pipeline's two guidance "
               "controls, and up to two runtime LoRAs with live strengths. "
               "One beat then done; with a trigger iport, one beat per "
               "inbound beat.",
  .display_name = "Ming-Image Model Config",
  .category  = StageCategory::ModelSpecificConfig,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

MingModelConfigStage::MingModelConfigStage(
    const vpipe::SessionContextIntf* s,
    std::string                      id,
    std::vector<vpipe::InEdge>       iports,
    FlexData                         config)
  : vpipe::ModelConfigSourceStage<MingModelConfigStage>(
        s, std::move(id), std::move(iports), std::move(config))
{
  allocate_oports(spec().oports.size());
}

const StageSpec&
MingModelConfigStage::spec() const noexcept
{
  return kSpec;
}

const StageSpec&
MingModelConfigStage::stage_spec() noexcept
{
  return kSpec;
}

FlexData
MingModelConfigStage::resolved_config() const
{
  FlexData fd = model_config::make_config("ming-image");
  auto o = fd.as_object();
  // FORWARDED ONLY WHEN SET: an unset key must leave the family's own
  // value alone rather than replace it with a default that merely looks
  // configured.
  const FlexData& c = config();
  if (c.is_object()) {
    const auto co = c.as_object();
    for (const char* k : {kCfgShift, kCfgTrunc, kCfgNorm}) {
      if (!co.contains(k)) { continue; }
      const double v = co.at(k).as_real(0.0);
      if (v > 0.0) { o.insert_or_assign(k, FlexData::make_real(v)); }
    }
    // The adapters on the same rule. A strength is forwarded as it is --
    // 0 (off) and a negative one are both requests, not unset values.
    for (const auto& kk : kCfgLoraKeys) {
      if (co.contains(kk[0])) {
        o.insert_or_assign(
            kk[0], FlexData::make_string(std::string(
                       co.at(kk[0]).as_string(""))));
      }
      if (co.contains(kk[1])) {
        o.insert_or_assign(kk[1],
                           FlexData::make_real(co.at(kk[1]).as_real(1.0)));
      }
    }
  }
  return fd;
}

}  // namespace ming
