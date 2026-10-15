#ifndef VPIPE_MING_PROFILES_H
#define VPIPE_MING_PROFILES_H

// What this family tells the host about its checkpoints through the
// open profile bag (generative-models/family-profile.h) -- and, as
// importantly, what it deliberately does NOT tell it.
//
// THE `detect` DOMAIN, WITHOUT A CLASS NAME. The host recognises a
// diffusers checkpoint on disk by ONE fact: its transformer
// `_class_name`, matched against the profiles' `class_names`
// (model-detect's dit_tag_ -> detect::model_type_for_class). Nothing
// structural is consulted. This DiT's class is "DiffusionTransformer",
// diffusers' generic word -- a lookalike with that class and none of the
// Ming contract would be labelled Ming-Image, and a picker would then
// offer it to a stage that refuses it. So the profile names no class.
//
// What it still does, safely, is answer for a model whose TYPE is
// already known to be "ming-image": a catalogue fetch, a model-quantize
// pack (registered under this family's tag), or a model-register with
// an explicit `model_type`. Those get the family / version labels and
// the text -> image modalities from here (detect::for_model_type,
// catalog_default_io), which no host table can know for a plugin.
//
// A Ming-Image directory that arrives any other way is NOT recognised
// by the host today. The fix belongs in the host, not in a looser key
// here: model-detect asking ImageModelRegistry::claim_for(root) -- the
// structural test this family loads by -- when no class name answers,
// and taking the claiming family's tag as the type. The labels and
// modalities below would then apply to it with no change on this side.
//
// THE `quantize` DOMAIN IS NOT REGISTERED, for the same class-name
// reason: see ming-quant-family.h, which takes the structural seam.

#include "common/flex-data.h"

namespace ming {

// The runtime model type: the image family's tag, the catalogue row's
// `model_type`, and what ming-image-conditioner's picker filters on.
inline constexpr const char* kModelType = "ming-image";

// `_vpipe_component` stamped into a quantized transformer/config.json.
inline constexpr const char* kDitComponentTag = "ming-image-dit";

// The `detect` domain's bag (generative-models/detect-profile.h).
vpipe::FlexData detect_profile();

}  // namespace ming

#endif
