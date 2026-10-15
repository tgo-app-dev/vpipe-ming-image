// vpipe-ming-image -- inclusionAI's Ming-Image-0.1-Design text-to-image
// model as a vpipe PLUGIN.
//
// What it registers, and why each is the right seam:
//
//   register_image_family     Ming-Image is an IMAGE family, so it plugs
//                             into `generate-image`: the same ports, the
//                             same latent out. The family owns its whole
//                             denoise loop below that.
//   register_catalog_entries  so the checkpoint is discoverable and
//                             fetchable like a built-in's.
//   register_quantize_family  so `model-quantize` writes a w8 / w4 pack
//                             of the DiT as a self-contained checkpoint,
//                             found by the structural identity test.
//   register_family_profile   `detect` only: labels and modalities for a
//                             model already typed "ming-image".
//
// What it does NOT register:
//
//   * NO QUANTIZE PROFILE, and NO CLASS NAME in the detect profile. The
//     host matches both on the DiT's `_class_name`, which here is the
//     generic "DiffusionTransformer" (ming-profiles.h).
//   * NO VAE FAMILY. The checkpoint's VAE is diffusers'
//     AutoencoderKLQwenImage at 4 channels (RGBA), which the host's
//     `vae-decode` already decodes -- given that it reads
//     `input_channels` and a scalar `scaling_factor`, which is a host
//     fix rather than a second copy of a 2k-line VAE here.
//
// The conditioner (`ming-image-conditioner`: a BailingMoeV2 MLLM and a
// Qwen2 connector) arrives with the port that needs it. The DiT's own
// kernels are registered below (register_metal_library).

#include "plugin/plugin-abi.h"
#include "plugin/plugin-context.h"

#include "ming-catalog.h"
#include "ming-conditioner-stage.h"
#include "ming-dit.h"
#include "ming-family.h"
#include "ming-model-config-stage.h"
#include "ming-ops.h"
#include "ming-profiles.h"
#include "ming-quant-family.h"

#include "generative-models/detect-profile.h"

#include <memory>
#include <string>

// The DiT's own kernels (src/metal/ming_ops.metal), embedded by
// vpipe_add_metal_library.
extern "C" const unsigned char ming_ops_bf16_metallib[];
extern "C" const unsigned long ming_ops_bf16_metallib_len;
extern "C" const unsigned char ming_ops_f16_metallib[];
extern "C" const unsigned long ming_ops_f16_metallib_len;
// The conditioning encoder's kernels (src/metal/ming_llm.metal).
extern "C" const unsigned char ming_llm_metallib[];
extern "C" const unsigned long ming_llm_metallib_len;

// Beyond the ABI integer: the DiT dispatches host kernels by name --
// including, through the toolkit's runtime LoRA, the accumulate GEMM
// twin that revision 2 of the kernel contract added -- and reads its
// acceleration tiers out of the request's FlexData bag.
static const char* const kRequires[] = {
    VPIPE_FEATURE_KERNEL_CONTRACT,
    VPIPE_FEATURE_KERNEL_CONTRACT_2,
    VPIPE_FEATURE_ACCEL_BAG,
    nullptr,
};

static const VpipePluginInfo kInfo = {
    VPIPE_PLUGIN_INFO_SCHEMA,
    "ming-image",
    "0.1.0",
    "T-Go LLC",
    "Apache-2.0 (plugin); the Ming-Image weights are published by "
    "inclusionAI under the MIT License",
    "Ming-Image-0.1-Design: a 6B text-to-image model for UI, posters and "
    "other text-rich designs, with RGBA output, for generate-image",
    kRequires,
};

static void
ming_register(vpipe::VpipePluginContext* ctx)
{
  if (ctx == nullptr) { return; }
  // The family FIRST: generate-image consults the registry before its
  // built-in dispatch, and without this a Ming directory falls through
  // to the host's "krea2" default and fails on missing tensors.
  ctx->register_image_family(std::make_unique<ming::MingImageFamily>());
  // The DiT's own kernels, under the names it resolves with
  // load_library(). Without them the DiT REFUSES to load (it validates
  // every function), rather than dispatching nothing. Registration is
  // first-wins and cannot shadow a built-in name.
  ctx->register_metal_library(ming::kMetalLibBf16, ming_ops_bf16_metallib,
                              ming_ops_bf16_metallib_len);
  ctx->register_metal_library(ming::kMetalLibF16, ming_ops_f16_metallib,
                              ming_ops_f16_metallib_len);
  // The conditioning encoder's, likewise validated at its load.
  ctx->register_metal_library(ming::kLlmMetalLib, ming_llm_metallib,
                              ming_llm_metallib_len);
  // The conditioning encoder: the host's diffusion-conditioner has no
  // seam for a third-party encoder, and this one is a 16B MoE MLLM with
  // a second (connector) transformer behind it. Registered through the
  // plugin path so its StageSpec reaches the composer.
  ctx->register_stage<ming::MingConditionerStage>(
      ming::MingConditionerStage::stage_spec());
  // The family's own knobs (the static shift override and the reference
  // pipeline's two guidance controls), on generate-image's model_config
  // port rather than as keys of generate-image.
  ctx->register_stage<ming::MingModelConfigStage>(
      &ming::MingModelConfigStage::stage_spec());
  ctx->register_catalog_entries(ming::catalog_entries());
  // `model-quantize`: a w8 / w4 pack of the DiT as a self-contained
  // checkpoint. Found by the same STRUCTURAL identity test the image
  // family loads by -- not by a quantize profile, which the host keys on
  // the DiT's class name, here the generic "DiffusionTransformer"
  // (ming-quant-family.h).
  ctx->register_quantize_family(std::make_unique<ming::MingQuantFamily>());
  // Labels and modalities for a model already TYPED "ming-image" (a
  // fetch, a pack, an explicit model-register). Deliberately no class
  // name, so detection never labels a checkpoint Ming-Image by its DiT's
  // generic class alone (ming-profiles.h). Cosmetic, so a host without
  // profiles keeps the plugin rather than refusing it.
  if (vpipe::host_has_feature(VPIPE_FEATURE_FAMILY_PROFILES)) {
    ctx->register_family_profile(
        std::string(vpipe::genai::detect::kDomain), ming::kModelType,
        ming::detect_profile());
  }
}

VPIPE_PLUGIN_DEFINE(&kInfo, ming_register)
