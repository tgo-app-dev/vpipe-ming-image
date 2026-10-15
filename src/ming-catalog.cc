#include "ming-catalog.h"

#include <string>
#include <utility>

namespace ming {

std::vector<vpipe::ModelCatalogEntry>
catalog_entries()
{
  std::vector<vpipe::ModelCatalogEntry> v;
  vpipe::ModelCatalogEntry e;
  e.family      = "Ming-Image";
  e.version     = "0.1";
  e.param_class = "6B";
  e.variant     = "Design (bf16)";
  e.hf_path     = "inclusionAI/Ming-Image-0.1-Design";
  e.model_type  = "ming-image";
  e.inputs      = {"text"};
  e.outputs     = {"image"};
  // Pinned so a fetch skips the repo's showcase assets. The MLLM's
  // vision tower ships inside its shards and is never run for
  // text-to-image; it cannot be skipped without repacking.
  e.files = {
      "README.md",
      "LICENSE",
      "scheduler/scheduler_config.json",
      "transformer/config.json",
      "transformer/diffusion_pytorch_model.safetensors.index.json",
      "transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
      "transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
      "transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
      "transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
      "transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
      "vae/config.json",
      "vae/diffusion_pytorch_model.safetensors",
      "mlp/config.json",
      "mlp/model.safetensors",
      "connector/config.json",
      "connector/generation_config.json",
      "connector/model.safetensors.index.json",
      "connector/model-00001-of-00002.safetensors",
      "connector/model-00002-of-00002.safetensors",
      "mllm/config.json",
      "mllm/preprocessor_config.json",
      "mllm/special_tokens_map.json",
      "mllm/tokenizer.json",
      "mllm/tokenizer_config.json",
      "mllm/model.safetensors.index.json",
      "mllm/model-00001-of-00007.safetensors",
      "mllm/model-00002-of-00007.safetensors",
      "mllm/model-00003-of-00007.safetensors",
      "mllm/model-00004-of-00007.safetensors",
      "mllm/model-00005-of-00007.safetensors",
      "mllm/model-00006-of-00007.safetensors",
      "mllm/model-00007-of-00007.safetensors",
  };
  e.needs_tokenizer_json = false;
  v.push_back(std::move(e));
  return v;
}

}  // namespace ming
