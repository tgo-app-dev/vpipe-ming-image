// The encoder's metallib bytes, and their registration for a host that
// has no plugin context. Kept out of the shared model objects so a test
// that never runs the encoder does not have to link the metallib.

#include "ming-ops.h"

// Embedded by vpipe_add_metal_library (CMakeLists.txt).
extern "C" const unsigned char ming_llm_metallib[];
extern "C" const unsigned long ming_llm_metallib_len;

namespace ming {

bool
register_llm_kernels(vpipe::metal_compute::MetalCompute* mc)
{
  if (mc == nullptr) { return false; }
  return mc->register_metal_library(kLlmMetalLib, ming_llm_metallib,
                                    (std::size_t)ming_llm_metallib_len);
}

}  // namespace ming
