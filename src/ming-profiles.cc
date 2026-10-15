#include "ming-profiles.h"

#include "generative-models/detect-profile.h"

#include <string_view>
#include <vector>

namespace ming {

namespace {

void
set_str(vpipe::FlexData& p, std::string_view k, const char* v)
{
  p.as_object().insert_or_assign(k, vpipe::FlexData::make_string(v));
}

void
set_strs(vpipe::FlexData& p, std::string_view k,
         const std::vector<const char*>& vs)
{
  vpipe::FlexData a = vpipe::FlexData::make_array();
  for (const char* v : vs) {
    a.as_array().push_back(vpipe::FlexData::make_string(v));
  }
  p.as_object().insert_or_assign(k, a);
}

}  // namespace

vpipe::FlexData
detect_profile()
{
  namespace d_ = vpipe::genai::detect;
  vpipe::FlexData p = vpipe::FlexData::make_object();
  // NO `class_names` -- see ming-profiles.h. Absent, the host's
  // class-name lookup never reaches this family, so no checkpoint is
  // labelled Ming-Image by its DiT's class alone.
  set_str(p, d_::kModelType, kModelType);
  // The catalogue row's spelling, so a pack or a hand-registered copy
  // sorts beside the fetched checkpoint in a browser.
  set_str(p, d_::kLabelFamily, "Ming-Image");
  set_str(p, d_::kLabelVersion, "0.1");
  // Text to image. The checkpoint's MLLM carries a vision tower, but
  // text-to-image never runs it and nothing here takes a picture in.
  set_strs(p, d_::kInputs, {"text"});
  set_strs(p, d_::kOutputs, {"image"});
  return p;
}

}  // namespace ming
