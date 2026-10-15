// Runtime LoRA through the PRODUCTION plumbing.
//
// The arithmetic is ming-dit-test's -- the DiT against a FUSED reference,
// every publisher spelling, streaming, the ANE's rows. This is the wiring
// that sits above it and that no DiT test can reach:
//
//   * MingImageFamily::load reads `lora` / `lora2` and their strengths off
//     ImageModelCreateArgs.model_config (what ming-image-model-config
//     emits), resolves each with the host's resolver -- a directory
//     holding one .safetensors here -- and binds them POSITIONALLY;
//   * the generator applies a strength from ImageGenRequest.model_config
//     LIVE, per request: 0 must reproduce the un-adapted model EXACTLY;
//   * a beat naming a different adapter after load is IGNORED and SAID so;
//   * an adapter for another model binds nothing and is SAID so.
//
// Two forwards per generation at 512x512 from the fp32 P1 conditioning;
// the latents are compared bit for bit, not to a tolerance.
//
// Needs VPIPE_MING_TEST_MODEL_PATH (the checkpoint) and VPIPE_MING_GOLDEN
// (conditioning/p1/fp32 + dit_real/lora from dump_ming_golden.py --stage
// real_lora); SKIPPED without them.

#include "ming-config.h"
#include "ming-dit.h"
#include "ming-family.h"
#include "ming-generator.h"
#include "ming-model-config-stage.h"
#include "ming-test.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/image-model-registry.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

extern "C" const unsigned char ming_ops_bf16_metallib[];
extern "C" const unsigned long ming_ops_bf16_metallib_len;

namespace fs = std::filesystem;
using vpipe::FlexData;
using vpipe::metal_compute::MetalCompute;

namespace {

// The least session the family runs under: a model manager for the weight
// set, and every line it logs kept, so a test can ask what was SAID.
class Services final : public vpipe::SessionServicesIntf {
 public:
  void* find_service(std::string_view id) const override
  {
    using vpipe::ServiceId;
    if (id == ServiceId<vpipe::genai::GenerativeModelManager>::value) {
      return genai.get();
    }
    if (id == ServiceId<MetalCompute>::value) { return mc; }
    return nullptr;
  }
  std::unique_ptr<vpipe::genai::GenerativeModelManager> genai;
  MetalCompute* mc = nullptr;
};

class Session final : public vpipe::SessionContextIntf {
 public:
  explicit Session(MetalCompute* mc)
  {
    _svc.mc = mc;
    _svc.genai = std::make_unique<vpipe::genai::GenerativeModelManager>(this);
  }
  ~Session() override { _svc.genai.reset(); }
  void error(const vpipe::VpipeFormat& m) const override { out_("E", m); }
  void warn(const vpipe::VpipeFormat& m) const override { out_("W", m); }
  void info(const vpipe::VpipeFormat& m) const override { out_("I", m); }
  void log_debug(const vpipe::VpipeFormat&) const override {}
  void log_verbose(const vpipe::VpipeFormat&) const override {}
  void log_normal(const vpipe::VpipeFormat& m) const override
  {
    out_("L", m);
  }
  void log_always(const vpipe::VpipeFormat& m) const override
  {
    out_("L", m);
  }
  vpipe::ThreadPool* thread_pool() const noexcept override
  {
    return nullptr;
  }
  vpipe::SessionServicesIntf* services() const override { return &_svc; }
  // Every line logged since the last mark() that contains `needle`.
  bool said(const std::string& needle) const
  {
    std::lock_guard<std::mutex> g(_mu);
    for (std::size_t i = _mark; i < _lines.size(); ++i) {
      if (_lines[i].find(needle) != std::string::npos) { return true; }
    }
    return false;
  }
  void mark()
  {
    std::lock_guard<std::mutex> g(_mu);
    _mark = _lines.size();
  }

 private:
  void out_(const char* lvl, const vpipe::VpipeFormat& m) const
  {
    const std::string s = m();
    std::lock_guard<std::mutex> g(_mu);
    _lines.push_back(s);
    if (*lvl == 'W' || s.find("LoRA") != std::string::npos) {
      std::printf("    [%s] %s\n", lvl, s.c_str());
    }
  }
  mutable Services _svc;
  mutable std::mutex _mu;
  mutable std::vector<std::string> _lines;
  std::size_t _mark = 0;
};

std::vector<float>
read_f32(const fs::path& dir, const std::string& name)
{
  std::vector<float> v;
  std::ifstream m(dir / "manifest.json");
  if (!m) { return v; }
  const FlexData man = FlexData::from_json(m);
  if (!man.is_object() || !man.as_object().contains(name)) { return v; }
  const FlexData e = man.as_object().at(name);
  const std::string file(e.as_object().at("file").as_string(""));
  std::ifstream in(dir / file, std::ios::binary | std::ios::ate);
  if (!in) { return v; }
  const std::streamsize n = in.tellg();
  in.seekg(0);
  v.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

std::uint16_t
f32_to_bf16(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) { return (std::uint16_t)0x7fc0; }
  u += 0x7fffu + ((u >> 16) & 1u);
  return (std::uint16_t)(u >> 16);
}

// A model_config beat as ming-image-model-config emits it.
FlexData
config(std::initializer_list<std::pair<const char*, FlexData>> kv)
{
  FlexData c = FlexData::make_object();
  auto o = c.as_object();
  o.insert_or_assign("model_family", FlexData::make_string("ming-image"));
  for (const auto& [k, v] : kv) { o.insert_or_assign(k, v); }
  return c;
}

}  // namespace

int
main()
{
  const char* model = std::getenv("VPIPE_MING_TEST_MODEL_PATH");
  const char* groot = std::getenv("VPIPE_MING_GOLDEN");
  if (model == nullptr || *model == '\0' || groot == nullptr ||
      *groot == '\0') {
    std::printf("SKIPPED: VPIPE_MING_TEST_MODEL_PATH and VPIPE_MING_GOLDEN "
                "are both needed -- the LoRA plumbing was NOT checked.\n");
    return 0;
  }
  const fs::path G(groot);
  const fs::path adir = G / "dit_real" / "lora";
  if (!fs::exists(adir / "adapter_diffusers.safetensors")) {
    std::printf("SKIPPED: no %s (dump_ming_golden.py --stage real_lora) -- "
                "the LoRA plumbing was NOT checked.\n", adir.c_str());
    return 0;
  }
  MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no usable Metal device. NOTHING was checked.\n");
    return 0;
  }
  CHECK(mc.register_metal_library(ming::kMetalLibBf16,
                                  ming_ops_bf16_metallib,
                                  ming_ops_bf16_metallib_len));
  Session ss(&mc);
  const ming::Checkpoint ck = ming::read_checkpoint(model);
  CHECK(ck.valid);
  const fs::path cdir = G / "conditioning" / "p1" / "fp32";
  const std::vector<float> capA = read_f32(cdir, "cap_feats");
  const std::vector<float> capB = read_f32(cdir, "cap_feats_2");
  CHECK(!capA.empty() && !capB.empty());
  if (!ck.valid || capA.empty() || capB.empty()) {
    return mingtest::summary("ming-lora");
  }
  const int A = ck.cond.query_rows(), W = ck.dit.dim;
  const int cw = ck.dit.cap_feat_dim;
  const int N = (int)(capB.size() / (std::size_t)W);
  std::vector<std::uint16_t> cond((std::size_t)(A + N) * W, 0);
  for (int r = 0; r < A; ++r) {
    for (int c = 0; c < cw; ++c) {
      cond[(std::size_t)r * W + c] =
          f32_to_bf16(capA[(std::size_t)r * cw + c]);
    }
  }
  for (std::size_t i = 0; i < capB.size(); ++i) {
    cond[(std::size_t)A * W + i] = f32_to_bf16(capB[i]);
  }
  FlexData sb = FlexData::make_object();
  sb.as_object().insert(ming::kSbQueryRows, FlexData::make_int(A));

  ming::MingImageFamily fam;
  auto load = [&](const FlexData* cfg) {
    vpipe::genai::ImageModelCreateArgs args;
    args.root = model;
    args.metal = &mc;
    args.session = &ss;
    args.model_config = cfg;
    return fam.load(args);
  };
  auto gen = [&](vpipe::genai::ImageGenerator& g, const FlexData* cfg,
                 std::vector<float>* out) {
    vpipe::genai::ImageGenRequest req;
    req.height = 512;
    req.width = 512;
    req.steps = 2;
    req.seed = 42;
    req.cond = cond.data();
    req.cond_rows = A + N;
    req.cond_dim = W;
    req.cond_elem_size = 2;
    req.cond_is_bf16 = true;
    req.cond_sideband = &sb;
    req.model_config = cfg;
    vpipe::genai::ImageGenResult r;
    const bool ok = g.generate(req, &r);
    *out = std::move(r.latent);
    return ok && !out->empty();
  };
  const int want = ming::MingDit::lora_module_capacity(ck.dit);
  // A DIRECTORY holding one .safetensors (and a meta.json): the host's
  // resolver has to find the file in it.
  const FlexData dir_ref = FlexData::make_string(adir.string());

  CASE("no model_config: the base model");
  std::vector<float> base;
  {
    auto g = load(nullptr);
    CHECK(g != nullptr);
    if (!g || !gen(*g, nullptr, &base)) {
      CHECK(!"the base generation failed");
      return mingtest::summary("ming-lora");
    }
  }

  CASE("`lora` binds through the family; its strength is LIVE per request");
  {
    const FlexData c1 = config({{ming::kCfgLora, dir_ref},
                                {ming::kCfgLoraScale,
                                 FlexData::make_real(1.0)}});
    ss.mark();
    auto g = load(&c1);
    CHECK(g != nullptr);
    const std::string bound =
        "runtime LoRA modules bound: lora " + std::to_string(want);
    const bool did_bind = ss.said(bound);
    CHECK(did_bind);
    std::vector<float> l1, l0, l1b, lign;
    if (g) {
      CHECK(gen(*g, &c1, &l1));
      // The SAME generator, strength 0 from the request: the base, bit
      // for bit -- the live path through the generator, and the caption
      // cache dropped with the change.
      const FlexData off = config({{ming::kCfgLoraScale,
                                    FlexData::make_real(0.0)}});
      CHECK(gen(*g, &off, &l0));
      CHECK(gen(*g, &c1, &l1b));
      // A beat naming ANOTHER adapter once the DiT is up: ignored, said.
      ss.mark();
      const FlexData other = config(
          {{ming::kCfgLora, FlexData::make_string("/nonexistent/x."
                                                  "safetensors")},
           {ming::kCfgLoraScale, FlexData::make_real(1.0)}});
      CHECK(gen(*g, &other, &lign));
      CHECK(ss.said("IS IGNORED") || ss.said("is IGNORED"));
    }
    const double moved = mingtest::rel_l2(l1, base);
    std::printf("    bound %s; 2 steps adapted move the latent %.4f; "
                "strength 0 from the request == base: %s; back to 1 == "
                "first: %s; a different `lora` mid-run ignored: %s\n",
                did_bind ? std::to_string(want).c_str() : "NOTHING", moved,
                l0 == base ? "bit-identical" : "DIFFERS",
                l1b == l1 ? "bit-identical" : "DIFFERS",
                lign == l1 ? "unchanged" : "CHANGED");
    CHECK(moved > 0.01);
    CHECK(l0 == base);
    CHECK(l1b == l1);
    CHECK(lign == l1);
  }

  CASE("`lora2` alone takes slot 1, and `lora2_scale` reaches it");
  {
    const FlexData c2 = config({{ming::kCfgLora2, dir_ref},
                                {ming::kCfgLora2Scale,
                                 FlexData::make_real(0.5)}});
    ss.mark();
    auto g = load(&c2);
    CHECK(g != nullptr);
    CHECK(ss.said("runtime LoRA modules bound: lora2 " +
                  std::to_string(want)));
    std::vector<float> h, z;
    if (g) {
      CHECK(gen(*g, &c2, &h));
      const FlexData off = config({{ming::kCfgLora2Scale,
                                    FlexData::make_real(0.0)}});
      CHECK(gen(*g, &off, &z));
    }
    std::printf("    at 0.5 moves the latent %.4f; `lora2_scale` 0 == base: "
                "%s\n", mingtest::rel_l2(h, base),
                z == base ? "bit-identical" : "DIFFERS");
    CHECK(mingtest::rel_l2(h, base) > 0.005);
    CHECK(z == base);
  }

  CASE("an adapter for ANOTHER model binds nothing, and is SAID to");
  {
    // FLUX.2-shaped names: nothing this model has.
    const fs::path fdir = fs::temp_directory_path() / "ming_lora_flux";
    std::error_code ec;
    fs::create_directories(fdir, ec);
    const fs::path ff = fdir / "flux_style.safetensors";
    {
      const std::string hdr0 =
          "{\"transformer_blocks.0.attn.to_q.lora_A.weight\":{\"dtype\":"
          "\"BF16\",\"shape\":[1,8],\"data_offsets\":[0,16]}}";
      std::string hdr = hdr0;
      while (hdr.size() % 8 != 0) { hdr += ' '; }
      std::ofstream o(ff, std::ios::binary);
      const std::uint64_t hl = hdr.size();
      o.write(reinterpret_cast<const char*>(&hl), 8);
      o.write(hdr.data(), (std::streamsize)hdr.size());
      const char zeros[16] = {};
      o.write(zeros, 16);
    }
    const FlexData cf = config({{ming::kCfgLora,
                                 FlexData::make_string(ff.string())}});
    ss.mark();
    auto g = load(&cf);
    CHECK(g != nullptr);
    CHECK(ss.said("bound NO modules"));
    std::vector<float> l;
    if (g) { CHECK(gen(*g, &cf, &l)); }
    CHECK(l == base);
    std::printf("    warned: %s; the base model ran: %s\n",
                ss.said("bound NO modules") ? "yes" : "NO",
                l == base ? "yes" : "NO");
    fs::remove_all(fdir, ec);
  }
  return mingtest::summary("ming-lora");
}
