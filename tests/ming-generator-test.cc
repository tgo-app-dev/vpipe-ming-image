// The whole denoise, through the production generator, against the
// reference's own trajectory: the e2e golden (512x512, seed 42, the
// reference's "12 steps", cfg 1) driven by the fp32 P1 conditioning.
//
// What only this test can catch, because the DiT test feeds one forward
// its inputs directly: the schedule (11 forwards for the reference's 12),
// the sign of the velocity, the Euler step, torch-identical noise, the
// patch packing, and the conditioning contract as the GENERATOR builds
// it from a conditioner's tensor and sideband.
//
// Needs VPIPE_MING_TEST_MODEL_PATH (the checkpoint) and VPIPE_MING_GOLDEN
// (tools/dump_ming_golden.py's output root); SKIPPED without them.

#include "ming-config.h"
#include "ming-dit.h"
#include "ming-family.h"
#include "ming-generator.h"
#include "ming-test.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "generative-models/image-model-registry.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

extern "C" const unsigned char ming_ops_bf16_metallib[];
extern "C" const unsigned long ming_ops_bf16_metallib_len;

namespace fs = std::filesystem;
using vpipe::FlexData;
using vpipe::metal_compute::MetalCompute;

namespace {

// A golden directory: manifest.json + raw little-endian files.
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

}  // namespace

int
main()
{
  const char* model = std::getenv("VPIPE_MING_TEST_MODEL_PATH");
  const char* groot = std::getenv("VPIPE_MING_GOLDEN");
  if (model == nullptr || *model == '\0' || groot == nullptr ||
      *groot == '\0') {
    std::printf("SKIPPED: VPIPE_MING_TEST_MODEL_PATH and VPIPE_MING_GOLDEN "
                "are both needed -- the denoise was NOT checked.\n");
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
  const fs::path G(groot);
  const fs::path cdir = G / "conditioning" / "p1" / "fp32";
  const fs::path edir = G / "e2e" / "fp32";

  CASE("e2e: 11 forwards reproduce the reference's 12-step trajectory");
  const ming::Checkpoint ck = ming::read_checkpoint(model);
  CHECK(ck.valid);
  const std::vector<float> capA = read_f32(cdir, "cap_feats");     // 256x2560
  const std::vector<float> capB = read_f32(cdir, "cap_feats_2");   // Nx3840
  const std::vector<float> want = read_f32(edir, "final_latent");
  const std::vector<float> traj = read_f32(edir, "latents");       // 12 steps
  CHECK(!capA.empty() && !capB.empty() && !want.empty() && !traj.empty());
  if (!ck.valid || capA.empty() || capB.empty() || want.empty()) {
    return mingtest::summary("ming-generator");
  }
  const int A = ck.cond.query_rows(), W = ck.dit.dim;
  const int cw = ck.dit.cap_feat_dim;
  const int N = (int)(capB.size() / (std::size_t)W);
  CHECK((int)(capA.size() / (std::size_t)cw) == A);

  // The conditioning exactly as `ming-image-conditioner` emits it.
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
  {
    auto o = sb.as_object();
    o.insert(ming::kSbQueryRows, FlexData::make_int(A));
  }

  vpipe::genai::ImageModelCreateArgs args;
  args.root = model;
  args.metal = &mc;
  auto gen = ming::MingImageGenerator::load(ck, args);
  CHECK(gen != nullptr);
  if (!gen) { return mingtest::summary("ming-generator"); }

  vpipe::genai::ImageGenRequest req;
  req.height = 512;
  req.width = 512;
  req.steps = ming::kDefaultSteps;
  req.seed = 42;
  req.guidance_scale = 1.0;
  req.cond = cond.data();
  req.cond_rows = A + N;
  req.cond_dim = W;
  req.cond_elem_size = 2;
  req.cond_is_bf16 = true;
  req.cond_sideband = &sb;
  std::vector<std::vector<float>> steps;
  req.step_latent = [&](int, int, const float* l,
                        const std::vector<int>& shape) {
    std::size_t n = 1;
    for (int s : shape) { n *= (std::size_t)s; }
    steps.emplace_back(l, l + n);
  };
  int progressed = 0;
  req.progress = [&](int step, int total) {
    progressed = step;
    return total == ming::kDefaultSteps;
  };
  vpipe::genai::ImageGenResult out;
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = gen->generate(req, &out);
  const double secs = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  CHECK(ok);
  CHECK(progressed == ming::kDefaultSteps);
  CHECK(out.shape == std::vector<int>({16, 64, 64}));
  CHECK(out.latent.size() == want.size());
  CHECK((int)steps.size() == ming::kDefaultSteps);
  if (!ok || out.latent.size() != want.size()) {
    return mingtest::summary("ming-generator");
  }
  // Per step: the reference's latents[i] is the latent after its step i;
  // its 12th step is a no-op, so 11 forwards land on latents[10].
  const std::size_t per = want.size();
  for (std::size_t i = 0; i < steps.size() && (i + 1) * per <= traj.size();
       ++i) {
    const std::vector<float> ref(traj.begin() + (long)(i * per),
                                 traj.begin() + (long)((i + 1) * per));
    std::printf("    step %2zu: rel-L2 %.4f\n", i + 1,
                mingtest::rel_l2(steps[i], ref));
  }
  // THE BAR: the reference's own bf16 run lands 1.17% from its fp32 one
  // on this final latent. This port runs the DiT in bf16 too, so the
  // same order is expected; 3% leaves room for a different (but equally
  // bf16) rounding path while still failing a wrong sign, schedule,
  // noise or packing, each of which is O(1).
  const double r = mingtest::rel_l2(out.latent, want);
  std::printf("    final latent rel-L2 vs fp32 reference: %.4f (reference "
              "bf16: 0.0117), %.1f s for %d forwards\n",
              r, secs, ming::kDefaultSteps);
  CHECK(r < 0.03);

  CASE("a scheduler-select beat's default shift does NOT move the run");
  {
    // scheduler-select emits shift 1.15 / exponential when nothing was
    // set, so taking its shift would run this checkpoint (static 6.0) on
    // a curve it was never trained on whenever a graph pins its steps
    // that way -- the trap Z-Image fell into. Bit-identical, not close.
    FlexData sched = FlexData::make_object();
    {
      auto o = sched.as_object();
      o.insert("scheduler", FlexData::make_string("flow_match"));
      o.insert("type", FlexData::make_string("simple"));
      o.insert("steps", FlexData::make_int(ming::kDefaultSteps));
      o.insert("shift", FlexData::make_real(1.15));
      o.insert("shift_type", FlexData::make_string("exponential"));
    }
    vpipe::genai::ImageGenRequest r2 = req;
    r2.scheduler_spec = &sched;
    r2.step_latent = nullptr;
    vpipe::genai::ImageGenResult out2;
    CHECK(gen->generate(r2, &out2));
    CHECK(out2.latent == out.latent);
  }
  return mingtest::summary("ming-generator");
}
