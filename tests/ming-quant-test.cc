// Ming-Image under `model-quantize`, and what the host can tell about a
// Ming-Image directory on disk.
//
// THREE TIERS, each SKIPPED (and saying so) when what it needs is unset:
//
//   model-free     the quantize family's recipe; the detect profile names
//                  NO class (so the generic "DiffusionTransformer" never
//                  maps to this family); the diffusion-model channel
//                  carries "ming-image" once the image family registers.
//   checkpoint     VPIPE_MING_TEST_MODEL_PATH = the checkpoint ROOT. The
//                  quantize family claims it, and refuses lookalikes
//                  built from its own configs: no Ming contract, the
//                  Layer variant, the bare transformer directory.
//   packs          VPIPE_MING_W8_PACK / VPIPE_MING_W4_PACK = what
//                  model-quantize wrote for this checkpoint (the pack
//                  ROOT; a bare transformer/ also works, with the
//                  config then read from VPIPE_MING_TEST_MODEL_PATH),
//                  and VPIPE_MING_GOLDEN for the dit_real golden. Per
//                  pack: the tensor-by-tensor selection, the real 512^2
//                  forward against the reference's fp32 (bf16's own
//                  figure beside it), ENGAGEMENT of the quantized GEMM,
//                  and streamed == resident bit for bit (slots and the
//                  no-slot fallback).
//
// VPIPE_MING_QUANT_BENCH=1 adds per-forward timings: bf16, w8 and w4
// held together and run in interleaved rounds, the round order reversed
// every round, in one process -- ratios, not absolute claims.
//
// FLOORS. The qmm kernel serves every quantized GEMM on a box without
// matrix cores; on one with them (M5) the M >= 64 GEMMs dequantize once
// and take matmul2d, and only the M = 1 modulation GEMMs stay on qmm.
// The real golden's 1024 image rows clear matmul2d's 64-row floor, so
// both arms are the ones a real run takes.

#include "ming-config.h"
#include "ming-conditioner-stage.h"
#include "ming-dit.h"
#include "ming-family.h"
#include "ming-layout.h"
#include "ming-profiles.h"
#include "ming-quant-family.h"
#include "ming-test.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "generative-models/detect-profile.h"
#include "generative-models/family-profile.h"
#include "generative-models/image-model-registry.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/quantize-family-registry.h"
#include "generative-models/quantize-profile.h"
#include "generative-models/weight-set.h"
#include "pipeline/stage-config.h"
#include "pipeline/stage-registry.h"
#include "pipeline/stage-spec.h"
#include "stages/model-catalog.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

extern "C" const unsigned char ming_ops_bf16_metallib[];
extern "C" const unsigned long ming_ops_bf16_metallib_len;

namespace fs = std::filesystem;
using vpipe::FlexData;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

namespace {

const char*
env_(const char* name)
{
  const char* e = std::getenv(name);
  return e != nullptr && *e != '\0' ? e : nullptr;
}

// "0.0526", or "-" for a figure that was not measured.
std::string
num_(double v, bool have = true)
{
  if (!have) { return "-"; }
  char b[32];
  std::snprintf(b, sizeof b, "%.4f", v);
  return b;
}

bool
contains_csv_(const std::string& csv, const std::string& want)
{
  std::size_t i = 0;
  while (i <= csv.size()) {
    std::size_t c = csv.find(',', i);
    if (c == std::string::npos) { c = csv.size(); }
    if (csv.substr(i, c - i) == want) { return true; }
    i = c + 1;
  }
  return false;
}

// ---- bf16 + goldens ------------------------------------------------------

std::uint16_t
f32_to_bf16(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) { return (std::uint16_t)0x7fc0; }
  u += 0x7fffu + ((u >> 16) & 1u);
  return (std::uint16_t)(u >> 16);
}

float
bf16_to_f32(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// manifest.json -> {dtype, shape, file}; raw little-endian, C order.
class Golden {
 public:
  bool open(const fs::path& dir)
  {
    _dir = dir;
    std::ifstream in(dir / "manifest.json");
    if (!in) { return false; }
    _man = FlexData::from_json(in);
    return _man.is_object();
  }
  bool has(const std::string& n) const
  {
    return _man.is_object() && _man.as_object().contains(n);
  }
  std::vector<long> shape(const std::string& n) const
  {
    std::vector<long> s;
    if (!has(n)) { return s; }
    const FlexData e = _man.as_object().at(n);
    const FlexData a = e.as_object().at("shape");
    auto arr = a.as_array();
    for (std::size_t i = 0; i < arr.size(); ++i) {
      s.push_back((long)arr[i].as_int(0));
    }
    return s;
  }
  std::vector<float> f32(const std::string& n) const
  {
    std::vector<float> v;
    if (!has(n)) { return v; }
    const FlexData e = _man.as_object().at(n);
    const std::string file(e.as_object().at("file").as_string(""));
    std::size_t cnt = 1;
    for (long d : shape(n)) { cnt *= (std::size_t)d; }
    v.resize(cnt);
    std::ifstream in(_dir / file, std::ios::binary);
    if (!in) { v.clear(); return v; }
    in.read(reinterpret_cast<char*>(v.data()),
            (std::streamsize)(cnt * sizeof(float)));
    if (!in) { v.clear(); }
    return v;
  }

 private:
  fs::path _dir;
  FlexData _man;
};

struct Inputs {
  std::vector<float> x;      // [C, H, W]
  int C = 0, H = 0, W = 0;
  std::vector<float> cap_a;  // [Q, cap_feat_dim]
  int Q = 0;
  std::vector<float> cap_b;  // [B, dim]
  int B = 0;
  float t = 0.0f;            // the DiT's t = 1 - sigma
};

bool
load_inputs(const Golden& g, Inputs* in)
{
  const std::vector<long> xs = g.shape("x");
  const std::vector<long> as = g.shape("cap_feats");
  const std::vector<long> bs = g.shape("cap_feats_2");
  if (xs.size() != 3 || as.size() != 2 || bs.size() != 2) { return false; }
  in->x = g.f32("x");
  in->C = (int)xs[0]; in->H = (int)xs[1]; in->W = (int)xs[2];
  in->cap_a = g.f32("cap_feats");
  in->Q = (int)as[0];
  in->cap_b = g.f32("cap_feats_2");
  in->B = (int)bs[0];
  const std::vector<float> t = g.f32("t");
  if (t.size() != 1) { return false; }
  in->t = t[0];
  return !in->x.empty() && !in->cap_a.empty() && !in->cap_b.empty();
}

// ---- one forward -----------------------------------------------------------

struct Run {
  bool ok = false;
  std::string err;
  std::vector<std::uint16_t> raw;  // bf16 [img, patch_dim], as returned
  std::vector<float> z;            // unpacked [C, H, W]
  ming::DitEngagement eng;
  double ms = 0.0;
};

// The DiT's inputs, built once per (DiT config, golden): the packed
// latent rows and the conditioning tensor of the family contract.
struct Prepared {
  ming::Layout lay;
  SharedBuffer lat, cap;
  int cap_stride = 0;
  std::size_t out_elems = 0;
};

bool
prepare(MetalCompute& mc, const ming::DitConfig& d, const Inputs& in,
        Prepared* p, std::string* err)
{
  const int P = d.patch;
  if (!ming::build_layout(in.H / P, in.W / P, in.Q, in.B, d.axes_lens[0],
                          &p->lay, err)) {
    return false;
  }
  const int PD = P * P * in.C;
  std::vector<float> packed((std::size_t)p->lay.img_tokens * PD);
  ming::pack_latents(in.x.data(), in.C, in.H, in.W, P, packed.data());
  p->out_elems = packed.size();
  p->lat = mc.make_shared_buffer(packed.size() * 2);
  p->cap_stride = d.dim;
  p->cap = mc.make_shared_buffer((std::size_t)p->lay.cap_len * d.dim * 2);
  if (p->lat.empty() || p->cap.empty()) { *err = "alloc"; return false; }
  auto* lb = static_cast<std::uint16_t*>(p->lat.contents());
  for (std::size_t k = 0; k < packed.size(); ++k) {
    lb[k] = f32_to_bf16(packed[k]);
  }
  auto* cb = static_cast<std::uint16_t*>(p->cap.contents());
  std::memset(cb, 0, p->cap.byte_size());
  for (int i = 0; i < in.Q; ++i) {
    for (int c = 0; c < d.cap_feat_dim; ++c) {
      cb[(std::size_t)i * d.dim + c] =
          f32_to_bf16(in.cap_a[(std::size_t)i * d.cap_feat_dim + c]);
    }
  }
  for (int i = 0; i < in.B; ++i) {
    for (int c = 0; c < d.dim; ++c) {
      cb[(std::size_t)(in.Q + i) * d.dim + c] =
          f32_to_bf16(in.cap_b[(std::size_t)i * d.dim + c]);
    }
  }
  return true;
}

Run
run(ming::MingDit& dit, const Prepared& p, const Inputs& in)
{
  Run r;
  ming::MingDit::Request req;
  req.latents = &p.lat;
  req.cap = &p.cap;
  req.cap_stride = p.cap_stride;
  req.layout = &p.lay;
  req.sigma = 1.0f - in.t;
  const auto t0 = std::chrono::steady_clock::now();
  SharedBuffer o = dit.forward(req, &r.err);
  r.ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  r.eng = dit.last_engagement();
  if (o.empty()) { return r; }
  const auto* op = static_cast<const std::uint16_t*>(o.contents());
  r.raw.assign(op, op + p.out_elems);
  std::vector<float> rows(p.out_elems);
  for (std::size_t k = 0; k < rows.size(); ++k) {
    rows[k] = bf16_to_f32(r.raw[k]);
  }
  const int P = dit.config().patch;
  r.z.resize((std::size_t)in.C * in.H * in.W);
  ming::unpack_latents(rows.data(), in.C, in.H, in.W, P, r.z.data());
  r.ok = true;
  return r;
}

bool
same_bits(const Run& a, const Run& b)
{
  return a.ok && b.ok && a.raw.size() == b.raw.size() &&
         std::memcmp(a.raw.data(), b.raw.data(), a.raw.size() * 2) == 0;
}

void
print_eng(const ming::DitEngagement& e)
{
  std::printf("    gemm: qmm %d, mma2 %d, i8 %d, bm32 %d, bm64 %d, "
              "bm64bn64 %d; blocks held %d, streamed %d (prefetch %d)\n",
              e.gemm_qmm, e.gemm_mma2, e.gemm_i8, e.gemm_bm32, e.gemm_bm64,
              e.gemm_bm64bn64, e.blocks_held, e.blocks_streamed,
              e.prefetch_hits);
}

std::unique_ptr<ming::MingDit>
load_dit(MetalCompute& mc, const std::string& dir, const ming::DitConfig& d,
         bool stream, std::string* err)
{
  auto ws = vpipe::genai::WeightSet::open(dir, nullptr);
  if (!ws) {
    *err = "WeightSet::open failed";
    return nullptr;
  }
  return ming::MingDit::load(std::move(ws), &mc, d, nullptr, stream,
                             nullptr, err);
}

// ---- model-free ------------------------------------------------------------

void
test_recipe()
{
  CASE("the quantize family's recipe");
  ming::MingQuantFamily fam;
  CHECK(fam.tag() == ming::kModelType);
  const auto comps = fam.components();
  CHECK(comps.size() == 1);
  if (comps.size() == 1) {
    const auto& c = comps[0];
    CHECK(c.target == "dit" && c.role == "transformer");
    // Empty: a diffusers directory, never a single repack file.
    CHECK(c.meta_key.empty());
    // The wholesale rule needs a NON-EMPTY scope: an empty one sends the
    // stage back to the leaf rule, whose default set is an LM's.
    CHECK(!c.scope.empty() && c.all_in_scope);
    for (const char* e : {"all_x_embedder", "cap_embedder", "t_embedder.",
                          "all_final_layer"}) {
      CHECK(contains_csv_(c.exclude, e));
    }
    CHECK(c.component_tag == std::string(ming::kDitComponentTag));
  }

  CASE("the detect profile names no class; labels a typed model");
  {
    namespace d_ = vpipe::genai::detect;
    const FlexData p = ming::detect_profile();
    CHECK(!d_::has(&p, d_::kClassNames));
    CHECK(d_::text(&p, d_::kModelType) == ming::kModelType);
    CHECK(d_::text(&p, d_::kLabelFamily) == "Ming-Image");
    // Registered as the plugin does, the generic DiT class still maps to
    // nothing -- the whole point -- while the type answers.
    vpipe::genai::profile::Registry::get().add(
        std::string(d_::kDomain), ming::kModelType, p);
    CHECK(d_::model_type_for_class("DiffusionTransformer", false).empty());
    CHECK(d_::for_model_type(ming::kModelType) != nullptr);
    std::vector<std::string> in, out;
    vpipe::catalog_default_io(ming::kModelType, in, out);
    CHECK(in == std::vector<std::string>{"text"});
    CHECK(out == std::vector<std::string>{"image"});
    // ...and nothing registered a quantize PROFILE on the class name.
    CHECK(vpipe::genai::quant::family_for_class("DiffusionTransformer")
              .empty());
  }

  CASE("pickers: the diffusion-model channel carries ming-image");
  {
    vpipe::genai::ImageModelRegistry::get().add(
        std::make_unique<ming::MingImageFamily>());
    const auto& reg = vpipe::StageRegistry::get();
    auto hf_types = [&](const char* stage) -> std::string {
      const vpipe::StageSpec* sp = reg.spec(stage);
      if (sp == nullptr) { return "<no spec>"; }
      for (const auto& prm :
           vpipe::resolve_config_params(sp->attrs, FlexData())) {
        if (prm.key == "hf_dir") { return prm.suggest_db_type; }
      }
      return "<no hf_dir>";
    };
    // The channel SOURCE offers the union of its consumers and of the
    // families registered into existing stages -- this one included.
    const std::string sel = hf_types("model-select");
    CHECK(contains_csv_(sel, ming::kModelType));
    std::printf("    model-select hf_dir: %s ming-image\n",
                contains_csv_(sel, ming::kModelType) ? "offers"
                                                     : "DOES NOT offer");
    // CONSUMERS keep their own static lists: a host fact, reported.
    for (const char* st : {"generate-image", "vae-decode"}) {
      std::printf("    %s's own hf_dir picker: %s ming-image\n", st,
                  contains_csv_(hf_types(st), ming::kModelType)
                      ? "offers" : "does NOT offer");
    }
    const vpipe::StageSpec* cs = ming::MingConditionerStage::stage_spec();
    bool cond_ok = false;
    if (cs != nullptr) {
      for (const auto& k : cs->attrs) {
        if (k.key == std::string_view("hf_dir")) {
          cond_ok = k.suggest_db_type == std::string_view(ming::kModelType)
                    && k.model_channel == std::string_view("diffusion-model");
        }
      }
    }
    CHECK(cond_ok);
  }
}

// ---- the checkpoint and its lookalikes -------------------------------------

void
copy_json_(const fs::path& from, const fs::path& to,
           const std::map<std::string, FlexData>& set = {},
           const std::vector<std::string>& drop = {})
{
  std::ifstream in(from);
  FlexData j = FlexData::from_json(in);
  for (const auto& k : drop) { j.as_object().erase(k); }
  for (const auto& [k, v] : set) { j.as_object().insert_or_assign(k, v); }
  fs::create_directories(to.parent_path());
  std::ofstream(to) << j.to_json(true);
}

void
test_claims(const std::string& model)
{
  CASE("the quantize family claims the checkpoint, and only it");
  ming::MingQuantFamily fam;
  CHECK(fam.claims(model));
  const fs::path root(model);
  const fs::path tmp = fs::temp_directory_path() /
                       ("ming-quant-test-" + std::to_string(getpid()));
  fs::remove_all(tmp);
  // A lookalike that has every component config of the real one, with
  // the Ming contract taken off the DiT: the class alone must not do it.
  auto lookalike = [&](const char* name,
                       const std::map<std::string, FlexData>& set,
                       const std::vector<std::string>& drop) {
    const fs::path r = tmp / name;
    copy_json_(root / "transformer" / "config.json",
               r / "transformer" / "config.json", set, drop);
    for (const char* c : {"mllm/config.json", "connector/config.json",
                          "mlp/config.json", "vae/config.json",
                          "scheduler/scheduler_config.json"}) {
      copy_json_(root / c, r / c);
    }
    return r.string();
  };
  const std::string no_contract = lookalike(
      "no-contract", {}, {"alignment_padding_mode", "multi_frame_output"});
  const std::string layer = lookalike(
      "layer", {{"alignment_padding_mode", FlexData::make_string("learned")},
                {"multi_frame_output", FlexData::make_bool(true)}}, {});
  CHECK(!fam.claims(no_contract));
  CHECK(!fam.claims(layer));
  CHECK(!fam.claims((root / "transformer").string()));
  // The registry asks the same question the stage does.
  vpipe::genai::QuantizeFamilyRegistry::get().add(
      std::make_unique<ming::MingQuantFamily>());
  const auto* hit =
      vpipe::genai::QuantizeFamilyRegistry::get().claim_for(model);
  CHECK(hit != nullptr && hit->tag() == ming::kModelType);
  CHECK(vpipe::genai::QuantizeFamilyRegistry::get().claim_for(no_contract) ==
        nullptr);
  std::printf("    real: claimed; no-contract / Layer / bare transformer/: "
              "%s\n",
              !fam.claims(no_contract) && !fam.claims(layer)
                  ? "refused" : "CLAIMED");
  fs::remove_all(tmp);
}

// ---- the packs ---------------------------------------------------------

// The pack's transformer directory and DiT config.
bool
resolve_pack(const std::string& path, const char* model, std::string* dir,
             ming::DitConfig* d, bool* is_root)
{
  const ming::Checkpoint ck = ming::read_checkpoint(path);
  if (ck.valid) {
    *dir = ck.transformer_dir;
    *d = ck.dit;
    *is_root = true;
    return true;
  }
  *is_root = false;
  if (!fs::exists(fs::path(path) / "config.json") || model == nullptr) {
    return false;
  }
  const ming::Checkpoint ref = ming::read_checkpoint(model);
  if (!ref.valid) { return false; }
  *dir = path;
  *d = ref.dit;
  return true;
}

// The selection, tensor by tensor: every 2-D matrix of the block stacks
// carries codes + scales + biases at `bits`, every model-level weight is
// dense bf16, and nothing else changed shape.
void
check_selection(const std::string& dir, int bits)
{
  auto w = vpipe::genai::MetalLlamaWeights::open_model(dir);
  CHECK(w.has_value());
  if (!w) { return; }
  int quant = 0, dense_model = 0, bad = 0;
  for (const std::string& nm : w->tensor_names()) {
    if (nm.size() < 7 || nm.compare(nm.size() - 7, 7, ".weight") != 0) {
      continue;
    }
    const std::string base = nm.substr(0, nm.size() - 7);
    const auto* ti = w->info(nm);
    const bool block = nm.rfind("layers.", 0) == 0 ||
                       nm.rfind("noise_refiner.", 0) == 0 ||
                       nm.rfind("context_refiner.", 0) == 0;
    const auto* si = w->info(base + ".scales");
    const bool matrix_leaf = base.find("norm") == std::string::npos;
    if (block && matrix_leaf) {
      // codes [N, K * bits / 32], scales [N, K / 64]
      const bool ok = si != nullptr && ti->dtype == "U32" &&
                      w->info(base + ".biases") != nullptr &&
                      ti->shape.size() == 2 && si->shape.size() == 2 &&
                      ti->shape[1] * 32 == si->shape[1] * 64 * bits;
      if (ok) { ++quant; } else { ++bad; }
      if (!ok && bad <= 3) {
        std::printf("    NOT quantized: %s\n", nm.c_str());
      }
    } else if (!block) {
      const bool ok = si == nullptr && ti->dtype == "BF16";
      if (ok) { ++dense_model; } else { ++bad; }
      if (!ok && bad <= 3) {
        std::printf("    NOT dense: %s\n", nm.c_str());
      }
    } else if (si != nullptr) {
      ++bad;   // a norm vector with scales beside it
    }
  }
  std::printf("    %d block matrices at w%d, %d model-level weights dense, "
              "%d wrong\n", quant, bits, dense_model, bad);
  // 32 modulated blocks x (7 + adaLN) + 2 context-refiner blocks x 7.
  CHECK(quant == 32 * 8 + 2 * 7);
  // x_embedder, cap_embedder.0 (RMSNorm) + .1, t_embedder 0 + 2, final
  // adaLN + linear.
  CHECK(dense_model == 7);
  CHECK(bad == 0);
}

struct PackResult {
  bool ran = false;
  double rel = 0.0, rel_bar = 0.0, rel_vs_bf16 = 0.0, ms = 0.0;
};

PackResult
test_pack(MetalCompute& mc, const char* label, const std::string& path,
          int bits, const char* model, const Inputs& in,
          const std::vector<float>& want, const std::vector<float>& bar,
          const Run* bf16_run, double tol)
{
  PackResult out;
  std::string dir;
  ming::DitConfig d;
  bool is_root = false;
  std::printf("== %s pack: %s\n", label, path.c_str());
  CASE("the pack is a Ming-Image checkpoint the family still claims");
  CHECK(resolve_pack(path, model, &dir, &d, &is_root));
  if (dir.empty()) { return out; }
  if (is_root) {
    CHECK(ming::MingQuantFamily().claims(path));
    CHECK(ming::is_ming_image(path));
  }
  {
    std::ifstream cin(fs::path(dir) / "config.json");
    const FlexData c = FlexData::from_json(cin);
    auto o = c.as_object();
    const bool has_q = o.contains("quantization");
    CHECK(has_q);
    if (has_q) {
      const FlexData q = o.at("quantization");
      CHECK(q.as_object().at("bits").as_int(0) == bits);
      CHECK(q.as_object().at("group_size").as_int(0) == 64);
    }
    CHECK(o.contains("_vpipe_component") &&
          std::string(o.at("_vpipe_component").as_string("")) ==
              ming::kDitComponentTag);
    CHECK(std::string(o.at("_class_name").as_string("")) ==
          "DiffusionTransformer");
  }
  CASE("the selection, tensor by tensor");
  check_selection(dir, bits);

  CASE("the real 512x512 forward on the pack, resident");
  Run res;
  {
    std::string err;
    auto dit = load_dit(mc, dir, d, false, &err);
    CHECK(dit != nullptr);
    if (!dit) {
      std::printf("    load refused: %s\n", err.c_str());
      return out;
    }
    CHECK(dit->quant_bits() == bits);
    Prepared p;
    CHECK(prepare(mc, d, in, &p, &err));
    res = run(*dit, p, in);
    CHECK(res.ok);
    if (!res.ok) {
      std::printf("    forward failed: %s\n", res.err.c_str());
      return out;
    }
    out.ran = true;
    out.rel = mingtest::rel_l2(res.z, want);
    out.rel_bar = bar.empty() ? 0.0 : mingtest::rel_l2(res.z, bar);
    out.rel_vs_bf16 =
        bf16_run != nullptr && bf16_run->ok
            ? mingtest::rel_l2(res.z, bf16_run->z) : -1.0;
    out.ms = res.ms;
    std::printf("    w%d g64: rel-L2 vs reference fp32 %.4f (tol %.3f); vs "
                "reference bf16 %.4f; vs vpipe bf16 %s; %.0f ms (cold)\n",
                bits, out.rel, tol, out.rel_bar,
                num_(out.rel_vs_bf16, out.rel_vs_bf16 >= 0).c_str(), res.ms);
    CHECK(out.rel < tol);
    print_eng(res.eng);
    // ENGAGEMENT BEFORE ACCURACY'S MEANING: a pack whose GEMMs ran on
    // something else -- or on nothing -- can still look plausible.
    const int nm = d.n_refiner_layers + d.n_layers;
    const int n_quant = 8 * nm + 7 * d.n_refiner_layers;
    if (!mc.supports_matrix_cores()) {
      // Every quantized Linear on qmm, every dense one on a tile.
      CHECK(res.eng.gemm_qmm == n_quant);
      CHECK(res.eng.gemm_mma2 == 0 && res.eng.gemm_i8 == 0);
      CHECK(res.eng.gemm_bm32 + res.eng.gemm_bm64 +
                res.eng.gemm_bm64bn64 == 6);
    } else {
      // matmul2d takes M >= 64 off the dequantized codes; the M = 1
      // modulation GEMMs stay on qmm.
      CHECK(res.eng.gemm_qmm == nm);
      CHECK(res.eng.gemm_mma2 + res.eng.gemm_i8 > 0);
    }
  }
  CASE("streamed == resident, bit for bit");
  {
    std::string err;
    auto dit = load_dit(mc, dir, d, true, &err);
    CHECK(dit != nullptr);
    if (dit) {
      Prepared p;
      CHECK(prepare(mc, d, in, &p, &err));
      const Run s = run(*dit, p, in);
      CHECK(s.ok && s.eng.blocks_streamed ==
                        d.n_refiner_layers + d.n_layers);
      CHECK(s.eng.gemm_qmm == res.eng.gemm_qmm);
      CHECK(same_bits(s, res));
      std::printf("    streamed %d blocks (prefetch hits %d), %.0f ms: %s; "
                  "held %.0f MB (declared floor %.0f MB)\n",
                  s.eng.blocks_streamed, s.eng.prefetch_hits, s.ms,
                  same_bits(s, res) ? "identical" : "DIFFERS",
                  (double)dit->resident_bytes() / 1e6,
                  (double)ming::MingDit::streaming_floor_bytes(dir) / 1e6);
    }
  }
  CASE("VPIPE_MING_NO_SLOTS (per-block fallback) == resident");
  {
    setenv("VPIPE_MING_NO_SLOTS", "1", 1);
    std::string err;
    auto dit = load_dit(mc, dir, d, true, &err);
    CHECK(dit != nullptr);
    if (dit) {
      Prepared p;
      CHECK(prepare(mc, d, in, &p, &err));
      const Run s = run(*dit, p, in);
      CHECK(s.ok && s.eng.prefetch_hits == 0);
      CHECK(same_bits(s, res));
      std::printf("    no slots, %.0f ms: %s\n", s.ms,
                  same_bits(s, res) ? "identical" : "DIFFERS");
    }
    unsetenv("VPIPE_MING_NO_SLOTS");
  }
  return out;
}

// Interleaved timing: every arm held, rounds in alternating order, the
// caption served from the cache (as 10 of the 11 denoise steps are).
void
bench(MetalCompute& mc, const Inputs& in,
      const std::vector<std::pair<std::string, std::string>>& arms,
      const ming::DitConfig& d)
{
  CASE("bench: per-forward time, interleaved, one process");
  struct Arm {
    std::string label;
    std::unique_ptr<ming::MingDit> dit;
    Prepared p;
    std::vector<double> ms;
  };
  std::vector<Arm> v;
  for (const auto& [label, dir] : arms) {
    Arm a;
    a.label = label;
    std::string err;
    a.dit = load_dit(mc, dir, d, false, &err);
    if (!a.dit || !prepare(mc, d, in, &a.p, &err)) {
      std::printf("    %s: %s\n", label.c_str(), err.c_str());
      CHECK(!"bench arm failed to load");
      return;
    }
    // Warm: the first forward builds scratch and fills the caption.
    (void)run(*a.dit, a.p, in);
    v.push_back(std::move(a));
  }
  const int rounds = 6;
  for (int r = 0; r < rounds; ++r) {
    for (std::size_t i = 0; i < v.size(); ++i) {
      Arm& a = v[r % 2 == 0 ? i : v.size() - 1 - i];
      const Run x = run(*a.dit, a.p, in);
      CHECK(x.ok);
      a.ms.push_back(x.ms);
    }
  }
  auto median = [](std::vector<double> m) {
    std::sort(m.begin(), m.end());
    return m.empty() ? 0.0 : m[m.size() / 2];
  };
  const double base = median(v[0].ms);
  for (const Arm& a : v) {
    const double m = median(a.ms);
    std::printf("    %-5s median %.0f ms over %d forwards (min %.0f, max "
                "%.0f): %.2fx of %s\n", a.label.c_str(), m, (int)a.ms.size(),
                *std::min_element(a.ms.begin(), a.ms.end()),
                *std::max_element(a.ms.begin(), a.ms.end()),
                base > 0 ? base / m : 0.0, v[0].label.c_str());
  }
}

}  // namespace

int
main()
{
  test_recipe();

  const char* model = env_("VPIPE_MING_TEST_MODEL_PATH");
  if (model != nullptr) {
    test_claims(model);
  } else {
    std::printf("SKIPPED claims: VPIPE_MING_TEST_MODEL_PATH is unset\n");
  }

  const char* w8 = env_("VPIPE_MING_W8_PACK");
  const char* w4 = env_("VPIPE_MING_W4_PACK");
  const char* groot = env_("VPIPE_MING_GOLDEN");
  if ((w8 == nullptr && w4 == nullptr) || groot == nullptr) {
    std::printf("SKIPPED packs: %s -- no quantized DiT was loaded or run\n",
                groot == nullptr ? "VPIPE_MING_GOLDEN is unset"
                                 : "VPIPE_MING_W8_PACK and "
                                   "VPIPE_MING_W4_PACK are unset");
    return mingtest::summary("ming-quant");
  }
  MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED packs: no usable Metal device\n");
    return mingtest::summary("ming-quant");
  }
  CHECK(mc.register_metal_library(ming::kMetalLibBf16, ming_ops_bf16_metallib,
                                  ming_ops_bf16_metallib_len));
  const fs::path droot = fs::path(groot) / "dit_real";
  Golden g;
  Inputs in;
  if (!g.open(droot / "fp32") || !load_inputs(g, &in)) {
    std::printf("SKIPPED packs: no readable dit_real/fp32 golden under "
                "%s\n", groot);
    return mingtest::summary("ming-quant");
  }
  const std::vector<float> want = g.f32("out");
  Golden gbar;
  std::vector<float> bar;
  if (gbar.open(droot / "bf16")) { bar = gbar.f32("out"); }
  const double ref_bf16 = bar.empty() ? 0.0 : mingtest::rel_l2(bar, want);

  // VPIPE_MING_QUANT_BENCH=only: the timing rounds alone, for a quiet
  // window on a shared box (the checks above already passed).
  const char* bench_env = env_("VPIPE_MING_QUANT_BENCH");
  const bool bench_only =
      bench_env != nullptr && std::string(bench_env) == "only";

  // bf16, for the figure every pack is read against -- in this process.
  Run bf;
  ming::DitConfig dcfg;
  std::string bf_dir;
  if (model != nullptr && bench_only) {
    const ming::Checkpoint ck = ming::read_checkpoint(model);
    if (ck.valid) { dcfg = ck.dit; bf_dir = ck.transformer_dir; }
  } else if (model != nullptr) {
    CASE("bf16: the same forward, the unquantized checkpoint");
    const ming::Checkpoint ck = ming::read_checkpoint(model);
    CHECK(ck.valid);
    if (ck.valid) {
      dcfg = ck.dit;
      bf_dir = ck.transformer_dir;
      std::string err;
      auto dit = load_dit(mc, bf_dir, dcfg, false, &err);
      Prepared p;
      if (dit && prepare(mc, dcfg, in, &p, &err)) {
        bf = run(*dit, p, in);
      }
      CHECK(bf.ok);
      if (bf.ok) {
        std::printf("    bf16: rel-L2 vs reference fp32 %.4f (the "
                    "reference's own bf16 %.4f), %.0f ms (cold)\n",
                    mingtest::rel_l2(bf.z, want), ref_bf16, bf.ms);
      }
    }
  }
  const double bf_rel = bf.ok ? mingtest::rel_l2(bf.z, want) : 0.0526;

  // w8: held to the bf16 test's own bar (twice the reference's bf16).
  // MEASURED on the M4 Pro: 0.0583, bf16 0.0526.
  //
  // w4: a looser bar, stated. MEASURED 0.298 -- the 4-bit step is ~17x
  // the 8-bit one, and w8 adds ~0.025 in quadrature, so this is the
  // width rather than a fault (keeping the adaLN modulation dense made
  // it 0.341, not better: the modulation is not what costs). The bar
  // guards against a WRONG pack -- unbound codes, a misread group --
  // which lands near 1, not against w4 being coarse.
  const double tol8 = std::max(0.035, 2.0 * ref_bf16);
  const double tol4 = 0.35;
  PackResult r8, r4;
  if (bench_only) {
    std::printf("SKIPPED accuracy: VPIPE_MING_QUANT_BENCH=only\n");
  } else {
    if (w8 != nullptr) {
      r8 = test_pack(mc, "w8", w8, 8, model, in, want, bar,
                     bf.ok ? &bf : nullptr, tol8);
    } else {
      std::printf("SKIPPED w8 pack: VPIPE_MING_W8_PACK is unset\n");
    }
    if (w4 != nullptr) {
      r4 = test_pack(mc, "w4", w4, 4, model, in, want, bar,
                     bf.ok ? &bf : nullptr, tol4);
    } else {
      std::printf("SKIPPED w4 pack: VPIPE_MING_W4_PACK is unset\n");
    }
    if (r8.ran && r4.ran) {
      CASE("w4 is coarser than w8");
      CHECK(r4.rel > r8.rel);
    }
    std::printf("SUMMARY rel-L2 vs reference fp32: bf16 %.4f%s, w8 %s, w4 "
                "%s (reference's own bf16 %.4f)\n", bf_rel,
                bf.ok ? "" : " (README figure, not run here)",
                num_(r8.rel, r8.ran).c_str(), num_(r4.rel, r4.ran).c_str(),
                ref_bf16);
  }

  if (bench_env != nullptr) {
    std::vector<std::pair<std::string, std::string>> arms;
    if (!bf_dir.empty()) { arms.push_back({"bf16", bf_dir}); }
    ming::DitConfig d;
    bool is_root = false;
    std::string dir;
    if (w8 != nullptr && resolve_pack(w8, model, &dir, &d, &is_root)) {
      arms.push_back({"w8", dir});
    }
    if (w4 != nullptr && resolve_pack(w4, model, &dir, &d, &is_root)) {
      arms.push_back({"w4", dir});
    }
    if (arms.size() >= 2) {
      bench(mc, in, arms, bf_dir.empty() ? d : dcfg);
    } else {
      std::printf("SKIPPED bench: needs bf16 (VPIPE_MING_TEST_MODEL_PATH) "
                  "and at least one pack\n");
    }
  }
  return mingtest::summary("ming-quant");
}
