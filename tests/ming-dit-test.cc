// The DiT against the reference's own goldens.
//
// VPIPE_MING_GOLDEN = the goldens root tools/dump_ming_golden.py writes:
//
//   dit_small/   a 2-layer random-weight DiffusionTransformer (dim 1024,
//                8 heads x 128, 1 refiner each side, cap_feat_dim 256,
//                bf16 weights) and three cases at t = 0.3 (sigma 0.7):
//                  case_a  256 image rows, caption 64 + 37 = 101
//                  case_b  256 image rows, caption 64 + 70 = 134
//                  case_c  240 image rows (16 image pads), caption 101
//                Reference output in f32 compute from the bf16 weights,
//                and the reference's own bf16 run (its bf16 floor).
//   dit_real/    one forward of the REAL DiT at 512x512 (1024 image rows)
//                at sampler step 5, fp32 on CPU. Needs, in addition,
//                VPIPE_MING_TEST_MODEL_PATH = the checkpoint ROOT.
//
//   dit_small/lora/  RUNTIME LoRA (tools/dump_ming_golden.py --stage
//                dit_lora): one synthetic rank-8 alpha-16 adapter over
//                the 28 block projections, in the diffusers, pipeline and
//                kohya spellings; fused/ = the same factors folded into the
//                weights; <case>/lora_out = the FUSED model's fp32 output.
//   dit_real/lora/   a synthetic rank-4 adapter over the real checkpoint's
//                238 block projections (--stage real_lora), for binding
//                counts, live strength, Sol, streaming and the ANE.
//
// Either unset: the part that needs it SKIPS, and says so.
//
// FLOORS -- what these geometries do and do not exercise, because a
// golden below a fast path's floor verifies the arm nobody takes:
//
//   * dense tiles: bm64bn64 needs M >= 128 -- the image (256 / 240) and
//     joint (357 / 390 / 341) GEMMs clear it; the caption's 101 / 134
//     rows and every M=1 Linear take bm32. Both arms run.
//   * attention remainders: 101, 134, 341, 357, 390 and 240 rows are not
//     multiples of the steel kernel's 32-row query / 16-row key tiles,
//     so the unaligned path -- the one dropping the pad rows makes
//     ordinary -- runs in every case. 256 is aligned, so both do.
//   * matmul2d (M >= 64), NAX attention, i8_gemm (K >= 1024; dim 1024
//     clears it) and Sage (>= 512 rows) need MATRIX CORES. On a box
//     without them (the M4 Pro this was written on) none can engage and
//     the test asserts that they did not; on an M5 it asserts matmul2d
//     and NAX did. The small golden never reaches Sage's floor.
//   * Sol does NOT need matrix cores (its exact half runs on whichever
//     flash kernel the box has). Its floors are >= 1024 rows AND 16
//     64-key blocks, and a modulated block past `dense_layers` (1): the
//     real golden's noise refiner (1024 rows, exactly 16 blocks) and
//     main stack (1314 rows) clear both, so 31 of its 34 attentions
//     route; the context refiner (290 rows, and cached) never does.
//   * the ANE feed-forward needs ONE 2048-row chunk for the ANE and a
//     row left for the GPU: the real golden's 1314-row joint declines
//     it, so its case runs at 1024^2 (4096 image rows + the golden's 290
//     caption rows), random latents, and asserts it ARMED and took rows
//     before anything about the numbers.
//   * runtime LoRA: the small model's projections take the dense tiles
//     (M >= 128) and the 32x32 one (the caption's 101 / 134 rows) on an
//     M4, matmul2d on an M5 (M >= 64); the adapter accumulates after
//     whichever ran. The adapter's own GEMMs are the toolkit's
//     (steel 64-row tiles without matrix cores, mma pairs with them).
//   * the ffn width int(1024/3*8) = 2730 is not a multiple of 4 or 32:
//     the GEMMs' K-tail and the scalar swiglu fallback run here, where
//     the real model's 10240 takes the vec4 path.
//
// OPT-IN MEASUREMENTS (real model; speed is a property of the box, so it
// is printed, never asserted):
//
//   VPIPE_MING_BENCH=<rounds>   one-forward timings of several arms over
//       ONE weight set, warmed, then interleaved with the round order
//       reversed every other round. VPIPE_MING_BENCH_PX (default "1024",
//       e.g. "1024,2048") and VPIPE_MING_BENCH_ARMS (default
//       "dense,sol:0,sol:0.5,ane"; also "solane:<tau>").
//   VPIPE_MING_PROFILE_PX=1024,1536,2048   the per-category split of one
//       RESIDENT forward per size (VPIPE_MING_DIT_PROFILE), with each
//       category's FLOPs and achieved TFLOP/s.

#include "ming-config.h"
#include "ming-dit.h"
#include "ming-layout.h"
#include "ming-test.h"

#include "apple-silicon/coreml/coreml-model-manager.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "common/perf-flops.h"
#include "common/vpipe-format.h"
#include "generative-models/accel-keys.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/shared/accel-settings.h"
#include "generative-models/shared/dit-flops.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

// A test binary is not a plugin: it registers the DiT's own kernels
// itself, through the public API the plugin context forwards to.
extern "C" const unsigned char ming_ops_bf16_metallib[];
extern "C" const unsigned long ming_ops_bf16_metallib_len;

namespace fs = std::filesystem;
namespace accel = vpipe::genai::accel;
using vpipe::FlexData;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;
using vpipe::genai::WeightSet;

namespace {

// ---- a session the ANE tier can run under ---------------------------------
//
// The ANE tier reaches its worker through the session's CoreML manager
// and books its host rows into the generative-model manager's pool, so
// the null session every other case passes declines it -- correctly, and
// silently as far as a numeric bar can tell. This is the smallest context
// that carries both, built from the SDK's own constructors: a plugin
// cannot reach the host's Session. It prints what the model logs, which
// is also how the profile line reaches a reader.
class TestServices final : public vpipe::SessionServicesIntf {
 public:
  void* find_service(std::string_view id) const override
  {
    using vpipe::ServiceId;
    if (id == ServiceId<vpipe::CoreMLModelManager>::value) {
      return coreml.get();
    }
    if (id == ServiceId<vpipe::genai::GenerativeModelManager>::value) {
      return genai.get();
    }
    if (id == ServiceId<MetalCompute>::value) { return mc; }
    return nullptr;
  }
  std::unique_ptr<vpipe::CoreMLModelManager> coreml;
  std::unique_ptr<vpipe::genai::GenerativeModelManager> genai;
  MetalCompute* mc = nullptr;
};

class TestSession final : public vpipe::SessionContextIntf {
 public:
  explicit TestSession(MetalCompute* mc)
  {
    _svc.mc = mc;
    _svc.coreml = std::make_unique<vpipe::CoreMLModelManager>(this);
    _svc.genai = std::make_unique<vpipe::genai::GenerativeModelManager>(this);
  }
  ~TestSession() override
  {
    _svc.genai.reset();
    _svc.coreml.reset();
  }
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
  // The last line the model logged that starts with `prefix`.
  std::string last(const std::string& prefix) const
  {
    std::lock_guard<std::mutex> g(_mu);
    for (auto it = _lines.rbegin(); it != _lines.rend(); ++it) {
      if (it->rfind(prefix, 0) == 0) { return *it; }
    }
    return {};
  }
  void quiet(bool q) { _quiet = q; }

 private:
  void out_(const char* lvl, const vpipe::VpipeFormat& m) const
  {
    const std::string s = m();
    {
      std::lock_guard<std::mutex> g(_mu);
      _lines.push_back(s);
    }
    if (!_quiet) { std::printf("    [%s] %s\n", lvl, s.c_str()); }
  }
  mutable TestServices _svc;
  mutable std::mutex _mu;
  mutable std::vector<std::string> _lines;
  bool _quiet = false;
};

// VPIPE_MING_DIT_ONLY=<a,b,...> runs only the named parts -- small,
// stages, sol, ane, stream -- for a measurement that need not repeat the
// rest. Unset runs everything. (The real model's resident forward always
// runs: every part compares against it.)
bool
part(const char* name)
{
  const char* e = std::getenv("VPIPE_MING_DIT_ONLY");
  if (e == nullptr || *e == '\0') { return true; }
  const std::string s = std::string(",") + e + ",";
  return s.find(std::string(",") + name + ",") != std::string::npos;
}

// The acceleration bags generate-image would hand the family.
FlexData
sol_bag(float tau)
{
  FlexData b = FlexData::make_object();
  accel::set_flag(&b, accel::kSolAttn, true);
  accel::set_real(&b, accel::kSolTau, (double)tau);
  return b;
}

FlexData
ane_bag(double rows)
{
  FlexData b = FlexData::make_object();
  accel::set_flag(&b, accel::kAneFfn, true);
  accel::set_real(&b, accel::kAneRows, rows);
  return b;
}

// ---- bf16 ---------------------------------------------------------------

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

float
round_bf16(float f)
{
  return bf16_to_f32(f32_to_bf16(f));
}

// ---- a golden directory ---------------------------------------------------
//
// manifest.json maps a name to {dtype, shape, file}; the file is raw
// little-endian, C order, no header.
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
  bool has(const std::string& name) const
  {
    return _man.is_object() && _man.as_object().contains(name);
  }
  std::vector<long> shape(const std::string& name) const
  {
    std::vector<long> s;
    if (!has(name)) { return s; }
    const FlexData e = _man.as_object().at(name);
    const FlexData a = e.as_object().at("shape");
    auto arr = a.as_array();
    for (std::size_t i = 0; i < arr.size(); ++i) {
      s.push_back((long)arr[i].as_int(0));
    }
    return s;
  }
  template <typename T>
  std::vector<T> read(const std::string& name) const
  {
    std::vector<T> v;
    if (!has(name)) { return v; }
    const FlexData e = _man.as_object().at(name);
    const std::string file(e.as_object().at("file").as_string(""));
    std::size_t n = 1;
    for (long d : shape(name)) { n *= (std::size_t)d; }
    v.resize(n);
    std::ifstream in(_dir / file, std::ios::binary);
    if (!in) { v.clear(); return v; }
    in.read(reinterpret_cast<char*>(v.data()),
            (std::streamsize)(n * sizeof(T)));
    if (!in) { v.clear(); }
    return v;
  }
  std::vector<float> f32(const std::string& n) const { return read<float>(n); }
  std::vector<std::int32_t> i32(const std::string& n) const
  {
    return read<std::int32_t>(n);
  }

 private:
  fs::path _dir;
  FlexData _man;
};

// The small DiT's transformer/config.json, read for the keys
// ming::read_checkpoint reads for the real one (that reader wants a whole
// checkpoint root, which a small golden is not).
bool
read_dit_config(const fs::path& dir, ming::DitConfig* d)
{
  std::ifstream in(dir / "config.json");
  if (!in) { return false; }
  const FlexData fd = FlexData::from_json(in);
  if (!fd.is_object()) { return false; }
  auto o = fd.as_object();
  auto gi = [&](const char* k, int dflt) {
    return o.contains(k) ? (int)o.at(k).as_int(dflt) : dflt;
  };
  auto gr = [&](const char* k, float dflt) {
    return o.contains(k) ? (float)o.at(k).as_real(dflt) : dflt;
  };
  auto ints = [&](const char* k) {
    std::vector<int> v;
    if (!o.contains(k)) { return v; }
    const FlexData a = o.at(k);
    auto arr = a.as_array();
    for (std::size_t i = 0; i < arr.size(); ++i) {
      v.push_back((int)arr[i].as_int(0));
    }
    return v;
  };
  d->dim = gi("dim", d->dim);
  d->n_layers = gi("n_layers", d->n_layers);
  d->n_refiner_layers = gi("n_refiner_layers", d->n_refiner_layers);
  d->n_heads = gi("n_heads", d->n_heads);
  d->n_kv_heads = gi("n_kv_heads", d->n_kv_heads);
  d->in_channels = gi("in_channels", d->in_channels);
  d->cap_feat_dim = gi("cap_feat_dim", d->cap_feat_dim);
  d->norm_eps = gr("norm_eps", d->norm_eps);
  d->rope_theta = gr("rope_theta", d->rope_theta);
  d->t_scale = gr("t_scale", d->t_scale);
  d->qk_norm = o.contains("qk_norm") ? o.at("qk_norm").as_bool(true) : true;
  d->axes_dims = ints("axes_dims");
  d->axes_lens = ints("axes_lens");
  const std::vector<int> ps = ints("all_patch_size");
  const std::vector<int> fp = ints("all_f_patch_size");
  d->patch = ps.empty() ? 2 : ps.front();
  d->f_patch = fp.empty() ? 1 : fp.front();
  d->padding_mode = o.contains("alignment_padding_mode")
      ? std::string(o.at("alignment_padding_mode").as_string(""))
      : std::string();
  d->multi_frame_output = o.contains("multi_frame_output") &&
                          o.at("multi_frame_output").as_bool(false);
  return true;
}

// ---- one forward ----------------------------------------------------------

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

struct Run {
  bool ok = false;
  std::string err;
  std::vector<std::uint16_t> raw;  // bf16 [img, patch_dim], as returned
  std::vector<float> z;            // unpacked [C, H, W]
  ming::DitEngagement eng;
  double ms = 0.0;
  // What the host's FLOP/s meter was credited with on the GPU over the
  // forward, once its buffers retired.
  std::uint64_t credited = 0;
};

Run
run(ming::MingDit& dit, MetalCompute& mc, const Inputs& in,
    ming::Layout* lay_out = nullptr, float sigma_override = -1.0f)
{
  Run r;
  const ming::DitConfig& d = dit.config();
  const int P = d.patch;
  ming::Layout lay;
  std::string lerr;
  if (!ming::build_layout(in.H / P, in.W / P, in.Q, in.B, d.axes_lens[0],
                          &lay, &lerr)) {
    r.err = "layout: " + lerr;
    return r;
  }
  const int PD = P * P * in.C;
  std::vector<float> packed((std::size_t)lay.img_tokens * PD);
  ming::pack_latents(in.x.data(), in.C, in.H, in.W, P, packed.data());
  SharedBuffer lat = mc.make_shared_buffer(packed.size() * 2);
  // The conditioning contract (ming-family.h): one bf16 tensor of `dim`
  // columns; stream A in the first cap_feat_dim columns of its rows,
  // zero beyond them; stream B rows whole.
  const int CS = d.dim;
  SharedBuffer cap =
      mc.make_shared_buffer((std::size_t)lay.cap_len * CS * 2);
  if (lat.empty() || cap.empty()) { r.err = "alloc"; return r; }
  auto* lb = static_cast<std::uint16_t*>(lat.contents());
  for (std::size_t k = 0; k < packed.size(); ++k) {
    lb[k] = f32_to_bf16(packed[k]);
  }
  auto* cb = static_cast<std::uint16_t*>(cap.contents());
  std::memset(cb, 0, cap.byte_size());
  for (int i = 0; i < in.Q; ++i) {
    for (int c = 0; c < d.cap_feat_dim; ++c) {
      cb[(std::size_t)i * CS + c] =
          f32_to_bf16(in.cap_a[(std::size_t)i * d.cap_feat_dim + c]);
    }
  }
  for (int i = 0; i < in.B; ++i) {
    for (int c = 0; c < d.dim; ++c) {
      cb[(std::size_t)(in.Q + i) * CS + c] =
          f32_to_bf16(in.cap_b[(std::size_t)i * d.dim + c]);
    }
  }
  ming::MingDit::Request req;
  req.latents = &lat;
  req.cap = &cap;
  req.cap_stride = CS;
  req.layout = &lay;
  // The scheduler's sigma; the DiT feeds t_scale * (1 - sigma), which is
  // the golden's t_scale * t.
  req.sigma = sigma_override >= 0.0f ? sigma_override : 1.0f - in.t;
  namespace pf = vpipe::perf_flops;
  const bool meter = pf::available();
  const std::uint64_t g0 = meter ? pf::totals(pf::kGpu).flops : 0;
  const auto t0 = std::chrono::steady_clock::now();
  SharedBuffer o = dit.forward(req, &r.err);
  r.ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  r.eng = dit.last_engagement();
  if (meter) {
    // The last buffer's completion handler may run after the wait that
    // saw it; Sol's first count is credited at the forward's end.
    for (int i = 0; i < 400 && pf::totals(pf::kGpu).flops - g0 <
                                   r.eng.gpu_flops; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    r.credited = pf::totals(pf::kGpu).flops - g0;
  }
  if (o.empty()) { return r; }
  const auto* op = static_cast<const std::uint16_t*>(o.contents());
  r.raw.assign(op, op + packed.size());
  std::vector<float> rows(packed.size());
  for (std::size_t k = 0; k < rows.size(); ++k) {
    rows[k] = bf16_to_f32(r.raw[k]);
  }
  r.z.resize((std::size_t)in.C * in.H * in.W);
  ming::unpack_latents(rows.data(), in.C, in.H, in.W, P, r.z.data());
  r.ok = true;
  if (lay_out != nullptr) { *lay_out = std::move(lay); }
  return r;
}

// perf-flops: what a forward over `lay` hands the host's meter, from the
// structure -- every Linear at 2*M*N*K and every attention (square,
// unmasked) at 4*heads*head_dim per pair. A caption the cache served
// skips its embedder and the context refiner, so neither is counted.
std::uint64_t
flops_want(const ming::DitConfig& d, const ming::Layout& lay, bool hit)
{
  using vpipe::genai::attention_flops;
  using vpipe::genai::gemm_flops;
  const long long H = d.dim, FF = (long long)((double)d.dim / 3.0 * 8.0);
  const long long AD = std::min(d.dim, 256);
  const long long PD = (long long)d.f_patch * d.patch * d.patch *
                       d.in_channels;
  const long long IMG = lay.img_tokens, CAP = lay.cap_len;
  const long long JT = lay.joint_len;
  auto block = [&](long long rows) {
    return 4 * gemm_flops(rows, H, H) + 2 * gemm_flops(rows, FF, H) +
           gemm_flops(rows, H, FF) +
           attention_flops(d.n_heads, rows * rows, d.head_dim());
  };
  const std::uint64_t mod = gemm_flops(1, 4 * H, AD);
  std::uint64_t f = gemm_flops(1, 1024, 256) + gemm_flops(1, AD, 1024) +
                    gemm_flops(IMG, H, PD) +
                    (std::uint64_t)d.n_refiner_layers * (mod + block(IMG)) +
                    (std::uint64_t)d.n_layers * (mod + block(JT)) +
                    gemm_flops(1, H, AD) + gemm_flops(IMG, PD, H);
  if (!hit) {
    f += gemm_flops(lay.query_rows, H, d.cap_feat_dim) +
         (std::uint64_t)d.n_refiner_layers * block(CAP);
  }
  return f;
}

void
print_eng(const ming::DitEngagement& e)
{
  std::printf("    gemm: mma2 %d, i8 %d, bm32 %d, bm64 %d, bm64bn64 %d, "
              "qmm %d\n", e.gemm_mma2, e.gemm_i8, e.gemm_bm32, e.gemm_bm64,
              e.gemm_bm64bn64, e.gemm_qmm);
  std::printf("    attn: steel %d, nax %d, sage %d, sol %d; unaligned q %d, "
              "k %d\n", e.attn_steel, e.attn_nax, e.attn_sage, e.attn_sol,
              e.attn_unaligned_q, e.attn_unaligned_k);
  if (e.attn_sol > 0) {
    std::printf("    sol: kept %lld of %lld key blocks (%.1f%%)\n",
                e.sol_kept, e.sol_total,
                e.sol_total > 0 ? 100.0 * (double)e.sol_kept /
                                      (double)e.sol_total
                                : 0.0);
  }
  if (e.ane_split + e.ane_probe > 0) {
    std::printf("    ane: %d blocks split (%lld rows on the ANE), %d probed, "
                "%d lost rows\n", e.ane_split, e.ane_rows, e.ane_probe,
                e.ane_lost);
  }
  std::printf("    blocks: held %d, streamed %d (prefetch hits %d); caption "
              "cache %s\n", e.blocks_held, e.blocks_streamed,
              e.prefetch_hits, e.cap_cache_hit ? "HIT" : "miss");
  if (e.lora_proj + e.ane_lora > 0) {
    std::printf("    lora: %d projections adapted on the GPU, %d ANE splits "
                "carried the adapter\n", e.lora_proj, e.ane_lora);
  }
}

bool
same_bits(const Run& a, const Run& b)
{
  return a.ok && b.ok && a.raw.size() == b.raw.size() &&
         std::memcmp(a.raw.data(), b.raw.data(), a.raw.size() * 2) == 0;
}

using LoadOptions = ming::MingDit::LoadOptions;

std::unique_ptr<ming::MingDit>
load_dit(MetalCompute& mc, const std::string& dir, const ming::DitConfig& d,
         bool stream, std::string* err,
         std::shared_ptr<WeightSet>* keep_ws = nullptr,
         const LoadOptions& opts = LoadOptions(),
         const vpipe::SessionContextIntf* ss = nullptr)
{
  auto ws = WeightSet::open(dir, nullptr);
  if (!ws) {
    *err = "WeightSet::open failed";
    return nullptr;
  }
  if (keep_ws != nullptr) { *keep_ws = ws; }
  return ming::MingDit::load(std::move(ws), &mc, d, nullptr, stream, ss,
                             err, opts);
}

// Another model over a weight set ALREADY open: its cached tensors are
// shared per tensor, so a second arm costs its scratch, not 12 GB.
std::unique_ptr<ming::MingDit>
load_on(MetalCompute& mc, const std::shared_ptr<WeightSet>& ws,
        const ming::DitConfig& d, const FlexData* accel_bag,
        const vpipe::SessionContextIntf* ss, std::string* err,
        const LoadOptions& opts = LoadOptions())
{
  return ming::MingDit::load(ws, &mc, d, accel_bag, /*stream=*/false, ss,
                             err, opts);
}

// Load options naming one adapter (slot 0) at strength `s`.
LoadOptions
with_lora(const fs::path& file, float s = 1.0f)
{
  LoadOptions o;
  o.loras.push_back({file.string(), s});
  return o;
}

// The golden's caption and timestep over random latents at another size:
// N(0, 1), torch-like, from a fixed seed.
Inputs
at_size(const Inputs& g, int px, unsigned seed)
{
  Inputs in = g;
  in.H = px / 8;
  in.W = px / 8;
  in.x.assign((std::size_t)in.C * in.H * in.W, 0.0f);
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (float& v : in.x) { v = nd(rng); }
  return in;
}

// What a box WITHOUT matrix cores must have run, and what one with them
// must have -- the same forward, two different correct answers.
//
// `n_attn` is the attention dispatches the forward should have made: one
// per block, less the context refiner's when the caption was served from
// the cache (its blocks do not run at all then).
void
check_routes(const MetalCompute& mc, const ming::MingDit& dit,
             const ming::DitEngagement& e, int n_attn)
{
  // The A/B overrides turn a matrix-core route off on purpose.
  const bool cores = mc.supports_matrix_cores();
  const bool want_mma2 =
      cores && std::getenv("VPIPE_MING_NO_MMA2") == nullptr;
  const bool want_nax =
      cores && std::getenv("VPIPE_MING_NO_NAX_ATTN") == nullptr;
  CHECK(dit.uses_mma2() == want_mma2);
  CHECK(dit.uses_attn_nax() == want_nax);
  if (want_mma2) {
    CHECK(e.gemm_mma2 + e.gemm_i8 > 0);
  } else {
    CHECK(e.gemm_mma2 == 0 && e.gemm_i8 == 0);
  }
  if (want_nax) {
    CHECK(e.attn_nax == n_attn && e.attn_steel == 0);
  } else {
    CHECK(e.attn_steel == n_attn && e.attn_nax == 0);
  }
  if (!cores) {
    // int8 GEMMs and Sage need matrix cores. (Sol does not; these
    // forwards do not ask for it.)
    CHECK(!dit.uses_i8_gemm() && !dit.uses_sage_attn());
  }
  CHECK(e.attn_sol == 0 && !dit.uses_sol_attn());
  if (!want_mma2 && std::getenv("VPIPE_MING_GEMM_TILE") == nullptr) {
    // The widest dense tile is the default without matrix cores, and the
    // M >= 128 GEMMs must have taken it -- "not scalar" is not the bar.
    CHECK(dit.gemm_tile() == 2);
    CHECK(e.gemm_bm64bn64 > 0);
  }
  // The M = 1 Linears (timestep, modulation) stay on the 32x32 tile on
  // every box: matmul2d declines under 64 rows.
  CHECK(e.gemm_bm32 > 0);
  CHECK(e.gemm_qmm == 0);       // dense checkpoints only, here
}

// The golden's rows with its pads removed: the reference's sequence is
// [image | image pad | caption | caption pad] (or one half of it).
std::vector<float>
real_rows(const std::vector<float>& g, int dim, int img, int img_pad,
          int cap, bool with_image, bool with_caption)
{
  std::vector<float> out;
  if (with_image) {
    out.insert(out.end(), g.begin(), g.begin() + (std::size_t)img * dim);
  }
  if (with_caption) {
    const std::size_t c0 =
        with_image ? (std::size_t)(img + img_pad) * dim : 0;
    out.insert(out.end(), g.begin() + c0,
               g.begin() + c0 + (std::size_t)cap * dim);
  }
  return out;
}

std::vector<float>
read_trace(const fs::path& dir, const std::string& tag)
{
  std::vector<float> v;
  std::ifstream in(dir / (tag + ".f32"), std::ios::binary | std::ios::ate);
  if (!in) { return v; }
  const std::streamsize n = in.tellg();
  in.seekg(0);
  v.resize((std::size_t)n / sizeof(float));
  in.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

// ---- every stage against the reference's taps ---------------------------
//
// For LOCATING a gap rather than just measuring it. One traced forward
// (VPIPE_MING_DIT_TRACE dumps the joint sequence after every stage); the
// trace commits per stage but runs the same kernels, so its output must
// equal `untraced` bit for bit. Taps the golden does not carry (the real
// one keeps layers 0, 14 and 29) are skipped. The caption cache is off
// for this forward: a served caption has no context-refiner stage.
void
check_stages(ming::MingDit& dit, MetalCompute& mc, const Golden& g,
             const Inputs& in, const Run& untraced, double tol,
             double embed_tol)
{
  char tmpl[] = "/tmp/ming-dit-trace-XXXXXX";
  const char* tdir = mkdtemp(tmpl);
  CHECK(tdir != nullptr);
  if (tdir == nullptr) { return; }
  const ming::DitConfig& d = dit.config();
  dit.set_caption_cache(false);
  setenv("VPIPE_MING_DIT_TRACE", tdir, 1);
  ming::Layout lay;
  const Run r = run(dit, mc, in, &lay);
  unsetenv("VPIPE_MING_DIT_TRACE");
  dit.set_caption_cache(true);
  CHECK(same_bits(r, untraced));   // tracing changes nothing
  const int D = d.dim, IMG = lay.img_tokens, CAP = lay.cap_len;
  const int img_pad = (32 - IMG % 32) % 32;
  const fs::path td(tdir);
  // The embedding, piece by piece: image rows against x_embed, stream A
  // against cap_embed, stream B against its own bf16 input.
  {
    const std::vector<float> e = read_trace(td, "embed");
    const std::vector<float> xe = g.f32("x_embed");
    const std::vector<float> ce = g.f32("cap_embed");
    CHECK(e.size() == (std::size_t)(IMG + CAP) * D);
    if (e.size() == (std::size_t)(IMG + CAP) * D) {
      const auto i0 = e.begin();
      const auto a0 = i0 + (std::ptrdiff_t)IMG * D;
      const auto b0 = a0 + (std::ptrdiff_t)in.Q * D;
      const std::vector<float> ei(i0, a0), ea(a0, b0), eb(b0, e.end());
      std::vector<float> bb(in.cap_b.size());
      for (std::size_t k = 0; k < bb.size(); ++k) {
        bb[k] = round_bf16(in.cap_b[k]);
      }
      const double r1 = mingtest::rel_l2(
          ei, std::vector<float>(xe.begin(),
                                 xe.begin() + (std::ptrdiff_t)IMG * D));
      const double r2 = mingtest::rel_l2(ea, ce);
      const double r3 = mingtest::rel_l2(eb, bb);
      std::printf("    embed: image %.4f, stream A (cap_embedder) %.4f, "
                  "stream B (copied) %.1e\n", r1, r2, r3);
      CHECK(r1 < embed_tol && r2 < embed_tol);
      CHECK(r3 == 0.0);   // stream B bypasses cap_embedder EXACTLY
    }
  }
  struct Stage {
    std::string ours, theirs;
    bool img, cap;
  };
  const int NR = d.n_refiner_layers;
  std::vector<Stage> st;
  st.push_back({"ctx", "context_refiner" + std::to_string(NR - 1), false,
                true});
  for (int i = 0; i < NR; ++i) {
    st.push_back({"blk" + std::to_string(i),
                  "noise_refiner" + std::to_string(i), true, false});
  }
  for (int i = 0; i < d.n_layers; ++i) {
    st.push_back({"blk" + std::to_string(NR + i),
                  "layer" + std::to_string(i), true, true});
  }
  for (const Stage& s : st) {
    if (!g.has(s.theirs)) { continue; }
    const std::vector<float> ours = read_trace(td, s.ours);
    const std::vector<float> gt = g.f32(s.theirs);
    if (ours.empty() || gt.empty()) {
      CHECK(!"a trace or a tap is missing");
      continue;
    }
    // Ours: "ctx" is the caption rows alone, "blkN" the whole joint
    // [image | caption]; a noise-refiner tap is the image alone.
    std::vector<float> mine = ours;
    if (s.img && !s.cap) {
      mine.assign(ours.begin(), ours.begin() + (std::ptrdiff_t)IMG * D);
    }
    const std::vector<float> theirs =
        s.img && s.cap ? real_rows(gt, D, IMG, img_pad, CAP, true, true)
        : s.img        ? real_rows(gt, D, IMG, img_pad, 0, true, false)
                       : real_rows(gt, D, 0, 0, CAP, false, true);
    const double rel = mingtest::rel_l2(mine, theirs);
    std::printf("    %-6s vs %-17s rel-L2 %.4f\n", s.ours.c_str(),
                s.theirs.c_str(), rel);
    CHECK(rel < tol);
  }
  std::error_code ec;
  fs::remove_all(td, ec);
}

// ---- dit_small ------------------------------------------------------------

// The reference's OWN bf16 run sits 0.0158-0.0168 from its f32 compute
// on these three cases (meta.json, rel_l2_bf16_vs_f32): that is the bf16
// floor for this geometry. A vpipe forward is held to about TWICE it. A
// wrong position, a dropped norm, a swapped w1/w3 or a pad row that
// attends all land at O(0.1-1), far above this.
constexpr double kSmallTol = 0.035;

void
test_small(MetalCompute& mc, const fs::path& root)
{
  const fs::path mdir = root / "model";
  ming::DitConfig d;
  CASE("dit_small: config + load");
  CHECK(read_dit_config(mdir, &d));
  std::string err;
  auto dit = load_dit(mc, mdir.string(), d, /*stream=*/false, &err);
  CHECK(dit != nullptr);
  if (!dit) {
    std::printf("    load refused: %s\n", err.c_str());
    return;
  }
  const int n_blocks = 2 * d.n_refiner_layers + d.n_layers;
  std::printf("    dim %d, %d heads, %d layers + %d refiner, cap_feat_dim "
              "%d, ffn %d; matrix cores: %s\n", d.dim, d.n_heads,
              d.n_layers, d.n_refiner_layers, d.cap_feat_dim,
              (int)((double)d.dim / 3.0 * 8.0),
              mc.supports_matrix_cores() ? "yes" : "NO");

  std::map<std::string, Run> resident;
  for (const char* cname : {"case_a", "case_b", "case_c"}) {
    const std::string label = std::string("dit_small ") + cname;
    CASE(label.c_str());
    Golden g;
    Inputs in;
    if (!g.open(root / cname) || !load_inputs(g, &in)) {
      CHECK(!"golden case unreadable");
      continue;
    }
    ming::Layout lay;
    const Run r = run(*dit, mc, in, &lay);
    CHECK(r.ok);
    if (!r.ok) {
      std::printf("    forward failed: %s\n", r.err.c_str());
      continue;
    }
    // The layout against the reference's own position ids, real rows
    // only: the image's axis 0 must count the caption's PADS.
    {
      const std::vector<std::int32_t> xp = g.i32("x_pos_ids");
      const std::vector<std::int32_t> cp = g.i32("cap_pos_ids");
      bool pos_ok = xp.size() >= (std::size_t)lay.img_tokens * 3 &&
                    cp.size() >= (std::size_t)lay.cap_len * 3;
      for (int i = 0; pos_ok && i < lay.img_tokens; ++i) {
        const std::size_t j = (std::size_t)i;
        pos_ok = xp[j * 3] == lay.pos_t[j] &&
                 xp[j * 3 + 1] == lay.pos_h[j] &&
                 xp[j * 3 + 2] == lay.pos_w[j];
      }
      for (int i = 0; pos_ok && i < lay.cap_len; ++i) {
        const std::size_t j = (std::size_t)(lay.img_tokens + i);
        pos_ok = cp[(std::size_t)i * 3] == lay.pos_t[j] &&
                 cp[(std::size_t)i * 3 + 1] == 0 &&
                 cp[(std::size_t)i * 3 + 2] == 0;
      }
      CHECK(pos_ok);
      std::printf("    %d image + %d caption rows (joint %d, pads "
                  "dropped), image axis-0 position %d\n", lay.img_tokens,
                  lay.cap_len, lay.joint_len, lay.pos_t[0]);
    }
    const std::vector<float> want = g.f32("out");
    const std::vector<float> bar = g.f32("out_bf16_mps");
    const double rel = mingtest::rel_l2(r.z, want);
    const double ref_bf16 = mingtest::rel_l2(bar, want);
    const double vs_bar = mingtest::rel_l2(r.z, bar);
    std::printf("    rel-L2 vs reference f32 %.4f (reference's own bf16 "
                "%.4f; vpipe vs reference bf16 %.4f; tol %.3f), %.1f ms\n",
                rel, ref_bf16, vs_bar, kSmallTol, r.ms);
    CHECK(rel < kSmallTol);
    print_eng(r.eng);
    // The host's FLOP/s meter: the count exact from the structure, and
    // all of it credited once the buffers retired. (Zero, and skipped,
    // on a host built before the meter.)
    if (vpipe::perf_flops::available()) {
      const std::uint64_t want = flops_want(d, lay, r.eng.cap_cache_hit);
      std::printf("    perf-flops: counted %.6f TFLOP, derived %.6f, "
                  "credited %.6f\n", r.eng.gpu_flops / 1e12, want / 1e12,
                  r.credited / 1e12);
      CHECK(r.eng.gpu_flops == want);
      CHECK(r.credited == r.eng.gpu_flops);
    }
    // A caption already cached (case_c shares case_a's) skips the
    // context refiner -- which is the cache serving a caption across a
    // DIFFERENT picture size, and still matching the reference.
    const int nr = d.n_refiner_layers, nl = d.n_layers;
    const int ctx_runs = r.eng.cap_cache_hit ? 0 : nr;
    check_routes(mc, *dit, r.eng, n_blocks - nr + ctx_runs);
    // The three attention lengths: caption, image, joint. Each block
    // that runs at a length that is not a multiple of the tile is one
    // unaligned dispatch.
    const int qt = mc.supports_matrix_cores() ? 64 : 32;
    const int kt = mc.supports_matrix_cores() ? 32 : 16;
    auto off = [](int rows, int tile) { return (rows % tile) != 0 ? 1 : 0; };
    const int want_uq = ctx_runs * off(lay.cap_len, qt) +
                        nr * off(lay.img_tokens, qt) +
                        nl * off(lay.joint_len, qt);
    const int want_uk = ctx_runs * off(lay.cap_len, kt) +
                        nr * off(lay.img_tokens, kt) +
                        nl * off(lay.joint_len, kt);
    CHECK(r.eng.attn_unaligned_q == want_uq);
    CHECK(r.eng.attn_unaligned_k == want_uk);
    CHECK(r.eng.attn_unaligned_q > 0);
    CHECK(r.eng.blocks_held == nr + nl);
    resident[cname] = r;
  }
  if (resident.count("case_a") == 0 || resident.count("case_b") == 0) {
    return;
  }

  Golden ga, gb;
  Inputs a, b;
  ga.open(root / "case_a");
  load_inputs(ga, &a);
  gb.open(root / "case_b");
  load_inputs(gb, &b);

  CASE("dit_small case_a: every stage against the reference's taps");
  check_stages(*dit, mc, ga, a, resident["case_a"], kSmallTol, 0.01);

  // ---- the caption cache ------------------------------------------------
  CASE("dit_small: the caption cache is bit-identical to recomputing");
  {
    auto cold = [&]() {
      dit->set_caption_cache(false);    // drops every entry
      dit->set_caption_cache(true);
    };
    cold();
    const Run miss = run(*dit, mc, a);
    const Run hit = run(*dit, mc, a);
    CHECK(miss.ok && !miss.eng.cap_cache_hit);
    CHECK(hit.ok && hit.eng.cap_cache_hit);
    CHECK(same_bits(miss, resident["case_a"]));
    CHECK(same_bits(hit, resident["case_a"]));
    // Another sigma, cache on (a hit) vs off (a recompute).
    const Run on = run(*dit, mc, a, nullptr, 0.25f);
    dit->set_caption_cache(false);
    const Run off = run(*dit, mc, a, nullptr, 0.25f);
    CHECK(on.ok && on.eng.cap_cache_hit);
    CHECK(off.ok && !off.eng.cap_cache_hit);
    CHECK(same_bits(on, off));
    // Two captions alternating from cold -- the CFG shape: both stay
    // cached, and each is served as itself.
    cold();
    const Run a1 = run(*dit, mc, a);
    const Run b1 = run(*dit, mc, b);
    const Run a2 = run(*dit, mc, a);
    const Run b2 = run(*dit, mc, b);
    CHECK(!a1.eng.cap_cache_hit && !b1.eng.cap_cache_hit);
    CHECK(a2.eng.cap_cache_hit && b2.eng.cap_cache_hit);
    CHECK(same_bits(a2, resident["case_a"]));
    CHECK(same_bits(b1, resident["case_b"]) && same_bits(b2, b1));
    std::printf("    hit %s; another sigma, hit == recompute: %s; a/b "
                "alternation: %s, %s\n",
                same_bits(hit, resident["case_a"]) ? "identical" : "DIFFERS",
                same_bits(on, off) ? "identical" : "DIFFERS",
                a2.eng.cap_cache_hit && b2.eng.cap_cache_hit ? "both hit"
                                                             : "MISSED",
                same_bits(a2, resident["case_a"]) &&
                        same_bits(b2, resident["case_b"])
                    ? "identical"
                    : "DIFFERS");
  }

  // ---- stop and progress --------------------------------------------------
  CASE("dit_small: a stop is empty with no error, and leaves nothing "
       "behind");
  {
    dit->set_caption_cache(false);
    dit->set_caption_cache(true);   // start cold
    std::atomic<int> polls{0};
    // Past the context refiner, so the cache entry was ENCODED -- and
    // must not be served, since the pass never completed.
    const int stop_after = d.n_refiner_layers + 1;
    dit->set_stream_stop([&]() { return ++polls > stop_after; });
    const Run s = run(*dit, mc, a);
    dit->set_stream_stop(nullptr);
    CHECK(!s.ok && s.err.empty());
    const Run after = run(*dit, mc, a);
    CHECK(after.ok && !after.eng.cap_cache_hit);
    CHECK(same_bits(after, resident["case_a"]));
    std::vector<int> seen;
    std::atomic<int> published{0};
    dit->set_block_progress(
        [&](int done, int total) -> std::function<void()> {
          seen.push_back(done);
          CHECK(total == n_blocks);
          return [&published]() { ++published; };
        });
    const Run p = run(*dit, mc, a);
    dit->set_block_progress(nullptr);
    bool order = (int)seen.size() == n_blocks;
    for (int i = 0; order && i < n_blocks; ++i) {
      order = seen[(std::size_t)i] == i;
    }
    CHECK(p.ok && order);
    // Publishes ride command-buffer completion; every one has fired by
    // the time the forward's final wait returned.
    CHECK(published.load() == n_blocks);
    std::printf("    stop: empty, no error, cache not poisoned; progress "
                "0..%d of %d in order, %d published\n", n_blocks - 1,
                n_blocks, published.load());
  }
  dit.reset();

  // ---- streaming --------------------------------------------------------
  CASE("dit_small: streamed == resident, bit for bit");
  {
    auto sdit = load_dit(mc, mdir.string(), d, /*stream=*/true, &err);
    CHECK(sdit != nullptr && sdit->streaming());
    if (sdit) {
      const Run s = run(*sdit, mc, a);
      CHECK(s.ok);
      CHECK(s.eng.blocks_streamed == d.n_refiner_layers + d.n_layers);
      CHECK(same_bits(s, resident["case_a"]));
      std::printf("    streamed %d blocks, prefetch hits %d: %s\n",
                  s.eng.blocks_streamed, s.eng.prefetch_hits,
                  same_bits(s, resident["case_a"]) ? "identical"
                                                   : "DIFFERS");
      // Residency: with a reserve DECLARED, blocks are promoted out of
      // the slots and later forwards run them held.
      sdit->set_residency_reserve(0);
      sdit->set_residency_schedule(3);
      const Run p1 = run(*sdit, mc, a);
      const Run p2 = run(*sdit, mc, a);
      CHECK(sdit->resident_block_count() > 0);
      CHECK(p2.eng.blocks_held > 0);
      CHECK(same_bits(p1, resident["case_a"]) &&
            same_bits(p2, resident["case_a"]));
      std::printf("    residency: %d blocks promoted; the next forward "
                  "ran %d held + %d streamed: %s\n",
                  sdit->resident_block_count(), p2.eng.blocks_held,
                  p2.eng.blocks_streamed,
                  same_bits(p2, resident["case_a"]) ? "identical"
                                                    : "DIFFERS");
    }
  }
  CASE("dit_small: VPIPE_MING_NO_SLOTS (per-block fallback) == resident");
  {
    setenv("VPIPE_MING_NO_SLOTS", "1", 1);
    auto ndit = load_dit(mc, mdir.string(), d, /*stream=*/true, &err);
    CHECK(ndit != nullptr);
    if (ndit) {
      const Run s = run(*ndit, mc, a);
      CHECK(s.ok && s.eng.prefetch_hits == 0);
      CHECK(same_bits(s, resident["case_a"]));
      std::printf("    no slots: %s\n", same_bits(s, resident["case_a"])
                                            ? "identical" : "DIFFERS");
    }
    unsetenv("VPIPE_MING_NO_SLOTS");
  }
}

// ---- runtime LoRA on the small model --------------------------------------
//
// CHECKED AGAINST THE REFERENCE'S OWN DENSE ARITHMETIC: the dumper writes a
// synthetic adapter AND the same model with alpha/r * B@A folded into its
// weights, and runs the reference on the fused one in fp32. So this
// compares `base + runtime adapter` against `(base + delta) run densely`,
// not against a re-derivation of the same idea in this file.
//
// WHAT IT CANNOT ESTABLISH is that a PUBLISHED Ming-Image adapter will
// bind: none exists. The names are the reference DiT's own (it is a
// diffusers PeftAdapterMixin, and the diffusers file below is what its
// save_lora_adapter wrote), and the pipeline and kohya spellings are the
// two other conventions the toolkit maps. All three must bind the same
// modules AND give the same bits -- which also proves the file's alpha
// was READ in each (metadata for two, a per-module tensor for kohya):
// rank 8 against alpha 16, so an alpha taken as the rank halves the
// adapter.
//
// THE ASSERTIONS, in the order the checklist asks for them: it BOUND
// modules (all of them, and nothing skipped); the forward ENGAGED it at
// the dispatch site; it matches the fused reference at the bf16 floor;
// it MOVES the output far past that floor (or the match is vacuous);
// strength 0 is the base EXACTLY; streamed == resident, bit for bit.

// A safetensors file of bf16 tensors (the values do not matter here), for
// adapters this test makes up: one for ANOTHER model, one at the wrong
// shape.
bool
write_bf16_st(const fs::path& p,
              const std::vector<std::pair<std::string, std::vector<long>>>& ts)
{
  std::string hdr = "{";
  std::size_t off = 0;
  for (std::size_t i = 0; i < ts.size(); ++i) {
    std::size_t n = 1;
    for (long d : ts[i].second) { n *= (std::size_t)d; }
    std::string shape;
    for (long d : ts[i].second) {
      shape += (shape.empty() ? "" : ",") + std::to_string(d);
    }
    hdr += (i ? "," : "") + std::string("\"") + ts[i].first +
           "\":{\"dtype\":\"BF16\",\"shape\":[" + shape +
           "],\"data_offsets\":[" + std::to_string(off) + "," +
           std::to_string(off + n * 2) + "]}";
    off += n * 2;
  }
  hdr += "}";
  while (hdr.size() % 8 != 0) { hdr += ' '; }
  std::ofstream o(p, std::ios::binary);
  if (!o) { return false; }
  const std::uint64_t hl = hdr.size();
  o.write(reinterpret_cast<const char*>(&hl), 8);
  o.write(hdr.data(), (std::streamsize)hdr.size());
  const std::uint16_t one = f32_to_bf16(0.01f);
  for (std::size_t i = 0; i < off / 2; ++i) {
    o.write(reinterpret_cast<const char*>(&one), 2);
  }
  return (bool)o;
}

// The fused reference's output for one case, and the reference's own bf16
// run of base + adapter.
struct LoraGolden {
  std::vector<float> fused, bf16;
};

LoraGolden
lora_golden(const fs::path& dir)
{
  LoraGolden g;
  Golden m;
  if (!m.open(dir)) { return g; }
  g.fused = m.f32("lora_out");
  g.bf16 = m.f32("lora_out_bf16_mps");
  return g;
}

void
test_lora_small(MetalCompute& mc, const fs::path& root, TestSession* ts)
{
  const fs::path ldir = root / "lora";
  const fs::path mdir = root / "model";
  CASE("dit_small LoRA: every spelling binds every module, engages, and "
       "matches the FUSED reference at the bf16 floor");
  ming::DitConfig d;
  CHECK(read_dit_config(mdir, &d));
  int want_mods = 0;
  {
    std::ifstream in(ldir / "meta.json");
    const FlexData m = FlexData::from_json(in);
    if (m.is_object() && m.as_object().contains("modules")) {
      want_mods = (int)m.as_object().at("modules").as_int(0);
    }
  }
  // The capacity is the model's (7 x blocks); the golden states how many
  // it wrote. Both must agree, or the module list drifted.
  CHECK(want_mods == ming::MingDit::lora_module_capacity(d));
  CHECK(want_mods == 7 * (2 * d.n_refiner_layers + d.n_layers));
  const char* cases[] = {"case_a", "case_b", "case_c"};
  std::map<std::string, Inputs> in;
  std::map<std::string, LoraGolden> lg;
  for (const char* c : cases) {
    Golden g;
    Inputs x;
    CHECK(g.open(root / c) && load_inputs(g, &x));
    in[c] = x;
    lg[c] = lora_golden(ldir / c);
    CHECK(!lg[c].fused.empty() && !lg[c].bf16.empty());
  }
  if (lg["case_a"].fused.empty()) { return; }
  std::string err;
  // The UNADAPTED model in this port's own precision -- the arm strength
  // 0 has to reproduce, and the one "moved" is measured from.
  std::map<std::string, Run> base;
  {
    auto b = load_dit(mc, mdir.string(), d, false, &err);
    CHECK(b != nullptr);
    if (!b) { return; }
    for (const char* c : cases) { base[c] = run(*b, mc, in[c]); }
  }
  const int n_mod = d.n_refiner_layers + d.n_layers;

  // (a) + engagement + (b), per spelling; the first is kept for the rest.
  std::unique_ptr<ming::MingDit> m;
  Run first;
  for (const char* sp : {"diffusers", "pipeline", "kohya"}) {
    const fs::path f = ldir / (std::string("adapter_") + sp +
                               ".safetensors");
    auto mm = load_dit(mc, mdir.string(), d, false, &err, nullptr,
                       with_lora(f), ts);
    CHECK(mm != nullptr);
    if (!mm) { continue; }
    const Run r = run(*mm, mc, in["case_a"]);
    CHECK(r.ok);
    const double rel = mingtest::rel_l2(r.z, lg["case_a"].fused);
    std::printf("    %-9s bound %d of %d (skipped %d); %d projections "
                "adapted in the forward; vs fused fp32 %.4f%s\n", sp,
                mm->lora_modules(0), want_mods, mm->lora_skipped(0),
                r.eng.lora_proj, rel,
                first.ok ? (same_bits(r, first) ? ", bits == diffusers"
                                                : ", bits DIFFER")
                         : "");
    CHECK(mm->lora_modules(0) == want_mods);
    CHECK(mm->lora_skipped(0) == 0);
    // ENGAGED at the dispatch site: a cold caption cache runs every block,
    // so every bound projection was adapted exactly once.
    CHECK(r.eng.lora_proj == want_mods);
    CHECK(rel < kSmallTol);
    if (!first.ok) {
      first = r;
      m = std::move(mm);
    } else {
      CHECK(same_bits(r, first));
    }
  }
  if (!m || !first.ok) { return; }

  // (b) + (c) on every case: against the fused reference, against the
  // reference's own bf16 run of the same adapter, and how far it moved.
  for (const char* c : cases) {
    const Run r = std::string(c) == "case_a" ? first : run(*m, mc, in[c]);
    CHECK(r.ok);
    const double vs_fused = mingtest::rel_l2(r.z, lg[c].fused);
    const double ref_bf16 = mingtest::rel_l2(lg[c].bf16, lg[c].fused);
    const double moved = mingtest::rel_l2(base[c].z, r.z);
    const double base_vs_fused = mingtest::rel_l2(base[c].z, lg[c].fused);
    std::printf("    %s: vs fused fp32 %.4f (reference's own bf16 %.4f; "
                "tol %.3f); the adapter moves this port's output %.4f "
                "(base vs fused %.4f)\n", c, vs_fused, ref_bf16, kSmallTol,
                moved, base_vs_fused);
    CHECK(vs_fused < kSmallTol);
    // MOVED FAR PAST THE FLOOR, or "matches the fused reference" would
    // be a test of nothing: an adapter applied at the wrong strength, or
    // not at all, lands at O(moved), not at O(floor).
    CHECK(moved > 0.1 && moved > 10.0 * vs_fused);
    CHECK(base_vs_fused > 10.0 * vs_fused);
  }

  CASE("dit_small LoRA: strength 0 is the base EXACTLY; the strength is "
       "live and drops the caption cache");
  {
    // case_a's caption is cached by now (case_c shares it), built at
    // strength 1 through the context refiner's adapted projections.
    m->set_lora_scale(0, 0.0f);
    const Run z0 = run(*m, mc, in["case_a"]);
    const Run z1 = run(*m, mc, in["case_a"]);
    m->set_lora_scale(0, 1.0f);
    const Run back = run(*m, mc, in["case_a"]);
    CHECK(z0.ok && z1.ok && back.ok);
    CHECK(!z0.eng.cap_cache_hit);   // the change dropped the old rows
    CHECK(z1.eng.cap_cache_hit);    // ...and they are cached again
    CHECK(z0.eng.lora_proj == 0);   // 0 encodes nothing
    CHECK(same_bits(z0, base["case_a"]) && same_bits(z1, z0));
    CHECK(!back.eng.cap_cache_hit && same_bits(back, first));
    std::printf("    strength 0: %s the base (memcmp), %d adapter GEMMs; "
                "back to 1: %s the first adapted run; caption cache "
                "dropped on each change: %s\n",
                same_bits(z0, base["case_a"]) ? "IDENTICAL to" : "DIFFERS from",
                z0.eng.lora_proj,
                same_bits(back, first) ? "identical to" : "DIFFERS from",
                !z0.eng.cap_cache_hit && !back.eng.cap_cache_hit ? "yes"
                                                                 : "NO");
  }

  CASE("dit_small LoRA: two slots; the slots are positional");
  {
    const fs::path fd = ldir / "adapter_diffusers.safetensors";
    const fs::path fk = ldir / "adapter_kohya.safetensors";
    LoadOptions two;
    two.loras = {{fd.string(), 0.5f}, {fk.string(), 0.5f}};
    auto m2 = load_dit(mc, mdir.string(), d, false, &err, nullptr, two, ts);
    CHECK(m2 != nullptr);
    if (m2) {
      const Run r = run(*m2, mc, in["case_a"]);
      const double rel = mingtest::rel_l2(r.z, lg["case_a"].fused);
      std::printf("    the same adapter in both slots at 0.5 each: bound "
                  "%d + %d, %d projections adapted; vs fused %.4f\n",
                  m2->lora_modules(0), m2->lora_modules(1), r.eng.lora_proj,
                  rel);
      CHECK(m2->lora_modules(0) == want_mods &&
            m2->lora_modules(1) == want_mods);
      CHECK(r.ok && r.eng.lora_proj == want_mods && rel < kSmallTol);
    }
    // `lora2` alone: slot 0 stays EMPTY rather than the adapter moving
    // into it, so the second strength still reaches it.
    LoadOptions only2;
    only2.loras = {{"", 1.0f}, {fk.string(), 1.0f}};
    auto m3 = load_dit(mc, mdir.string(), d, false, &err, nullptr, only2,
                       ts);
    CHECK(m3 != nullptr);
    if (m3) {
      const Run r = run(*m3, mc, in["case_a"]);
      CHECK(m3->lora_modules(0) == 0 && m3->lora_modules(1) == want_mods);
      CHECK(same_bits(r, first));
      m3->set_lora_scale(0, 0.7f);  // an empty slot: no effect
      m3->set_lora_scale(1, 0.0f);
      const Run off = run(*m3, mc, in["case_a"]);
      CHECK(same_bits(off, base["case_a"]));
      std::printf("    `lora2` alone: slot 1 bound %d, == slot 0's bits: "
                  "%s; its strength 0 == base: %s\n", m3->lora_modules(1),
                  same_bits(r, first) ? "yes" : "NO",
                  same_bits(off, base["case_a"]) ? "yes" : "NO");
    }
    // `lora8` alone, on a host whose DiT holds eight: the seven slots
    // before it stay EMPTY, and the last one is the same adapter.
    if (ming::MingDit::kMaxLoraSlots >= 8) {
      LoadOptions only8;
      only8.loras.assign(7, ming::MingDit::LoraSpec{"", 1.0f});
      only8.loras.push_back({fk.string(), 1.0f});
      auto m8 = load_dit(mc, mdir.string(), d, false, &err, nullptr, only8,
                         ts);
      CHECK(m8 != nullptr);
      if (m8) {
        const Run r = run(*m8, mc, in["case_a"]);
        bool empty = true;
        for (int i = 0; i < 7; ++i) {
          empty = empty && m8->lora_modules(i) == 0;
        }
        CHECK(empty && m8->lora_modules(7) == want_mods);
        CHECK(same_bits(r, first));
        m8->set_lora_scale(7, 0.0f);
        const Run off = run(*m8, mc, in["case_a"]);
        CHECK(same_bits(off, base["case_a"]));
        std::printf("    `lora8` alone: slot 7 bound %d, == slot 0's bits: "
                    "%s; its strength 0 == base: %s\n",
                    m8->lora_modules(7), same_bits(r, first) ? "yes" : "NO",
                    same_bits(off, base["case_a"]) ? "yes" : "NO");
      }
    }
  }

  CASE("dit_small LoRA: an adapter for ANOTHER model binds nothing, one "
       "partly at the wrong shape is refused WHOLE, and both SAY so");
  {
    const fs::path tmp = fs::temp_directory_path();
    const fs::path foreign = tmp / "ming_lora_foreign.safetensors";
    const fs::path wrong = tmp / "ming_lora_wrong_shape.safetensors";
    const long H = d.dim;
    // A Krea-2 / Qwen-Image style name: nothing this model has.
    CHECK(write_bf16_st(foreign,
                        {{"transformer_blocks.0.attn.to_q.lora_A.weight",
                          {4, H}},
                         {"transformer_blocks.0.attn.to_q.lora_B.weight",
                          {H, 4}}}));
    // This model's names, one at another model's width: the module that
    // DOES fit must not be applied alone.
    CHECK(write_bf16_st(wrong,
                        {{"layers.0.attention.to_q.lora_A.weight",
                          {4, H / 2}},
                         {"layers.0.attention.to_q.lora_B.weight",
                          {H / 2, 4}},
                         {"layers.1.attention.to_q.lora_A.weight", {4, H}},
                         {"layers.1.attention.to_q.lora_B.weight",
                          {H, 4}}}));
    for (int k = 0; k < 2; ++k) {
      const fs::path f = k == 0 ? foreign : wrong;
      auto mf = load_dit(mc, mdir.string(), d, false, &err, nullptr,
                         with_lora(f), ts);
      CHECK(mf != nullptr);
      if (!mf) { continue; }
      const std::string said = ts->last("ming-image DiT: lora '");
      const Run r = run(*mf, mc, in["case_a"]);
      std::printf("    %s: bound %d, skipped %d; said \"%.90s...\"\n",
                  k == 0 ? "foreign" : "wrong shape", mf->lora_modules(0),
                  mf->lora_skipped(0), said.c_str());
      CHECK(mf->lora_modules(0) == 0);
      CHECK(mf->lora_skipped(0) == (k == 0 ? 0 : 1));
      CHECK(said.find(k == 0 ? "bound NO modules" : "is NOT applied") !=
            std::string::npos);
      CHECK(k == 0 || said.find("wrong shape (1 fit)") != std::string::npos);
      CHECK(r.eng.lora_proj == 0 && same_bits(r, base["case_a"]));
    }
    std::error_code ec;
    fs::remove(foreign, ec);
    fs::remove(wrong, ec);
  }

  CASE("dit_small LoRA: streamed == resident WITH the adapter, bit for bit");
  {
    const LoadOptions o = with_lora(ldir / "adapter_diffusers.safetensors");
    auto sm = load_dit(mc, mdir.string(), d, true, &err, nullptr, o);
    CHECK(sm != nullptr && sm->streaming());
    if (sm) {
      const Run s = run(*sm, mc, in["case_a"]);
      CHECK(s.ok && s.eng.blocks_streamed == n_mod);
      CHECK(s.eng.lora_proj == want_mods);
      CHECK(same_bits(s, first));
      // Promoted blocks are the model's own buffers; their adapter is
      // still the resident one.
      sm->set_residency_reserve(0);
      sm->set_residency_schedule(3);
      const Run p1 = run(*sm, mc, in["case_a"]);
      const Run p2 = run(*sm, mc, in["case_a"]);
      CHECK(p2.eng.blocks_held > 0);
      CHECK(same_bits(p1, first) && same_bits(p2, first));
      std::printf("    streamed %d blocks (prefetch hits %d): %s; then %d "
                  "held + %d streamed: %s\n", s.eng.blocks_streamed,
                  s.eng.prefetch_hits,
                  same_bits(s, first) ? "identical" : "DIFFERS",
                  p2.eng.blocks_held, p2.eng.blocks_streamed,
                  same_bits(p2, first) ? "identical" : "DIFFERS");
    }
    setenv("VPIPE_MING_NO_SLOTS", "1", 1);
    auto nm = load_dit(mc, mdir.string(), d, true, &err, nullptr, o);
    CHECK(nm != nullptr);
    if (nm) {
      const Run s = run(*nm, mc, in["case_a"]);
      CHECK(s.ok && s.eng.prefetch_hits == 0 && same_bits(s, first));
      std::printf("    no slots: %s\n",
                  same_bits(s, first) ? "identical" : "DIFFERS");
    }
    unsetenv("VPIPE_MING_NO_SLOTS");
  }

  CASE("dit_small LoRA: the ANE's rows carry the adapter");
  {
    // The ANE takes whole chunks of rows and leaves the GPU at least one,
    // so this 357-row joint needs a 256-row module (VPIPE_MING_ANE_CHUNK,
    // read when the tier is built) and a pinned share that rounds to one
    // chunk. Engagement first: a tier that declines passes everything.
    const LoadOptions o = with_lora(ldir / "adapter_diffusers.safetensors");
    auto ane_arm = [&](bool drop, Run* out) {
      const FlexData bag = ane_bag(0.8);
      if (drop) { setenv("VPIPE_MING_TEST_ANE_DROP_LORA", "1", 1); }
      auto ws = WeightSet::open(mdir.string(), nullptr);
      auto am = ws ? ming::MingDit::load(ws, &mc, d, &bag, false, ts, &err,
                                         o)
                   : nullptr;
      unsetenv("VPIPE_MING_TEST_ANE_DROP_LORA");
      if (!am) { return std::unique_ptr<ming::MingDit>(); }
      setenv("VPIPE_MING_ANE_CHUNK", "256", 1);
      *out = run(*am, mc, in["case_a"]);
      unsetenv("VPIPE_MING_ANE_CHUNK");
      return am;
    };
    Run a, x;
    auto am = ane_arm(false, &a);
    CHECK(am != nullptr);
    if (am && am->ane_armed()) {
      print_eng(a.eng);
      CHECK(a.ok && a.eng.ane_split == d.n_layers && a.eng.ane_lost == 0);
      CHECK(a.eng.ane_lora == a.eng.ane_split);
      CHECK(a.eng.lora_proj > 0);
      const double r = mingtest::rel_l2(a.z, lg["case_a"].fused);
      const double rg = mingtest::rel_l2(first.z, lg["case_a"].fused);
      auto xm = ane_arm(true, &x);
      const double rx = x.ok ? mingtest::rel_l2(x.z, lg["case_a"].fused)
                             : 1.0;
      std::printf("    ANE took %lld rows in %d blocks: vs fused fp32 %.4f "
                  "(GPU %.4f; tol %.3f). CONTROL -- the bare weights "
                  "staged, the ANE's rows unadapted: %.4f\n",
                  a.eng.ane_rows, a.eng.ane_split, r, rg, kSmallTol, rx);
      CHECK(r < kSmallTol);
      // The bar CAN fail: without the adapter on its rows the same
      // forward lands an order of magnitude further out.
      CHECK(x.ok && x.eng.ane_lora == 0 && rx > 5.0 * r);
    } else {
      std::printf("    NOT ARMED at this geometry -- the real model's "
                  "1024^2 case carries the ANE check\n");
    }
  }
}

// ---- Sol-Attn on the real model ----------------------------------------
//
// ENGAGEMENT FIRST, on THIS box: the in-tree file this DiT came from
// gated Sol on the matrix-core attention, so on an M4 it never ran and
// every numeric bar passed. Then the one case with a known answer --
// every key block kept, where the routing, both partials and the merge
// all still run and the output must be the dense one -- and then what
// each tau costs, against the dense run and against the reference.
//
// Sol is an APPROXIMATION: these bars say the routing works and how far
// it moves the output, not which tau is safe. That is a picture's call.
void
test_sol(MetalCompute& mc, const std::shared_ptr<WeightSet>& ws,
         const ming::DitConfig& d, const Inputs& in, const Run& dense,
         const std::vector<float>& ref)
{
  const int n_mod = d.n_refiner_layers + d.n_layers;
  const int n_attn = 2 * d.n_refiner_layers + d.n_layers;
  // dense_layers 1 keeps modulated block 0 (noise_refiner.0) dense, and
  // every other modulated block clears the floors at this geometry.
  const int want_sol = n_mod - 1;
  const double ref_dense = mingtest::rel_l2(dense.z, ref);
  std::unique_ptr<ming::MingDit> keep0;   // tau 0, for the sink A/B
  auto arm = [&](float tau, Run* out) {
    const FlexData bag = sol_bag(tau);
    std::string err;
    auto m = load_on(mc, ws, d, &bag, nullptr, &err);
    CHECK(m != nullptr && m->uses_sol_attn());
    if (!m) {
      std::printf("    load refused: %s\n", err.c_str());
      return false;
    }
    *out = run(*m, mc, in);
    CHECK(out->ok);
    if (!out->ok) {
      std::printf("    forward failed: %s\n", out->err.c_str());
      return false;
    }
    // ENGAGEMENT, before any number: routed on this box, the dense rest
    // on its own flash kernel, and the routing really chose.
    CHECK(out->eng.attn_sol == want_sol);
    CHECK(out->eng.attn_sol + out->eng.attn_steel + out->eng.attn_nax ==
          n_attn);
    CHECK(out->eng.sol_total > 0 && out->eng.sol_kept > 0);
    // perf-flops. A fresh model's FIRST forward has no keep ratio yet,
    // so it counts its routed attention at nothing as it encodes and
    // credits what the routing KEPT when it finishes: the dense
    // forward's attention over the routed blocks, at kept / total.
    if (vpipe::perf_flops::available()) {
      const std::uint64_t routed = dense.eng.gpu_flops - out->eng.gpu_flops;
      const double r = std::clamp((double)out->eng.sol_kept /
                                      (double)out->eng.sol_total, 0.0, 1.0);
      const std::uint64_t kept = (std::uint64_t)((double)routed * r);
      CHECK(routed > 0);
      CHECK(out->credited == out->eng.gpu_flops + kept);
    }
    if (tau == 0.0f) { keep0 = std::move(m); }
    return true;
  };
  auto pct = [](const Run& r) {
    return r.eng.sol_total > 0
               ? 100.0 * (double)r.eng.sol_kept / (double)r.eng.sol_total
               : 0.0;
  };

  Run all;
  if (arm(-1.0e30f, &all)) {
    print_eng(all.eng);
    const double r = mingtest::rel_l2(all.z, dense.z);
    std::printf("    EVERY BLOCK KEPT (tau -inf): %d of %d attentions "
                "routed, %.1f%% kept; vs dense rel-L2 %.3e (%s), %.0f ms "
                "against dense %.0f\n", all.eng.attn_sol, n_attn, pct(all),
                r, same_bits(all, dense) ? "bit-identical" : "not bitwise",
                all.ms, dense.ms);
    CHECK(all.eng.sol_kept == all.eng.sol_total);
    // BIT-IDENTICAL, MEASURED on this ALU arm (and on the NAX arm in
    // Qwen-Image-2.1's own test): with nothing approximated the merge
    // takes the exact partial whole, so the routed forward IS the dense
    // one -- 31 blocks of it, with the caption sink on.
    CHECK(same_bits(all, dense));
    CHECK(r == 0.0);
  }
  std::printf("    dense vs reference fp32: %.4f\n", ref_dense);
  double prev = -1.0;
  for (float tau : {-1.0f, 0.0f, 0.5f, 1.0f}) {
    Run s;
    if (!arm(tau, &s)) { continue; }
    const double vd = mingtest::rel_l2(s.z, dense.z);
    const double vr = mingtest::rel_l2(s.z, ref);
    std::printf("    tau %+.1f: kept %5.1f%% of key blocks; vs dense %.4f, "
                "vs reference fp32 %.4f (dense's %.4f); %.0f ms\n", tau,
                pct(s), vd, vr, ref_dense, s.ms);
    // It ROUTES (something was skipped, and it moved the output) ...
    CHECK(s.eng.sol_kept < s.eng.sol_total);
    CHECK(vd > 1e-4);
    // ... and HIGHER tau keeps FEWER blocks: a threshold, not a budget.
    CHECK(prev < 0.0 || pct(s) <= prev + 1e-9);
    prev = pct(s);
  }
  // OPT-IN (VPIPE_MING_SOL_TRACE=1): where the error comes from -- the
  // joint sequence after every block, dense against tau 0, so the first
  // routed block's own error and its growth through the stack are read
  // apart. Traced forwards commit per block but run the same kernels.
  if (keep0 && std::getenv("VPIPE_MING_SOL_TRACE") != nullptr) {
    char t1[] = "/tmp/ming-sol-dense-XXXXXX";
    char t2[] = "/tmp/ming-sol-tau0-XXXXXX";
    const char* d1 = mkdtemp(t1);
    const char* d2 = mkdtemp(t2);
    std::string err;
    auto dm = load_on(mc, ws, d, nullptr, nullptr, &err);
    if (d1 != nullptr && d2 != nullptr && dm) {
      dm->set_caption_cache(false);
      keep0->set_caption_cache(false);
      setenv("VPIPE_MING_DIT_TRACE", d1, 1);
      (void)run(*dm, mc, in);
      setenv("VPIPE_MING_DIT_TRACE", d2, 1);
      (void)run(*keep0, mc, in);
      unsetenv("VPIPE_MING_DIT_TRACE");
      keep0->set_caption_cache(true);
      std::printf("    per block, tau 0 vs dense (joint after the block):");
      for (int b = 0; b < n_mod; ++b) {
        const std::string tag = "blk" + std::to_string(b);
        const auto x = read_trace(d1, tag), y = read_trace(d2, tag);
        if (b % 8 == 0) { std::printf("\n     "); }
        std::printf(" %s %.4f", tag.c_str(), mingtest::rel_l2(y, x));
      }
      std::printf("\n");
    }
    std::error_code ec;
    if (d1 != nullptr) { fs::remove_all(d1, ec); }
    if (d2 != nullptr) { fs::remove_all(d2, ec); }
  }
  // THE CAPTION SINK, A/B at tau 0: the caption's keys exact for every
  // query (the default) against the caption routed like any image rows.
  if (keep0) {
    const Run on = run(*keep0, mc, in);
    setenv("VPIPE_MING_SOL_NO_SINK", "1", 1);
    const Run off = run(*keep0, mc, in);
    unsetenv("VPIPE_MING_SOL_NO_SINK");
    CHECK(on.ok && off.ok);
    if (on.ok && off.ok) {
      std::printf("    tau 0, caption sink ON: vs dense %.4f, kept %.1f%%; "
                  "OFF: vs dense %.4f, kept %.1f%%\n",
                  mingtest::rel_l2(on.z, dense.z), pct(on),
                  mingtest::rel_l2(off.z, dense.z), pct(off));
      // Keeping the caption exact keeps MORE blocks, never fewer.
      CHECK(on.eng.sol_kept >= off.eng.sol_kept);
    }
  }
}

// ---- the ANE feed-forward on the real model -----------------------------

// The adapter's strength for the ANE check: well past the ANE's own
// rounding through 30 blocks, or an adapter missing from half the rows
// could hide inside it.
constexpr float kAneLoraScale = 3.0f;
//
// ENGAGEMENT FIRST: the module's chunk is 2048 rows and the GPU keeps at
// least one, so the golden's 1314-row joint DECLINES the tier -- a test
// there passes having exercised nothing. 1024^2 is 4386 joint rows.
//
// The ANE's rows are fp16 and come back bf16, so the bar is a rounding
// difference through 30 blocks, not bit-identity.
//
// THE GPU FALLBACK is forced with the host's VPIPE_ANE_FAIL (read when
// the tier is built, at the first forward). The FIRST split losing every
// ANE row leaves the whole forward the GPU's -- which is what proves the
// tier latched and no later block split. A PINNED two-chunk share losing
// its SECOND chunk keeps the first chunk's ANE rows and recomputes the
// rest; the wrong range would land far from the GPU arm.
void
test_ane(MetalCompute& mc, const std::shared_ptr<WeightSet>& ws,
         const ming::DitConfig& d, const Inputs& golden, const Run& dense,
         const std::vector<float>& ref, double tol, TestSession* ts,
         const fs::path& adapter)
{
  std::string err;
  // AGAINST THE REFERENCE FIRST, which is the bar that means something:
  // the golden's own 512^2 geometry, where the default 2048-row chunk
  // would decline, with a 512-row module (VPIPE_MING_ANE_CHUNK, read when
  // the tier is built) and a pinned share of two chunks -- the ANE takes
  // 1024 of the 1314 rows of every main block. Held to the same tolerance
  // as the GPU forward: twice the reference's OWN bf16 error.
  {
    const FlexData b = ane_bag(0.8);
    auto m = load_on(mc, ws, d, &b, ts, &err);
    CHECK(m != nullptr);
    if (m) {
      setenv("VPIPE_MING_ANE_CHUNK", "512", 1);
      const Run r = run(*m, mc, golden);
      unsetenv("VPIPE_MING_ANE_CHUNK");
      CHECK(r.ok && m->ane_armed());
      CHECK(r.eng.ane_split == d.n_layers && r.eng.ane_lost == 0);
      CHECK(r.eng.ane_rows == (long long)d.n_layers * 1024);
      const double vr = mingtest::rel_l2(r.z, ref);
      std::printf("    512^2, 512-row chunks, %lld ANE rows over %d "
                  "blocks: vs reference fp32 %.4f (GPU %.4f; tol %.3f), "
                  "vs the GPU forward %.4f\n", r.eng.ane_rows,
                  r.eng.ane_split, vr, mingtest::rel_l2(dense.z, ref), tol,
                  mingtest::rel_l2(r.z, dense.z));
      CHECK(vr < tol);
    }
  }
  const Inputs in = at_size(golden, 1024, 1234u);
  auto gpu = load_on(mc, ws, d, nullptr, ts, &err);
  CHECK(gpu != nullptr);
  if (!gpu) {
    std::printf("    load refused: %s\n", err.c_str());
    return;
  }
  ming::Layout lay;
  const Run g = run(*gpu, mc, in, &lay);
  CHECK(g.ok && !gpu->ane_requested());
  CHECK(g.eng.ane_split == 0 && g.eng.ane_probe == 0);
  if (!g.ok) { return; }
  std::printf("    1024^2: %d image + %d caption rows (joint %d); GPU arm "
              "%.0f ms\n", lay.img_tokens, lay.cap_len, lay.joint_len, g.ms);
  // THE MODEL'S OWN SENSITIVITY at this input, which is the yardstick: the
  // same GPU forward with the modulation applied as its own pass instead
  // of folded into the norm gain -- exact algebra, a different bf16
  // rounding of one tensor per block, nothing else.
  double r_round = 0.0;
  {
    setenv("VPIPE_MING_NO_FOLD", "1", 1);
    auto nf = load_on(mc, ws, d, nullptr, ts, &err);
    unsetenv("VPIPE_MING_NO_FOLD");
    CHECK(nf != nullptr);
    if (nf) {
      const Run r = run(*nf, mc, in);
      CHECK(r.ok);
      r_round = mingtest::rel_l2(r.z, g.z);
      std::printf("    a rounding-level change on the GPU path (modulation "
                  "unfolded): rel-L2 %.4e vs the GPU arm\n", r_round);
    }
  }

  const FlexData bag = ane_bag(0.0);
  auto am = load_on(mc, ws, d, &bag, ts, &err);
  CHECK(am != nullptr && am->ane_requested());
  if (!am) { return; }
  const Run a = run(*am, mc, in);
  CHECK(a.ok);
  // Asked AFTER the forward: the module is built at the first one, when
  // the row count is finally known. Every Apple-silicon Mac has an ANE,
  // so declining here is a failure, not a skip.
  CHECK(am->ane_armed());
  if (!a.ok || !am->ane_armed()) { return; }
  print_eng(a.eng);
  // The balancer's opening: four split blocks, two probes, then whichever
  // measured better -- so at least four splits took ANE rows.
  CHECK(a.eng.ane_split >= 4 && a.eng.ane_rows >= 4LL * 2048);
  CHECK(a.eng.ane_lost == 0 && !am->ane_gpu_only());
  const double r_ff = mingtest::rel_l2(a.z, g.z);
  std::printf("    ANE split vs GPU-only: rel-L2 %.4e (%d splits, %lld ANE "
              "rows of %d x %d), %.0f ms incl. the module's build\n", r_ff,
              a.eng.ane_split, a.eng.ane_rows, lay.joint_len,
              a.eng.ane_split + a.eng.ane_probe, a.ms);
  // THE YARDSTICK IS THE MODEL'S OWN SENSITIVITY (r_round above): a
  // stack this deep, whose feed-forward outputs reach 1e6, moves its
  // output several percent for one tensor rounded differently per block.
  // MEASURED here: the ANE's rows are ~0.3% from the GPU's per block
  // (VPIPE_MING_ANE_VERIFY), the output ~2.7x the rounding change. A
  // swapped projection or a misplaced row lands at O(1).
  CHECK(r_ff > 0.0 && r_ff < 4.0 * r_round);
  // A second forward: the module is warm and the balancer has decided.
  const Run a2 = run(*am, mc, in);
  CHECK(a2.ok);
  print_eng(a2.eng);
  std::printf("    second forward %.0f ms (GPU arm %.0f): vs GPU-only "
              "rel-L2 %.4e\n", a2.ms, g.ms, mingtest::rel_l2(a2.z, g.z));

  // ---- THE ADAPTER ON THE ANE'S ROWS ----------------------------------
  //
  // The ANE is handed each block's weights, so a runtime adapter has to be
  // staged INTO its module (Tier's `<p>.lora<i>` bindings); left out, the
  // ANE's rows run the base model and the GPU's the adapted one. A pinned
  // half of every main block's rows, against the GPU arm with the same
  // adapter -- and the same pinned split WITHOUT an adapter as the
  // yardstick, and the bare weights staged (the test's own negative
  // control) as proof the bar can fail.
  if (!adapter.empty()) {
    const LoadOptions o = with_lora(adapter, kAneLoraScale);
    auto arm = [&](const FlexData* bag, const LoadOptions& lo, bool drop,
                   Run* out) {
      if (drop) { setenv("VPIPE_MING_TEST_ANE_DROP_LORA", "1", 1); }
      auto m = load_on(mc, ws, d, bag, ts, &err, lo);
      unsetenv("VPIPE_MING_TEST_ANE_DROP_LORA");
      if (!m) { return false; }
      *out = run(*m, mc, in);
      return out->ok && (bag == nullptr || m->ane_armed());
    };
    const FlexData half = ane_bag(0.5);
    Run gl, ab, al, ax;
    const bool ok = arm(nullptr, o, false, &gl) &&
                    arm(&half, LoadOptions(), false, &ab) &&
                    arm(&half, o, false, &al) && arm(&half, o, true, &ax);
    CHECK(ok);
    if (ok) {
      const double moved = mingtest::rel_l2(gl.z, g.z);
      const double r_base = mingtest::rel_l2(ab.z, g.z);
      const double r_l = mingtest::rel_l2(al.z, gl.z);
      const double r_x = mingtest::rel_l2(ax.z, gl.z);
      std::printf("    adapter x%.0f on the GPU moves the output %.4f; a "
                  "pinned half on the ANE: base vs GPU %.4f, ADAPTED vs "
                  "the adapted GPU %.4f (%d of %d splits carried it); "
                  "CONTROL, bare weights staged: %.4f\n", kAneLoraScale,
                  moved, r_base, r_l, al.eng.ane_lora, al.eng.ane_split,
                  r_x);
      CHECK(al.eng.ane_split == d.n_layers);
      CHECK(al.eng.ane_lora == al.eng.ane_split);
      CHECK(ax.eng.ane_split == d.n_layers && ax.eng.ane_lora == 0);
      // As close as the ANE gets WITHOUT an adapter...
      CHECK(r_l < 1.5 * r_base);
      // ...where the adapter left off its rows lands far further out.
      CHECK(r_x > 2.0 * r_l);
    }
  }

  auto failing = [&](const char* spec, double rows, Run* out,
                     bool* latched) {
    const FlexData b = ane_bag(rows);
    auto mf = load_on(mc, ws, d, &b, ts, &err);
    if (!mf) { return false; }
    setenv("VPIPE_ANE_FAIL", spec, 1);
    *out = run(*mf, mc, in);
    unsetenv("VPIPE_ANE_FAIL");
    *latched = mf->ane_gpu_only();
    return out->ok;
  };
  Run f0, f1;
  bool l0 = false, l1 = false;
  CHECK(failing("0:0", 0.0, &f0, &l0));
  CHECK(failing("0:1", 1.0, &f1, &l1));
  const double r_f0 = mingtest::rel_l2(f0.z, g.z);
  const double r_f1 = mingtest::rel_l2(f1.z, g.z);
  std::printf("    ANE rows lost -> GPU: first split, every row: rel-L2 "
              "%.3e vs GPU-only (%s); a pinned share's second chunk: "
              "%.3e (%lld ANE rows kept)\n", r_f0,
              same_bits(f0, g) ? "bit-identical" : "not bitwise", r_f1,
              f1.eng.ane_rows);
  CHECK(l0 && l1);                       // engaged, and latched
  CHECK(f0.eng.ane_lost == 1 && f1.eng.ane_lost == 1);
  CHECK(f0.eng.ane_rows == 0 && f1.eng.ane_rows == 2048);
  CHECK(r_f0 < 1e-6);                    // every row the GPU's
  // The ANE's first chunk of ONE block kept: a change, and less of one
  // than every block's split.
  CHECK(r_f1 > r_f0 && r_f1 < r_ff);
}

// ---- opt-in: interleaved one-forward timings -----------------------------
std::vector<std::string>
split_list(const char* s, const char* dflt)
{
  std::vector<std::string> out;
  std::stringstream ss((s != nullptr && *s != '\0') ? s : dflt);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) { out.push_back(item); }
  }
  return out;
}

void
bench(MetalCompute& mc, const std::shared_ptr<WeightSet>& ws,
      const ming::DitConfig& d, const Inputs& golden, TestSession* ts,
      int rounds)
{
  const auto pxs = split_list(std::getenv("VPIPE_MING_BENCH_PX"), "1024");
  const auto names = split_list(std::getenv("VPIPE_MING_BENCH_ARMS"),
                                "dense,sol:0,sol:0.5,ane");
  ts->quiet(true);
  for (const std::string& pxs_i : pxs) {
    const int px = std::atoi(pxs_i.c_str());
    if (px <= 0 || px % 16 != 0) { continue; }
    const Inputs in = at_size(golden, px, 4321u);
    std::printf("- bench %dx%d, %d rounds, arms interleaved and the order "
                "reversed every other round\n", px, px, rounds);
    struct Arm {
      std::string name;
      FlexData bag;
      std::unique_ptr<ming::MingDit> m;
      std::vector<double> ms;
      Run last;
    };
    std::vector<Arm> arms;
    for (const std::string& n : names) {
      Arm a;
      a.name = n;
      a.bag = FlexData::make_object();
      const auto colon = n.find(':');
      const std::string kind = n.substr(0, colon);
      const float tau = colon == std::string::npos
                            ? 0.0f
                            : (float)std::atof(n.c_str() + colon + 1);
      if (kind == "sol" || kind == "solane") {
        accel::set_flag(&a.bag, accel::kSolAttn, true);
        accel::set_real(&a.bag, accel::kSolTau, (double)tau);
      }
      if (kind == "ane" || kind == "solane") {
        accel::set_flag(&a.bag, accel::kAneFfn, true);
      }
      std::string err;
      a.m = load_on(mc, ws, d, &a.bag, ts, &err);
      if (!a.m) {
        std::printf("    %s: load refused: %s\n", n.c_str(), err.c_str());
        continue;
      }
      // WARM: the first forward fills the caption cache and builds the
      // ANE module; the balancer opens on it.
      a.last = run(*a.m, mc, in);
      std::printf("    warm %-10s %8.0f ms\n", n.c_str(), a.last.ms);
      arms.push_back(std::move(a));
    }
    for (int r = 0; r < rounds; ++r) {
      for (std::size_t j = 0; j < arms.size(); ++j) {
        Arm& a = arms[(r % 2) == 0 ? j : arms.size() - 1 - j];
        a.last = run(*a.m, mc, in);
        a.ms.push_back(a.last.ms);
      }
    }
    auto median = [](std::vector<double> v) {
      if (v.empty()) { return 0.0; }
      std::sort(v.begin(), v.end());
      const std::size_t n = v.size();
      return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    };
    const double base = arms.empty() ? 0.0 : median(arms[0].ms);
    for (Arm& a : arms) {
      std::string list;
      for (double x : a.ms) { list += std::to_string((int)(x + 0.5)) + " "; }
      const double md = median(a.ms);
      const ming::DitEngagement& e = a.last.eng;
      std::printf("    %-10s median %8.0f ms (%.3fx of %s) | %s| vs %s "
                  "rel-L2 %.4f", a.name.c_str(), md,
                  md > 0.0 ? base / md : 0.0, arms[0].name.c_str(),
                  list.c_str(), arms[0].name.c_str(),
                  mingtest::rel_l2(a.last.z, arms[0].last.z));
      if (e.attn_sol > 0) {
        std::printf(" | sol kept %.1f%%",
                    e.sol_total > 0 ? 100.0 * (double)e.sol_kept /
                                          (double)e.sol_total
                                    : 0.0);
      }
      if (e.ane_split + e.ane_probe > 0) {
        std::printf(" | ane %d split (%lld rows), %d probed",
                    e.ane_split, e.ane_rows, e.ane_probe);
      }
      std::printf("\n");
    }
  }
  ts->quiet(false);
}

// ---- opt-in: the per-category split by output size ----------------------
//
// RESIDENT, never streaming: every mark is a commit and a wait, which
// leaves a streamed block's prefetch nothing to hide under, and a slot's
// read would land in whichever bucket follows it. The profiled forward is
// a caption-cache HIT (the warm-up filled it), so the context refiner
// does not run and is not in the FLOPs.
void
profile(MetalCompute& mc, const std::shared_ptr<WeightSet>& ws,
        const ming::DitConfig& d, const Inputs& golden, TestSession* ts,
        const char* pxs_env)
{
  std::string err;
  auto m = load_on(mc, ws, d, nullptr, ts, &err);
  if (!m) { return; }
  const double H = d.dim;
  const double FFI = (double)(int)((double)d.dim / 3.0 * 8.0);
  for (const std::string& s : split_list(pxs_env, "1024")) {
    const int px = std::atoi(s.c_str());
    if (px <= 0 || px % 16 != 0) { continue; }
    const Inputs in = at_size(golden, px, 777u);
    ts->quiet(true);
    ming::Layout lay;
    const Run warm = run(*m, mc, in, &lay);
    setenv("VPIPE_MING_DIT_PROFILE", "1", 1);
    const Run p = run(*m, mc, in);
    unsetenv("VPIPE_MING_DIT_PROFILE");
    ts->quiet(false);
    if (!warm.ok || !p.ok) { continue; }
    const std::string line = ts->last("ming-image DiT profile");
    // FLOPs of THIS forward: the noise refiner over the image rows, the
    // main stack over the joint; 2 flops a multiply-add.
    const double I = lay.img_tokens, J = lay.joint_len;
    const double NR = d.n_refiner_layers, NL = d.n_layers;
    const double rows = NR * I + NL * J;
    std::map<std::string, double> gf;
    gf["gemm.qkv"] = 6.0 * rows * H * H / 1e9;
    gf["gemm.out"] = 2.0 * rows * H * H / 1e9;
    gf["gemm.ff"] = 6.0 * rows * H * FFI / 1e9;
    gf["attn"] = 4.0 * (NR * I * I + NL * J * J) * H / 1e9;
    std::printf("- profile %dx%d: joint %d (%d image + %d caption), "
                "unprofiled %.0f ms (warm-up incl. the caption)\n", px, px,
                lay.joint_len, lay.img_tokens, lay.cap_len, warm.ms);
    // " | <name> <ms> ms (<pct>%, <n> marks)" per bucket.
    double total = 0.0;
    std::vector<std::pair<std::string, double>> buckets;
    for (std::size_t at = line.find(" | "); at != std::string::npos;
         at = line.find(" | ", at + 3)) {
      std::istringstream is(line.substr(at + 3));
      std::string name;
      double ms = 0.0;
      if (is >> name >> ms) {
        buckets.emplace_back(name, ms);
        total += ms;
      }
    }
    for (const auto& b : buckets) {
      const auto it = gf.find(b.first);
      std::printf("    %-10s %8.0f ms %5.1f%%", b.first.c_str(), b.second,
                  total > 0.0 ? 100.0 * b.second / total : 0.0);
      if (it != gf.end() && b.second > 0.0) {
        std::printf("   %9.0f GFLOP  %5.2f TFLOP/s", it->second,
                    it->second / b.second);
      }
      std::printf("\n");
    }
    std::printf("    total      %8.0f ms (profiled; %s)\n", total,
                line.empty() ? "NO PROFILE LINE" : "marks commit+wait");
  }
}

// ---- runtime LoRA on the real model ---------------------------------------
//
// A SYNTHETIC rank-4 adapter over every block projection of the real
// checkpoint, in the diffusers spelling, each delta 5% of its weight's
// norm (dump_ming_golden.py --stage real_lora). Not an accuracy bar --
// the small golden is that -- but what only the real checkpoint can say:
// the binding COUNT on the real names and shapes (7 x 34 = 238),
// engagement, strength 0 exactly off on the real stack, and the adapter
// beside Sol (attention is untouched, the projections are not: with
// every key block kept the adapted forward is the adapted dense one).
Run
test_lora_real(MetalCompute& mc, const std::shared_ptr<WeightSet>& ws,
               const ming::DitConfig& d, const Inputs& in, const Run& dense,
               const fs::path& adapter, TestSession* ts)
{
  const int want = ming::MingDit::lora_module_capacity(d);
  std::string err;
  auto m = load_on(mc, ws, d, nullptr, ts, &err, with_lora(adapter));
  CHECK(m != nullptr);
  if (!m) { return Run{}; }
  const Run r = run(*m, mc, in);
  CHECK(r.ok);
  const double moved = mingtest::rel_l2(r.z, dense.z);
  std::printf("    bound %d of %d (skipped %d); %d projections adapted; "
              "moves the output %.4f; %.0f ms (dense %.0f)\n",
              m->lora_modules(0), want, m->lora_skipped(0), r.eng.lora_proj,
              moved, r.ms, dense.ms);
  CHECK(want == 238);
  CHECK(m->lora_modules(0) == want && m->lora_skipped(0) == 0);
  CHECK(r.eng.lora_proj == want);
  CHECK(moved > 0.01);
  m->set_lora_scale(0, 0.0f);
  const Run z = run(*m, mc, in);
  m->set_lora_scale(0, 1.0f);
  CHECK(z.ok && z.eng.lora_proj == 0 && same_bits(z, dense));
  std::printf("    strength 0: %s the dense forward (memcmp)\n",
              same_bits(z, dense) ? "IDENTICAL to" : "DIFFERS from");
  m.reset();
  for (float tau : {-1.0e30f, 0.0f}) {
    const FlexData bag = sol_bag(tau);
    auto ms = load_on(mc, ws, d, &bag, nullptr, &err, with_lora(adapter));
    CHECK(ms != nullptr);
    if (!ms) { continue; }
    const Run s = run(*ms, mc, in);
    CHECK(s.ok && s.eng.attn_sol > 0 && s.eng.lora_proj == want);
    std::printf("    Sol tau %s + adapter: %d attentions routed, %d "
                "projections adapted; vs the dense adapted forward %.4f%s\n",
                tau < -1.0f ? "-inf" : "0", s.eng.attn_sol, s.eng.lora_proj,
                mingtest::rel_l2(s.z, r.z),
                tau < -1.0f ? (same_bits(s, r) ? " (bit-identical)"
                                               : " (NOT bitwise)")
                            : "");
    if (tau < -1.0f) { CHECK(same_bits(s, r)); }
  }
  return r;
}

// A quantized pack, with the same adapter: it rides the QUANTIZED GEMM's
// output, so it is the qmm route's turn. The pack is the same model, so
// the adapter must move it as it moves the bf16 one -- the pack's adapted
// forward sits about as far from the bf16 adapted one as the two base
// forwards sit from each other.
void
test_lora_pack(MetalCompute& mc, const char* label, const std::string& pack,
               const Inputs& in, const Run& bf_base, const Run& bf_lora,
               const fs::path& adapter, TestSession* ts)
{
  const ming::Checkpoint ck = ming::read_checkpoint(pack);
  CHECK(ck.valid);
  if (!ck.valid) {
    std::printf("    %s: %s\n", label, ck.why.c_str());
    return;
  }
  std::string err;
  std::shared_ptr<WeightSet> ws;
  auto b = load_dit(mc, ck.transformer_dir, ck.dit, false, &err, &ws);
  CHECK(b != nullptr);
  if (!b) { return; }
  const Run base = run(*b, mc, in);
  b.reset();
  const int want = ming::MingDit::lora_module_capacity(ck.dit);
  auto m = load_on(mc, ws, ck.dit, nullptr, ts, &err, with_lora(adapter));
  CHECK(m != nullptr);
  if (!m) { return; }
  const Run r = run(*m, mc, in);
  CHECK(base.ok && r.ok);
  CHECK(m->quant_bits() > 0 && r.eng.gemm_qmm > 0);
  CHECK(m->lora_modules(0) == want && r.eng.lora_proj == want);
  const double moved = mingtest::rel_l2(r.z, base.z);
  const double d_base = mingtest::rel_l2(base.z, bf_base.z);
  const double d_lora = mingtest::rel_l2(r.z, bf_lora.z);
  const double bf_moved = mingtest::rel_l2(bf_lora.z, bf_base.z);
  std::printf("    %s pack (w%d): bound %d, %d adapted after %d qmm GEMMs; "
              "moves it %.4f (bf16: %.4f); pack vs bf16 adapted %.4f, base "
              "%.4f\n", label, m->quant_bits(), m->lora_modules(0),
              r.eng.lora_proj, r.eng.gemm_qmm, moved, bf_moved, d_lora,
              d_base);
  CHECK(moved > 0.5 * bf_moved && moved < 2.0 * bf_moved);
  CHECK(d_lora < 1.5 * d_base + 0.01);
  m->set_lora_scale(0, 0.0f);
  const Run z = run(*m, mc, in);
  CHECK(z.ok && same_bits(z, base));
  std::printf("    %s strength 0: %s the pack's base (memcmp)\n", label,
              same_bits(z, base) ? "IDENTICAL to" : "DIFFERS from");
}

// ---- dit_real -------------------------------------------------------------

void
test_real(MetalCompute& mc, const fs::path& root, const char* model,
          TestSession* ts)
{
  CASE("dit_real: the real DiT at 512x512, resident");
  Golden g;
  Inputs in;
  if (!g.open(root / "fp32") || !load_inputs(g, &in)) {
    CHECK(!"dit_real/fp32 is unreadable");
    return;
  }
  const ming::Checkpoint ck = ming::read_checkpoint(model);
  CHECK(ck.valid);
  if (!ck.valid) {
    std::printf("    checkpoint: %s\n", ck.why.c_str());
    return;
  }
  const std::vector<float> want = g.f32("out");
  Golden gbar;
  std::vector<float> bar;
  if (gbar.open(root / "bf16")) { bar = gbar.f32("out"); }
  const double ref_bf16 = bar.empty() ? 0.0 : mingtest::rel_l2(bar, want);
  // Twice the reference's own bf16 run, with the small golden's bar as
  // the floor when the reference did not record one.
  const double tol = std::max(kSmallTol, 2.0 * ref_bf16);
  const int n_blocks = 2 * ck.dit.n_refiner_layers + ck.dit.n_layers;
  // The synthetic real-size adapter, when the golden carries one.
  const fs::path adapter = root / "lora" / "adapter_diffusers.safetensors";
  const bool have_adapter = fs::exists(adapter);
  Run res, res_lora;
  {
    std::string err;
    std::shared_ptr<WeightSet> ws;
    auto dit = load_dit(mc, ck.transformer_dir, ck.dit, false, &err, &ws);
    CHECK(dit != nullptr);
    if (!dit) {
      std::printf("    load refused: %s\n", err.c_str());
      return;
    }
    ming::Layout lay;
    res = run(*dit, mc, in, &lay);
    CHECK(res.ok);
    if (!res.ok) {
      std::printf("    forward failed: %s\n", res.err.c_str());
      return;
    }
    const double rel = mingtest::rel_l2(res.z, want);
    std::printf("    %d image + %d caption rows (joint %d), image axis-0 "
                "%d\n", lay.img_tokens, lay.cap_len, lay.joint_len,
                lay.pos_t[0]);
    std::printf("    rel-L2 vs reference fp32 %.4f (reference's own bf16 "
                "%.4f%s; tol %.3f), %.0f ms\n", rel, ref_bf16,
                bar.empty() ? " -- not recorded" : "", tol, res.ms);
    if (!bar.empty()) {
      std::printf("    vpipe vs reference bf16 %.4f\n",
                  mingtest::rel_l2(res.z, bar));
    }
    CHECK(rel < tol);
    print_eng(res.eng);
    check_routes(mc, *dit, res.eng, n_blocks);   // a cold cache: all run
    // A second forward of the same caption serves it from the cache,
    // bit for bit.
    const Run again = run(*dit, mc, in);
    CHECK(again.eng.cap_cache_hit && same_bits(again, res));
    std::printf("    second forward (caption cached): %.0f ms, %s\n",
                again.ms, same_bits(again, res) ? "identical" : "DIFFERS");
    // The host's FLOP/s meter, both ways: exact from the structure, and
    // credited whole. The cached forward skips the caption path.
    if (vpipe::perf_flops::available()) {
      const std::uint64_t cold = flops_want(ck.dit, lay, false);
      const std::uint64_t warm = flops_want(ck.dit, lay, true);
      std::printf("    perf-flops: %.4f TFLOP (derived %.4f, credited "
                  "%.4f) -> %.2f TFLOP/s; cached caption %.4f (derived "
                  "%.4f)\n", res.eng.gpu_flops / 1e12, cold / 1e12,
                  res.credited / 1e12,
                  res.credited / 1e12 / (res.ms / 1000.0),
                  again.eng.gpu_flops / 1e12, warm / 1e12);
      CHECK(res.eng.gpu_flops == cold && res.credited == cold);
      CHECK(again.eng.gpu_flops == warm && again.credited == warm);
    }
    if (part("stages")) {
      CASE("dit_real: every stage against the reference's taps");
      // The embedders are single GEMMs from bf16 operands: within bf16
      // of an f32 compute however deep the model is.
      check_stages(*dit, mc, g, in, res, tol, 0.01);
    }
    dit.reset();
    // The arms below share `ws`: a second model over it costs its
    // scratch, not a second copy of the checkpoint.
    if (part("sol")) {
      CASE("dit_real: Sol-Attn engages without matrix cores; every key "
           "block kept is the dense answer; what each tau costs");
      test_sol(mc, ws, ck.dit, in, res, want);
    }
    if (part("lora") && have_adapter) {
      CASE("dit_real LoRA: a synthetic adapter binds all 238 projections, "
           "engages, is exactly off at 0, and rides beside Sol");
      res_lora = test_lora_real(mc, ws, ck.dit, in, res, adapter, ts);
    } else if (part("lora")) {
      std::printf("SKIPPED dit_real LoRA: no %s (dump_ming_golden.py "
                  "--stage real_lora)\n", adapter.string().c_str());
    }
    if (part("ane")) {
      CASE("dit_real: the ANE feed-forward at 1024^2 -- engaged, against "
           "the GPU, and its lost-rows fallback");
      test_ane(mc, ws, ck.dit, in, res, want, tol, ts,
               have_adapter ? adapter : fs::path());
    }
    if (const char* pe = std::getenv("VPIPE_MING_PROFILE_PX")) {
      CASE("dit_real: the per-category split by output size (opt-in)");
      profile(mc, ws, ck.dit, in, ts, pe);
    }
    if (const char* be = std::getenv("VPIPE_MING_BENCH")) {
      CASE("dit_real: interleaved one-forward timings (opt-in)");
      bench(mc, ws, ck.dit, in, ts, std::max(1, std::atoi(be)));
    }
  }
  if (!part("stream")) { return; }
  CASE("dit_real: streamed == resident, bit for bit");
  {
    std::string err;
    auto dit = load_dit(mc, ck.transformer_dir, ck.dit, true, &err);
    CHECK(dit != nullptr);
    if (dit) {
      const Run s = run(*dit, mc, in);
      CHECK(s.ok && s.eng.blocks_streamed ==
                        ck.dit.n_refiner_layers + ck.dit.n_layers);
      CHECK(same_bits(s, res));
      std::printf("    streamed %d blocks (prefetch hits %d) in %.0f ms: "
                  "%s\n", s.eng.blocks_streamed, s.eng.prefetch_hits, s.ms,
                  same_bits(s, res) ? "identical" : "DIFFERS");
      std::printf("    held while streaming: %.0f MB (declared floor "
                  "%.0f MB)\n", (double)dit->resident_bytes() / 1e6,
                  (double)ming::MingDit::streaming_floor_bytes(
                      ck.transformer_dir) / 1e6);
    }
  }
  CASE("dit_real: VPIPE_MING_NO_SLOTS (per-block fallback) == resident");
  {
    setenv("VPIPE_MING_NO_SLOTS", "1", 1);
    std::string err;
    auto dit = load_dit(mc, ck.transformer_dir, ck.dit, true, &err);
    CHECK(dit != nullptr);
    if (dit) {
      const Run s = run(*dit, mc, in);
      CHECK(s.ok && s.eng.prefetch_hits == 0);
      CHECK(same_bits(s, res));
      std::printf("    no slots, %.0f ms: %s\n", s.ms,
                  same_bits(s, res) ? "identical" : "DIFFERS");
    }
    unsetenv("VPIPE_MING_NO_SLOTS");
  }
  if (!part("lora") || !have_adapter || !res_lora.ok) { return; }
  CASE("dit_real LoRA: streamed == resident WITH the adapter, bit for bit");
  {
    std::string err;
    auto dit = load_dit(mc, ck.transformer_dir, ck.dit, true, &err, nullptr,
                        with_lora(adapter), ts);
    CHECK(dit != nullptr);
    if (dit) {
      const Run s = run(*dit, mc, in);
      CHECK(s.ok && s.eng.blocks_streamed ==
                        ck.dit.n_refiner_layers + ck.dit.n_layers);
      CHECK(s.eng.lora_proj == dit->lora_modules(0));
      CHECK(same_bits(s, res_lora));
      std::printf("    streamed %d blocks with the adapter (%d projections "
                  "adapted) in %.0f ms: %s\n", s.eng.blocks_streamed,
                  s.eng.lora_proj, s.ms,
                  same_bits(s, res_lora) ? "identical" : "DIFFERS");
    }
  }
  for (const char* env : {"VPIPE_MING_W8_PACK", "VPIPE_MING_W4_PACK"}) {
    const char* pk = std::getenv(env);
    if (pk == nullptr || *pk == '\0') {
      std::printf("SKIPPED dit_real LoRA on a pack: %s is unset\n", env);
      continue;
    }
    CASE("dit_real LoRA: on a quantized pack, after the qmm GEMM");
    test_lora_pack(mc, env, pk, in, res, res_lora, adapter, ts);
  }
}

}  // namespace

int
main()
{
  MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no usable Metal device. NOTHING was checked.\n");
    return 0;
  }
  // For the ANE tier (and the profile line); every other case runs with
  // no session at all, as before.
  TestSession ts(&mc);
  CASE("the plugin's own kernels register and every entry resolves");
  {
    const bool reg = mc.register_metal_library(
        ming::kMetalLibBf16, ming_ops_bf16_metallib,
        ming_ops_bf16_metallib_len);
    CHECK(reg);
    const auto lib = mc.load_library(ming::kMetalLibBf16);
    CHECK(lib.valid());
    for (const char* fn :
         {"gated_residual_tanh_f16", "gated_residual_tanh_v4_f16",
          "swiglu_v4_f16", "residual_add_f16", "residual_add_v4_f16",
          "copy_rect_f16"}) {
      const bool ok = lib.valid() && lib.function(fn).valid();
      if (!ok) { std::printf("    unresolved: %s\n", fn); }
      CHECK(ok);
    }
  }

  const char* groot = std::getenv("VPIPE_MING_GOLDEN");
  if (groot == nullptr || *groot == '\0') {
    std::printf("SKIPPED: VPIPE_MING_GOLDEN is unset -- the DiT was NOT "
                "checked against any reference.\n");
    return mingtest::summary("ming-dit");
  }
  const fs::path root(groot);
  if (!part("small")) {
    std::printf("dit_small: not run (VPIPE_MING_DIT_ONLY)\n");
  } else if (fs::exists(root / "dit_small" / "case_a" / "manifest.json")) {
    test_small(mc, root / "dit_small");
  } else {
    std::printf("SKIPPED dit_small: no %s\n",
                (root / "dit_small").string().c_str());
  }
  if (!part("lora")) {
    std::printf("dit_small LoRA: not run (VPIPE_MING_DIT_ONLY)\n");
  } else if (fs::exists(root / "dit_small" / "lora" / "meta.json")) {
    test_lora_small(mc, root / "dit_small", &ts);
  } else {
    std::printf("SKIPPED dit_small LoRA: no %s (dump_ming_golden.py "
                "--stage dit_lora)\n",
                (root / "dit_small" / "lora").string().c_str());
  }
  const char* model = std::getenv("VPIPE_MING_TEST_MODEL_PATH");
  const bool have_real =
      fs::exists(root / "dit_real" / "fp32" / "manifest.json");
  if (have_real && model != nullptr && *model != '\0') {
    test_real(mc, root / "dit_real", model, &ts);
  } else {
    std::printf("SKIPPED dit_real: %s\n",
                !have_real ? "no dit_real/fp32 golden"
                           : "VPIPE_MING_TEST_MODEL_PATH is unset");
  }
  return mingtest::summary("ming-dit");
}
