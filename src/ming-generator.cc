#include "ming-generator.h"

#include "ming-family.h"
#include "ming-layout.h"
#include "ming-model-config-stage.h"
#include "ming-torch-randn.h"

#include "common/flex-data.h"
#include "common/perf-event.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "generative-models/gen-input.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/shared/accel-settings.h"
#include "generative-models/spec-keys.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-memory.h"
#include "stages/model-registry.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <string>

namespace ming {

using vpipe::fmt;

namespace {

// The adapter slots this build runs: the stage's keys, as far as the DiT
// built against this host's lora::Stack holds them.
constexpr int kLoraSlots = std::min(kCfgLoraSlots, MingDit::kMaxLoraSlots);

using vpipe::metal_compute::SharedBuffer;

std::uint16_t
f32_to_bf16(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) { return (std::uint16_t)0x7fc0; }
  u += 0x7fffu + ((u >> 16) & 1u);   // round to nearest even
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

// The conditioning tensor as bf16 rows of `dim`. The conditioner emits
// bf16; f32 is accepted and narrowed (a graph that converted it on the
// way is still the same conditioning).
std::vector<std::uint16_t>
cond_to_bf16(const void* p, int rows, int dim, int elem, bool is_bf16)
{
  std::vector<std::uint16_t> v;
  if (p == nullptr || rows <= 0 || dim <= 0) { return v; }
  const std::size_t n = (std::size_t)rows * dim;
  if (elem == 2 && is_bf16) {
    v.resize(n);
    std::memcpy(v.data(), p, n * 2);
  } else if (elem == 4) {
    v.resize(n);
    const float* f = static_cast<const float*>(p);
    for (std::size_t i = 0; i < n; ++i) { v[i] = f32_to_bf16(f[i]); }
  }
  return v;
}

int
query_rows_of(const vpipe::FlexData* sb, int dflt)
{
  if (sb == nullptr || !sb->is_object()) { return dflt; }
  auto o = sb->as_object();
  if (!o.contains(kSbQueryRows)) { return dflt; }
  return (int)o.at(kSbQueryRows).as_int(dflt);
}

// The graph's scheduler / sampler selections, read against what this
// family runs: the reference's flow-match Euler over a STATIC linear
// shift of linspace(1, 1/S, S). Only `steps` is taken from a scheduler
// beat (generate-image already folds it into the request). Its `shift`
// and `shift_type` are NOT: scheduler-select always emits both -- 1.15
// and "exponential" when nothing was set -- so a beat cannot say whether
// they were chosen, and taking them would run this checkpoint (static
// 6.0) on a curve it was never trained on whenever a graph pins its
// steps that way. A deliberate shift goes on `ming-image-model-config`.
// What is not taken is said out loud: dropping a request silently reads
// as the model ignoring it.
void
note_unhonoured_(const vpipe::FlexData* sched, const vpipe::FlexData* samp,
                 float shift, const std::function<void(std::string)>& note)
{
  namespace sk = vpipe::genai::scheduler_spec;
  namespace pk = vpipe::genai::sampler_spec;
  if (sched != nullptr && sched->is_object()) {
    auto o = sched->as_object();
    if (o.contains(sk::kShift) &&
        std::fabs(o.at(sk::kShift).as_real(shift) - shift) > 1e-6) {
      note(fmt("the scheduler beat's shift {:.3f} is not taken (a "
               "scheduler-select emits 1.15 when unset); running {:.3f} -- "
               "set `shift` on ming-image-model-config to change it",
               o.at(sk::kShift).as_real(0.0), shift)());
    }
    if (o.contains(sk::kType)) {
      const std::string t(o.at(sk::kType).as_string(sk::kSimple));
      if (t != sk::kSimple) {
        note("scheduler type '" + t + "' is not implemented -- running "
             "'simple' (linspace), as the reference does");
      }
    }
    if (o.contains(sk::kDynamicShift) &&
        o.at(sk::kDynamicShift).as_bool(false)) {
      note("dynamic_shift was asked for; this checkpoint runs a static "
           "shift at every size (the reference's own dynamic override is "
           "a no-op) -- running static");
    }
  }
  if (samp != nullptr && samp->is_object()) {
    auto o = samp->as_object();
    if (o.contains(pk::kMethod)) {
      const std::string m(o.at(pk::kMethod).as_string(pk::kEuler));
      if (m != pk::kEuler) {
        note("sampler method '" + m + "' is not implemented -- running "
             "euler, as the reference does");
      }
    }
  }
}

// A real-valued key of the family's model_config beat, or `dflt`.
double
config_real_(const vpipe::FlexData* cfg, const char* key, double dflt)
{
  if (cfg == nullptr || !cfg->is_object()) { return dflt; }
  auto o = cfg->as_object();
  if (!o.contains(key)) { return dflt; }
  const double v = o.at(key).as_real(dflt);
  return v > 0.0 ? v : dflt;
}

}  // namespace

std::vector<float>
schedule_sigmas(int steps, float shift)
{
  // diffusers: timesteps = linspace(sigma_max * N, sigma_min * N, S + 1)
  // with sigma_min = 0 forced by the pipeline, shifted in double, cast
  // to f32, plus a terminal 0. The (S + 1)-th grid point is sigma 0 and
  // its forward is a no-op, so S forwards use the first S points.
  std::vector<float> s;
  if (steps <= 0) { return s; }
  s.reserve((std::size_t)steps + 1);
  for (int k = 0; k < steps; ++k) {
    const double u = 1.0 - (double)k / (double)steps;
    const double sig = (double)shift * u / (1.0 + ((double)shift - 1.0) * u);
    s.push_back((float)sig);
  }
  s.push_back(0.0f);
  return s;
}

void
MingImageGenerator::warn_(const std::string& m) const
{
  if (_session != nullptr) { _session->warn(fmt("ming-image: {}", m)); }
}

void
MingImageGenerator::warn_(const vpipe::VpipeFormat& m) const
{
  warn_(m());
}

void
MingImageGenerator::warn_once_(const std::string& m)
{
  if (_noted.insert(m).second) { warn_(m); }
}

std::unique_ptr<MingImageGenerator>
MingImageGenerator::load(const Checkpoint& ck,
                         const vpipe::genai::ImageModelCreateArgs& args)
{
  const vpipe::SessionContextIntf* session = args.session;
  auto* mgr = session != nullptr && session->services() != nullptr
                  ? session->services()->generative_model_manager()
                  : nullptr;
  const std::size_t dit_bytes =
      vpipe::model_memory::dir_weights_bytes(ck.transformer_dir);
  const std::size_t dit_floor =
      MingDit::streaming_floor_bytes(ck.transformer_dir);
  // kStreamHeadroom: block streaming is an argument to the load and
  // cannot be revised afterwards, so it takes the bigger cushion.
  const vpipe::model_memory::StreamPlan plan =
      vpipe::model_memory::plan_streaming(
          session, ck.transformer_dir, /*enc_dir=*/"",
          vpipe::model_memory::kStreamHeadroom);
  const bool stream = args.prefer_streaming || plan.stream;
  if (session != nullptr) {
    session->info(fmt("ming-image: {} the DiT ({:.1f} GB on disk)",
                      stream ? "streaming" : "preloading",
                      (double)dit_bytes / 1e9));
  }
  std::shared_ptr<vpipe::genai::WeightSet> ws =
      vpipe::genai::open_weight_set(ck.transformer_dir, session);
  if (!ws) {
    if (session != nullptr) {
      session->warn(fmt("ming-image: could not open '{}'",
                        ck.transformer_dir));
    }
    return nullptr;
  }
  // ---- what the load takes beyond the weights -------------------------
  MingDit::LoadOptions opts;
  // The label the HOST booked the ANE module under, when it did: it then
  // read the grant itself and the accel bag already says yes or no.
  opts.coreml_label = vpipe::genai::accel::text(
      args.borrowed_extra, vpipe::genai::create_args::kCoreMLLabel);
  // The adapters. LOAD-TIME: they bind into the blocks as they are
  // built. Positional -- slot 1 is `lora2` whether or not `lora` is set.
  std::string lora_ref[MingDit::kMaxLoraSlots];
  float lora_scale[MingDit::kMaxLoraSlots];
  std::fill(std::begin(lora_scale), std::end(lora_scale), 1.0f);
  if (args.model_config != nullptr && args.model_config->is_object()) {
    const auto o = args.model_config->as_object();
    bool any = false;
    std::vector<MingDit::LoraSpec> specs((std::size_t)MingDit::kMaxLoraSlots);
    // A slot the config names past what this host's DiT holds is said
    // out loud, not read off the end of either table.
    for (int i = kLoraSlots; i < kCfgLoraSlots; ++i) {
      if (o.contains(kCfgLoraKeys[i][0]) && session != nullptr) {
        session->warn(fmt("ming-image: `{}` is set, but this host's DiT "
                          "runs {} adapters at once; it is NOT applied",
                          kCfgLoraKeys[i][0], kLoraSlots));
      }
    }
    for (int i = 0; i < kLoraSlots; ++i) {
      const char* pk = kCfgLoraKeys[i][0];
      const char* sk = kCfgLoraKeys[i][1];
      if (o.contains(sk)) {
        lora_scale[i] = (float)o.at(sk).as_real(1.0);
      }
      specs[(std::size_t)i].scale = lora_scale[i];
      if (!o.contains(pk)) { continue; }
      lora_ref[i] = std::string(o.at(pk).as_string(""));
      if (lora_ref[i].empty()) { continue; }
      // A registry key, a directory holding one .safetensors, or a file
      // -- the host's own resolver, the same one its families use (a
      // null session resolves paths only).
      std::string rerr;
      const std::string file =
          vpipe::resolve_adapter_file(session, lora_ref[i], &rerr);
      if (file.empty()) {
        if (session != nullptr) {
          session->warn(fmt("ming-image: `{}` '{}' did not resolve to an "
                            "adapter file ({}); it is NOT applied", pk,
                            lora_ref[i], rerr.empty() ? "not found" : rerr));
        }
        continue;
      }
      specs[(std::size_t)i].path = file;
      any = true;
    }
    if (any) { opts.loras = std::move(specs); }
  }
  std::string err;
  std::unique_ptr<MingDit> dit =
      MingDit::load(std::move(ws), args.metal, ck.dit, args.accel, stream,
                    session, &err, opts);
  if (!dit) {
    if (session != nullptr) {
      session->warn(fmt("ming-image: the DiT at '{}' did not load ({}); "
                        "this family is inert", ck.transformer_dir, err));
    }
    return nullptr;
  }
  // A STREAMING DiT HOLDS ITS FLOOR, NOT ITS CHECKPOINT, and says so --
  // generate-image does not revise a registered family's declaration.
  if (stream && mgr != nullptr) {
    const std::size_t held = std::max<std::size_t>(
        dit_floor, (std::size_t)dit->resident_bytes());
    mgr->revise_declaration(ck.transformer_dir, held);
  }
  std::unique_ptr<MingImageGenerator> g(new MingImageGenerator());
  g->_ck = ck;
  g->_mc = args.metal;
  g->_session = session;
  g->_dit = std::move(dit);
  for (int i = 0; i < MingDit::kMaxLoraSlots; ++i) {
    g->_lora_ref[i] = lora_ref[i];
    g->_lora_scale[i] = lora_scale[i];
  }
  return g;
}

void
MingImageGenerator::apply_lora_keys_(const vpipe::FlexData* cfg)
{
  if (cfg == nullptr || !cfg->is_object() || !_dit) { return; }
  const auto o = cfg->as_object();
  for (int i = 0; i < kLoraSlots; ++i) {
    const char* pk = kCfgLoraKeys[i][0];
    const char* sk = kCfgLoraKeys[i][1];
    // An ABSENT key keeps what is in force: a beat carrying other keys
    // and not these is saying nothing about the adapter.
    if (o.contains(pk) &&
        std::string(o.at(pk).as_string("")) != _lora_ref[i]) {
      warn_once_(fmt("the DiT is already loaded, so the model_config "
                     "beat's `{}` '{}' is IGNORED -- it is a load-time "
                     "argument (loaded: '{}'). `{}` still applies", pk,
                     std::string(o.at(pk).as_string("")), _lora_ref[i],
                     sk)());
    }
    if (!o.contains(sk)) { continue; }
    const float s = (float)o.at(sk).as_real(1.0);
    if (s == _lora_scale[i]) { continue; }
    _lora_scale[i] = s;
    _dit->set_lora_scale(i, s);
    if (_session != nullptr && _dit->lora_modules(i) > 0) {
      _session->info(fmt("ming-image: `{}` -> {:.3f} ({} modules)", sk, s,
                         _dit->lora_modules(i)));
    }
  }
}

void
MingImageGenerator::release_idle()
{
  if (_dit) { _dit->release_resident_blocks((std::size_t)-1); }
}

std::uint64_t
MingImageGenerator::resident_bytes() const
{
  return _dit ? _dit->resident_bytes() : 0;
}

bool
MingImageGenerator::generate(const vpipe::genai::ImageGenRequest& req,
                             vpipe::genai::ImageGenResult* out)
{
  if (out == nullptr || !_dit || _mc == nullptr) { return false; }
  const DitConfig& d = _ck.dit;
  const int Z = d.in_channels, P = d.patch;
  const int lh = req.height / kVaeScale, lw = req.width / kVaeScale;
  if (lh <= 0 || lw <= 0 || lh % P != 0 || lw % P != 0) {
    warn_(fmt("{}x{} is not a whole number of {}-pixel patches",
              req.width, req.height, kVaeScale * P));
    return false;
  }
  const int gh = lh / P, gw = lw / P;
  const int img = gh * gw;
  const int PD = P * P * Z;

  // ---- conditioning ------------------------------------------------
  if (req.cond == nullptr || req.cond_rows <= 0) {
    warn_("no conditioning -- wire `ming-image-conditioner` into "
          "generate-image's conditioning port");
    return false;
  }
  if (req.cond_dim != d.dim) {
    warn_(fmt("conditioning is {} wide where this DiT takes {} -- the "
              "graph's conditioner is not `ming-image-conditioner`",
              req.cond_dim, d.dim));
    return false;
  }
  const int qrows = query_rows_of(req.cond_sideband,
                                  _ck.cond.query_rows());
  if (qrows < 0 || qrows > req.cond_rows) {
    warn_(fmt("conditioning has {} rows but claims {} query rows",
              req.cond_rows, qrows));
    return false;
  }
  const std::vector<std::uint16_t> pos = cond_to_bf16(
      req.cond, req.cond_rows, req.cond_dim, req.cond_elem_size,
      req.cond_is_bf16);
  if (pos.empty()) {
    warn_("conditioning is neither bf16 nor f32");
    return false;
  }
  // THE NEGATIVE IS ZEROS unless the graph wired one: the reference's
  // own negative is the positive's two streams times zero.
  const bool cfg = req.guidance_scale > 1.0;
  std::vector<std::uint16_t> neg;
  int neg_rows = 0;
  if (cfg) {
    if (req.neg != nullptr && req.neg_rows > 0) {
      neg = cond_to_bf16(req.neg, req.neg_rows, req.cond_dim,
                         req.cond_elem_size, req.cond_is_bf16);
      neg_rows = req.neg_rows;
    } else {
      neg.assign(pos.size(), 0);
      neg_rows = req.cond_rows;
    }
  }

  Layout lay, lay_neg;
  std::string lerr;
  if (!build_layout(gh, gw, qrows, req.cond_rows - qrows, d.axes_lens[0],
                    &lay, &lerr) ||
      (cfg && !build_layout(gh, gw, std::min(qrows, neg_rows),
                            neg_rows - std::min(qrows, neg_rows),
                            d.axes_lens[0], &lay_neg, &lerr))) {
    warn_(fmt("the joint sequence does not add up: {}", lerr));
    return false;
  }
  auto upload = [&](const std::vector<std::uint16_t>& v) {
    SharedBuffer b = _mc->make_shared_buffer(v.size() * 2);
    if (!b.empty()) { std::memcpy(b.contents(), v.data(), v.size() * 2); }
    return b;
  };
  const SharedBuffer cap_pos = upload(pos);
  const SharedBuffer cap_neg = cfg ? upload(neg) : SharedBuffer();
  if (cap_pos.empty() || (cfg && cap_neg.empty())) { return false; }

  // A TEXT-TO-IMAGE checkpoint has nowhere to put a reference latent: no
  // image stream in its sequence and no edit training. Said, because a
  // reference that does nothing reads as an edit the model ignored.
  if (!req.references.empty()) {
    warn_(fmt("{} reference latent(s) wired, but Ming-Image-0.1-Design is "
              "text-to-image only -- ignored", req.references.size()));
  }

  // ---- the adapters' strengths, live --------------------------------
  apply_lora_keys_(req.model_config);

  // ---- schedule and noise ------------------------------------------
  const int S = req.steps > 0 ? req.steps : kDefaultSteps;
  const float shift =
      (float)config_real_(req.model_config, kCfgShift, _ck.shift);
  note_unhonoured_(req.scheduler_spec, req.sampler_spec, shift,
                   [this](std::string m) { warn_once_(m); });
  // The reference pipeline's two guidance controls (both off by default).
  const double cfg_trunc = config_real_(req.model_config, kCfgTrunc, 1.0);
  const double cfg_norm = config_real_(req.model_config, kCfgNorm, 0.0);
  const std::vector<float> sig = schedule_sigmas(S, shift);
  // RESERVE FIRST: the schedule sizes its per-forward admission cap from
  // the reserve in force, so the other order sizes the first generation
  // against nothing.
  _dit->set_residency_reserve(vpipe::model_memory::vae_decode_scratch_bytes(
      _ck.root, req.width, req.height));
  _dit->set_residency_schedule(S);
  std::vector<float> packed((std::size_t)img * PD);
  {
    TorchCpuRandn rng(req.seed);
    const std::vector<float> nchw = rng.randn((std::size_t)Z * lh * lw);
    if (nchw.size() != (std::size_t)Z * lh * lw) { return false; }
    pack_latents(nchw.data(), Z, lh, lw, P, packed.data());
  }
  SharedBuffer latbuf = _mc->make_shared_buffer(packed.size() * 2);
  if (latbuf.empty()) { return false; }

  // ---- stop and progress, detached on every return -----------------
  auto stopped = std::make_shared<std::atomic<bool>>(false);
  struct Detach {
    MingDit* d;
    ~Detach()
    {
      d->set_stream_stop(nullptr);
      d->set_block_progress(nullptr);
    }
  } detach{_dit.get()};
  _dit->set_stream_stop([stopped]() { return stopped->load(); });
  if (req.block_progress) {
    const auto bp = req.block_progress;
    _dit->set_block_progress(
        [bp, stopped](int done, int total) -> std::function<void()> {
          return [bp, stopped, done, total]() {
            if (!bp(done, total)) { stopped->store(true); }
          };
        });
  }

  auto forward = [&](const SharedBuffer& cap, const Layout& L, float sigma,
                     std::vector<float>* v) -> bool {
    MingDit::Request r;
    r.latents = &latbuf;
    r.cap = &cap;
    r.cap_stride = d.dim;
    r.layout = &L;
    r.sigma = sigma;
    std::string e;
    // On the LLM perf lane, as the built-in DiTs book each forward, so
    // the denoise is not a blank stretch of a profile.
    vpipe::PerfAuxScope perf(_session, vpipe::kPerfLaneLLM,
                             vpipe::kGvidLlmDit, vpipe::kPerfLlmDitBegin,
                             (std::uint64_t)L.joint_len);
    SharedBuffer o = _dit->forward(r, &e);
    if (o.empty()) {
      if (!e.empty()) { warn_(fmt("DiT forward failed: {}", e)); }
      return false;
    }
    const auto* op = static_cast<const std::uint16_t*>(o.contents());
    v->resize(packed.size());
    for (std::size_t k = 0; k < v->size(); ++k) {
      (*v)[k] = bf16_to_f32(op[k]);
    }
    return true;
  };

  const auto t0 = std::chrono::steady_clock::now();
  std::vector<float> v, vn;
  const float g = (float)req.guidance_scale;
  for (int i = 0; i < S; ++i) {
    if (stopped->load()) { return false; }
    // The reference casts the f32 latent to bf16 for the DiT.
    auto* lb = static_cast<std::uint16_t*>(latbuf.contents());
    for (std::size_t k = 0; k < packed.size(); ++k) {
      lb[k] = f32_to_bf16(packed[k]);
    }
    if (!forward(cap_pos, lay, sig[(std::size_t)i], &v)) { return false; }
    // TRUNCATION, the reference's: past (1 - sigma) > cfg_truncation the
    // step runs unguided -- and then the negative pass is not run at all.
    const bool guide = cfg && (1.0 - (double)sig[(std::size_t)i]) <= cfg_trunc;
    if (guide) {
      if (!forward(cap_neg, lay_neg, sig[(std::size_t)i], &vn)) {
        return false;
      }
      double pn = 0.0, gn = 0.0;
      for (std::size_t k = 0; k < v.size(); ++k) {
        const float p = v[k];
        v[k] = p + g * (p - vn[k]);     // pos + g*(pos - neg)
        pn += (double)p * p;
        gn += (double)v[k] * v[k];
      }
      // RENORMALIZATION, the reference's: the guided prediction's norm is
      // clamped to cfg_normalization x the conditional one's.
      if (cfg_norm > 0.0 && gn > 0.0) {
        const double maxn = std::sqrt(pn) * cfg_norm;
        const double newn = std::sqrt(gn);
        if (newn > maxn) {
          const float k = (float)(maxn / newn);
          for (auto& x : v) { x *= k; }
        }
      }
    }
    // NEGATED, then the ordinary Euler step toward sigma 0.
    const float dt = sig[(std::size_t)i + 1] - sig[(std::size_t)i];
    for (std::size_t k = 0; k < packed.size(); ++k) {
      packed[k] += dt * -v[k];
    }
    if (req.step_latent) {
      std::vector<float> l((std::size_t)Z * lh * lw);
      unpack_latents(packed.data(), Z, lh, lw, P, l.data());
      req.step_latent(i + 1, S, l.data(), {Z, lh, lw});
    }
    if (req.progress && !req.progress(i + 1, S)) { return false; }
  }
  const double secs = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  if (_session != nullptr) {
    _session->info(fmt(
        "ming-image: {}x{} latent in {:.2f} s ({} forwards, {} image + {} "
        "caption rows, shift {:.1f}, {})",
        req.width, req.height, secs, S, img, lay.cap_len, shift,
        cfg ? fmt("cfg {:.2f}", req.guidance_scale)()
            : std::string("no cfg")));
  }
  out->latent.resize((std::size_t)Z * lh * lw);
  unpack_latents(packed.data(), Z, lh, lw, P, out->latent.data());
  out->shape = {Z, lh, lw};
  return true;
}

}  // namespace ming
