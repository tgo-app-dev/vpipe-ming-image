#include "ming-conditioner-stage.h"

#include "ming-family.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/perf-event.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "generative-models/generative-model-manager.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-registry.h"

#include <cstring>
#include <utility>

namespace ming {

using vpipe::ConfigKey;
using vpipe::ConfigType;
using vpipe::FlexData;
using vpipe::FlexDataPayload;
using vpipe::InEdge;
using vpipe::Job;
using vpipe::PortSpec;
using vpipe::ResourceClaim;
using vpipe::RuntimeContext;
using vpipe::SessionContextIntf;
using vpipe::StageCategory;
using vpipe::StageSpec;
using vpipe::TensorBeat;
using vpipe::TensorBeatPayload;
using vpipe::fmt;
namespace mm = vpipe::model_memory;

namespace {

constexpr unsigned kPromptPort = 0;
constexpr unsigned kModelPort  = 1;
constexpr unsigned kCondOut    = 0;

const PortSpec kIports[] = {
  {.name = "prompt", .doc = "prompt text (FlexData string or {text: ...})",
   .type = &typeid(FlexDataPayload), .tags = "text", .clock_group = 0},
  {.name = "model",
   .doc = "OPTIONAL model reference (model-select); overrides hf_dir",
   .type = &typeid(FlexDataPayload), .tags = "model", .clock_group = 0},
};

const PortSpec kOports[] = {
  {.name = "conditioning",
   .doc = "bf16 [256 + N, 3840] for generate-image's conditioning port: "
          "the connector's 256 caption rows (2560 wide, zero-padded) and "
          "then the MLLM's N template rows. The sideband's "
          "`ming_query_rows` says where the first block ends. Only a "
          "Ming-Image DiT reads it",
   .type = &typeid(TensorBeatPayload), .tags = "conditioning",
   .clock_group = 0},
};

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "the Ming-Image checkpoint root (the directory holding mllm/, "
          "connector/, mlp/ and transformer/). A model-select source on "
          "the `model` iport overrides it",
   .def_str = "",
   .suggest_db = vpipe::kModelRegistryDb,
   .suggest_db_type = "ming-image",
   .model_channel = "diffusion-model"},
  {.key = "unload_when_idle", .type = ConfigType::Text, .required = false,
   .doc = "what happens to the ~32 GB MLLM between prompts -- it is idle "
          "for the whole denoise. \"auto\" (default) lets it go when it "
          "and the DiT do not fit this machine's RAM together with "
          "headroom; \"destroy\" always frees it (a reload per new "
          "prompt); \"keep\" holds it. A repeated prompt never reloads: "
          "its conditioning is kept",
   .def_str = "auto"},
  {.key = "stream", .type = ConfigType::Text, .required = false,
   .doc = "whether the MLLM's layers are read from disk per prompt "
          "rather than held: \"auto\" (default) streams when the box is "
          "short of room for it, \"on\" / \"off\" force it",
   .def_str = "auto"},
};

const StageSpec kSpec = {
  .type_name = MingConditionerStage::kTypeName,
  .doc = "Ming-Image text conditioning: a prompt becomes the DiT's "
         "two-stream caption through the BailingMoeV2 MLLM (with 256 "
         "learned query tokens) and a Qwen2 connector. Feeds "
         "generate-image with a Ming-Image checkpoint.",
  .display_name = "Ming-Image Conditioner",
  .category = StageCategory::Generative,
  .iports = kIports,
  .oports = kOports,
  .attrs = kAttrs,
};

std::string
prompt_of_(const FlexData& fd)
{
  if (fd.is_string()) { return std::string(fd.as_string("")); }
  if (fd.is_object()) {
    const auto o = fd.as_object();
    for (const char* k : {"text", "prompt", "caption"}) {
      if (o.contains(k)) {
        const FlexData v = o.at(k);
        if (v.is_string()) { return std::string(v.as_string("")); }
      }
    }
  }
  return {};
}

// The encoder's three weight sets, in claim order.
std::vector<std::string>
encoder_dirs_(const Checkpoint& ck)
{
  return {ck.mllm_dir, ck.connector_dir, ck.mlp_dir};
}

}  // namespace

const StageSpec*
MingConditionerStage::stage_spec() noexcept
{
  return &kSpec;
}

const StageSpec&
MingConditionerStage::spec() const noexcept
{
  return kSpec;
}

MingConditionerStage::MingConditionerStage(const SessionContextIntf* session,
                                           std::string               id,
                                           std::vector<InEdge>       iports,
                                           FlexData                  config)
  : vpipe::TypedStage<MingConditionerStage>(session, std::move(id),
                                            std::move(iports),
                                            std::move(config))
{
  // The spec only DESCRIBES the oports; without this the stage reports
  // none and every downstream edge is refused at launch.
  this->allocate_oports(kSpec.oports.size());
  const FlexData& c = this->config();
  if (!c.is_object()) { return; }
  const auto o = c.as_object();
  if (o.contains("hf_dir")) {
    _hf_dir = std::string(o.at("hf_dir").as_string(""));
  }
  if (o.contains("unload_when_idle")) {
    bool bad = false;
    _unload_cfg = mm::parse_unload_policy(
        std::string(o.at("unload_when_idle").as_string("auto")), &bad);
    if (bad) {
      fail_config(fmt("{}: unload_when_idle must be auto, destroy or keep",
                      this->id()));
    }
  }
  if (o.contains("stream")) {
    _stream_cfg = std::string(o.at("stream").as_string("auto"));
    if (_stream_cfg != "auto" && _stream_cfg != "on" &&
        _stream_cfg != "off") {
      fail_config(fmt("{}: stream must be auto, on or off", this->id()));
    }
  }
}

MingConditionerStage::~MingConditionerStage() = default;

std::string
MingConditionerStage::model_root_() const
{
  return vpipe::resolve_model_dir(this->session(), _hf_dir);
}

void
MingConditionerStage::apply_constant(unsigned iport, const FlexData& beat)
{
  if (iport != kModelPort) { return; }
  std::string ref;
  if (vpipe::apply_model_select_beat(beat, ref) && !ref.empty()) {
    _hf_dir = ref;
  }
}

bool
MingConditionerStage::releases_(const Checkpoint& ck) const
{
  switch (_unload_cfg) {
    case mm::UnloadPolicy::kDestroy: return true;
    case mm::UnloadPolicy::kKeep:    return false;
    default: break;
  }
  // auto: the encoder and the DiT, together, against total RAM. Parking
  // is not offered -- the encoder reads its weights into its own
  // buffers, so a park would hand the kernel nothing.
  std::vector<std::string> dirs = encoder_dirs_(ck);
  dirs.push_back(ck.transformer_dir);
  return mm::bounded(this->session(), dirs, mm::kHeadroom);
}

bool
MingConditionerStage::streams_(const Checkpoint& ck) const
{
  if (_stream_cfg == "on") { return true; }
  if (_stream_cfg == "off") { return false; }
  // A prefill-only encoder that is let go after each prompt competes
  // with nothing but itself; one that is kept competes with the DiT.
  std::vector<std::string> dirs = encoder_dirs_(ck);
  if (!releases_(ck)) { dirs.push_back(ck.transformer_dir); }
  return mm::bounded(this->session(), dirs, mm::kStreamHeadroom);
}

std::vector<ResourceClaim>
MingConditionerStage::declare_resources() const
{
  if (_hf_dir.empty()) { return {}; }
  const Checkpoint ck = read_checkpoint(model_root_());
  if (!ck.valid) { return {}; }
  // The MLLM streams its layers when the box is short of room, so its
  // claim carries the floor it streams at. Only in the CONDITION phase
  // when it is let go after the prompt: a peer sizing the denoise must
  // not count an encoder that will be gone by then.
  std::vector<ResourceClaim> out;
  const bool phased = releases_(ck);
  const std::size_t floor = MingEncoder::streaming_floor_bytes(ck);
  if (!phased) {
    out.push_back(floor > 0
                      ? mm::weight_claim_streamable(ck.mllm_dir, floor)
                      : mm::weight_claims({ck.mllm_dir}).front());
    for (auto& c : mm::weight_claims({ck.connector_dir, ck.mlp_dir})) {
      out.push_back(std::move(c));
    }
    return out;
  }
  return mm::weight_claims_in_phase(encoder_dirs_(ck), mm::kPhaseCondition);
}

vpipe::StageMemory
MingConditionerStage::declare_memory() const
{
  vpipe::StageMemory m;
  if (_hf_dir.empty()) { return m; }
  const Checkpoint ck = read_checkpoint(model_root_());
  if (!ck.valid) { return m; }
  const bool rel = releases_(ck);
  // What the encoder HOLDS, from its own allocation arithmetic -- not the
  // directories' sizes: mllm/ carries a vision tower and an lm_head that
  // text-to-image never loads, and connector/ and mlp/ are f32 on disk
  // and bf16 here (2.4x apart). The connector's figure covers the heads
  // in mlp/ too, so mlp/ holds nothing of its own on this ledger.
  m.hold(ck.mllm_dir, MingEncoder::resident_weight_bytes(ck),
         MingEncoder::streaming_floor_bytes(ck), rel);
  m.hold(ck.connector_dir, MingEncoder::connector_weight_bytes(ck), 0, rel);
  // The conditioning beat: a few MB at the longest prompts the
  // reference advertises (~1.4k template tokens).
  m.outputs = {(std::size_t)(ck.cond.query_rows() + 2048) *
               (std::size_t)ck.cond.inner_dim * 2};
  return m;
}

void
MingConditionerStage::reset_run_state()
{
  _emitted = 0;
}

bool
MingConditionerStage::ensure_loaded_(const Checkpoint& ck)
{
  if (_enc && _loaded_root == ck.root) { return true; }
  _enc.reset();
  auto* mc = this->session()->services()->metal_compute();
  if (mc == nullptr) {
    this->session()->warn(fmt("{}: no Metal device", this->id()));
    return false;
  }
  MingEncoder::Options opt;
  opt.stream_layers = streams_(ck);
  std::string err;
  _enc = MingEncoder::load(ck, mc, this->session(), opt, &err);
  if (!_enc) {
    this->session()->warn(fmt("{}: the encoder did not load: {}",
                              this->id(), err));
    return false;
  }
  _loaded_root = ck.root;
  this->session()->info(fmt(
      "{}: Ming-Image encoder loaded ({}, {:.1f} GB held)", this->id(),
      opt.stream_layers ? "MLLM layers streamed" : "resident",
      (double)_enc->resident_bytes() / 1e9));
  return true;
}

Job
MingConditionerStage::process(RuntimeContext& ctx)
{
  if (ctx.num_iports() > kModelPort && ctx.iport_connected(kModelPort)) {
    auto mb = co_await ctx.read(kModelPort);
    if (const auto* mp = mb ? dynamic_cast<const FlexDataPayload*>(mb.get())
                            : nullptr) {
      std::string ref;
      if (vpipe::apply_model_select_beat(mp->data, ref) && !ref.empty()) {
        _hf_dir = ref;
      }
    }
  }
  auto pb = co_await ctx.read(kPromptPort);
  if (!pb) {
    // EOS: SIGNAL DONE. A bare co_return is re-entered forever.
    ctx.signal_done();
    co_return;
  }
  const auto* fp = dynamic_cast<const FlexDataPayload*>(pb.get());
  if (fp == nullptr) { co_return; }
  const std::string prompt = prompt_of_(fp->data);
  if (prompt.empty()) {
    this->session()->warn(fmt("{}: the prompt beat carries no text; "
                              "emitting nothing", this->id()));
    co_return;
  }
  if (_hf_dir.empty()) {
    this->session()->warn(fmt("{}: no `hf_dir` and no model reference; "
                              "emitting nothing", this->id()));
    co_return;
  }
  const Checkpoint ck = read_checkpoint(model_root_());
  if (!ck.valid) {
    this->session()->warn(fmt("{}: '{}' is not a Ming-Image checkpoint: {}",
                              this->id(), model_root_(), ck.why));
    co_return;
  }

  // A repeat of the last prompt reuses its conditioning: it depends on
  // the prompt alone, and re-encoding would reload a released MLLM.
  if (!(prompt == _last_prompt && ck.root == _last_root &&
        !_last.cond.empty())) {
    if (!ensure_loaded_(ck)) { co_return; }
    MingEncoder::Result r;
    std::string err;
    bool ok = false;
    {
      // On the LLM perf lane, where the host's conditioners book their
      // text encodes.
      vpipe::PerfAuxScope perf(this->session(), vpipe::kPerfLaneLLM,
                               vpipe::kGvidLlmDitText,
                               vpipe::kPerfLlmDitTextBegin,
                               (std::uint64_t)prompt.size());
      ok = _enc->encode(prompt, &r, &err);
    }
    if (!ok) {
      this->session()->warn(fmt("{}: encode failed: {}", this->id(), err));
      co_return;
    }
    _last = std::move(r);
    _last_prompt = prompt;
    _last_root = ck.root;
    // RELEASE BEFORE PUBLISHING: generate-image loads its DiT on the
    // first conditioning beat and decides streaming there, so the
    // encoder's bytes must be gone -- and its declaration with them --
    // before that beat lands.
    if (releases_(ck)) {
      _enc.reset();
      _loaded_root.clear();
      if (auto* mgr = this->session()->services()
                          ->generative_model_manager()) {
        for (const std::string& d : encoder_dirs_(ck)) {
          mgr->revise_declaration(d, 0);
        }
      }
    }
  }

  TensorBeat tb;
  tb.dtype = TensorBeat::DType::Bf16;
  tb.shape = {(std::int64_t)_last.rows, (std::int64_t)_last.dim};
  tb.resize_contiguous((std::size_t)_last.rows * _last.dim);  // ELEMENTS
  std::memcpy(tb.data.data(), _last.cond.data(), _last.cond.size() * 2);
  tb.sideband = FlexData::make_object();
  {
    auto so = tb.sideband.as_object();
    so.insert(kSbQueryRows, FlexData::make_int(_last.query_rows));
  }
  this->session()->info(fmt(
      "{}: '{}' -> [{}, {}] ({} connector + {} text rows)", this->id(),
      prompt.size() > 48 ? prompt.substr(0, 45) + "..." : prompt,
      _last.rows, _last.dim, _last.query_rows, _last.text_rows));
  co_await ctx.write(kCondOut,
                     vpipe::make_payload<TensorBeatPayload>(std::move(tb)));
  ++_emitted;
}

}  // namespace ming
