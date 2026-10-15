// Checkpoint-free facts about the model, and -- with
// VPIPE_MING_TEST_MODEL_PATH set to the real checkpoint -- the identity
// test against it and against lookalikes.

#include "ming-config.h"
#include "ming-dit.h"
#include "ming-family.h"
#include "ming-generator.h"
#include "ming-layout.h"
#include "ming-model-config-stage.h"
#include "ming-test.h"
#include "ming-torch-randn.h"

#include "stages/model-memory.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

const char*
model_path()
{
  const char* p = std::getenv("VPIPE_MING_TEST_MODEL_PATH");
  return (p != nullptr && *p != '\0') ? p : nullptr;
}

void
write_file(const fs::path& p, const std::string& s)
{
  fs::create_directories(p.parent_path());
  std::ofstream(p) << s;
}

}  // namespace

int
main()
{
  CASE("schedule: 11 forwards at shift 6 are the reference's 12 steps");
  {
    // docs/MING-IMAGE.md 3.1: sigma = 6m / (11 + 5m), m = 11 - k, and the
    // reference's 12th point (sigma 0) is the terminal, not a forward.
    const std::vector<float> s = ming::schedule_sigmas(11, 6.0f);
    CHECK(s.size() == 12);
    if (s.size() == 12) {
      for (int k = 0; k < 11; ++k) {
        const double m = 11 - k;
        CHECK(mingtest::near(s[(std::size_t)k], 6 * m / (11 + 5 * m), 1e-7));
      }
      CHECK(s[0] == 1.0f && s[10] == 0.375f && s[11] == 0.0f);
      CHECK(mingtest::near(s[1], 60.0 / 61.0, 1e-7));
    }
  }

  CASE("layout: pads dropped, positions keep the padded caption run");
  {
    // The sample prompt: 256 query rows + 34 template tokens = 290 real
    // caption rows, padded to 320 for POSITIONS only.
    ming::Layout L;
    std::string err;
    CHECK(ming::build_layout(32, 32, 256, 34, 20480, &L, &err));
    CHECK(L.img_tokens == 1024 && L.cap_len == 290 && L.cap_total == 320);
    CHECK(L.joint_len == 1024 + 290);
    CHECK(L.pos_t.size() == (std::size_t)L.joint_len);
    if (L.pos_t.size() == (std::size_t)L.joint_len) {
      // Image row (h, w) = (cap_total + 1, h, w): 321, NOT 291.
      CHECK(L.pos_t[0] == 321 && L.pos_h[0] == 0 && L.pos_w[0] == 0);
      CHECK(L.pos_t[33] == 321 && L.pos_h[33] == 1 && L.pos_w[33] == 1);
      // Caption row i = (1 + i, 0, 0).
      CHECK(L.pos_t[1024] == 1 && L.pos_h[1024] == 0);
      CHECK(L.pos_t[1024 + 289] == 290 && L.pos_w[1024 + 289] == 0);
    }
    // A caption that is already a multiple of 32 has no gap.
    CHECK(ming::build_layout(2, 2, 256, 32, 20480, &L, &err));
    CHECK(L.cap_total == 288 && L.pos_t[0] == 289);
    // The rotary table bounds the caption.
    CHECK(!ming::build_layout(2, 2, 256, 20300, 20480, &L, &err));
  }

  CASE("packing: channel fastest, and unpack inverts it");
  {
    const int C = 16, H = 4, W = 6, P = 2;
    std::vector<float> z((std::size_t)C * H * W);
    for (std::size_t i = 0; i < z.size(); ++i) { z[i] = (float)i; }
    std::vector<float> rows(z.size()), back(z.size());
    ming::pack_latents(z.data(), C, H, W, P, rows.data());
    // Row (0,0), patch offset (0,1), channel 2 = z[2][0][1].
    CHECK(rows[(std::size_t)(1 * C + 2)] == z[(std::size_t)(2 * H * W + 1)]);
    ming::unpack_latents(rows.data(), C, H, W, P, back.data());
    CHECK(back == z);
  }

  CASE("noise: torch.Generator('cpu').manual_seed(42).randn, bit for bit");
  {
    ming::TorchCpuRandn rng(42);
    const std::vector<float> x = rng.randn(16 * 8 * 8);
    CHECK(x.size() == 1024);
    if (x.size() == 1024) {
      CHECK(x[0] == 1.9269150495529175f && x[1] == 1.4872841835021973f);
      CHECK(x[3] == -2.1055214405059814f && x[1023] == 1.5817502737045288f);
    }
  }

  CASE("claims: lookalikes are not ours");
  {
    const fs::path d = fs::temp_directory_path() /
        ("vpipe-ming-claims-" + std::to_string((long)::getpid()));
    std::error_code ec;
    fs::remove_all(d, ec);
    // A Z-Image-shaped DiT that does not declare the Ming contract.
    write_file(d / "transformer" / "config.json",
               R"({"_class_name": "DiffusionTransformer", "dim": 3840})");
    ming::MingImageFamily fam;
    CHECK(!fam.claims(d.string(), ""));
    CHECK(!fam.claims(d.string(), "ming-image"));
    // The contract alone, with no conditioning MLLM beside it.
    write_file(d / "transformer" / "config.json",
               R"({"_class_name": "DiffusionTransformer",
                   "alignment_padding_mode": "zero_masked",
                   "multi_frame_output": false})");
    CHECK(!fam.claims(d.string(), ""));
    fs::remove_all(d, ec);
  }

  CASE("model config: only keys that were SET are forwarded");
  {
    vpipe::FlexData c = vpipe::FlexData::make_object();
    {
      auto o = c.as_object();
      o.insert(ming::kCfgTrunc, vpipe::FlexData::make_real(0.8));
    }
    ming::MingModelConfigStage st(nullptr, "cfg", {}, c);
    const vpipe::FlexData fd = st.resolved_config();
    CHECK(fd.is_object());
    if (fd.is_object()) {
      const auto o = fd.as_object();
      CHECK(o.contains("model_family") &&
            o.at("model_family").as_string("") == "ming-image");
      CHECK(o.contains(ming::kCfgTrunc) &&
            o.at(ming::kCfgTrunc).as_real(0.0) == 0.8);
      // Unset: absent, so the family's own value (the checkpoint's 6.0)
      // stands rather than a default that merely looks configured.
      CHECK(!o.contains(ming::kCfgShift));
      CHECK(!o.contains(ming::kCfgNorm));
    }
  }

  if (const char* root = model_path()) {
    CASE("checkpoint: the published Design checkpoint");
    const ming::Checkpoint ck = ming::read_checkpoint(root);
    CHECK(ck.valid);
    if (!ck.valid) { std::printf("    why: %s\n", ck.why.c_str()); }
    CHECK(ck.text_to_image());
    CHECK(ck.dit.dim == 3840 && ck.dit.n_layers == 30 &&
          ck.dit.n_refiner_layers == 2 && ck.dit.n_heads == 30);
    CHECK(ck.dit.head_dim() == 128 && ck.dit.cap_feat_dim == 2560);
    CHECK(ck.dit.axes_lens.size() == 3 && ck.dit.axes_lens[0] == 20480);
    CHECK(ck.cond.query_rows() == 256);
    CHECK(ck.cond.taps == std::vector<int>({5, 12, 20}));
    CHECK(ck.shift == 6.0f && ck.vae_channels == 4);
    CHECK(mingtest::near(ck.vae_scaling, 8.0064, 1e-6));
    ming::MingImageFamily fam;
    CHECK(fam.claims(root, ""));
    int gh = 0, gw = 0;
    fam.size_grid(root, &gh, &gw);
    CHECK(gh == 16 && gw == 16);
    CHECK(fam.latent_bytes(root, 1024, 1024) == 16ull * 128 * 128 * 4);
    // The DiT (streamable, every phase), the encoder's three sets in
    // the CONDITION phase -- the family answers for the encoder, so
    // generate-image adds no unphased claim of its own -- and the ANE
    // feed-forward module: one CoreML unit in the DENOISE phase.
    const auto claims = fam.declare_resources(root);
    CHECK(claims.size() == 5);
    if (claims.size() == 5) {
      std::string label;
      std::size_t ub = 0;
      int un = 0;
      CHECK(vpipe::model_memory::parse_coreml_claim(claims[4], &label, &ub,
                                                    &un));
      CHECK(label == ming::kAneClaimLabel && ub > 0 && un == 1 &&
            claims[4].phase == "denoise");
      std::printf("    ANE module: %.0f MB, one unit\n", ub / 1048576.0);
    }
    if (claims.size() >= 4) {
      CHECK(claims[0].key == ck.transformer_dir && claims[0].phase.empty());
      CHECK(claims[0].floor_bytes > 0);
      CHECK(claims[1].key == ck.mllm_dir && claims[1].phase == "condition");
      CHECK(claims[1].floor_bytes > 0);
      CHECK(claims[2].key == ck.connector_dir &&
            claims[2].phase == "condition");
      CHECK(claims[3].key == ck.mlp_dir && claims[3].phase == "condition");
    }
    const auto holds = fam.declare_holdings(root);
    CHECK(holds.size() == 1);
    if (holds.size() == 1) {
      std::printf("    DiT: %.0f MB on disk, floor %.0f MB\n",
                  holds[0].preload / 1048576.0, holds[0].floor / 1048576.0);
      CHECK(holds[0].floor > 0 && holds[0].floor < holds[0].preload / 4);
    }
  } else {
    std::printf("- checkpoint cases SKIPPED (VPIPE_MING_TEST_MODEL_PATH)\n");
  }

  return mingtest::summary("ming-config");
}
