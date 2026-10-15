// The conditioning encoder against the reference's own goldens.
//
// VPIPE_MING_TEST_MODEL_PATH = the checkpoint ROOT (mllm/, connector/,
// mlp/ ...). VPIPE_MING_GOLDEN = the goldens root
// tools/dump_ming_golden.py writes:
//
//   tokenizer/<key>/      prompt.txt, ids_template, ids_bare -- ~10
//                         prompts (ASCII, Chinese, the RGBA prefixes,
//                         digits, whitespace, the cabin JSON, emoji, NFD
//                         accents, mixed quotes)
//   conditioning/<p1|cabin>/<variant>/
//                         every intermediate of one reference
//                         conditioning pass, all rows; variants fp32 (the
//                         truth), bf16 (the reference as it runs under
//                         CUDA autocast: bf16 ROUTER logits) and
//                         bf16_f32router (the reference code as read: f32
//                         router). report.json = each bf16 variant's
//                         rel-L2 against fp32 -- THE BAR.
//
// THE BAR, and why it is a bar and not a tolerance. This MLLM routes
// with sigmoid + bias top-8 over 256 experts, and near-ties flip under
// any rounding: the reference's OWN bf16 run picks different experts for
// 4-6% of its slots with an f32 router and ~45% with the CUDA one, and
// the hidden states drift accordingly (hs20 rel-L2 0.15 / 0.41 on P1).
// So the port is held to "no worse than the reference's own bf16",
// variant by variant: this encoder's router is f32, so the
// bf16_f32router numbers are the bar it must meet, and it is stated
// against both. Routing is reported as an agreement RATE with the fp32
// routing, per layer -- exactness would be a wrong bar.
//
// The connector is ALSO checked teacher-forced: the fp32 golden's own
// connector input through this connector, against its output -- one
// component, no upstream drift, held to the reference's teacher-forced
// bf16 number (report_tf.json heads.bf16.connector).
//
// Either env var unset: the part that needs it SKIPS and says so.
//
// FLOORS -- what these geometries exercise: P1 is L = 292 rows (34
// text), the cabin 1665 (1407 text). Both clear the 64-row wide-GEMM
// tile everywhere but nowhere else; neither is a multiple of the steel
// attention's 32-row query tile, so its remainder path runs; every
// expert sees fewer than 32 rows on P1 (one tile each) and several
// tiles on the cabin, so both arms of the grouped GEMM's tiling run.

#include "ming-config.h"
#include "ming-connector.h"
#include "ming-encoder.h"
#include "ming-ops.h"
#include "ming-test.h"
#include "ming-tokenize.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "generative-models/weight-set.h"

#include <mach/mach.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using vpipe::FlexData;
using vpipe::metal_compute::MetalCompute;
using mingtest::rel_l2;

namespace {

// VPIPE_MING_TEST_PARTS: a comma list of the parts to run (tok, conn,
// resident, tf, stream); unset runs everything. A development knob --
// every resident part reloads ~32 GB.
bool
part(const char* name)
{
  const char* e = std::getenv("VPIPE_MING_TEST_PARTS");
  if (e == nullptr || *e == '\0') { return true; }
  const std::string s = std::string(",") + e + ",";
  return s.find(std::string(",") + name + ",") != std::string::npos;
}

// ---- a golden directory ---------------------------------------------------
// manifest.json maps a name to {dtype, shape, file}; raw little-endian,
// C order, no header.
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
  template <typename T>
  std::vector<T> read(const std::string& n) const
  {
    std::vector<T> v;
    if (!has(n)) { return v; }
    const FlexData e = _man.as_object().at(n);
    const std::string file(e.as_object().at("file").as_string(""));
    std::size_t count = 1;
    for (long d : shape(n)) { count *= (std::size_t)d; }
    v.resize(count);
    std::ifstream in(_dir / file, std::ios::binary);
    if (!in) { v.clear(); return v; }
    in.read(reinterpret_cast<char*>(v.data()),
            (std::streamsize)(count * sizeof(T)));
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

FlexData
read_json(const fs::path& p)
{
  std::ifstream in(p);
  if (!in) { return {}; }
  try {
    return FlexData::from_json(in);
  } catch (...) {
    return {};
  }
}

std::string
read_text(const fs::path& p)
{
  std::ifstream in(p, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// A number out of a nested report: report.variants.<v>.rel_l2_vs_fp32.<k>.
double
bar_of(const FlexData& rep, const char* variant, const char* key)
{
  if (!rep.is_object()) { return -1.0; }
  try {
    const FlexData v = rep.as_object().at("variants").as_object().at(variant);
    const FlexData r = v.as_object().at("rel_l2_vs_fp32");
    return r.as_object().at(key).as_real(-1.0);
  } catch (...) {
    return -1.0;
  }
}

double
routing_bar(const FlexData& rep, const char* variant,
            const char* what = "overlap_mean")
{
  try {
    const FlexData v = rep.as_object().at("variants").as_object().at(variant);
    const FlexData r = v.as_object().at("routing_vs_fp32");
    return r.as_object().at("all_rows").as_object().at(what).as_real(-1.0);
  } catch (...) {
    return -1.0;
  }
}

// Rows [r0, r1) of a [rows, dim] matrix.
std::vector<float>
rows_of(const std::vector<float>& m, int dim, int r0, int r1)
{
  if (r1 <= r0 || m.size() < (std::size_t)r1 * dim) { return {}; }
  return std::vector<float>(m.begin() + (std::ptrdiff_t)r0 * dim,
                            m.begin() + (std::ptrdiff_t)r1 * dim);
}

double
gb(std::size_t b)
{
  return (double)b / 1e9;
}

// This process's physical footprint now and at its peak (what Activity
// Monitor's "Memory" column and the jetsam limit both read).
void
footprint(double* now_gb, double* peak_gb)
{
  task_vm_info_data_t vi{};
  mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
  *now_gb = *peak_gb = 0.0;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vi, &n) ==
      KERN_SUCCESS) {
    *now_gb = gb(vi.phys_footprint);
    *peak_gb = gb(vi.ledger_phys_footprint_peak);
  }
}

double
seconds_since(std::chrono::steady_clock::time_point t0)
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

// ---- tokenizer -----------------------------------------------------------

// docs/MING-IMAGE.md 1.2, VERIFIED there with `tokenizers` on this file.
const std::int32_t kP1Ids[] = {
    157151, 90827, 157152, 651, 4929, 82317, 10618, 37862, 311, 198, 198,
    14136, 5381, 6350, 928, 156895, 157151, 39, 116171, 157152, 32, 2258,
    3862, 33821, 366, 259, 4999, 3742, 156895, 157151, 8469, 7342, 5468,
    157152};

void
test_tokenizer(const fs::path& model, const char* groot)
{
  CASE("tokenizer: id-exact against the reference");
  std::string err;
  auto tok = ming::LingTokenizer::load(
      (model / "mllm" / "tokenizer.json").string(), &err);
  CHECK(tok != nullptr);
  if (!tok) {
    std::printf("  tokenizer did not load: %s\n", err.c_str());
    return;
  }
  const std::vector<std::int32_t> want(std::begin(kP1Ids), std::end(kP1Ids));
  CHECK(tok->encode(ming::ming_t2i_template(
            "A small red cube on a white table")) == want);
  if (groot == nullptr) {
    std::printf("  SKIPPED golden prompts: VPIPE_MING_GOLDEN unset "
                "(only the docs/MING-IMAGE.md sample was checked)\n");
    return;
  }
  const fs::path troot = fs::path(groot) / "tokenizer";
  int n = 0, ok_t = 0, ok_b = 0;
  std::error_code ec;
  for (const auto& de : fs::directory_iterator(troot, ec)) {
    if (!de.is_directory()) { continue; }
    Golden g;
    if (!g.open(de.path())) { continue; }
    const std::string prompt = read_text(de.path() / "prompt.txt");
    const auto t = g.i32("ids_template");
    const auto b = g.i32("ids_bare");
    const auto mt = tok->encode(ming::ming_t2i_template(prompt));
    const auto mb = tok->encode(prompt);
    ++n;
    ok_t += (mt == t);
    ok_b += (mb == b);
    std::printf("  %-14s template %4zu ids %s, bare %4zu ids %s\n",
                de.path().filename().c_str(), t.size(),
                mt == t ? "exact" : "MISMATCH", b.size(),
                mb == b ? "exact" : "MISMATCH");
  }
  std::printf("  golden prompts: %d, template exact %d, bare exact %d\n", n,
              ok_t, ok_b);
  CHECK(n >= 8);
  CHECK(ok_t == n);
  CHECK(ok_b == n);
}

// ---- one prompt against conditioning/<key> -------------------------------

struct Bars {
  double f32r = -1.0;   // bf16_f32router: the reference with THIS router
  double cuda = -1.0;   // bf16: the reference under CUDA autocast
};

Bars
bars(const FlexData& rep, const char* key)
{
  return {bar_of(rep, "bf16_f32router", key), bar_of(rep, "bf16", key)};
}

// Per-row ||a_r - b_r|| / ||b_r|| over [rows, dim], as the goldens
// compute it (dump_ming_golden.py per_row_rel).
std::vector<double>
per_row(const std::vector<float>& a, const std::vector<float>& b, int dim)
{
  std::vector<double> r;
  if (a.size() != b.size() || dim <= 0) { return r; }
  for (std::size_t o = 0; o + (std::size_t)dim <= a.size(); o += dim) {
    double n = 0, d = 0;
    for (int k = 0; k < dim; ++k) {
      const double e = (double)a[o + k] - b[o + k];
      n += e * e;
      d += (double)b[o + k] * b[o + k];
    }
    r.push_back(std::sqrt(n / std::max(d, 1e-60)));
  }
  return r;
}

double
quantile(std::vector<double> v, double q)
{
  if (v.empty()) { return 1.0; }
  std::sort(v.begin(), v.end());
  // numpy / torch "linear" interpolation.
  const double x = q * (double)(v.size() - 1);
  const std::size_t i = (std::size_t)x;
  const double f = x - (double)i;
  return i + 1 < v.size() ? v[i] * (1 - f) + v[i + 1] * f : v[i];
}

// One tensor against fp32: all-rows rel-L2 and the per-row median, each
// held to the reference's own bf16 number (f32 router -- this encoder's
// router) with 10% grace. The CUDA-router bar is printed as the looser
// reference. The per-row median is the sharper bar: this model's
// residual carries massive-activation sink rows (norm ~1e4 against
// ~1e2) that dominate any all-rows figure.
void
report(const FlexData& rep, const char* name, const std::vector<float>& mine,
       const std::vector<float>& gold, int dim)
{
  const double all = rel_l2(mine, gold);
  const auto pr = per_row(mine, gold, dim);
  const double med = quantile(pr, 0.5), p90 = quantile(pr, 0.9);
  const std::string mk = std::string(name) + "_row_median";
  const Bars b = bars(rep, name);
  const Bars bm = bars(rep, mk.c_str());
  const bool bad = (b.f32r >= 0 && all > 1.10 * b.f32r) ||
                   (bm.f32r >= 0 && med > 1.10 * bm.f32r);
  std::printf("  %-14s all-rows %.4f (bar %.4f, CUDA %.4f) | row median "
              "%.4f (bar %.4f, CUDA %.4f) | p90 %.4f%s\n",
              name, all, b.f32r, b.cuda, med, bm.f32r, bm.cuda, p90,
              bad ? "   ABOVE BAR" : "");
  if (b.f32r >= 0.0) { CHECK(all <= 1.10 * b.f32r); }
  if (bm.f32r >= 0.0) { CHECK(med <= 1.10 * bm.f32r); }
}

// Information only: a row subset's all-rows rel-L2 beside the
// reference's. NOT checked -- a subset is ruled by whichever sink row it
// holds (P1's text rows by row 9, whose one channel near -14800 carries
// a bf16 compute error of a few tenths of a percent either way).
void
report_info(const char* what, double v, Bars b)
{
  std::printf("  %-22s all-rows %.4f   (ref bf16 f32-router %.4f, CUDA "
              "%.4f)   info\n", what, v, b.f32r, b.cuda);
}

// Mean over rows of |chosen ∩ golden| / k, per MoE layer.
std::vector<double>
routing_overlap(const std::vector<std::vector<std::int32_t>>& mine,
                const std::vector<std::int32_t>& gold, int L, int K,
                double* exact_rows)
{
  std::vector<double> per;
  long exact = 0, total = 0;
  for (std::size_t l = 0; l < mine.size(); ++l) {
    double acc = 0.0;
    for (int r = 0; r < L; ++r) {
      std::set<int> a, b;
      for (int k = 0; k < K; ++k) {
        a.insert(mine[l][(std::size_t)r * K + k]);
        b.insert(gold[(l * (std::size_t)L + r) * K + k]);
      }
      int hit = 0;
      for (int e : a) { hit += (int)b.count(e); }
      acc += (double)hit / K;
      exact += (hit == K);
      ++total;
    }
    per.push_back(acc / L);
  }
  *exact_rows = total > 0 ? (double)exact / (double)total : 0.0;
  return per;
}

void
check_prompt(ming::MingEncoder& enc, const fs::path& croot,
             const char* key, ming::MingEncoder::Result* keep)
{
  const fs::path dir = croot / key;
  Golden g;
  if (!g.open(dir / "fp32")) {
    std::printf("  SKIPPED %s: no %s\n", key, (dir / "fp32").c_str());
    return;
  }
  const FlexData rep = read_json(dir / "report.json");
  const std::string prompt = read_text(dir / "prompt.txt");
  ming::MingEncoder::Debug dbg;
  ming::MingEncoder::Result res;
  std::string err;
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = enc.encode_debug(prompt, &res, &dbg, &err);
  const double secs = seconds_since(t0);
  CHECK(ok);
  if (!ok) {
    std::printf("  encode failed: %s\n", err.c_str());
    return;
  }
  // VPIPE_MING_TEST_DUMP=<dir>: this encoder's taps and routing as raw
  // f32 / i32, for analysis next to the goldens.
  if (const char* dd = std::getenv("VPIPE_MING_TEST_DUMP")) {
    const fs::path d = fs::path(dd) / key;
    fs::create_directories(d);
    auto put = [&](const char* n, const void* p, std::size_t bytes) {
      std::ofstream o(d / n, std::ios::binary);
      o.write(static_cast<const char*>(p), (std::streamsize)bytes);
    };
    const char* tn[3] = {"hs5.f32", "hs12.f32", "hs20.f32"};
    for (std::size_t t = 0; t < dbg.taps.size() && t < 3; ++t) {
      put(tn[t], dbg.taps[t].data(), dbg.taps[t].size() * 4);
    }
    std::vector<std::int32_t> ex;
    for (const auto& v : dbg.experts) {
      ex.insert(ex.end(), v.begin(), v.end());
    }
    put("experts.i32", ex.data(), ex.size() * 4);
    put("cap_feats_2.f32", dbg.cap_feats_2.data(), dbg.cap_feats_2.size() * 4);
    put("cap_feats.f32", dbg.cap_feats.data(), dbg.cap_feats.size() * 4);
  }
  const auto ids = g.i32("input_ids");
  const int L = (int)ids.size();
  const int N = res.text_rows;
  const int H = 2048;
  std::printf("  %s: N %d, L %d, encode (with debug readback) %.2f s\n",
              key, N, L, secs);
  CHECK(dbg.ids == ids);
  CHECK(res.rows == 256 + N && res.dim == 3840 && res.query_rows == 256);
  if (dbg.ids != ids) {
    std::printf("  input ids DIFFER from the golden; nothing below is "
                "comparable\n");
    return;
  }
  const int q0 = N + 1, q1 = N + 257;
  const char* tapn[3] = {"hs5", "hs12", "hs20"};
  for (int t = 0; t < 3 && t < (int)dbg.taps.size(); ++t) {
    const auto gold = g.f32(tapn[t]);
    const auto& mine = dbg.taps[(std::size_t)t];
    report(rep, tapn[t], mine, gold, H);
    char kt[32], kq[32];
    std::snprintf(kt, sizeof(kt), "%s_text_rows", tapn[t]);
    std::snprintf(kq, sizeof(kq), "%s_query_rows", tapn[t]);
    report_info(kt, rel_l2(rows_of(mine, H, 0, N), rows_of(gold, H, 0, N)),
                bars(rep, kt));
    report_info(kq, rel_l2(rows_of(mine, H, q0, q1),
                           rows_of(gold, H, q0, q1)), bars(rep, kq));
  }
  const auto gidx = g.i32("moe_topk_idx");
  double exact = 0.0;
  const auto per = routing_overlap(dbg.experts, gidx, L, 8, &exact);
  double mean = 0.0;
  std::printf("  routing overlap with fp32, per MoE layer:");
  for (std::size_t l = 0; l < per.size(); ++l) {
    std::printf("%s%.3f", l % 10 == 0 ? "\n    " : " ", per[l]);
    mean += per[l];
  }
  mean /= per.empty() ? 1.0 : (double)per.size();
  const double rb = routing_bar(rep, "bf16_f32router");
  const double eb = routing_bar(rep, "bf16_f32router", "exact_rows");
  std::printf("\n  routing overlap mean %.4f (bar %.4f, CUDA %.4f); rows "
              "with the identical 8-set %.4f (bar %.4f)\n",
              mean, rb, routing_bar(rep, "bf16"), exact, eb);
  CHECK(per.size() == 19);
  if (rb >= 0.0) { CHECK(mean >= rb - 0.02); }
  if (eb >= 0.0) { CHECK(exact >= eb - 0.05); }

  report(rep, "cap_feats_2", dbg.cap_feats_2, g.f32("cap_feats_2"), 3840);
  report(rep, "connector_in", dbg.connector_in, g.f32("connector_in"), 1536);
  report(rep, "connector_out", dbg.connector_out, g.f32("connector_out"),
         1536);
  report(rep, "cap_feats", dbg.cap_feats, g.f32("cap_feats"), 2560);

  // The packed contract: stream A in columns [0, 2560) of the first 256
  // rows and zero beyond; stream B after.
  bool zeros = true;
  for (int r = 0; r < 256 && zeros; ++r) {
    for (int c = 2560; c < 3840; ++c) {
      if (res.cond[(std::size_t)r * 3840 + c] != 0) { zeros = false; break; }
    }
  }
  CHECK(zeros);
  std::vector<float> a(256 * 2560), b((std::size_t)N * 3840);
  for (int r = 0; r < 256; ++r) {
    for (int c = 0; c < 2560; ++c) {
      a[(std::size_t)r * 2560 + c] =
          ming::bf16_to_f32(res.cond[(std::size_t)r * 3840 + c]);
    }
  }
  for (std::size_t k = 0; k < b.size(); ++k) {
    b[k] = ming::bf16_to_f32(res.cond[(std::size_t)256 * 3840 + k]);
  }
  CHECK(a == dbg.cap_feats);
  CHECK(b == dbg.cap_feats_2);
  if (keep != nullptr) { *keep = std::move(res); }
}

// ---- the MLLM, teacher-forced --------------------------------------------
//
// Every layer fed the fp32 golden's own input (hs_all[i]) and compared
// with its output (hs_all[i+1]; hs20_prenorm for the last), so each
// layer is judged alone. Twice: with the routing FORCED to the fp32
// golden's (the arithmetic alone; bar report_tf bf16_forced) and with
// this encoder's own f32 router (bar bf16_f32router). `delta` is the
// error of the layer's UPDATE (out - in): the residual otherwise
// dominates and hides a wrong layer.

double
tf_bar(const FlexData& tf, int layer, const char* mode, const char* key)
{
  try {
    return tf.as_object().at("layers").as_object()
        .at(std::to_string(layer)).as_object().at(mode).as_object()
        .at(key).as_real(-1.0);
  } catch (...) {
    return -1.0;
  }
}

void
check_teacher_forced(ming::MingEncoder& enc, const fs::path& croot,
                     const char* key)
{
  Golden g;
  if (!g.open(croot / key / "fp32") || !g.has("hs_all")) {
    std::printf("  SKIPPED %s: no hs_all golden\n", key);
    return;
  }
  const FlexData tf = read_json(croot / key / "report_tf.json");
  const std::string prompt = read_text(croot / key / "prompt.txt");
  const auto hs_all = g.f32("hs_all");
  const auto pre = g.f32("hs20_prenorm");
  const auto idx = g.i32("moe_topk_idx");
  const auto wts = g.f32("moe_topk_w");
  const int L = (int)g.i32("input_ids").size();
  const std::size_t LH = (std::size_t)L * 2048;
  for (int forced = 1; forced >= 0; --forced) {
    ming::MingEncoder::Debug dbg;
    dbg.tf_hs = hs_all.data();
    if (forced != 0) {
      dbg.tf_idx = idx.data();
      dbg.tf_w = wts.data();
    }
    ming::MingEncoder::Result res;
    std::string err;
    const bool ok = enc.encode_debug(prompt, &res, &dbg, &err);
    CHECK(ok);
    CHECK(dbg.layer_out.size() == 20);
    if (!ok || dbg.layer_out.size() != 20) {
      std::printf("  teacher-forced run failed: %s\n", err.c_str());
      return;
    }
    const char* mode = forced != 0 ? "bf16_forced" : "bf16_f32router";
    std::printf("  %s, routing %s; per layer: rel-L2 (bar) | delta rel-L2 "
                "(bar)\n", key, forced != 0 ? "FORCED to fp32" : "own f32");
    int above = 0;
    for (int i = 0; i < 20; ++i) {
      const float* in = hs_all.data() + (std::size_t)i * LH;
      const float* gold = i < 19 ? hs_all.data() + (std::size_t)(i + 1) * LH
                                 : pre.data();
      const auto& out = dbg.layer_out[(std::size_t)i];
      double num = 0, den = 0, dnum = 0, dden = 0;
      for (std::size_t k = 0; k < LH; ++k) {
        const double e = (double)out[k] - gold[k];
        num += e * e;
        den += (double)gold[k] * gold[k];
        const double u = (double)gold[k] - in[k];
        dnum += e * e;
        dden += u * u;
      }
      const double r = std::sqrt(num / den), dr = std::sqrt(dnum / dden);
      // The text rows alone, where the stream-B taps are read.
      double tn = 0, td = 0;
      for (std::size_t k = 0; k < (std::size_t)res.text_rows * 2048; ++k) {
        const double e = (double)out[k] - gold[k];
        const double u = (double)gold[k] - in[k];
        tn += e * e;
        td += u * u;
      }
      const double tr = std::sqrt(tn / td);
      const double rb = tf_bar(tf, i, mode, "rel_l2");
      const double db = tf_bar(tf, i, mode, "delta_rel_l2");
      const bool bad = (rb >= 0 && r > 1.25 * rb + 1e-4) ||
                       (db >= 0 && dr > 1.25 * db + 1e-4);
      above += bad;
      std::printf("    L%-2d %.5f (%.5f) | %.5f (%.5f) | text delta %.5f%s\n",
                  i, r, rb, dr, db, tr, bad ? "  ABOVE" : "");
    }
    // Held per layer with 25% grace: these are single-layer bf16 errors
    // of a few 1e-3, where two correct bf16 implementations rounding at
    // different points differ by that much.
    CHECK(above == 0);
    if (forced != 0) { continue; }
    // ROUTING, teacher-forced: every layer sees the fp32 golden's own
    // input, so a pick that differs from fp32 can only be a near-tie the
    // bf16 normed row tipped. Each disagreeing (layer, row) is looked up
    // in the fp32 router margins (8th - 9th score; 4th - 5th group).
    const auto mar = g.f32("moe_margin");
    double ov = 0.0;
    long flips = 0, rows_all = 0, tiny = 0;
    double worst = 0.0;
    std::vector<double> fm;
    for (std::size_t l = 0; l < dbg.experts.size(); ++l) {
      for (int r = 0; r < L; ++r) {
        std::set<int> a, b;
        for (int k = 0; k < 8; ++k) {
          a.insert(dbg.experts[l][(std::size_t)r * 8 + k]);
          b.insert(idx[(l * (std::size_t)L + r) * 8 + k]);
        }
        int hit = 0;
        for (int e : a) { hit += (int)b.count(e); }
        ov += hit / 8.0;
        ++rows_all;
        if (hit == 8) { continue; }
        ++flips;
        const std::size_t mi = (l * (std::size_t)L + r) * 2;
        const double m = std::min((double)mar[mi], (double)mar[mi + 1]);
        fm.push_back(m);
        worst = std::max(worst, m);
        tiny += (m < 4e-3);
      }
    }
    std::printf("  %s routing, teacher-forced: overlap %.4f, rows with a "
                "flip %.4f; their fp32 margin median %.2e, max %.2e, "
                "%.1f%% below 4e-3\n", key, ov / std::max(1L, rows_all),
                (double)flips / std::max(1L, rows_all), quantile(fm, 0.5),
                worst, 100.0 * tiny / std::max(1L, flips));
    // A flip on a decisive pick would be a routing BUG, not rounding.
    CHECK(worst < 1e-2);
  }
}

// ---- the connector, teacher-forced ---------------------------------------

void
test_connector_tf(MetalCompute& mc, const ming::Checkpoint& ck,
                  const fs::path& croot)
{
  CASE("connector teacher-forced: fp32 golden input -> its output");
  Golden g;
  if (!g.open(croot / "p1" / "fp32")) {
    std::printf("  SKIPPED: no conditioning/p1/fp32 golden\n");
    return;
  }
  const FlexData tf = read_json(croot / "p1" / "report_tf.json");
  double bar = -1.0, bar_med = -1.0;
  try {
    const FlexData c = tf.as_object().at("heads").as_object().at("bf16")
                           .as_object().at("connector");
    bar = c.as_object().at("rel_l2").as_real(-1.0);
    bar_med = c.as_object().at("row_median").as_real(-1.0);
  } catch (...) {
  }
  ming::LlmOps ops;
  std::string err;
  CHECK(ops.init(&mc, &err));
  ming::ConnectorConfig cc;
  CHECK(ming::read_connector_config(ck.connector_dir, &cc, &err));
  auto ws = vpipe::genai::WeightSet::open(ck.connector_dir, nullptr);
  auto conn = ming::MingConnector::load(ws, &ops, cc, &err);
  CHECK(conn != nullptr);
  if (!conn) {
    std::printf("  connector did not load: %s\n", err.c_str());
    return;
  }
  const auto in = g.f32("connector_in");
  const auto want = g.f32("connector_out");
  const int S = 256;
  // The reference feeds the connector bf16 (proj_in's output).
  auto inb = ops.upload_bf16(in.data(), in.size());
  auto outb = ops.alloc((std::size_t)S * cc.hidden * 2);
  auto stream = mc.make_command_stream();
  auto enc = stream.begin_compute();
  CHECK(conn->encode(enc, inb, outb, S, &err));
  enc.end();
  CHECK(stream.commit().wait_ok(&err));
  std::vector<float> got((std::size_t)S * cc.hidden);
  const auto* p = static_cast<const std::uint16_t*>(outb.contents());
  for (std::size_t k = 0; k < got.size(); ++k) {
    got[k] = ming::bf16_to_f32(p[k]);
  }
  const double r = rel_l2(got, want);
  const double med = quantile(per_row(got, want, cc.hidden), 0.5);
  std::printf("  connector all-rows %.4f (bar %.4f) | row median %.4f "
              "(bar %.4f)   [bars: the reference's bf16 connector on the "
              "same input]\n", r, bar, med, bar_med);
  if (bar >= 0.0) { CHECK(r <= 1.10 * bar); }
  if (bar_med >= 0.0) { CHECK(med <= 1.10 * bar_med); }
  std::printf("  connector holds %.2f GB (bf16; %.2f GB f32 on disk)\n",
              gb(conn->weight_bytes()), gb(2 * conn->weight_bytes()));
  // The set OUTLIVES the model here, as it does under the model manager
  // (which keeps an unborrowed set, parked). Its derived products --
  // never parkable -- must go with the model, or a released encoder
  // keeps its GBs.
  const std::size_t held = ws->stats().bytes;
  conn.reset();
  const std::size_t after = ws->stats().bytes;
  std::printf("  weight set: %.2f GB cached with the connector, %.3f GB "
              "after it is destroyed\n", gb(held), gb(after));
  CHECK(held > 2000000000u);
  CHECK(after < (16u << 20));
}

}  // namespace

int
main()
{
  // Line-buffered: a run is minutes long, and a log that only appears at
  // exit cannot tell a slow load from a hang.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  const char* model = std::getenv("VPIPE_MING_TEST_MODEL_PATH");
  const char* groot = std::getenv("VPIPE_MING_GOLDEN");
  if (model == nullptr) {
    std::printf("SKIPPED: VPIPE_MING_TEST_MODEL_PATH is unset -- the "
                "encoder was NOT tested.\n");
    return 0;
  }
  const fs::path mroot(model);
  if (part("tok")) { test_tokenizer(mroot, groot); }

  MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no usable Metal device. The GPU half was NOT "
                "checked.\n");
    return mingtest::summary("ming-encoder");
  }
  CHECK(ming::register_llm_kernels(&mc));
  const ming::Checkpoint ck = ming::read_checkpoint(model);
  CHECK(ck.valid);
  if (!ck.valid) {
    std::printf("  not a Ming-Image checkpoint: %s\n", ck.why.c_str());
    return mingtest::summary("ming-encoder");
  }

  CASE("sizes: what a load holds, from the loader's own arithmetic");
  const std::size_t res_b = ming::MingEncoder::resident_weight_bytes(ck);
  const std::size_t floor_b = ming::MingEncoder::streaming_floor_bytes(ck);
  const std::size_t conn_b = ming::MingEncoder::connector_weight_bytes(ck);
  std::printf("  MLLM resident %.2f GB, streaming floor %.2f GB; connector "
              "+ heads %.2f GB\n", gb(res_b), gb(floor_b), gb(conn_b));
  CHECK(res_b > 30e9 && res_b < 33e9);
  CHECK(floor_b > 2e9 && floor_b < 6e9);

  const fs::path croot = groot != nullptr
                             ? fs::path(groot) / "conditioning"
                             : fs::path();
  if (groot != nullptr && part("conn")) { test_connector_tf(mc, ck, croot); }

  // ---- resident ------------------------------------------------------
  ming::MingEncoder::Result p1_resident;
  if (part("resident") || part("tf") || part("stream")) {
    CASE("resident encoder: P1 and the cabin JSON against the fp32 "
         "goldens");
    std::string err;
    ming::MingEncoder::Options opt;
    const auto t0 = std::chrono::steady_clock::now();
    auto enc = ming::MingEncoder::load(ck, &mc, nullptr, opt, &err);
    const double load_s = seconds_since(t0);
    CHECK(enc != nullptr);
    if (!enc) {
      std::printf("  load failed: %s\n", err.c_str());
      return mingtest::summary("ming-encoder");
    }
    double now = 0, peak = 0;
    footprint(&now, &peak);
    std::printf("  loaded in %.1f s; holds %.2f GB; footprint %.2f GB "
                "(peak %.2f)\n", load_s, gb(enc->resident_bytes()), now,
                peak);
    CHECK(enc->resident_bytes() >= res_b);
    if (groot != nullptr) {
      check_prompt(*enc, croot, "p1", &p1_resident);
      if (part("resident")) { check_prompt(*enc, croot, "cabin", nullptr); }
      if (part("tf")) {
        CASE("MLLM teacher-forced, layer by layer");
        check_teacher_forced(*enc, croot, "p1");
        check_teacher_forced(*enc, croot, "cabin");
      }
    } else {
      std::printf("  SKIPPED golden comparison: VPIPE_MING_GOLDEN unset\n");
      CHECK(enc->encode("A small red cube on a white table", &p1_resident,
                        &err));
    }
    // Timings without the debug readback, warm.
    const std::string cabin =
        read_text(mroot.parent_path().parent_path().parent_path() /
                  "ming-image" / "Ming-Image" / "assets" /
                  "t2i_four_seasons_cabin_prompt.json");
    for (const char* which : {"P1", "P1", "cabin", "cabin"}) {
      if (!part("resident")) { break; }
      const std::string p = std::strcmp(which, "P1") == 0
                                ? std::string("A small red cube on a white "
                                              "table")
                                : (croot.empty()
                                       ? cabin
                                       : read_text(croot / "cabin" /
                                                   "prompt.txt"));
      if (p.empty()) { continue; }
      ming::MingEncoder::Result r;
      const auto t1 = std::chrono::steady_clock::now();
      CHECK(enc->encode(p, &r, &err));
      std::printf("  encode %-5s (N %4d): %.3f s\n", which, r.text_rows,
                  seconds_since(t1));
      // Where it goes: a profiled run (it syncs after every command
      // buffer, so its total runs a little over the plain one).
      ming::MingEncoder::Debug d;
      d.profile = true;
      d.tap_layers.clear();
      CHECK(enc->encode_debug(p, &r, &d, &err));
      std::printf("    profile: GPU attention+shared %.0f ms, GPU routed "
                  "experts %.0f ms, CPU routing %.0f ms, CPU embed %.0f ms, "
                  "GPU heads+connector %.0f ms\n", d.gpu_attn_ms,
                  d.gpu_moe_ms, d.cpu_route_ms, d.cpu_embed_ms, d.heads_ms);
    }
    footprint(&now, &peak);
    std::printf("  footprint after encodes %.2f GB (peak %.2f)\n", now, peak);
  }

  // ---- streamed --------------------------------------------------------
  if (part("stream")) {
    CASE("streamed encoder: bit-identical to resident, and its cost");
    std::string err;
    ming::MingEncoder::Options opt;
    opt.stream_layers = true;
    const auto t0 = std::chrono::steady_clock::now();
    auto enc = ming::MingEncoder::load(ck, &mc, nullptr, opt, &err);
    const double load_s = seconds_since(t0);
    CHECK(enc != nullptr);
    if (!enc) {
      std::printf("  load failed: %s\n", err.c_str());
      return mingtest::summary("ming-encoder");
    }
    double now = 0, peak = 0;
    footprint(&now, &peak);
    std::printf("  loaded in %.1f s; holds %.2f GB (MLLM floor %.2f + "
                "connector/heads %.2f); footprint %.2f GB\n", load_s,
                gb(enc->resident_bytes()), gb(floor_b), gb(conn_b), now);
    CHECK(enc->resident_bytes() <= floor_b + conn_b + (64u << 20));
    ming::MingEncoder::Result r;
    const auto t1 = std::chrono::steady_clock::now();
    CHECK(enc->encode("A small red cube on a white table", &r, &err));
    std::printf("  encode P1 streamed: %.2f s (reads the 19 MoE layers, "
                "~30 GB)\n", seconds_since(t1));
    footprint(&now, &peak);
    std::printf("  footprint after the streamed encode %.2f GB\n", now);
    CHECK(!p1_resident.cond.empty());
    const bool same = r.cond == p1_resident.cond;
    std::printf("  streamed == resident: %s\n", same ? "bit-identical" :
                                                      "DIFFERENT");
    CHECK(same);
  }
  return mingtest::summary("ming-encoder");
}
