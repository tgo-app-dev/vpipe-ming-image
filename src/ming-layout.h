#ifndef VPIPE_MING_LAYOUT_H
#define VPIPE_MING_LAYOUT_H

// The joint image/caption sequence Ming-Image's DiT runs over, as pure
// integer arithmetic -- no GPU, no weights.
//
// The reference's sequence is Z-Image's:
//
//     [ image | image pad | caption | caption pad ]      pads to x32
//
// with one difference that changes what a port must build. Z-Image
// fills the pads with LEARNED tokens that attend like any other row;
// this checkpoint is "zero_masked": the pad rows are zero and are
// excluded as attention KEYS everywhere, and a pad QUERY row's output is
// never read (the final layer takes the first img_tokens rows). So the
// pads contribute nothing to any real row, and this layout DROPS them:
//
//     [ image (img_tokens) | caption (cap_len) ]
//
// which is exact -- with one condition that every test must pin: the
// POSITIONS still count the pads. The image's axis-0 position is
// roundup32(cap_len) + 1, not cap_len + 1, because the reference gave
// the caption a padded position run before it put the image after it.
// (docs/MING-IMAGE.md 2.3.)
//
// The caption is the conditioning contract's two streams in order:
// `query_rows` connector rows (which the DiT's cap_embedder maps to its
// width) and then `text_rows` MLLM rows (already DiT-wide).

#include <cstdint>
#include <string>
#include <vector>

namespace ming {

inline constexpr int kSeqMultiple = 32;

struct Layout {
  int grid_h = 0;       // patch rows    (latent H / patch)
  int grid_w = 0;       // patch columns (latent W / patch)
  int img_tokens = 0;   // grid_h * grid_w
  int query_rows = 0;   // stream A rows (through cap_embedder)
  int text_rows = 0;    // stream B rows (pre-embedded)
  int cap_len = 0;      // query_rows + text_rows: the REAL caption rows
  int cap_total = 0;    // roundup32(cap_len): positions only, no rows
  int joint_len = 0;    // img_tokens + cap_len: the rows actually run

  // Rotary coordinates per joint row, in the order the rows are laid
  // out: image rows (cap_total + 1, h, w), then caption row i (1 + i,
  // 0, 0).
  std::vector<std::int32_t> pos_t, pos_h, pos_w;

  int img_off() const { return 0; }
  int cap_off() const { return img_tokens; }
};

// `grid_h`/`grid_w` are PATCH counts. `axis0_len` is the checkpoint's
// axes_lens[0] (20480): the image's axis-0 position must stay inside the
// rotary table. Returns false with a reason rather than throwing.
bool build_layout(int grid_h, int grid_w, int query_rows, int text_rows,
                  int axis0_len, Layout* out, std::string* err);

// Patch packing, the reference's (C, F, H, W) -> (H_t, W_t, pH, pW, C)
// flatten: the CHANNEL runs fastest within a row. `rows` is
// [(lat_h/patch) * (lat_w/patch), patch*patch*channels].
void pack_latents(const float* z, int channels, int lat_h, int lat_w,
                  int patch, float* rows);
void unpack_latents(const float* rows, int channels, int lat_h, int lat_w,
                    int patch, float* z);

}  // namespace ming

#endif
