#include "ming-layout.h"

#include <cstddef>

namespace ming {

namespace {

int
round_up_(int n, int mult)
{
  const int r = n % mult;
  return r == 0 ? n : n + (mult - r);
}

}  // namespace

bool
build_layout(int grid_h, int grid_w, int query_rows, int text_rows,
             int axis0_len, Layout* out, std::string* err)
{
  auto fail = [&](const char* why) {
    if (err != nullptr) { *err = why; }
    return false;
  };
  if (out == nullptr) { return fail("no output layout"); }
  if (grid_h <= 0 || grid_w <= 0) {
    return fail("patch grid must be positive on both axes");
  }
  if (query_rows < 0 || text_rows < 0 || query_rows + text_rows <= 0) {
    return fail("caption must have at least one row");
  }
  Layout L;
  L.grid_h = grid_h;
  L.grid_w = grid_w;
  L.img_tokens = grid_h * grid_w;
  L.query_rows = query_rows;
  L.text_rows = text_rows;
  L.cap_len = query_rows + text_rows;
  L.cap_total = round_up_(L.cap_len, kSeqMultiple);
  L.joint_len = L.img_tokens + L.cap_len;
  // The image sits AFTER the padded caption run on axis 0, and the
  // rotary table must reach it.
  if (axis0_len > 0 && L.cap_total + 1 >= axis0_len) {
    return fail("caption too long for the DiT's axis-0 rotary table");
  }

  L.pos_t.assign((std::size_t)L.joint_len, 0);
  L.pos_h.assign((std::size_t)L.joint_len, 0);
  L.pos_w.assign((std::size_t)L.joint_len, 0);
  const std::int32_t img_t = (std::int32_t)(L.cap_total + 1);
  for (int h = 0; h < grid_h; ++h) {
    for (int w = 0; w < grid_w; ++w) {
      const std::size_t r = (std::size_t)(h * grid_w + w);
      L.pos_t[r] = img_t;
      L.pos_h[r] = (std::int32_t)h;
      L.pos_w[r] = (std::int32_t)w;
    }
  }
  for (int i = 0; i < L.cap_len; ++i) {
    L.pos_t[(std::size_t)(L.img_tokens + i)] = (std::int32_t)(i + 1);
  }
  *out = std::move(L);
  return true;
}

void
pack_latents(const float* z, int channels, int lat_h, int lat_w, int patch,
             float* rows)
{
  if (z == nullptr || rows == nullptr || patch <= 0) { return; }
  const int gh = lat_h / patch, gw = lat_w / patch;
  const int rw = patch * patch * channels;
  for (int ht = 0; ht < gh; ++ht) {
    for (int wt = 0; wt < gw; ++wt) {
      float* dst = rows + (std::size_t)(ht * gw + wt) * rw;
      for (int ph = 0; ph < patch; ++ph) {
        for (int pw = 0; pw < patch; ++pw) {
          const int h = ht * patch + ph, w = wt * patch + pw;
          float* d = dst + (std::size_t)(ph * patch + pw) * channels;
          for (int c = 0; c < channels; ++c) {
            d[c] = z[((std::size_t)c * lat_h + h) * lat_w + w];
          }
        }
      }
    }
  }
}

void
unpack_latents(const float* rows, int channels, int lat_h, int lat_w,
               int patch, float* z)
{
  if (z == nullptr || rows == nullptr || patch <= 0) { return; }
  const int gh = lat_h / patch, gw = lat_w / patch;
  const int rw = patch * patch * channels;
  for (int ht = 0; ht < gh; ++ht) {
    for (int wt = 0; wt < gw; ++wt) {
      const float* src = rows + (std::size_t)(ht * gw + wt) * rw;
      for (int ph = 0; ph < patch; ++ph) {
        for (int pw = 0; pw < patch; ++pw) {
          const int h = ht * patch + ph, w = wt * patch + pw;
          const float* s = src + (std::size_t)(ph * patch + pw) * channels;
          for (int c = 0; c < channels; ++c) {
            z[((std::size_t)c * lat_h + h) * lat_w + w] = s[c];
          }
        }
      }
    }
  }
}

}  // namespace ming
