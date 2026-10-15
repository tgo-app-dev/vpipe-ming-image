#ifndef VPIPE_MING_TOKENIZE_H
#define VPIPE_MING_TOKENIZE_H

// The Ling (BailingMM2) tokenizer, id-exact against HF `tokenizers` on
// mllm/tokenizer.json, and the text-to-image prompt template.
//
// WHY NOT THE HOST'S genai::Tokenizer. It is close but not exact here,
// in three ways that each change ids:
//
//   * it does not split added tokens out of the text ("callers prepend
//     the appropriate id explicitly"), while the template is a string
//     with five special tokens in it -- and a prompt that happens to
//     contain one is split by the reference too;
//   * it does not apply the declared NFC normalizer;
//   * its pre-tokenizer is a hand-coded scanner with ASCII-only
//     whitespace and range-table letters, so U+3000 / U+00A0 and any
//     letter outside its ranges are classified differently from the
//     reference's Oniguruma (\s and \p{L} are Unicode-wide there).
//
// So this owns the whole pipeline -- added-token split, NFC,
// pre-tokenizer, byte-level BPE -- and every stage is the reference's
// own definition. The Ling pre-tokenizer regex
//
//   '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}
//   | ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
//
// is matched by a scanner that walks its seven alternatives in order
// (ming-tokenize.cc says why each possessive quantifier changes
// nothing), with \p{L} / \p{N} from the OS's Unicode tables and \s as
// Oniguruma defines it.

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ming {

class LingTokenizer {
public:
  // Parse mllm/tokenizer.json. Null with `err` on failure.
  static std::unique_ptr<LingTokenizer>
  load(const std::string& tokenizer_json, std::string* err);

  // HF `encode(text, add_special_tokens=False)`: added tokens are split
  // out first (leftmost-longest), every other span is NFC-normalized,
  // pre-tokenized and BPE-merged. No BOS/EOS; nothing is truncated.
  std::vector<std::int32_t> encode(std::string_view text) const;

  // An added token's id, or a plain vocab entry's; -1 when unknown.
  std::int32_t id_of(std::string_view token) const;

  std::size_t vocab_size() const noexcept { return _vocab.size(); }

  // Exposed for the tests: the pre-tokenizer's pieces of one
  // (already-normalized) span.
  std::vector<std::string> pre_tokenize(std::string_view text) const;

private:
  LingTokenizer() = default;
  void bpe_(std::string_view piece, std::vector<std::int32_t>* out) const;

  std::unordered_map<std::string, std::int32_t> _vocab;
  // (left id << 32 | right id) -> (rank << 32 | merged id)
  std::unordered_map<std::uint64_t, std::uint64_t> _merges;
  // Added tokens, longest first within each leading byte.
  std::vector<std::pair<std::string, std::int32_t>> _added[256];
  // GPT-2 byte-level alphabet: byte -> its single-char token id.
  std::int32_t _byte_id[256] = {};
};

// The text-to-image template (docs/MING-IMAGE.md 1.2): the processor's
// own apply_chat_template, not the jinja one in tokenizer_config.json.
std::string ming_t2i_template(std::string_view prompt);

}  // namespace ming

#endif
