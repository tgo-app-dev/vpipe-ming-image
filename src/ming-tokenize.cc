#include "ming-tokenize.h"

#include "common/flex-data.h"

#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace ming {

namespace {

// ---- Unicode classes ------------------------------------------------
//
// The reference regex runs in Oniguruma (the `tokenizers` default
// backend), whose \p{L} / \p{N} are Unicode General Categories L* / N*.
// CoreFoundation states its predefined sets in those terms:
//   Letter       = L* + M*     NonBase = M*     AlphaNumeric = L*+M*+N*
// so L* = Letter - NonBase and N* = AlphaNumeric - Letter, exactly, at
// whatever Unicode version the OS carries.
struct UniSets {
  CFCharacterSetRef letter = nullptr;
  CFCharacterSetRef nonbase = nullptr;
  CFCharacterSetRef alnum = nullptr;
  UniSets()
  {
    letter = CFCharacterSetGetPredefined(kCFCharacterSetLetter);
    nonbase = CFCharacterSetGetPredefined(kCFCharacterSetNonBase);
    alnum = CFCharacterSetGetPredefined(kCFCharacterSetAlphaNumeric);
  }
};

const UniSets&
uni()
{
  static const UniSets s;
  return s;
}

bool
is_letter(std::uint32_t c)
{
  if (c < 0x80) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  }
  // Tangut (Lo) is the one block CoreFoundation's letter set leaves out:
  // MEASURED against Python's unicodedata over every assigned codepoint,
  // these 6145 are the only disagreement on L* or N*.
  if ((c >= 0x17000 && c <= 0x187F7) || (c >= 0x18D00 && c <= 0x18D08)) {
    return true;
  }
  const UniSets& u = uni();
  return CFCharacterSetIsLongCharacterMember(u.letter, c) &&
         !CFCharacterSetIsLongCharacterMember(u.nonbase, c);
}

bool
is_number(std::uint32_t c)
{
  if (c < 0x80) { return c >= '0' && c <= '9'; }
  const UniSets& u = uni();
  return CFCharacterSetIsLongCharacterMember(u.alnum, c) &&
         !CFCharacterSetIsLongCharacterMember(u.letter, c);
}

// Oniguruma's Unicode \s: U+0009..000D, U+0085 and the Zs / Zl / Zp
// categories. Not the ASCII set -- an ideographic space (U+3000) in a
// Chinese prompt is whitespace to the reference.
bool
is_space(std::uint32_t c)
{
  if (c == ' ' || (c >= 0x09 && c <= 0x0D)) { return true; }
  if (c < 0x85) { return false; }
  return c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
         c == 0x202F || c == 0x205F || c == 0x3000;
}

bool
is_nl(std::uint32_t c)
{
  return c == '\r' || c == '\n';
}

// "Punctuation" in the regex's own sense: [^\s\p{L}\p{N}].
bool
is_other(std::uint32_t c)
{
  return !is_space(c) && !is_letter(c) && !is_number(c);
}

// UTF-8 decode, one codepoint. An invalid byte decodes as itself (as a
// Latin-1 codepoint) and consumes one byte, so the scanner always makes
// progress; NFC has already replaced invalid input where CoreFoundation
// could read it.
std::uint32_t
next_cp(std::string_view s, std::size_t i, std::size_t* len)
{
  const auto b0 = (unsigned char)s[i];
  if (b0 < 0x80) { *len = 1; return b0; }
  int n = 0;
  std::uint32_t cp = 0;
  if ((b0 & 0xE0) == 0xC0) { n = 2; cp = b0 & 0x1F; }
  else if ((b0 & 0xF0) == 0xE0) { n = 3; cp = b0 & 0x0F; }
  else if ((b0 & 0xF8) == 0xF0) { n = 4; cp = b0 & 0x07; }
  else { *len = 1; return b0; }
  if (i + (std::size_t)n > s.size()) { *len = 1; return b0; }
  for (int k = 1; k < n; ++k) {
    const auto b = (unsigned char)s[i + (std::size_t)k];
    if ((b & 0xC0) != 0x80) { *len = 1; return b0; }
    cp = (cp << 6) | (b & 0x3F);
  }
  *len = (std::size_t)n;
  return cp;
}

// NFC, the tokenizer.json normalizer. ASCII is NFC already, which is
// most prompts, so that skips CoreFoundation entirely.
std::string
nfc(std::string_view s)
{
  bool ascii = true;
  for (char c : s) {
    if ((unsigned char)c >= 0x80) { ascii = false; break; }
  }
  if (ascii) { return std::string(s); }
  CFStringRef str = CFStringCreateWithBytes(
      kCFAllocatorDefault, (const UInt8*)s.data(), (CFIndex)s.size(),
      kCFStringEncodingUTF8, false);
  if (str == nullptr) { return std::string(s); }   // not valid UTF-8
  CFMutableStringRef m =
      CFStringCreateMutableCopy(kCFAllocatorDefault, 0, str);
  CFRelease(str);
  if (m == nullptr) { return std::string(s); }
  CFStringNormalize(m, kCFStringNormalizationFormC);
  const CFIndex n = CFStringGetLength(m);
  CFIndex need = 0;
  CFStringGetBytes(m, CFRangeMake(0, n), kCFStringEncodingUTF8, 0, false,
                   nullptr, 0, &need);
  std::string out((std::size_t)need, '\0');
  CFStringGetBytes(m, CFRangeMake(0, n), kCFStringEncodingUTF8, 0, false,
                   (UInt8*)out.data(), need, nullptr);
  CFRelease(m);
  return out;
}

// GPT-2's bytes_to_unicode: the printable Latin-1 bytes map to
// themselves, the other 68 to U+0100 upwards, in byte order.
std::uint32_t
byte_char(int b)
{
  static std::uint32_t table[256];
  static const bool built = [] {
    int n = 0;
    for (int c = 0; c < 256; ++c) {
      const bool keep = (c >= 33 && c <= 126) || (c >= 161 && c <= 172) ||
                        (c >= 174 && c <= 255);
      table[c] = keep ? (std::uint32_t)c : (std::uint32_t)(256 + n++);
    }
    return true;
  }();
  (void)built;
  return table[b];
}

void
append_utf8(std::string* s, std::uint32_t cp)
{
  if (cp < 0x80) {
    s->push_back((char)cp);
  } else if (cp < 0x800) {
    s->push_back((char)(0xC0 | (cp >> 6)));
    s->push_back((char)(0x80 | (cp & 0x3F)));
  } else {
    s->push_back((char)(0xE0 | (cp >> 12)));
    s->push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    s->push_back((char)(0x80 | (cp & 0x3F)));
  }
}

std::uint64_t
pair_key(std::int32_t a, std::int32_t b)
{
  return ((std::uint64_t)(std::uint32_t)a << 32) | (std::uint32_t)b;
}

// A codepoint view of one span, so the alternatives below can look
// ahead by codepoints and report byte lengths.
struct Cps {
  std::vector<std::uint32_t> cp;
  std::vector<std::size_t>   off;   // byte offset of cp[i]; off[n] = end
};

Cps
decode(std::string_view s)
{
  Cps c;
  std::size_t i = 0;
  while (i < s.size()) {
    std::size_t l = 1;
    c.cp.push_back(next_cp(s, i, &l));
    c.off.push_back(i);
    i += l;
  }
  c.off.push_back(s.size());
  return c;
}

char
lower_ascii(std::uint32_t c)
{
  return (c >= 'A' && c <= 'Z') ? (char)(c + 32) : (char)c;
}

// One match of the Ling regex at codepoint `i`, as a codepoint count.
// Every codepoint is L, N, \s or "other", and each class starts some
// alternative, so a match always exists -- the HF Split can therefore
// never leave an unmatched gap, and this never returns 0.
std::size_t
match_at(const Cps& c, std::size_t i)
{
  const std::size_t n = c.cp.size();
  auto at = [&](std::size_t k) -> std::uint32_t {
    return k < n ? c.cp[k] : 0xFFFFFFFFu;
  };
  // 1: '(?i:[sdmt]|ll|ve|re)
  if (at(i) == '\'') {
    const char a = lower_ascii(at(i + 1));
    const char b = lower_ascii(at(i + 2));
    if (a == 's' || a == 'd' || a == 'm' || a == 't') { return 2; }
    if ((a == 'l' && b == 'l') || (a == 'v' && b == 'e') ||
        (a == 'r' && b == 'e')) {
      return 3;
    }
  }
  // 2: [^\r\n\p{L}\p{N}]?+\p{L}+. The possessive ?+ cannot change the
  // answer: it gives back a consumed prefix only for \p{L}+ to start
  // AT that prefix, which is not a letter by construction.
  {
    std::size_t s = i;
    if (i < n && !is_letter(c.cp[i])) {
      if (!is_nl(c.cp[i]) && !is_number(c.cp[i]) && i + 1 < n &&
          is_letter(c.cp[i + 1])) {
        s = i + 1;
      } else {
        s = n;   // no letter run here
      }
    }
    if (s < n && is_letter(c.cp[s])) {
      std::size_t e = s;
      while (e < n && is_letter(c.cp[e])) { ++e; }
      return e - i;
    }
  }
  // 3: \p{N} -- ONE digit; the vocabulary has no multi-digit tokens.
  if (i < n && is_number(c.cp[i])) { return 1; }
  // 4: ' ?[^\s\p{L}\p{N}]++[\r\n]*'. ++ is again inert: [\r\n]* can
  // match empty, so nothing ever asks the run to give a char back.
  {
    std::size_t s = i;
    if (at(i) == ' ' && i + 1 < n && is_other(c.cp[i + 1])) { s = i + 1; }
    if (s < n && is_other(c.cp[s])) {
      std::size_t e = s;
      while (e < n && is_other(c.cp[e])) { ++e; }
      while (e < n && is_nl(c.cp[e])) { ++e; }
      return e - i;
    }
  }
  // The whitespace run [i, j) the last three alternatives work on.
  std::size_t j = i;
  while (j < n && is_space(c.cp[j])) { ++j; }
  if (j > i) {
    // 5: \s*[\r\n] -- up to and including the run's LAST newline.
    for (std::size_t k = j; k > i; --k) {
      if (is_nl(c.cp[k - 1])) { return k - i; }
    }
    // 6: \s+(?!\S) -- the whole run at the end of the text, else all
    // but its last char (which stays to lead the next word).
    if (j == n) { return j - i; }
    if (j - i >= 2) { return j - 1 - i; }
    // 7: \s+
    return j - i;
  }
  return 1;   // unreachable: every class is covered above
}

}  // namespace

std::unique_ptr<LingTokenizer>
LingTokenizer::load(const std::string& path, std::string* err)
{
  auto fail = [err](std::string m) -> std::unique_ptr<LingTokenizer> {
    if (err != nullptr) { *err = std::move(m); }
    return nullptr;
  };
  std::ifstream in(path, std::ios::binary);
  if (!in) { return fail("cannot open " + path); }
  vpipe::FlexData root;
  try {
    root = vpipe::FlexData::from_json(in);
  } catch (const std::exception& e) {
    return fail(path + ": " + e.what());
  }
  if (!root.is_object()) { return fail(path + " is not a JSON object"); }
  auto tok = std::unique_ptr<LingTokenizer>(new LingTokenizer());
  try {
    // Every lookup below is a VALUE (FlexData copies deeply), so the
    // big subtrees are bound once rather than re-copied per access.
    const vpipe::FlexData model = root.as_object().at("model");
    const auto mo = model.as_object();
    if (std::string(mo.at("type").as_string()) != "BPE") {
      return fail(path + ": not a BPE model");
    }
    const vpipe::FlexData vocab = mo.at("vocab");
    const auto vo = vocab.as_object();
    tok->_vocab.reserve(vo.size() + 512);
    for (auto it = vo.begin(); it != vo.end(); ++it) {
      const auto e = *it;
      tok->_vocab.emplace(std::string(e.first),
                          (std::int32_t)e.second.as_int(-1));
    }
    const vpipe::FlexData merges = mo.at("merges");
    const auto ma = merges.as_array();
    tok->_merges.reserve(ma.size());
    std::int64_t rank = 0;
    for (auto it = ma.begin(); it != ma.end(); ++it, ++rank) {
      const vpipe::FlexData m = *it;
      std::string a, b;
      if (m.is_array() && m.as_array().size() == 2) {
        a = std::string(m.as_array()[0].as_string());
        b = std::string(m.as_array()[1].as_string());
      } else if (m.is_string()) {   // the older "a b" spelling
        const std::string s(m.as_string());
        const auto sp = s.find(' ');
        if (sp == std::string::npos) { continue; }
        a = s.substr(0, sp);
        b = s.substr(sp + 1);
      } else {
        continue;
      }
      const auto ia = tok->_vocab.find(a);
      const auto ib = tok->_vocab.find(b);
      const auto iab = tok->_vocab.find(a + b);
      if (ia == tok->_vocab.end() || ib == tok->_vocab.end() ||
          iab == tok->_vocab.end()) {
        return fail(path + ": merge '" + a + " " + b +
                    "' names a token missing from the vocab");
      }
      // First wins: a duplicated pair keeps its best rank.
      tok->_merges.emplace(pair_key(ia->second, ib->second),
                           ((std::uint64_t)rank << 32) |
                               (std::uint32_t)iab->second);
    }
    const vpipe::FlexData added = root.as_object().at("added_tokens");
    const auto aa = added.as_array();
    for (auto it = aa.begin(); it != aa.end(); ++it) {
      const vpipe::FlexData t = *it;
      const auto to = t.as_object();
      const std::string content(to.at("content").as_string());
      const auto id = (std::int32_t)to.at("id").as_int(-1);
      if (content.empty() || id < 0) { continue; }
      // Every added token in this file is special, un-normalized and
      // strips nothing; a file that says otherwise is not one this
      // splitter reproduces.
      if (to.at("lstrip").as_bool(false) || to.at("rstrip").as_bool(false) ||
          to.at("single_word").as_bool(false) ||
          to.at("normalized").as_bool(false)) {
        return fail(path + ": added token '" + content +
                    "' has strip/normalize flags this does not implement");
      }
      tok->_added[(unsigned char)content[0]].emplace_back(content, id);
      tok->_vocab.emplace(content, id);
    }
  } catch (const std::exception& e) {
    return fail(path + ": " + e.what());
  }
  for (auto& v : tok->_added) {
    std::stable_sort(v.begin(), v.end(), [](const auto& x, const auto& y) {
      return x.first.size() > y.first.size();
    });
  }
  for (int b = 0; b < 256; ++b) {
    std::string s;
    append_utf8(&s, byte_char(b));
    const auto it = tok->_vocab.find(s);
    if (it == tok->_vocab.end()) {
      return fail(path + ": the byte-level alphabet is incomplete");
    }
    tok->_byte_id[b] = it->second;
  }
  return tok;
}

std::int32_t
LingTokenizer::id_of(std::string_view token) const
{
  const auto it = _vocab.find(std::string(token));
  return it == _vocab.end() ? -1 : it->second;
}

std::vector<std::string>
LingTokenizer::pre_tokenize(std::string_view text) const
{
  std::vector<std::string> out;
  const Cps c = decode(text);
  std::size_t i = 0;
  while (i < c.cp.size()) {
    const std::size_t m = match_at(c, i);
    out.emplace_back(text.substr(c.off[i], c.off[i + m] - c.off[i]));
    i += m;
  }
  return out;
}

void
LingTokenizer::bpe_(std::string_view piece,
                    std::vector<std::int32_t>* out) const
{
  std::vector<std::int32_t> sym;
  sym.reserve(piece.size());
  for (char ch : piece) { sym.push_back(_byte_id[(unsigned char)ch]); }
  // Merge the lowest-ranked adjacent pair, leftmost on a tie, until no
  // pair has a merge -- the order HF's heap of (rank, position) yields.
  for (;;) {
    std::uint64_t best = std::numeric_limits<std::uint64_t>::max();
    std::size_t at = 0;
    for (std::size_t k = 0; k + 1 < sym.size(); ++k) {
      const auto it = _merges.find(pair_key(sym[k], sym[k + 1]));
      if (it != _merges.end() && it->second < best) {
        best = it->second;
        at = k;
      }
    }
    if (best == std::numeric_limits<std::uint64_t>::max()) { break; }
    sym[at] = (std::int32_t)(best & 0xFFFFFFFFu);
    sym.erase(sym.begin() + (std::ptrdiff_t)at + 1);
  }
  out->insert(out->end(), sym.begin(), sym.end());
}

std::vector<std::int32_t>
LingTokenizer::encode(std::string_view text) const
{
  std::vector<std::int32_t> ids;
  auto encode_span = [&](std::string_view span) {
    if (span.empty()) { return; }
    const std::string norm = nfc(span);
    for (const std::string& p : pre_tokenize(norm)) { bpe_(p, &ids); }
  };
  // Added tokens first, leftmost-longest, on the RAW text: they are
  // declared un-normalized, so NFC never sees them.
  std::size_t start = 0, i = 0;
  while (i < text.size()) {
    const auto& cands = _added[(unsigned char)text[i]];
    const std::pair<std::string, std::int32_t>* hit = nullptr;
    for (const auto& c : cands) {
      if (text.compare(i, c.first.size(), c.first) == 0) { hit = &c; break; }
    }
    if (hit == nullptr) { ++i; continue; }
    encode_span(text.substr(start, i - start));
    ids.push_back(hit->second);
    i += hit->first.size();
    start = i;
  }
  encode_span(text.substr(start));
  return ids;
}

std::string
ming_t2i_template(std::string_view prompt)
{
  // SYSTEM_PROMPT_LINGV2_FLASH_NOTHINK + eos, then the one HUMAN turn
  // and the ASSISTANT prefix ([R] processing_bailingmm2.py:57-61,
  // 351-415).
  std::string s =
      "<role>SYSTEM</role>\xe4\xbd\xa0\xe6\x98\xaf\xe4\xb8\x80\xe4\xb8\xaa"
      "\xe5\x8f\x8b\xe5\xa5\xbd\xe7\x9a\x84" "AI"
      "\xe5\x8a\xa9\xe6\x89\x8b\xe3\x80\x82\n\ndetailed thinking off"
      "<|role_end|><role>HUMAN</role>";
  s.append(prompt);
  s += "<|role_end|><role>ASSISTANT</role>";
  return s;
}

}  // namespace ming
