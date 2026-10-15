#ifndef VPIPE_MING_TEST_H
#define VPIPE_MING_TEST_H

// A three-macro harness. A plugin cannot reach the host's minitest --
// it is not installed and would not link -- so these tests are ordinary
// main()s, and this is the least that keeps them readable.
//
// CHECK RECORDS AND CONTINUES; it does not abort. That is the same
// behaviour the host's ASSERT_TRUE has, and it is worth stating because
// it is the one that surprises: everything after a failed check runs
// against the state that just failed, so a test with dependent steps
// must guard them itself.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace mingtest {

inline int g_failed = 0;
inline int g_checks = 0;
inline const char* g_case = "";

inline void
check(bool ok, const char* expr, const char* file, int line)
{
  ++g_checks;
  if (ok) { return; }
  ++g_failed;
  std::printf("  FAIL %s:%d [%s] %s\n", file, line, g_case, expr);
}

inline void
begin(const char* name)
{
  g_case = name;
  std::printf("- %s\n", name);
}

inline int
summary(const char* suite)
{
  std::printf("%s: %d checks, %d failed\n", suite, g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}

inline bool
near(double a, double b, double tol)
{
  return std::fabs(a - b) <= tol;
}

// Relative L2 of `a` against `b`, the measure every numerical bar in
// this tree is stated in.
inline double
rel_l2(const std::vector<float>& a, const std::vector<float>& b)
{
  if (a.size() != b.size() || a.empty()) { return 1.0; }
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace mingtest

#define CHECK(x) ::mingtest::check((x), #x, __FILE__, __LINE__)
#define CASE(name) ::mingtest::begin(name)

#endif
