#include "base.h"
#include <cmath>

int32 g_verbose_level = 0;

bool ApproxEqual(BaseFloat a, BaseFloat b, BaseFloat tol) {
  if (std::abs(a - b) > a * tol)
    return false;
  return true;
}
