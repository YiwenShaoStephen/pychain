#include "base.h"
#include <cmath>

int g_verbose_level = 0;

bool ApproxEqual(float a, float b, float tol) {
  if (std::abs(a - b) < a * tol)
    return true;
  return false;
}
