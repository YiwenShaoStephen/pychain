#ifndef BASE_H_
#define BASE_H_

#include<cmath>

#define CU1DBLOCK   256

#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209290e-7f
#endif

static const float kMinLogDiffFloat = std::log(FLT_EPSILON);  // negative!

inline float LogAdd(float x, float y) {
   float diff;

   if (x < y) {
     diff = x - y;
     x = y;
   } else {
     diff = y - x;
   }
   // diff is negative.  x is now the larger one.

   if (diff >= kMinLogDiffFloat) {
     float res;
     res = x + std::log1p(std::exp(diff));
     return res;
   } else {
     return x;  // return the larger one.
   }
}

extern int g_verbose_level;

inline int GetVerboseLevel() {
  return g_verbose_level;
}

inline void SetVerboseLevel(int level) {
  g_verbose_level = level;
}

/** Number of blocks in which the task of size 'size' is splitted **/
inline int n_blocks(int size, int block_size) {
  return size / block_size + ((size % block_size == 0)? 0 : 1);
}

bool ApproxEqual(float a, float b, float tol = 0.01);

#endif
