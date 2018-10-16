#ifndef BASE_H_
#define BASE_H_

#define HAVE_CUDA 0

#if HAVE_CUDA == 1
#define CPU_OR_CUDA CUDA
#else
#define CPU_OR_CUDA CPU
#endif
typedef int int32;
typedef long int64;
typedef float BaseFloat;

extern int32 g_verbose_level;

inline int32 GetVerboseLevel() {
  return g_verbose_level;
}

inline void SetVerboseLevel(int32 level) {
  g_verbose_level = level;
}

bool ApproxEqual(BaseFloat a, BaseFloat b, BaseFloat tol = 0.01);

#endif
