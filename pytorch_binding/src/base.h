#ifndef BASE_H_
#define BASE_H_

#define HAVE_CUDA 1

#if HAVE_CUDA == 1
#define CU1DBLOCK   256
#endif
typedef int int32;
typedef long int64;
typedef float BaseFloat;
typedef int int32_cuda;

extern int32 g_verbose_level;

inline int32 GetVerboseLevel() {
  return g_verbose_level;
}

inline void SetVerboseLevel(int32 level) {
  g_verbose_level = level;
}

/** Number of blocks in which the task of size 'size' is splitted **/
inline int32 n_blocks(int32 size, int32 block_size) {
  return size / block_size + ((size % block_size == 0)? 0 : 1);
}

bool ApproxEqual(BaseFloat a, BaseFloat b, BaseFloat tol = 0.01);

#endif
