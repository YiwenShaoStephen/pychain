#ifndef BASE_H_
#define BASE_H_

#define CU1DBLOCK   256

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
