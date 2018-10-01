#ifndef BASE_H_
#define BASE_H_

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

#endif
