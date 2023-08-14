#include <cuda_runtime.h>

#define CUDA_CHECK(condition)                                                  \
    /* Code block avoids redefinition of cudaError_t error */                  \
    do {                                                                       \
        cudaError_t error = condition;                                         \
        if (error != cudaSuccess) {                                            \
            printf("%s in %s at line %d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

void randomInit(float* data, int len) {
  for (int i = 0; i < len; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

bool all_close(float* a, float* b, int len) {
  for (int i = 0; i < len; i++) {
    if (std::abs(a[i] - b[i]) > 1e-3) {
      return false;
    }
  }
  return true;
}
