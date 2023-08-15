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

// generate random values between [min_val, max_val)
void randomInit(int* data, int len, int min_val, int max_val) {
  for (int i = 0; i < len; i++) {
    data[i] = min_val + (rand() % (max_val - min_val));
  }
}

bool all_close(float* a, float* b, int len) {
  for (int i = 0; i < len; i++) {
    if (std::abs(a[i] - b[i]) > 1e-3) {
      return false;
    }
  }
  return true;
}

bool all_equal(int* a, int* b, int len) {
  for (int i = 0; i < len; i++) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}
