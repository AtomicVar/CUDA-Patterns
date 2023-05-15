#include <iostream>

#include "./gemm_cpu.h"
#include "./gemm_naive.cuh"
#include "./gemm_tiled.cuh"

#define M 1024
#define K 1024
#define N 1024

// Allocates a matrix with random float entries.
void randomInit(float* data, int size) {
  for (int i = 0; i < size; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

int main() {
  float* A_h = new float[M * K];
  float* B_h = new float[K * N];
  float* C_h = new float[M * N];

  randomInit(A_h, M * K);
  randomInit(B_h, K * N);

  return 0;
}