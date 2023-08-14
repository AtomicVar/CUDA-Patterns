#include <iostream>

#include "../common/common.hpp"
#include "../common/Timer.hpp"
#include "./gemm_cpu.h"
#include "./gemm_naive.cuh"
#include "./gemm_tiled.cuh"

#define M 1024
#define K 1024
#define N 1024
#define TILE_WIDTH 32

int main() {
  // ############# Make CPU/GPU buffers and randomly initialize inputs A, B #############
  float* A_h = new float[M * K];
  float* B_h = new float[K * N];
  float* C_h = new float[M * N];
  float* C_h_copy = new float[M * N];
  float *A_d, *B_d, *C_d;

  CUDA_CHECK(cudaMalloc((void**)&A_d, sizeof(float) * M * K));
  CUDA_CHECK(cudaMalloc((void**)&B_d, sizeof(float) * K * N));
  CUDA_CHECK(cudaMalloc((void**)&C_d, sizeof(float) * M * N));

  randomInit(A_h, M * K);
  randomInit(B_h, K * N);

  CUDA_CHECK(cudaMemcpy(A_d, A_h, sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_d, B_h, sizeof(float) * K * N, cudaMemcpyHostToDevice));

  // ############# CPU gemm #############
  Timer::Get()->Start("gemm_cpu");
  gemm_cpu(C_h, A_h, B_h, M, K, N);
  Timer::Get()->End("gemm_cpu");

  if (!all_close(C_h, C_h_copy, M * N)) {
    printf("[OK] Before naive kernel, C_h and C_h_copy are not equal.\n");
  }

  // ############# GPU gemm #############
  dim3 block(TILE_WIDTH, TILE_WIDTH);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  // ############# Naive kernel #############
  CUDA_CHECK(cudaDeviceSynchronize());
  Timer::Get()->Start("gemm_naive");
  gemm_naive<<<grid, block>>>(C_d, A_d, B_d, M, K, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  Timer::Get()->End("gemm_naive");

  CUDA_CHECK(cudaMemcpy(C_h_copy, C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
  if (!all_close(C_h, C_h_copy, M * N)) {
    fprintf(stderr, "naive kernel does not match CPU result!\n");
    exit(1);
  }

  // ############# Tiled kernel #############
  randomInit(C_h_copy, M * N);
  if (!all_close(C_h, C_h_copy, M * N)) {
    printf("[OK] Before tiled kernel, C_h and C_h_copy are not equal.\n");
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  Timer::Get()->Start("gemm_tiled");
  gemm_tiled<TILE_WIDTH><<<grid, block>>>(C_d, A_d, B_d, M, K, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  Timer::Get()->End("gemm_tiled");

  CUDA_CHECK(cudaMemcpy(C_h_copy, C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
  if (!all_close(C_h, C_h_copy, M * N)) {
    fprintf(stderr, "tiled kernel does not match CPU result!\n");
    exit(1);
  }

  // ############# Clean-up #############
  CUDA_CHECK(cudaFree(A_d));
  CUDA_CHECK(cudaFree(B_d));
  CUDA_CHECK(cudaFree(C_d));

  delete[] A_h;
  delete[] B_h;
  delete[] C_h;
  delete[] C_h_copy;

  return 0;
}