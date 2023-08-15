#include <iostream>

#include "./transpose_cpu.h"
#include "./transpose_tiled.cuh"
#include "./transpose_naive.cuh"
#include "./transpose_no_bank_conflicts.cuh"

#include "../common/common.hpp"
#include "../common/Timer.hpp"

#define M         4096
#define N         4096
#define TILE_SIZE 32

float *x_cpu, *y_cpu;
float *x_gpu, *y_gpu, *y_gpu_copyback;

void prepare_input() {
  x_cpu          = new float[M * N];
  y_cpu          = new float[M * N];
  y_gpu_copyback = new float[M * N];
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++)
      x_cpu[i * N + j] = i * N + j;
  }

  CUDA_CHECK(cudaMalloc(&x_gpu, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&y_gpu, M * N * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(x_gpu, x_cpu, sizeof(float) * M * N, cudaMemcpyHostToDevice));
}

void cleanup() {
  delete[] x_cpu;
  delete[] y_cpu;
  delete[] y_gpu_copyback;
  CUDA_CHECK(cudaFree(x_gpu));
  CUDA_CHECK(cudaFree(y_gpu));
}

void test_transpose_cpu() {
  Timer::Get()->Start("transpose_cpu");

  transpose_cpu(x_cpu, y_cpu, M, N);

  Timer::Get()->End("transpose_cpu");
}

void test_transpose_naive() {
  randomInit(y_gpu_copyback, M * N);
  CUDA_CHECK(cudaDeviceSynchronize());

  Timer::Get()->Start("transpose_naive");

  dim3 block = {TILE_SIZE, TILE_SIZE, 1};
  dim3 grid  = {ceil_div(N, TILE_SIZE), ceil_div(M, TILE_SIZE), 1};
  transpose_naive<<<grid, block>>>(y_gpu, x_gpu, M, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  Timer::Get()->End("transpose_naive");

  CUDA_CHECK(cudaMemcpy(y_gpu_copyback, y_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

  if (!all_close(y_cpu, y_gpu_copyback, M * N)) {
    fprintf(stderr, "Error: transpose_naive does not match CPU result!\n");
    exit(1);
  }
}

void test_transpose_tiled() {
  randomInit(y_gpu_copyback, M * N);
  CUDA_CHECK(cudaDeviceSynchronize());

  Timer::Get()->Start("transpose_tiled");

  dim3 block = {TILE_SIZE, TILE_SIZE, 1};
  dim3 grid  = {ceil_div(N, TILE_SIZE), ceil_div(M, TILE_SIZE), 1};
  transpose_tiled<TILE_SIZE><<<grid, block>>>(y_gpu, x_gpu, M, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  Timer::Get()->End("transpose_tiled");

  CUDA_CHECK(cudaMemcpy(y_gpu_copyback, y_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

  if (!all_close(y_cpu, y_gpu_copyback, M * N)) {
    fprintf(stderr, "Error: transpose_tiled does not match CPU result!\n");
    // print cpu result
    printf("CPU:\n");
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++)
        printf("%f ", y_cpu[i * N + j]);
      printf("\n");
    }
    // print gpu result
    printf("GPU:\n");
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++)
        printf("%f ", y_gpu_copyback[i * N + j]);
      printf("\n");
    }
    exit(1);
  }
}

void test_transpose_no_bc() {
  randomInit(y_gpu_copyback, M * N);
  CUDA_CHECK(cudaDeviceSynchronize());

  Timer::Get()->Start("transpose_no_bc");

  dim3 block = {TILE_SIZE, TILE_SIZE, 1};
  dim3 grid  = {ceil_div(N, TILE_SIZE), ceil_div(M, TILE_SIZE), 1};
  transpose_no_bank_conflicts<TILE_SIZE><<<grid, block>>>(y_gpu, x_gpu, M, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  Timer::Get()->End("transpose_no_bc");

  CUDA_CHECK(cudaMemcpy(y_gpu_copyback, y_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

  if (!all_close(y_cpu, y_gpu_copyback, M * N)) {
    fprintf(stderr, "Error: transpose_no_bc does not match CPU result!\n");
    exit(1);
  }
}

int main() {
  prepare_input();

  test_transpose_cpu();

  test_transpose_naive();
  test_transpose_tiled();
  test_transpose_no_bc();

  cleanup();
  return 0;
}