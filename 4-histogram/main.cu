#include "../common/common.hpp"
#include "../common/Timer.hpp"
#include "./histogram_cpu.h"
#include "./histogram_blocked.cuh"
#include "./histogram_interleaved.cuh"
#include "./histogram_privatized.cuh"

#include <stdio.h>

#define N                 (1920 * 1080)
#define BINS              256
#define THREADS_PER_BLOCK 512

int x_cpu[N];
int *x_gpu, *hist_gpu;
int hist_cpu[BINS];
int hist_gpu_copyback[BINS];

int NUM_BLOCKS(int total_threads) {
  return (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

void prepare_inputs() {
  // prepare CPU input
  randomInit(x_cpu, N, 0, BINS);

  // prepare GPU input/output
  CUDA_CHECK(cudaMalloc(&x_gpu, sizeof(int) * N));
  CUDA_CHECK(cudaMalloc(&hist_gpu, sizeof(int) * BINS));
  CUDA_CHECK(cudaMemcpy(x_gpu, x_cpu, sizeof(int) * N, cudaMemcpyHostToDevice));
}

void cleanup() {
  CUDA_CHECK(cudaFree(x_gpu));
  CUDA_CHECK(cudaFree(hist_gpu));
}

void test_cpu_histogram() {
  Timer::Get()->Start("histogram_cpu");

  histogram_cpu(x_cpu, hist_cpu, N);

  Timer::Get()->End("histogram_cpu");
}

void test_gpu_histogram_blocked() {
  Timer::Get()->Start("histogram_blocked");

  histogram_blocked<<<NUM_BLOCKS(N), THREADS_PER_BLOCK>>>(x_gpu, hist_gpu, N, BINS);

  CUDA_CHECK(cudaDeviceSynchronize());
  Timer::Get()->End("histogram_blocked");

  CUDA_CHECK(cudaMemcpy(hist_gpu_copyback, hist_gpu, sizeof(int) * BINS, cudaMemcpyDeviceToHost));

  if (!all_equal(hist_cpu, hist_gpu_copyback, BINS)) {
    fprintf(stderr, "Error: histogram_blocked does not match CPU result!\n");
    exit(1);
  }
}

void test_gpu_histogram_interleaved() {
  Timer::Get()->Start("histogram_interleaved");
  histogram_interleaved<<<NUM_BLOCKS(N), THREADS_PER_BLOCK>>>(x_gpu, hist_gpu, N, BINS);
  CUDA_CHECK(cudaDeviceSynchronize());
  Timer::Get()->End("histogram_interleaved");
  CUDA_CHECK(cudaMemcpy(hist_gpu_copyback, hist_gpu, sizeof(int) * BINS, cudaMemcpyDeviceToHost));

  if (!all_equal(hist_cpu, hist_gpu_copyback, BINS)) {
    fprintf(stderr, "Error: histogram_interleaved does not match CPU result!\n");
    exit(1);
  }
}

void test_gpu_histogram_privatized() {
  Timer::Get()->Start("histogram_privatized");
  histogram_privatized<<<NUM_BLOCKS(N), THREADS_PER_BLOCK, sizeof(int) * BINS>>>(
      x_gpu, hist_gpu, N, BINS);
  CUDA_CHECK(cudaDeviceSynchronize());
  Timer::Get()->End("histogram_privatized");
  CUDA_CHECK(cudaMemcpy(hist_gpu_copyback, hist_gpu, sizeof(int) * BINS, cudaMemcpyDeviceToHost));

  if (!all_equal(hist_cpu, hist_gpu_copyback, BINS)) {
    fprintf(stderr, "Error: histogram_privatized does not match CPU result!\n");
    exit(1);
  }
}

int main() {
  prepare_inputs();

  test_cpu_histogram();

  test_gpu_histogram_blocked();

  CUDA_CHECK(cudaMemset(hist_gpu, 0, sizeof(int) * BINS));

  test_gpu_histogram_interleaved();

  CUDA_CHECK(cudaMemset(hist_gpu, 0, sizeof(int) * BINS));

  test_gpu_histogram_privatized();

  cleanup();

  return 0;
}