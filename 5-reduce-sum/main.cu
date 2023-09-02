#include "./reduce_cpu.h"
#include "./reduce_V1.cuh"
#include "./reduce_V2.cuh"
#include "./reduce_V3.cuh"
#include "./reduce_V4.cuh"
#include "./reduce_V5.cuh"
#include "./reduce_V6.cuh"
#include "./reduce_V7.cuh"

#include "../common/common.hpp"
#include "../common/Timer.hpp"

constexpr int N = 2560*1440*10;
// constexpr int MIN_N = 40000;  // better to use CPU for small N
constexpr int BLOCK_SIZE = 1024;
constexpr int MAX_NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

int *x_cpu, result_cpu;
int *x_gpu, *inWorkspace, *outWorkspace;

void prepare_input() {
  x_cpu          = new int[N];
  randomInit(x_cpu, N, 0, 5);

  CUDA_CHECK(cudaMalloc(&x_gpu, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&inWorkspace, MAX_NUM_BLOCKS * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&outWorkspace, MAX_NUM_BLOCKS * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(x_gpu, x_cpu, sizeof(int) * N, cudaMemcpyHostToDevice));
}

void cleanup() {
  delete[] x_cpu;
  CUDA_CHECK(cudaFree(x_gpu));
  CUDA_CHECK(cudaFree(inWorkspace));
  CUDA_CHECK(cudaFree(outWorkspace));
}

void test_reduce_cpu() {
  Timer::Get()->Start("reduce_cpu");

  result_cpu = 0.0;
  reduce_cpu(x_cpu, N, &result_cpu);

  Timer::Get()->End("reduce_cpu");
}

void test_reduce_V1(bool is_warmup = false) {
  if (!is_warmup)
    Timer::Get()->Start("reduce_V1");

  int num_blocks = MAX_NUM_BLOCKS;
  reduce_V1<<<num_blocks, BLOCK_SIZE, sizeof(int)*BLOCK_SIZE>>>(x_gpu, outWorkspace, N);

  int s = num_blocks;
  while (s > 1) {
    num_blocks = (s + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CHECK(cudaMemcpy(inWorkspace, outWorkspace, sizeof(int) * s, cudaMemcpyDeviceToDevice));
    reduce_V1<<<num_blocks, BLOCK_SIZE, sizeof(int)*BLOCK_SIZE>>>(inWorkspace, outWorkspace, s);

    s = num_blocks;
  }

  int result;
  CUDA_CHECK(cudaMemcpy(&result, outWorkspace, sizeof(int), cudaMemcpyDeviceToHost));

  CHECK(result == result_cpu) << "Reduce V1: Wrong result!"
                              << " expected: " << result_cpu << " got: " << result << std::endl;

  if (!is_warmup)
    Timer::Get()->End("reduce_V1");
}

void test_reduce_V2() {
  Timer::Get()->Start("reduce_V2");

  int num_blocks = MAX_NUM_BLOCKS;
  reduce_V2<<<num_blocks, BLOCK_SIZE, sizeof(int)*BLOCK_SIZE>>>(x_gpu, outWorkspace, N);

  int s = num_blocks;
  while (s > 1) {
    num_blocks = (s + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CHECK(cudaMemcpy(inWorkspace, outWorkspace, sizeof(int) * s, cudaMemcpyDeviceToDevice));
    reduce_V2<<<num_blocks, BLOCK_SIZE, sizeof(int)*BLOCK_SIZE>>>(inWorkspace, outWorkspace, s);

    s = num_blocks;
  }

  int result;
  CUDA_CHECK(cudaMemcpy(&result, outWorkspace, sizeof(int), cudaMemcpyDeviceToHost));

  CHECK(result == result_cpu) << "Reduce V2: Wrong result!"
                              << " expected: " << result_cpu << " got: " << result << std::endl;

  Timer::Get()->End("reduce_V2");
}

void test_reduce_V3() {
  Timer::Get()->Start("reduce_V3");

  int num_blocks = MAX_NUM_BLOCKS;
  reduce_V3<<<num_blocks, BLOCK_SIZE, sizeof(int)*BLOCK_SIZE>>>(x_gpu, outWorkspace, N);

  int s = num_blocks;
  while (s > 1) {
    num_blocks = (s + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CHECK(cudaMemcpy(inWorkspace, outWorkspace, sizeof(int) * s, cudaMemcpyDeviceToDevice));
    reduce_V3<<<num_blocks, BLOCK_SIZE, sizeof(int)*BLOCK_SIZE>>>(inWorkspace, outWorkspace, s);

    s = num_blocks;
  }

  int result;
  CUDA_CHECK(cudaMemcpy(&result, outWorkspace, sizeof(int), cudaMemcpyDeviceToHost));

  CHECK(result == result_cpu) << "Reduce V3: Wrong result!"
                              << " expected: " << result_cpu << " got: " << result << std::endl;

  Timer::Get()->End("reduce_V3");
}

void test_reduce_V4() {
  Timer::Get()->Start("reduce_V4");

  int num_blocks = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
  reduce_V4<<<num_blocks, BLOCK_SIZE, sizeof(int)*BLOCK_SIZE>>>(x_gpu, outWorkspace, N);

  int s = num_blocks;
  while (s > 1) {
    num_blocks = (s + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    CUDA_CHECK(cudaMemcpy(inWorkspace, outWorkspace, sizeof(int) * s, cudaMemcpyDeviceToDevice));
    reduce_V4<<<num_blocks, BLOCK_SIZE, sizeof(int)*BLOCK_SIZE>>>(inWorkspace, outWorkspace, s);

    s = num_blocks;
  }

  int result;
  CUDA_CHECK(cudaMemcpy(&result, outWorkspace, sizeof(int), cudaMemcpyDeviceToHost));

  CHECK(result == result_cpu) << "Reduce V4: Wrong result!"
                              << " expected: " << result_cpu << " got: " << result << std::endl;

  Timer::Get()->End("reduce_V4");
}

void test_reduce_V5() {
  Timer::Get()->Start("reduce_V5");

  int num_blocks = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
  reduce_V5<<<num_blocks, BLOCK_SIZE, sizeof(int)*BLOCK_SIZE>>>(x_gpu, outWorkspace, N);

  int s = num_blocks;
  while (s > 1) {
    num_blocks = (s + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    CUDA_CHECK(cudaMemcpy(inWorkspace, outWorkspace, sizeof(int) * s, cudaMemcpyDeviceToDevice));
    reduce_V5<<<num_blocks, BLOCK_SIZE, sizeof(int)*BLOCK_SIZE>>>(inWorkspace, outWorkspace, s);

    s = num_blocks;
  }

  int result;
  CUDA_CHECK(cudaMemcpy(&result, outWorkspace, sizeof(int), cudaMemcpyDeviceToHost));

  CHECK(result == result_cpu) << "Reduce V5: Wrong result!"
                              << " expected: " << result_cpu << " got: " << result << std::endl;

  Timer::Get()->End("reduce_V5");
}

void test_reduce_V6() {
  Timer::Get()->Start("reduce_V6");

  int num_blocks = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
  reduce_V6<<<num_blocks, BLOCK_SIZE, sizeof(int)*BLOCK_SIZE>>>(x_gpu, outWorkspace, N);

  int s = num_blocks;
  while (s > 1) {
    num_blocks = (s + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    CUDA_CHECK(cudaMemcpy(inWorkspace, outWorkspace, sizeof(int) * s, cudaMemcpyDeviceToDevice));
    reduce_V6<<<num_blocks, BLOCK_SIZE, sizeof(int)*BLOCK_SIZE>>>(inWorkspace, outWorkspace, s);

    s = num_blocks;
  }

  int result;
  CUDA_CHECK(cudaMemcpy(&result, outWorkspace, sizeof(int), cudaMemcpyDeviceToHost));

  CHECK(result == result_cpu) << "Reduce V6: Wrong result!"
                              << " expected: " << result_cpu << " got: " << result << std::endl;

  Timer::Get()->End("reduce_V6");
}

void test_reduce_V7() {
  Timer::Get()->Start("reduce_V7");

  int num_blocks = 64;
  reduce_V7<<<num_blocks, BLOCK_SIZE, sizeof(int)*BLOCK_SIZE>>>(x_gpu, outWorkspace, N);

  int s = num_blocks;
  while (s > 1) {
    // num_blocks = (s + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    CUDA_CHECK(cudaMemcpy(inWorkspace, outWorkspace, sizeof(int) * s, cudaMemcpyDeviceToDevice));
    reduce_V7<<<num_blocks, BLOCK_SIZE, sizeof(int)*BLOCK_SIZE>>>(inWorkspace, outWorkspace, s);

    s = (s + BLOCK_SIZE - 1) / BLOCK_SIZE;
  }

  int result;
  CUDA_CHECK(cudaMemcpy(&result, outWorkspace, sizeof(int), cudaMemcpyDeviceToHost));

  CHECK(result == result_cpu) << "Reduce V7: Wrong result!"
                              << " expected: " << result_cpu << " got: " << result << std::endl;

  Timer::Get()->End("reduce_V7");
}

void warm_up() {
  for (int i = 0; i < 4; i++) {
    test_reduce_V1(true);
  }
}

int main() {
  prepare_input();

  test_reduce_cpu();

  // warmup GPU
  warm_up();

  test_reduce_V1();
  test_reduce_V2();
  test_reduce_V3();
  test_reduce_V4();
  test_reduce_V5();
  test_reduce_V6();
  test_reduce_V7();

  cleanup();

  return 0;
}