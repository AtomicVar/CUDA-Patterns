#pragma once

/**
 * @brief 计算矩阵乘法 C = A·B（使用 Tiling 技巧优化）
 * TODO: 越界检查
 *
 * @param C 输出矩阵，形状为 [hA, wB]
 * @param A 输入矩阵，形状为 [hA, wA]
 * @param B 输入矩阵，形状为 [wA, wB]
 * @param hA A 和 C 的行数
 * @param wA A 的列数和 B 的行数
 * @param wB B 和 C 的列数
 */
template <int BLOCK_SIZE>
__global__ void gemm_tiled(float* C, const float* A, const float* B, int hA, int wA, int wB) {
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  int aBegin = by * BLOCK_SIZE * wA;
  int aStep  = BLOCK_SIZE;
  int aEnd   = aBegin + wA - 1;

  int bBegin = bx * BLOCK_SIZE;
  int bStep  = BLOCK_SIZE * wB;

  float Csum = 0.0;
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    As[ty][tx] = A[a + ty * wA + tx];
    Bs[ty][tx] = B[b + ty * wB + tx];

    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
      Csum += As[ty][k] * Bs[k][tx];
    }

    __syncthreads();
  }

  int cBegin               = by * BLOCK_SIZE * wB + bx * BLOCK_SIZE;
  C[cBegin + ty * wB + tx] = Csum;
}