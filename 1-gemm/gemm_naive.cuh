#pragma once

/**
 * @brief 计算矩阵乘法 C = A·B（原始版本）
 *
 * @param C 输出矩阵，形状为 [hA, wB]
 * @param A 输入矩阵，形状为 [hA, wA]
 * @param B 输入矩阵，形状为 [wA, wB]
 * @param hA A 和 C 的行数
 * @param wA A 的列数和 B 的行数
 * @param wB B 和 C 的列数
 */
__global__ void gemm_naive(float* C, const float* A, const float* B, int hA, int wA, int wB) {
  int bx = blockIdx.x, tx = threadIdx.x;
  int by = blockIdx.y, ty = threadIdx.y;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;

  if (row < hA && col < wB) {
    float sum = 0.0;
    for (int i = 0; i < wA; i++) {
      sum += A[row * wA + i] * B[wB * i + col];
    }
    C[row * wB + col] = sum;
  }
}