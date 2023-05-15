#pragma once

/**
 * @brief 矩阵转置（原始版本）
 *
 * @param odata 输出矩阵的首地址
 * @param idata 输入矩阵的首地址
 * @param h 输入矩阵的行数
 * @param w 输入矩阵的列数
 */
__global__ void transpose_naive(float* odata, const float* idata, int h, int w) {
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  // 每个线程的全局下标，同时也是每个线程的读取位置
  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;

  if (row < h && col < w) {
    odata[col * h + row] = idata[row * w + col];
  }
}