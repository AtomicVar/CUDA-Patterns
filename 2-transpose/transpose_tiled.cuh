#pragma once

/**
 * @brief 矩阵转置（使用 Shared Memory 优化，读写均能访存合并）
 *
 * @param odata 输出矩阵的首地址
 * @param idata 输入矩阵的首地址
 * @param h 输入矩阵的行数
 * @param w 输入矩阵的列数
 */
template <int BLOCK_SIZE>
__global__ void transpose_tiled(float* odata, const float* idata, int h, int w) {
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;

  __shared__ tile[BLOCK_SIZE][BLOCK_SIZE];

  if (row < h && col < w) {
    tile[ty][tx] = idata[row * w + col];
    __syncthreads();

    odata[col * h + row] = tile[ty][tx];
  }
}
