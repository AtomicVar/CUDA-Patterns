#pragma once

#define BLOCK_ROWS 1

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

  int row_in = by * blockDim.y + ty;
  int col_in = bx * blockDim.x + tx;

  int row_out = bx * blockDim.x + ty;
  int col_out = by * blockDim.y + tx;

  __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];

  if (row_in < h && col_in < w) {
    tile[ty][tx] = idata[row_in * w + col_in];
    __syncthreads();

    odata[row_out * w + col_out] = tile[tx][ty];
  }
}
