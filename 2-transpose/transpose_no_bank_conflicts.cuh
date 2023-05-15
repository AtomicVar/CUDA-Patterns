#pragma once

/**
 * @brief 矩阵转置（对 Shared Memory 的访问进行优化，避免 Bank Conflicts）
 *
 * @param odata 输出矩阵的首地址
 * @param idata 输入矩阵的首地址
 * @param h 输入矩阵的行数
 * @param w 输入矩阵的列数
 */
template <int BLOCK_SIZE>
__global__ void transpose_no_bank_conflicts(float* odata, const float* idata, int h, int w) {
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;

  __shared__ tile[BLOCK_SIZE][BLOCK_SIZE + 1];

  if (row < h && col < w) {
    tile[ty][tx] = idata[row * w + col];
    __syncthreads();

    odata[col * h + row] = tile[ty][tx];
  }
}
