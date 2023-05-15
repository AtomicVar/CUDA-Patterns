#pragma once

/**
 * @brief 1D 卷积（Tiled 版本）
 *
 * @param odata
 * @param idata
 * @param mask
 * @param w
 * @param mask_w
 * @return __global__
 */
__global__ void convolution_tiled(float* odata,
                                  const float* idata,
                                  const float* mask,
                                  int w,
                                  int mask_w) {
  int bx = blockIdx.x, tx = threadIdx.x;
  int i = bx * blockDim.x + tx;

  __shared__ float tile[blockDim.x];

  tile[tx] = idata[i];
  __syncthreads();

  int tile_start = bx * blockDim.x;
  int tile_end = (bx + 1) * blockDim.x; 

  float sum = 0.0;
  int begin = i - mask_w / 2;
  for (int j = 0; j < mask_w; j++) {
    int idata_idx = begin + j;
    if (idata_idx >= 0 && idata_idx < w) {
      if (idata_idx >= tile_start && idata_idx < tile_end) {
        sum += tile[tx] * mask[j];  // TODO: 这里 tile[tx] 不对
      } else {
        sum += idata[begin + j] * mask[j];
      }
    }
  }

  odata[i] = sum;
}
