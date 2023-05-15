#pragma once

/**
 * @brief 1D 卷积（原始版本）
 *
 * @param odata
 * @param idata
 * @param mask
 * @param w
 * @param mask_w
 * @return __global__
 */
__global__ void convolution_naive(float* odata,
                                  const float* idata,
                                  const float* mask,
                                  int w,
                                  int mask_w) {
  int bx = blockIdx.x, tx = threadIdx.x;
  int i = bx * blockDim.x + tx;

  float sum = 0.0;

  int begin = i - mask_w / 2;
  for (int j = 0; j < mask_w; j++) {
    if (begin + j >= 0 && begin + j < w) {
      sum += idata[begin + j] * mask[j];
    }
  }

  odata[i] = sum;
}