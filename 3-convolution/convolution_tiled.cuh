#pragma once

#define MASK_MAX_LEN 32
__constant__ float M[MASK_MAX_LEN];

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
template<int TILE_SIZE>
__global__ void conv1d(float* data, float* output, int len, int mask_len) {
    __shared__ float data_s[TILE_SIZE + MASK_MAX_LEN - 1];

    int tx = threadIdx.x, bx = blockIdx.x;
    int tid = bx * TILE_SIZE + tx;

    // 1. Load to shared memory
    int n = mask_len / 2;
    if (tx >= TILE_SIZE - n) {
        float value = 0.0f;
        if (bx > 0) {
            value = data[(bx - 1) * TILE_SIZE + tx];
        }
        data_s[tx - (TILE_SIZE - n)] = value;
    }
    data_s[n + tx] = data[tid];
    if (tx < n) {
        float value = 0.0;
        if ((bx + 1) * TILE_SIZE + tx < len) {
            value = data[(bx + 1) * TILE_SIZE + tx];
        }
        data_s[TILE_SIZE + n + tx] = value;
    }

    __syncthreads();

    // 2. conv1d on shared memory
    float sum = 0.0;
    for (int i = 0; i < mask_len; i++) {
        sum += M[i] * data_s[tx + i];
    }

    // 3. Commit to global memory
    output[tid] = sum;
}
