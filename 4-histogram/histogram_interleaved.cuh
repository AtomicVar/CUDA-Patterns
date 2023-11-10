#pragma once

// 实际上就是 Grid-Stride Looping
__global__ void histogram_interleaved(int* data, int* histogram, int len, int num_bins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = tid; i < len; i += total_threads) {
        int val = data[i];
        atomicAdd(histogram + val, 1);
    }
}
