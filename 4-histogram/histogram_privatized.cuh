#pragma once

__global__ void histogram_privatized(int* data, int* histogram, int len, int num_bins) {
    extern __shared__ int histogram_s[];

    // set histogram_s to all 0s cooperatively
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        histogram_s[i] = 0;
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = tid; i < len; i += total_threads) {
        int val = data[i];
        atomicAdd(&histogram_s[val], 1);
    }
    __syncthreads();

    // commit results to global memory
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&histogram[i], histogram_s[i]);
    }
}