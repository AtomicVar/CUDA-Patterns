#pragma once

__global__ void histogram_blocked(int* data, int* histogram, int len, int num_bins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    int section_size = (len + total_threads - 1) / total_threads;
    int offset = tid * section_size;

    for (int i = 0; i < section_size; i++) {
        if (offset + i < len) {
            int val = data[offset + i];
            atomicAdd(histogram + val, 1);
        }
    }
}