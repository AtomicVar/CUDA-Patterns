#include "common.cuh"

/**
 * @brief 1D block reduction V6 (Completely unrolled)
*/
__global__ void reduce_V6(int* idata, int* odata, int len) {
  extern __shared__ int sdata[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  int a = (i < len) ? idata[i] : 0;
  int b = (i + blockDim.x < len) ? idata[i + blockDim.x] : 0;
  sdata[tid] = a + b;
  __syncthreads();

  if (tid < 512) {
    sdata[tid] += sdata[tid + 512];
  }
  __syncthreads();
  if (tid < 256) {
    sdata[tid] += sdata[tid + 256];
  }
  __syncthreads();
  if (tid < 128) {
    sdata[tid] += sdata[tid + 128];
  }
  __syncthreads();
  if (tid < 64) {
    sdata[tid] += sdata[tid + 64];
  }
  __syncthreads();

  if (tid < 32) {
    warpReduce(sdata, tid);
  }

  if (tid == 0) {
    odata[blockIdx.x] = sdata[0];
  }
}