#include "common.cuh"

/**
 * @brief 1D block reduction V7 (Multiple add)
*/
__global__ void reduce_V7(int* idata, int* odata, int len) {
  extern __shared__ int sdata[];

  int tid = threadIdx.x;

  // grid-stride loop
  int sum = 0;
  for (int i = blockIdx.x * blockDim.x + tid; i < len; i += gridDim.x * blockDim.x) {
    sum += idata[i];
  }
  sdata[tid] = sum;
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