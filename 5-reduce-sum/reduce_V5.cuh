#include "common.cuh"

/**
 * @brief 1D block reduction V5 (Unroll the last warp)
*/
__global__ void reduce_V5(int* idata, int* odata, int len) {
  extern __shared__ int sdata[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  int a = (i < len) ? idata[i] : 0;
  int b = (i + blockDim.x < len) ? idata[i + blockDim.x] : 0;
  sdata[tid] = a + b;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s /= 2) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    warpReduce(sdata, tid);
  }

  if (tid == 0) {
    odata[blockIdx.x] = sdata[0];
  }
}