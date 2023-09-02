/**
 * @brief 1D block reduction V3 (Continuous Addressing without bank conflicts)
 *        Problem: Half of the threads only do one load from global memory to shared memory
*/
__global__ void reduce_V3(int* idata, int* odata, int len) {
  extern __shared__ int sdata[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < len) ? idata[i] : 0;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s /= 2) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    odata[blockIdx.x] = sdata[0];
  }
}