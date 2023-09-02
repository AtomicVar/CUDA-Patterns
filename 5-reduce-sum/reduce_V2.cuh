/**
 * @brief 1D block reduction V2 (Interleaved Addressing without divergent warps)
 *        Problem: bank conflicts
*/
__global__ void reduce_V2(int* idata, int* odata, int len) {
  extern __shared__ int sdata[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < len) ? idata[i] : 0;
  __syncthreads();

  for (int s = 1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;

    if (index < blockDim.x) {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    odata[blockIdx.x] = sdata[0];
  }
}