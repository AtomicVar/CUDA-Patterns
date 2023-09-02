/**
 * @brief 1D block reduction V1 (Interleaved Addressing)
 *        Problem: warp divergence
*/
__global__ void reduce_V1(int* idata, int* odata, int len) {
  extern __shared__ int sdata[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < len) ? idata[i] : 0;
  __syncthreads();

  for (int s = 1; s < blockDim.x; s *= 2) {
    //! warp divergence: threads in the same warp take different execution paths
    if (tid % (2 * s) == 0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    odata[blockIdx.x] = sdata[0];
  }
}