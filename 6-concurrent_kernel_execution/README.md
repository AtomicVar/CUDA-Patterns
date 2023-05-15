# Concurrent Kernel Execution

一些 Compute Capability 2.x 或更高的显卡支持**核函数的并发执行**：同时执行多个核函数。

## 如何判断是否支持？

我们可以通过检查 [device property](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html) 结构体里的 `concurrentKernels` 是否为 `1` 来判断当前显卡设备是否支持这个功能：

```c++
cudaDeviceProp deviceProp;
...
if ((deviceProp.concurrentKernels == 0)) {
  printf("> GPU does not support concurrent kernel execution\n");
  printf("  CUDA kernel runs will be serialized\n");
}
```

## 当前显卡最大支持多少 Kernels 并发执行？

当前显卡支持的最大 Kernels 并发数取决于当前显卡的 Compute Capability，具体数值在官方的 **CUDA C++ Programming Guide** 的 Compute Capabilities 章可以查阅。[点我查看](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability)

目前主流显卡（Compute Capability >= 7.5）支持的最大并发数是 **128**。