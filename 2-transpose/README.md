# 矩阵转置（Matrix Transpose）

## 实验环境

- GPU: NVIDIA GeForce RTX 3080
- CPU: Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz x 40
- CUDA: 11.7
- OS: Ubuntu 20.04

## 性能数据

|                              版本                               | 耗时（us） | 加速比 |
| :-------------------------------------------------------------: | :--------: | :----: |
|               [transpose_cpu](./transpose_cpu.h)                |   363335   |   1    |
|            [transpose_naive](./transpose_naive.cuh)             |    722     | 503.2  |
|            [transpose_tiled](./transpose_tiled.cuh)             |    375     | 968.9  |
| [transpose_no_bank_conflict](./transpose_no_bank_conflicts.cuh) |    323     | 1124.9 |

## 算法说明

TODO

## 参考

- [An Efficient Matrix Transpose in CUDA C/C++An Efficient Matrix Transpose in CUDA C/C++ | NVIDIA Technical Blog](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples): `Samples/6_Performance/transpose/transpose.cu`