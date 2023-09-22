# 矩阵转置（Matrix Transpose）

## 实验环境

- GPU: NVIDIA GeForce RTX 3080
- CPU: Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz x 40
- CUDA/NVCC: 11.7
- OS: Ubuntu 20.04
- Host Compiler: g++ (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
  - Compiler Options: `-O3 -std=c++17`

## 性能数据

输入数据尺寸：`[10240, 1024]`

|                              版本                               | 耗时（us） | 加速比 |
| :-------------------------------------------------------------: | :--------: | :----: |
|               [transpose_cpu](./transpose_cpu.h)                |   75764    |   1    |
|     [transpose_naive_read_coalesced](./transpose_naive.cuh)     |    480     | 157.8  |
|    [transpose_naive_write_coalesced](./transpose_naive.cuh)     |    228     | 332.3  |
|            [transpose_tiled](./transpose_tiled.cuh)             |    234     | 323.7  |
| [transpose_no_bank_conflict](./transpose_no_bank_conflicts.cuh) |    175     | 432.8  |

## 算法说明

TODO

## 参考

- [An Efficient Matrix Transpose in CUDA C/C++An Efficient Matrix Transpose in CUDA C/C++ | NVIDIA Technical Blog](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples): `Samples/6_Performance/transpose/transpose.cu`