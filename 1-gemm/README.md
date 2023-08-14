# 通用矩阵乘法（General Matrix Multiplication）

## 实验环境

- GPU: NVIDIA GeForce RTX 3080
- CPU: Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz x 40
- CUDA: 11.7
- OS: Ubuntu 20.04

## 性能数据

|              版本              | 耗时（us） | 加速比 |
| :----------------------------: | :--------: | :----: |
|    [gemm_cpu](./gemm_cpu.h)    | 7,559,124  |   1    |
| [gemm_naive](./gemm_naive.cuh) |   1,377    | 5,489  |
| [gemm_tiled](./gemm_tiled.cuh) |    979     | 7,721  |

## 算法说明

TODO

## 参考

- [Programming Massively Parallel Processors](https://book.douban.com/subject/4265432/) (3rd Edition)
- [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples): `Samples/0_Introduction/matrixMul/matrixMul.cu`