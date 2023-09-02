# 规约（Reduction）

## 实验环境

- GPU: NVIDIA GeForce RTX 3080
- CPU: Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz x 40
- CUDA/NVCC: 11.7
- OS: Ubuntu 20.04
- Host Compiler: g++ (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
  - Compiler Options: `-O3 -std=c++17`

## 性能数据

|             版本             | 耗时（us） | 加速比 |
| :--------------------------: | :--------: | :----: |
| [reduce_cpu](./reduce_cpu.h) |   22,461   |   1    |
| [reduce_V1](./reduce_V1.cuh) |   2,127    |  10.6  |
| [reduce_V2](./reduce_V2.cuh) |   1,261    |  17.8  |
| [reduce_V3](./reduce_V3.cuh) |   1,203    |  18.7  |
| [reduce_V4](./reduce_V4.cuh) |    646     |  34.8  |
| [reduce_V5](./reduce_V5.cuh) |    431     |  52.1  |
| [reduce_V6](./reduce_V6.cuh) |    391     |  57.4  |
| [reduce_V7](./reduce_V7.cuh) |    250     |  89.9  |

## 算法说明

TODO

## 参考

- [Programming Massively Parallel Processors](https://book.douban.com/subject/4265432/) (3rd Edition)