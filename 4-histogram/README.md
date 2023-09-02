# 直方图（Histogram）

## 实验环境

- GPU: NVIDIA GeForce RTX 3080
- CPU: Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz x 40
- CUDA/NVCC: 11.7
- OS: Ubuntu 20.04
- Host Compiler: g++ (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
  - Compiler Options: `-O3 -std=c++17`

## 性能数据

|                         版本                         | 耗时（us） | 加速比 |
| :--------------------------------------------------: | :--------: | :----: |
|          [histogram_cpu](./histogram_cpu.h)          |   1,454    |   1    |
|     [histogram_blocked](./histogram_blocked.cuh)     |    654     |  2.2   |
| [histogram_interleaved](./histogram_interleaved.cuh) |    636     |  2.3   |
|  [histogram_privatized](./histogram_privatized.cuh)  |     59     |  24.6  |

## 算法说明

TODO

## 参考

- [Programming Massively Parallel Processors](https://book.douban.com/subject/4265432/) (3rd Edition)