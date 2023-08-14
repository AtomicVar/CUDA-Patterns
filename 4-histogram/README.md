# 直方图（Histogram）

## 实验环境

- GPU: NVIDIA GeForce RTX 3080
- CPU: Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz x 40
- CUDA: 11.7
- OS: Ubuntu 20.04

## 性能数据

|                         版本                         | 耗时（us） | 加速比 |
| :--------------------------------------------------: | :--------: | :----: |
|          [histogram_cpu](./histogram_cpu.h)          |   5,660    |   1    |
|     [histogram_blocked](./histogram_blocked.cuh)     |    668     |  8.47  |
| [histogram_interleaved](./histogram_interleaved.cuh) |    635     |  8.91  |
|  [histogram_privatized](./histogram_privatized.cuh)  |     59     |  95.9  |

## 算法说明

TODO

## 参考

- [Programming Massively Parallel Processors](https://book.douban.com/subject/4265432/) (3rd Edition)