## 1. C++ 题目

> 见 [C++ 面试题目](https://github.com/MrDotZLC/Algorithm/tree/main/interview_practice)

## 2. CUDA 题目

### 2.1 基础规约

- Reduce Sum（shared memory 归约 + 分层 kernel）
- Block Reduce
- 大矩阵 Reduce：$(10^6, 128) \rightarrow (1, 128)$
- 100 万浮点数相加
- 前缀和
- 行 TOP-1
- Vector Dot Product
- 向量加

### 2.2 矩阵运算

- GEMM
- 二维矩阵转置（逐步优化）
- Concat（四维）

### 2.3 激活 / 归一化

- RMSNorm
- Softmax / Online Softmax
- Layer Norm

### 2.4 Attention 系列

- MHA
- MGA（GQA）
- MLA
- Flash Attention v1
- Linear Attention

### 2.5 其他

- uint8 数组第 K 大
- CUDA 快速排序
- CUDA 线性插值 / 双线性插值
- MoE
