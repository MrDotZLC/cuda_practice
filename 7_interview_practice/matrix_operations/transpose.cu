#include <cuda_runtime.h>

static constexpr int TILE = 32;

// ── V2：Shared Memory 转置（读写均 Coalesced）────────────────────────────
// 策略：
//   1. 协作读 A 的一个 TILE×TILE 块到 smem（行连续读，Coalesced）
//   2. 从 smem 中转置读取后写到 B（列连续写，Coalesced）
__global__ void transposeShared(const float* __restrict__ A,
                                 float*       __restrict__ B,
                                 int R, int C) {
    // +1 消除 Bank Conflict：
    // 转置后同一列元素映射到同一 Bank，+1 使步长为质数，打散 Bank 访问
    __shared__ float smem[TILE][TILE + 1];

    // 读阶段：按 Block 坐标从 A 加载（行优先，Coalesced）
    int read_row = blockIdx.y * TILE + threadIdx.y;
    int read_col = blockIdx.x * TILE + threadIdx.x;
    if (read_row < R && read_col < C)
        smem[threadIdx.y][threadIdx.x] = A[read_row * C + read_col];
    __syncthreads();

    // 写阶段：Block 坐标互换（x↔y），从 smem 中转置读出后按列写 B
    // B 的目标 Tile：行起点 = blockIdx.x * TILE，列起点 = blockIdx.y * TILE
    int write_row = blockIdx.x * TILE + threadIdx.y;   // 注意：x↔y 互换
    int write_col = blockIdx.y * TILE + threadIdx.x;
    if (write_row < C && write_col < R)
        // 从 smem 转置读：smem[threadIdx.x][threadIdx.y]（x,y 互换）
        B[write_row * R + write_col] = smem[threadIdx.x][threadIdx.y];
}

// ── Host 封装（V2 推荐版）─────────────────────────────────────────────────
void transpose(const float* d_A, float* d_B, int R, int C) {
    dim3 block(TILE, TILE);
    dim3 grid((C + TILE - 1) / TILE, (R + TILE - 1) / TILE);
    transposeShared<<<grid, block>>>(d_A, d_B, R, C);
}