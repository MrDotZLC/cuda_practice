#include <iostream>

#include "common/tester.h"
#include "common/common.h"
// #include "mma.h"
using namespace nvcuda;

// 索引变换级别 swizzle 函数
// i: shared memory 中的行索引（row index）
// j: shared memory 中的列索引（col index）
//
// 该函数的职责：
//   - 仅对“列索引 j”进行 swizzle
//   - 行索引 i 只作为扰动因子参与 XOR
//
// 设计目标：
//   - 消除 ldmatrix / 向量化 load 的 shared memory bank conflict
//   - 完全保持逻辑矩阵不变（只改变物理存储顺序）
template <const int kColStride = 16, const int kStep = 8>
static __device__ __forceinline__ int swizzle_permuted_j(int i, int j) {
    // Tensor Core 的 MMA atom 在 K 维最大为 16
    // swizzle 模式只针对 k <= 16 的 ldmatrix 访问
    static_assert(kColStride <= 16, "kColStride must <= 16");

    // kStep = 8 : 128-bit load（8 * half）
    // kStep = 4 : 64-bit  load（4 * half）
    static_assert(kStep == 4 || kStep == 8, "kStep must be 8 or 4.");

    // 列宽必须是 step 的整数倍，否则会破坏向量 load 对齐
    static_assert(kColStride % kStep == 0,
                  "kColStride must be multiple of kStep.");

    if constexpr (kStep == 8) {
        // j >> 3 : 第几个 8-half 向量
        // i >> 2 : 每 4 行切换一次 XOR phase（匹配 ldmatrix 的 row grouping）
        // XOR    : 生成 ZigZag / Checkerboard 访问模式
        // %      : 保证索引落在合法列范围内
        // << 3   : 还原成 half 索引
        return (((j >> 3) ^ (i >> 2)) % (kColStride >> 3)) << 3;
    } else {
        // kStep == 4 时逻辑完全一致，只是粒度变成 4 half
        return (((j >> 2) ^ (i >> 2)) % (kColStride >> 2)) << 2;
    }
}

// i: shared memory 行索引（0 ~ 15，对应 MMA atom 的 M 维）
// j: shared memory 列索引（0 或 8）
//
// 专门为 MMA m16n8k16 的 A operand 定制的 swizzle
// 目的：
//   - 消除 ldmatrix.x4 从 s_a 读取时的 bank conflict
//   - 精确匹配 warp 内 lane_id 的访问模式
template <const int kMmaAtomK = 16>
static __device__ __forceinline__ int swizzle_permuted_A_j(int i, int j) {
    // swizzle 产生的实际效果（step = 8）：
    //
    // row  0~3   : (0, 8)
    // row  4~7   : (8, 0)
    // row  8~11  : (0, 8)
    // row 12~15  : (8, 0)
    //
    // 这正好匹配：
    //   - lane_id % 16 决定行
    //   - lane_id / 16 决定列（0 或 8）
    return swizzle_permuted_j<kMmaAtomK, 8>(i, j);
}

// 地址变换级别 swizzle 函数（未在代码中使用）
// S：SShift, right shift the addr for swizzling 
// B：BShift, bits to be swizzled 
// M: MBase，bits keep the same
// S: SShift
//    从 addr 右移 S 位后，选取其中一段 bit 参与 swizzle
//
// B: BShift
//    需要参与 swizzle 的 bit 数量（连续 B 位）
//
// M: MBase
//    swizzle 发生的 bit 起始位置（低 M 位保持不变）
template <uint32_t S, uint32_t B, uint32_t M>
__device__ __forceinline__ uint32_t swizzle(uint32_t addr) {

  // 构造一个 bit mask：
  //   - (1 << B) - 1  : 生成 B 个连续的 1（如 B=3 -> 0b111）
  //   - << M          : 将这 B 位左移到 [M, M+B) 区间
  //
  // 示例：
  //   B = 3, M = 5
  //   Bmask = 0b00000011100000
  constexpr auto Bmask = ((1 << B) - 1) << M;

  // swizzle 核心逻辑：
  //
  // 1. addr >> S
  //    - 将地址右移 S 位
  //    - 选取“较高位”的一部分作为扰动源
  //
  // 2. (addr >> S) & Bmask
  //    - 只保留 mask 覆盖的那 B 位
  //    - 其余位清零
  //
  // 3. ^ addr
  //    - 用选中的高位 bit 去 XOR 原地址
  //    - 只会翻转 Bmask 覆盖的那一段 bit
  //
  // 效果：
  //   - addr 的低 M 位保持不变（保证对齐）
  //   - addr 的 [M, M+B) 位被高位信息扰动
  //   - 高于该区间的位保持不变
  return ((addr >> S) & Bmask) ^ addr;
}

// Kernel with Swizzle
template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
          const int MMA_TILE_M = 2, const int MMA_TILE_N = 4,
          const int WARP_TILE_M = 4, const int WARP_TILE_N = 4,
          const int A_PAD = 0, const int B_PAD = 0>
__global__ void __launch_bounds__(256)
    hgemm_mma_m16n8k16_mma2x4_warp4x4_swizzle_kernel(half *A, half *B, half *C,
                                                     int M, int N, int K) {
    // block 在 grid 中的二维索引
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // K 方向 tile 数
    const int NUM_K_TILES = div_ceil(K, MMA_K);

    // block 级别负责的 C tile 尺寸：128 x 128
    constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;  // 16*2*4 = 128
    constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;  // 8*4*4  = 128
    constexpr int BK = MMA_K;                             // 16

    // shared memory：A 为 [BM][BK]，B 为 [BK][BN]
    // A_PAD / B_PAD 用于避免 bank conflict（与 swizzle 可叠加）
    __shared__ half s_a[BM][BK + A_PAD];
    __shared__ half s_b[BK][BN + B_PAD];

    // 线程在线程块内的线性索引
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // warp 索引（一个 block 8 个 warp）
    const int warp_id = tid / 32;

    // warp 内 lane 索引
    const int lane_id = tid % 32;

    // warp 在 block tile 中的二维坐标
    const int warp_m = warp_id % 2;  // 行方向
    const int warp_n = warp_id / 2;  // 列方向

    // 每个线程加载 s_a 中的一个 8-half 向量
    int load_smem_a_m = tid / 2;                 // 行索引
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;  // 列索引（0 或 8）

    // 每个线程加载 s_b 中的一个 8-half 向量
    int load_smem_b_k = tid / 16;        // 行索引
    int load_smem_b_n = (tid % 16) * 8;  // 列索引

    // 映射到全局内存 A / B 的位置
    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;

    // 越界保护
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    // 每个 warp tile 的累加寄存器
    uint32_t RC[WARP_TILE_M][WARP_TILE_N][2] = {};

#pragma unroll
    for (int k = 0; k < NUM_K_TILES; ++k) {
        // A: gmem -> smem（写入时进行 swizzle）
        int load_gmem_a_k = k * BK + load_smem_a_k;
        int swizzled_a_k = swizzle_permuted_A_j(load_smem_a_m, load_smem_a_k);
        LDST128BITS(s_a[load_smem_a_m][swizzled_a_k]) =
            LDST128BITS(A[load_gmem_a_m * K + load_gmem_a_k]);

        // B: gmem -> smem（写入时进行 swizzle）
        int load_gmem_b_k = k * BK + load_smem_b_k;
        int swizzled_b_n =
            swizzle_permuted_j<BK, 8>(load_smem_b_k, load_smem_b_n);
        LDST128BITS(s_b[load_smem_b_k][swizzled_b_n]) =
            LDST128BITS(B[load_gmem_b_k * N + load_gmem_b_n]);

        __syncthreads();

        // 每个 warp 从 smem 读取到寄存器 fragment
        uint32_t RA[WARP_TILE_M][4], RB[WARP_TILE_N][2];

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
            int lane_smem_a_k = (lane_id / 16) * 8;

            // 读取时使用与写入完全一致的 swizzle
            int swizzled_lane_a_k =
                swizzle_permuted_A_j(lane_smem_a_m, lane_smem_a_k);

            uint32_t lane_smem_a_ptr = __cvta_generic_to_shared(
                &s_a[lane_smem_a_m][swizzled_lane_a_k]);

            // ldmatrix.x4：加载 A fragment
            LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3],
                        lane_smem_a_ptr);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            int lane_smem_b_k = lane_id % 16;
            int lane_smem_b_n = warp_smem_b_n;

            int swizzled_lane_b_n =
                swizzle_permuted_j<MMA_K, 8>(lane_smem_b_k, lane_smem_b_n);

            uint32_t lane_smem_b_ptr = __cvta_generic_to_shared(
                &s_b[lane_smem_b_k][swizzled_lane_b_n]);

            // ldmatrix.x2.trans：加载 B fragment
            LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                // Tensor Core MMA 指令
                HMMA16816(RC[i][j][0], RC[i][j][1], RA[i][0], RA[i][1],
                          RA[i][2], RA[i][3], RB[j][0], RB[j][1], RC[i][j][0],
                          RC[i][j][1]);
            }
        }
        __syncthreads();
    }

    // reg -> gmem
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int store_warp_smem_c_m =
                warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            int store_warp_smem_c_n =
                warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            int store_lane_gmem_c_m =
                by * BM + store_warp_smem_c_m + lane_id / 4;
            int store_lane_gmem_c_n =
                bx * BN + store_warp_smem_c_n + (lane_id % 4) * 2;
            int addr0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
            int addr1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
            // 每个 lane 写回两个 half2
            LDST32BITS(C[addr0]) = LDST32BITS(RC[i][j][0]);
            LDST32BITS(C[addr1]) = LDST32BITS(RC[i][j][1]);
        }
    }
}

// 128x128, mma2x4, warp4x4(64,32,16)
void hgemm_mma_m16n8k16_mma2x4_warp4x4_swizzle(half *A, half *B, half *C, int M,
                                               int N, int K) {
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int MMA_TILE_M = 2;
    constexpr int MMA_TILE_N = 4;
    constexpr int WARP_TILE_M = 4;
    constexpr int WARP_TILE_N = 4;
    // bank conflicts free via pad = 8, reject fantasy, trust the profile.
    // ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld
    // ./hgemm_mma_stage.89.debug.bin ncu --metrics
    // sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm
    // ./hgemm_mma_stage.89.debug.bin
    constexpr int A_PAD = 8;
    constexpr int B_PAD = 8;
    constexpr int NUM_THREADS =
        (MMA_TILE_M * MMA_TILE_N * WARP_SIZE);  // 2 * 4 * 32 = 256

    dim3 block(NUM_THREADS);
    dim3 grid(div_ceil(N, MMA_N * MMA_TILE_N * WARP_TILE_N),
              div_ceil(M, MMA_M * MMA_TILE_M * WARP_TILE_M));

    hgemm_mma_m16n8k16_mma2x4_warp4x4_swizzle_kernel<
        MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M, WARP_TILE_N,
        A_PAD, B_PAD><<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char *argv[]) {
    Tester tester(512, 2048, 1024, 1, 10, 100, true);
    tester.evaluate(hgemm_mma_m16n8k16_mma2x4_warp4x4_swizzle,
                    "hgemm_mma_m16n8k16_mma2x4_warp4x4_swizzle");
}
