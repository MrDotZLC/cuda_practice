#include <iostream>

#include "common/tester.h"
#include "common/common.h"
// #include "mma.h"
using namespace nvcuda;

// only 1 warp per block(32 threads), m16n8k16. A, B, C: all row_major.
template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16>
__global__ void hgemm_mma_m16n8k16_naive_kernel(half *A, half *B, half *C,
                                                int M, int N, int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K);
  constexpr int BM = MMA_M; // 16
  constexpr int BN = MMA_N; // 8
  constexpr int BK = MMA_K; // 16

  __shared__ half s_a[MMA_M][MMA_K]; // 16x16
  __shared__ half s_b[MMA_K][MMA_N]; // 16x8
  __shared__ half s_c[MMA_M][MMA_N]; // 16x8

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int lane_id = tid % WARP_SIZE;                    // 0~31

  // s_a[16][16], 每行16，每线程load 8，需要2线程，共16行，需2x16=32线程
  const int load_smem_a_m = tid / 2;       // row 0~15
  const int load_smem_a_k = (tid % 2) * 8; // col 0,8
  // s_b[16][8], 每行8，每线程load
  // 8，需要1线程，共16行，需16线程，只需一半线程加载
  const int load_smem_b_k = tid; // row 0~31, but only use 0~15
  const int load_smem_b_n = 0;   // col 0
  const int load_gmem_a_m = by * BM + load_smem_a_m; // global m
  const int load_gmem_b_n = bx * BN + load_smem_b_n; // global n
  if (load_gmem_a_m >= M && load_gmem_b_n >= N)
    return;

  uint32_t RC[2] = {0, 0};

#pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    // gmem_a -> smem_a
    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) =
        (LDST128BITS(A[load_gmem_a_addr]));

    // gmem_b -> smem_b
    if (lane_id < MMA_K) {
      int load_gmem_b_k = k * MMA_K + load_smem_b_k; // global row of b
      int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
      LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) =
          (LDST128BITS(B[load_gmem_b_addr]));
    }
    __syncthreads();

    uint32_t RA[4];
    uint32_t RB[2];

    // ldmatrix for s_a, ldmatrix.trans for s_b.
    // s_a: (0,1)*8 -> 0,8 -> [(0~15),(0,8)]
    uint32_t load_smem_a_ptr =
        __cvta_generic_to_shared(&s_a[lane_id % 16][(lane_id / 16) * 8]);
    LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], load_smem_a_ptr);
    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(&s_b[lane_id % 16][0]);
    LDMATRIX_X2_T(RB[0], RB[1], load_smem_b_ptr);

    HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0],
              RC[1]);

    __syncthreads();
  }

  // s_c[16][8],
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
  // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
  // [0~7][0~3 u32 -> 0~7 f16], [8~15][0~3 u32 -> 0~7 f16]
  LDST32BITS(s_c[lane_id / 4][(lane_id % 4) * 2]) = LDST32BITS(RC[0]);
  LDST32BITS(s_c[lane_id / 4 + 8][(lane_id % 4) * 2]) = LDST32BITS(RC[1]);

  __syncthreads();

  // store s_c[16][8]
  if (lane_id < MMA_M) {
    // store 128 bits per memory issue.
    int store_gmem_c_m = by * BM + lane_id;
    int store_gmem_c_n = bx * BN;
    int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
    LDST128BITS(C[store_gmem_c_addr]) = (LDST128BITS(s_c[lane_id][0]));
  }
}

// only 1 warp per block(32 threads), m16n8k16. A, B, C: all row_major.
void hgemm_mma_m16n8k16_naive(half *A, half *B, half *C, int M, int N,
                                int K) {
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;

  dim3 block(WARP_SIZE);
  dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));

  hgemm_mma_m16n8k16_naive_kernel<MMA_M, MMA_N, MMA_K>
      <<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char *argv[]) {
    Tester tester(512, 2048, 1024, 1, 10, 100, true);
    tester.evaluate(hgemm_mma_m16n8k16_naive, "hgemm_mma_m16n8k16_naive");
}