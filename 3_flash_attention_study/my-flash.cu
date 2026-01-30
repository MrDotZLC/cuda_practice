#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void forward_kernel(const float *Q, const float *K,
                               const float *V, const int N,
                               const int d, const int Tc,
                               const int Tr, const int Bc,
                               const int Br,
                               const float softmax_scale, float *l,
                               float *m, float *O) {
    int thread_num_per_block = blockDim.x * blockDim.y;
    int tid = threadIdx.x;
    int bid = gridDim.x * blockIdx.y + blockIdx.x;

    const float *Q_start = Q + bid * d * N;
    const float *K_start = K + bid * d * N;
    const float *V_start = V + bid * d * N;
    float *O_start = O + bid * d * N;

    float *m_start = m + bid * N;
    float *l_start = l + bid * N;

    extern __shared__ float smem[];
    int offset = 0;
    float *Qs = smem + offset;
    offset += Br * d;
    float *Ks = smem + offset;
    offset += d * Bc;
    float *Vs = smem + offset;
    offset += d * Bc;
    float *Ss = smem + offset;

    // 遍历 K/V tiles
    for (int tc = 0; tc < Tc; tc++) {
        /* ---- 1. 载入 K/V tile 到 shared memory ---- */
        for (int idx = tid; idx < d * Bc;
             idx += thread_num_per_block) {
            Ks[idx] = K_start[tc * Bc * d + idx];
            Vs[idx] = V_start[tc * Bc * d + idx];

            // 初始化m_start、l_start、O_start
            m_start[idx] = -INFINITY;
            l_start[idx] = 0.f;
            O_start[idx] = 0.f;
        }
        __syncthreads();

        for (int tr = 0; tr < Tr; tr++) {
            int q_row = tr * Br + tid;
            if (q_row >= N) continue;
            /* ---- 2. 载入 Q row ---- */
            for (int idx = tid; idx < d * Br;
                 idx += thread_num_per_block) {
                Qs[idx] = Q_start[q_row * d + idx];
            }
            __syncthreads();

            /* ---- 3. 计算 S = QK^T + scale ---- */
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float acc = 0.f;
                for (int x = 0; x < d; x++) {
                    acc += Qs[tid * d + x] * Ks[y * d + x];
                }
                acc *= softmax_scale;
                Ss[tid * Bc + y] = acc;
                row_m = max(row_m, acc);
            }

            /* ---- 4. softmax（减 max）---- */
            float row_l = 0.f;
            for (int y = 0; y < Bc; y++) {
                float v = __expf(Ss[tid * Bc + y] - row_m);
                Ss[tid * Bc + y] = v;
                row_l += v;
            }

            /* ---- 5. online softmax 合并 ---- */
            float row_m_pre = m_start[q_row];
            float row_l_pre = l_start[q_row];

            float row_m_new = max(row_m_pre, row_m);
            float row_l_new =
                (__expf(row_m_pre - row_m_new) * row_l_pre) +
                (__expf(row_m - row_m_new) * row_l);
            row_l_new = max(row_l_new, 1e-6f); // 防止除零

            /* ---- 6. 计算 PV 并更新 O ---- */
            for (int x = 0; x < d; x++) {
                float pv = 0.f;
                for (int y = 0; y < Bc; y++) {
                    pv += Ss[Bc * tid + y] * Vs[y * d + x];
                }
                int o_idx = q_row * d + x;
                float o_prev = O_start[o_idx];
                O_start[o_idx] =
                    (row_l_pre * __expf(row_m_pre - row_m_new) *
                         o_prev +
                     __expf(row_m - row_m_new) * pv) /
                    row_l_new;
            }

            /* ---- 7. 写回 m / l ---- */
            m[q_row] = row_m_new;
            l[q_row] = row_l_new;
        }
        __syncthreads();
    }
}