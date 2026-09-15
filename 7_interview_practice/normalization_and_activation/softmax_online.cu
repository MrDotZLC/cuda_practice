
int get_block_size(int hidden_dim) {
    int block_size = 32;
    while (block_size < hidden_dim && block_size < 1024) {
        block_size <<= 1;
    }
    return std::min(block_size, 1024); // 上限取决于架构
}

__device__ void warp_reduce_md(float& m, float& d) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float om = __shfl_down_sync(0xffffffff, m, offset);
        float od = __shfl_down_sync(0xffffffff, d, offset);
        // d = d * expf(m - m_new) + ∑exp(x-m_new) =
        //   = d * expf(m - m_new) + ∑exp(x-om) * exp(om-m_new)
        //   = d * expf(m - m_new) + od exp(om-m_new)
        if (om > m) {
            d = d * expf(m - om) + od;
            m = om;
        } else {
            d = d + od * expf(om - m);
        }
    }
}


// 逐元素追加
// d = d_o*exp(m_o-m) + ∑exp(x-m)，x∈B
__global__ void softmax_online_kernel(const float* __restrict__ in,
             float*       __restrict__ out,
             int num_rows, int row_len) {
    int row_id = blockIdx.x;
    const float* x = in  + row_id * row_len;
    float*       y = out + row_id * row_len;

    float m = -__FLT_MAX__;
    float d = 0.f;
    // Pass 1：每线程单次遍历，维护局部 (m, d)
    for (int i = threadIdx.x; i < row_len; i += blockDim.x) {
        float xi = x[i];
        float m_new = fmaxf(m, xi);
        d = d * expf(m - m_new) + expf(xi - m_new);
        m = m_new;
    }
    // Warp 内合并
    warp_reduce_md(m, d);

    __shared__ float s_m[32];
    __shared__ float s_d[32];
    
    // 跨 Warp 合并
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    int num_warps = (blockDim.x + 31) / 32;
    if (lane_id == 0) {
        s_m[warp_id] = m;
        s_d[warp_id] = d;
    }
    __syncthreads();

    if (warp_id == 0) {
        m = (tid < num_warps) ? s_m[tid] : -__FLT_MAX__;
        d = (tid < num_warps) ? s_d[tid] : 0.f;
        warp_reduce_md(m, d);
        if (lane_id == 0) {
            s_m[0] = m;
            s_d[0] = d;
        }
    }
    __syncthreads();

    float gm = s_m[0];
    float ginv = 1.f / s_d[0];
    // Pass 2：输出归一化结果
    for (int i = tid; i < row_len; i += blockDim.x) {
        y[i] = expf(x[i] - gm) * ginv;
    }
}

void softmax_online(const float* __restrict__ d_in,
             float*       __restrict__ d_out,
             int num_rows, int row_len) {

    const int block_size = get_block_size(row_len);

    dim3 block(block_size);
    dim3 grid(num_rows);
    softmax_online_kernel<<<grid, block>>>(d_in, d_out, num_rows, row_len);
}