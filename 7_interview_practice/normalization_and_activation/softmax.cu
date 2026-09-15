
enum class OperatorType{
    MAX = 0,
    SUM = 1
};

int get_block_size(int hidden_dim) {
    int block_size = 32;
    while (block_size < hidden_dim && block_size < 1024) {
        block_size <<= 1;
    }
    return std::min(block_size, 1024);
}

template <OperatorType T>
__device__ __forceinline__ float warp_reduce(float v) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(mask, v, offset);
        if constexpr (T == OperatorType::MAX) {
            v = fmaxf(v, other);
        } else {
            v += other;
        }
    }
    return v;
}

template <OperatorType t>
__device__ __forceinline__ float identity() {
    if constexpr (t == OperatorType::MAX)
        return -__FLT_MAX__;
    else
        return 0.0f;
}

template <OperatorType T>
__device__ float block_reduce(float v) {
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int lane_id = tid % 32; // & 31
    int warp_id = tid / 32; // >> 5
    int num_warps = (blockDim.x + 32 - 1) / 32;
    v = warp_reduce<T>(v);
    if (lane_id == 0) {
        smem[warp_id] = v;
    }
    __syncthreads();
    if (tid < num_warps) {
        v = smem[tid];
    } else {
        v = identity<T>();
    }
    if (warp_id == 0) {
        v = warp_reduce<T>(v);
    }
    return v;
}

__global__ void softmax_kernel(const float* __restrict__ in,
                               float* __restrict__ out, int num_rows,
                               int row_len) {
    int row_id = blockIdx.x;
    const float* x = in  + row_id * row_len;
    float*       y = out + row_id * row_len;

    float m = -__FLT_MAX__;
    for (int i = threadIdx.x; i < row_len; i += blockDim.x) {
        m = fmaxf(m, x[i]);
    }
    m = block_reduce<OperatorType::MAX>(m);
    __shared__ float s_max;
    if (threadIdx.x == 0) {
        s_max = m;
    }
    __syncthreads();

    float max_val = s_max;
    float sum_exp = 0.f;
    for (int i = threadIdx.x; i < row_len; i += blockDim.x) {
        float e = expf(x[i] - max_val);
        y[i] = e;
        sum_exp += e;
    }
    sum_exp = block_reduce<OperatorType::SUM>(sum_exp);
    __shared__ float s_sum_exp;
    if (threadIdx.x == 0) {
        s_sum_exp = sum_exp;
    }
    __syncthreads();
    float inv_sum = 1.f / s_sum_exp;

    for (int i = threadIdx.x; i < row_len; i += blockDim.x) {
        y[i] *= inv_sum;
    }
}

void softmax(const float* __restrict__ d_in,
             float*       __restrict__ d_out,
             int num_rows, int row_len) {

    const int block_size = get_block_size(row_len);

    dim3 block(block_size);
    dim3 grid(num_rows);
    softmax_kernel<<<grid, block>>>(d_in, d_out, num_rows, row_len);
}