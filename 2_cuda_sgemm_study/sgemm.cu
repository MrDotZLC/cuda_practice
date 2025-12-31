#include "error.cuh"
#include <cstdio>

const int NUM_REPEATS = 3;
const int M = 2048 / 4;
const int N = 2048 / 4;
const int K = 2048 / 4;
const int TILE = 16;
const int STRIDE = 2; 
const int M_NUM_PER_BLOCK = 32;
const int N_NUM_PER_BLOCK = 32;
const int K_NUM_PER_BLOCK = 32;
const int M_NUM_PER_BLOCK1 = 64;
const int N_NUM_PER_BLOCK1 = 64;
const int K_NUM_PER_BLOCK1 = 64;
const int NUM_PER_THREAD = 4;
const int M_NUM_PER_THREAD = 4;
const int N_NUM_PER_THREAD = 4;
const int K_NUM_PER_THREAD = 4;
const int NUM_REG = NUM_PER_THREAD / 2;

#define A(i, j) a[(i) * n + (j)]
#define B(i, j) b[(i) * n + (j)]

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

void random_matrix(int m, int n, float *a) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
#if 1
            A(i, j) = 2.0 * (float)drand48() - 1.0;
#else
            A(i, j) = (j - i) % 3;
#endif
        }
    }
}   

float compare_matrices(int m, int n, float *a, float *b) {
    // int i, j;
    float max_diff = 0.0, diff;
    int printed = 0;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            diff = abs(A(i, j) - B(i, j));
            max_diff = diff > max_diff ? diff : max_diff;
            if (printed == 0) {
                if (max_diff > 0.5f) {
                    printf("\nError: i %d j %d diff %f got %f expect %f\n", i, j, max_diff, A(i, j), B(i, j));
                    printed = 1;
                }
            }
        }
    }
    return max_diff;
}

// 使用CPU计算
void sgemm0(const float *a, const float *b, float *c) {
    for (int m = 0 ; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float temp = 0.f;
            for (int k = 0; k < K; k++) {
                temp += a[m * K + k] * b[k * N + n];
            }
            c[m * N + n] = temp;
        }
    }
}

// 使用GPU全局内存
__global__ void sgemm1(float *a, float *b, float *c) {
    const int col= threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;
    float *a_begin = a + blockDim.y * blockIdx.y * K;
    float *b_begin = b + blockDim.x * blockIdx.x;
    float temp = 0.f;

    for (int k = 0; k < K; k++) {
        temp += a_begin[threadIdx.y * K + k] * b_begin[k * N + threadIdx.x];
    }
    c[row * N + col] = temp;
}

// 使用GPU共享内存：整个A所需的TILE*K和B所需的K*TILE存入共享内存
// 会出现内存不够分配 或者 恰好够存储但不够计算（结果为0或者debug跳过核函数）
__global__ void sgemm2(float *a, float *b, float *c) {
    const int col = threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;
    float *a_begin = a + blockDim.y * blockIdx.y * K; // a行固定列遍历 
    float *b_begin = b + blockDim.x * blockIdx.x;     // b列固定行遍历

    __shared__ float a_shared[TILE][K];
    __shared__ float b_shared[K][TILE];
    
    for (int s = 0; s < K; s += TILE) {
        a_shared[threadIdx.y][threadIdx.x + s] = 
                a_begin[threadIdx.y * K + threadIdx.x + s];
        b_shared[threadIdx.y + s][threadIdx.x] = 
                b_begin[(threadIdx.y + s) * N + threadIdx.x];
    }
    __syncthreads();

    float temp = 0.f;
    for (int k = 0; k < K; k++) {
        temp += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
    }
    if (row < M && col < N) {
        c[row * N + col] = temp;
    }
}

// 使用GPU共享内存：分块存入Shared Memory并计算
__global__ void sgemm3(float *a, float *b, float *c) {
    const int col = threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;
    float *a_begin = a + blockDim.y * blockIdx.y * K; // a行固定列遍历 
    float *b_begin = b + blockDim.x * blockIdx.x;     // b列固定行遍历

    __shared__ float a_shared[TILE][TILE];
    __shared__ float b_shared[TILE][TILE];
    
    float temp = 0.f;
    for (int s = 0; s < K; s += TILE) {
        a_shared[threadIdx.y][threadIdx.x] = 
                a_begin[threadIdx.y * K + threadIdx.x + s];
        b_shared[threadIdx.y][threadIdx.x] = 
                b_begin[(threadIdx.y + s) * N + threadIdx.x];
        __syncthreads();
        for (int k = 0; k < TILE; k++) {
            temp += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        c[row * N + col] = temp;
    }
}

// 每个线程增加工作量：减少Block数量
__global__ void sgemm4(float *a, float *b, float *c) {
    const int step = TILE * STRIDE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float *a_begin = a + step * blockIdx.y * K; // a行固定列遍历 
    float *b_begin = b + step * blockIdx.x;     // b列固定行遍历
    float *c_begin = c + step * blockIdx.y * M + step * blockIdx.x;

    __shared__ float a_shared[step][step];
    __shared__ float b_shared[step][step];
    
    float temp[STRIDE][STRIDE] = {{0.f, 0.f}, {0.f, 0.f}};

    for (int s = 0; s < K; s += step) {
        for (int i = 0; i < STRIDE; i++) {
            for (int j = 0; j < STRIDE; j++) {
                a_shared[ty + i * TILE][tx + j * TILE] = a_begin[(ty + i * TILE) * K + tx + j * TILE + s];
                b_shared[ty + i * TILE][tx + j * TILE] = b_begin[(ty + i * TILE + s) * N + tx + j * TILE];
            }
        }
        __syncthreads();
        for (int i = 0; i < STRIDE; i++) {
            for (int j = 0; j < STRIDE; j++) {
                for (int k = 0; k < step; k++) {
                    temp[i][j] += a_shared[ty + i * TILE][k] * b_shared[k][tx + j * TILE];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < STRIDE; i++) {
        for (int j = 0; j < STRIDE; j++) {
            c_begin[(ty + i * TILE) * N + tx + j * TILE] = temp[i][j];
        }
    }
}

// 降低数据精度：共享内存使用float4，则每个线程处理NUM_PER_THREAD
__global__ void sgemm5(float *a, float *b, float *c) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    float *a_begin = a + blockIdx.y * M_NUM_PER_BLOCK * K;
    float *b_begin = b + blockIdx.x * N_NUM_PER_BLOCK;

    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    float temp[NUM_PER_THREAD] = {0.f};

    for (int s = 0; s < K; s += K_NUM_PER_BLOCK) {
        FETCH_FLOAT4(a_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(a_begin[ty * K + tx * NUM_PER_THREAD + s]);
        // a_shared[ty][tx * NUM_PER_THREAD] = a_begin[ty * K + tx * NUM_PER_THREAD + s];
        // a_shared[ty][tx * NUM_PER_THREAD + 1] = a_begin[ty * K + tx * NUM_PER_THREAD + s + 1];
        // a_shared[ty][tx * NUM_PER_THREAD + 2] = a_begin[ty * K + tx * NUM_PER_THREAD + s + 2];
        // a_shared[ty][tx * NUM_PER_THREAD + 3] = a_begin[ty * K + tx * NUM_PER_THREAD + s + 3];
        FETCH_FLOAT4(b_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(b_begin[(ty + s) * N + tx * NUM_PER_THREAD]);
        // b_shared[ty][tx * NUM_PER_THREAD] = b_begin[(ty + s) * N + tx * NUM_PER_THREAD];
        // b_shared[ty][tx * NUM_PER_THREAD + 1] = b_begin[(ty + s) * N + tx * NUM_PER_THREAD + 1];
        // b_shared[ty][tx * NUM_PER_THREAD + 2] = b_begin[(ty + s) * N + tx * NUM_PER_THREAD + 2];
        // b_shared[ty][tx * NUM_PER_THREAD + 3] = b_begin[(ty + s) * N + tx * NUM_PER_THREAD + 3];
        __syncthreads();
        for (int i = 0; i < NUM_PER_THREAD; i++) {
            for (int k = 0; k < K_NUM_PER_BLOCK; k++) {
                temp[i] += a_shared[ty][k] * b_shared[k][tx * NUM_PER_THREAD + i];
            }
        }
        __syncthreads();
    }

    float *c_begin = c + blockIdx.y * M_NUM_PER_BLOCK * N + blockIdx.x * N_NUM_PER_BLOCK;
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        c_begin[ty * N + tx * NUM_PER_THREAD + i] = temp[i]; 
    }
}

// 寄存器+外积：寄存器减少共享内存读取，使用外积计算
__global__ void sgemm6(float *a, float *b, float *c) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int ctx = tid % (K_NUM_PER_BLOCK / NUM_REG);
    const int cty = tid / (K_NUM_PER_BLOCK / NUM_REG);
    float *a_begin = a + blockIdx.y * M_NUM_PER_BLOCK * K;
    float *b_begin = b + blockIdx.x * N_NUM_PER_BLOCK;

    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    float a_reg[NUM_REG] = {0.f};
    float b_reg[NUM_REG] = {0.f};
    
    float temp[NUM_REG][NUM_REG] = {0.f};

    for (int s = 0; s < K; s += K_NUM_PER_BLOCK) {
        FETCH_FLOAT4(a_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(a_begin[ty * K + tx * NUM_PER_THREAD + s]);
        // a_shared[ty][tx * NUM_PER_THREAD] = a_begin[ty * K + tx * NUM_PER_THREAD + s];
        // a_shared[ty][tx * NUM_PER_THREAD + 1] = a_begin[ty * K + tx * NUM_PER_THREAD + s + 1];
        // a_shared[ty][tx * NUM_PER_THREAD + 2] = a_begin[ty * K + tx * NUM_PER_THREAD + s + 2];
        // a_shared[ty][tx * NUM_PER_THREAD + 3] = a_begin[ty * K + tx * NUM_PER_THREAD + s + 3];
        FETCH_FLOAT4(b_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(b_begin[(ty + s) * N + tx * NUM_PER_THREAD]);
        // b_shared[ty][tx * NUM_PER_THREAD] = b_begin[(ty + s) * N + tx * NUM_PER_THREAD];
        // b_shared[ty][tx * NUM_PER_THREAD + 1] = b_begin[(ty + s) * N + tx * NUM_PER_THREAD + 1];
        // b_shared[ty][tx * NUM_PER_THREAD + 2] = b_begin[(ty + s) * N + tx * NUM_PER_THREAD + 2];
        // b_shared[ty][tx * NUM_PER_THREAD + 3] = b_begin[(ty + s) * N + tx * NUM_PER_THREAD + 3];
        __syncthreads();
        for (int k = 0; k < K_NUM_PER_BLOCK; k++) {
            // 使用外积计算
            a_reg[0] = a_shared[cty * NUM_REG][k];
            a_reg[1] = a_shared[cty * NUM_REG + 1][k];
            b_reg[0] = b_shared[k][ctx * NUM_REG];
            b_reg[1] = b_shared[k][ctx * NUM_REG + 1];
            for (int i = 0; i < NUM_REG; i++) {
                for (int j = 0; j < NUM_REG; j++) {
                    temp[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    float *c_begin = c + blockIdx.y * M_NUM_PER_BLOCK * N + blockIdx.x * N_NUM_PER_BLOCK;

    for (int i = 0; i < NUM_REG; i++) {
        for (int j = 0; j < NUM_REG; j++) {
            c_begin[(cty * NUM_REG + i) * N + ctx * NUM_REG + j] = temp[i][j];
        }
    }
}

// 基于sgemm6的改进版本：寄存器存储4*4，blockDim=(16,16)，NUM_PER_BLOCK=(64,64)
__global__ void sgemm7(float *a, float *b, float *c) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    float *a_begin = a + blockIdx.y * M_NUM_PER_BLOCK1 * K; // 一行
    float *b_begin = b + blockIdx.x * N_NUM_PER_BLOCK1;     // 一列

    __shared__ float a_shared[M_NUM_PER_BLOCK1][K_NUM_PER_BLOCK1];
    __shared__ float b_shared[K_NUM_PER_BLOCK1][N_NUM_PER_BLOCK1];

    float a_reg[M_NUM_PER_THREAD] = {0.f};
    float b_reg[N_NUM_PER_THREAD] = {0.f};
    
    float temp[M_NUM_PER_THREAD][N_NUM_PER_THREAD] = {0.f};

    for (int s = 0; s < K; s += K_NUM_PER_BLOCK1) {
        // 每个线程从a取 M_NUM_PER_THREAD*K_NUM_PER_THREAD 个，存到共享内存
        // 每次循环取K_NUM_PER_THREAD个，列向是连续的
        for (int i = 0; i < M_NUM_PER_THREAD; i++) {
            FETCH_FLOAT4(a_shared[ty * M_NUM_PER_THREAD + i][tx * K_NUM_PER_THREAD]) = 
                FETCH_FLOAT4(a_begin[(ty * M_NUM_PER_THREAD + i) * K + tx * K_NUM_PER_THREAD + s]);
        }
        // 每个线程从a取 K_NUM_PER_THREAD*N_NUM_PER_THREAD 个，存到共享内存
        for (int i = 0; i < K_NUM_PER_THREAD; i++) {
            FETCH_FLOAT4(b_shared[ty * K_NUM_PER_THREAD + i][tx * N_NUM_PER_THREAD]) = 
                FETCH_FLOAT4(b_begin[(ty * K_NUM_PER_THREAD + i + s) * N + tx * N_NUM_PER_THREAD]);
        }
        __syncthreads();
        for (int k = 0; k < K_NUM_PER_BLOCK1; k++) {
            // 使用外积计算
            // a是按列取的，只能按列存到寄存器
            a_reg[0] = a_shared[ty * M_NUM_PER_THREAD][k];
            a_reg[1] = a_shared[ty * M_NUM_PER_THREAD + 1][k];
            a_reg[2] = a_shared[ty * M_NUM_PER_THREAD + 2][k];
            a_reg[3] = a_shared[ty * M_NUM_PER_THREAD + 3][k];
            // b是按行取的，可以用float4一次性存到寄存器
            FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(b_shared[k][tx * N_NUM_PER_THREAD]);
            
            for (int i = 0; i < M_NUM_PER_THREAD; i++) {
                for (int j = 0; j < N_NUM_PER_THREAD; j++) {
                    temp[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    float *c_begin = c + blockIdx.y * M_NUM_PER_BLOCK1 * N + blockIdx.x * N_NUM_PER_BLOCK1;

    for (int i = 0; i < M_NUM_PER_THREAD; i++) {
        FETCH_FLOAT4(c_begin[(ty * M_NUM_PER_THREAD + i) * N + tx * N_NUM_PER_THREAD]) = FETCH_FLOAT4(temp[i][0]);
        // 连续的，可以一次性存储
        // for (int j = 0; j < N_NUM_PER_THREAD; j++) {
        //     c_begin[(ty * M_NUM_PER_THREAD + i) * N + tx * N_NUM_PER_THREAD + j] = temp[i][j];
        // }
    }
}

// 使用A转置提高访问比：寄存器读A，再将A的转置存入共享内存，计算时使用float4一次性存入寄存器计算
__global__ void sgemm8(float *a, float *b, float *c) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    float *a_begin = a + blockIdx.y * M_NUM_PER_BLOCK1 * K; // 一行
    float *b_begin = b + blockIdx.x * N_NUM_PER_BLOCK1;     // 一列

    __shared__ float a_shared[M_NUM_PER_BLOCK1][K_NUM_PER_BLOCK1];
    __shared__ float b_shared[K_NUM_PER_BLOCK1][N_NUM_PER_BLOCK1];

    float a_reg[M_NUM_PER_THREAD] = {0.f};
    float b_reg[N_NUM_PER_THREAD] = {0.f};
    float a_trans[K_NUM_PER_THREAD] = {0.f};
    
    float temp[M_NUM_PER_THREAD][N_NUM_PER_THREAD] = {0.f};

    for (int s = 0; s < K; s += K_NUM_PER_BLOCK1) {
        // 每个线程从a取 M_NUM_PER_THREAD*K_NUM_PER_THREAD 个，转置后存到寄存器
        // 每次循环取K_NUM_PER_THREAD个，列向是连续的
        for (int i = 0; i < M_NUM_PER_THREAD; i++) {
            // FETCH_FLOAT4(a_shared[ty * M_NUM_PER_THREAD + i][tx * K_NUM_PER_THREAD]) = 
            //     FETCH_FLOAT4(a_begin[(ty * M_NUM_PER_THREAD + i) * K + tx * K_NUM_PER_THREAD + s]);
            FETCH_FLOAT4(a_trans[0]) = 
                FETCH_FLOAT4(a_begin[(ty * M_NUM_PER_THREAD + i) * K + tx * K_NUM_PER_THREAD + s]);
            a_shared[tx * K_NUM_PER_THREAD + 0][ty * M_NUM_PER_THREAD + i] = a_trans[0];
            a_shared[tx * K_NUM_PER_THREAD + 1][ty * M_NUM_PER_THREAD + i] = a_trans[1];
            a_shared[tx * K_NUM_PER_THREAD + 2][ty * M_NUM_PER_THREAD + i] = a_trans[2];
            a_shared[tx * K_NUM_PER_THREAD + 3][ty * M_NUM_PER_THREAD + i] = a_trans[3];
        }
        // 每个线程从a取 K_NUM_PER_THREAD*N_NUM_PER_THREAD 个，存到共享内存
        for (int i = 0; i < K_NUM_PER_THREAD; i++) {
            FETCH_FLOAT4(b_shared[ty * K_NUM_PER_THREAD + i][tx * N_NUM_PER_THREAD]) = 
                FETCH_FLOAT4(b_begin[(ty * K_NUM_PER_THREAD + i + s) * N + tx * N_NUM_PER_THREAD]);
        }
        __syncthreads();
        for (int k = 0; k < K_NUM_PER_BLOCK1; k++) {
            // 使用外积计算
            // a是按列取的，只能按列存到寄存器
            // a_reg[0] = a_shared[ty * M_NUM_PER_THREAD][k];
            // a_reg[1] = a_shared[ty * M_NUM_PER_THREAD + 1][k];
            // a_reg[2] = a_shared[ty * M_NUM_PER_THREAD + 2][k];
            // a_reg[3] = a_shared[ty * M_NUM_PER_THREAD + 3][k];
            FETCH_FLOAT4(a_reg[0]) = FETCH_FLOAT4(a_shared[k][ty * M_NUM_PER_THREAD]);
            // b是按行取的，可以用float4一次性存到寄存器
            FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(b_shared[k][tx * N_NUM_PER_THREAD]);
            
            for (int i = 0; i < M_NUM_PER_THREAD; i++) {
                for (int j = 0; j < N_NUM_PER_THREAD; j++) {
                    temp[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    float *c_begin = c + blockIdx.y * M_NUM_PER_BLOCK1 * N + blockIdx.x * N_NUM_PER_BLOCK1;

    for (int i = 0; i < M_NUM_PER_THREAD; i++) {
        // 连续的，可以一次性存储
        FETCH_FLOAT4(c_begin[(ty * M_NUM_PER_THREAD + i) * N + tx * N_NUM_PER_THREAD]) = FETCH_FLOAT4(temp[i][0]);
        // for (int j = 0; j < N_NUM_PER_THREAD; j++) {
        //     c_begin[(ty * M_NUM_PER_THREAD + i) * N + tx * N_NUM_PER_THREAD + j] = temp[i][j];
        // }
    }
}

void sgemm(float *a, float *b, float *c, const int method) {
    dim3 block0(TILE, TILE);
    dim3 block1(8, 32); // 32行，每行存放8个float4，共256个
    dim3 block2(16, 16);
    // 设置列0主序，threadIdx.x为列下标且连续，因历史原因cuda中矩阵常为列主序
    dim3 grid0((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    dim3 grid1((N + TILE - 1) / TILE / STRIDE, (M + TILE - 1) / TILE / STRIDE);
    dim3 grid2((N + N_NUM_PER_BLOCK - 1) / N_NUM_PER_BLOCK, (M + M_NUM_PER_BLOCK - 1) / M_NUM_PER_BLOCK);
    dim3 grid3((N + N_NUM_PER_BLOCK1 - 1) / N_NUM_PER_BLOCK1, (M + M_NUM_PER_BLOCK1 - 1) / M_NUM_PER_BLOCK1);

    switch (method) {
    case 0:
        sgemm0(a, b, c);
        break;
    case 1:
        sgemm1<<<grid0, block0>>>(a, b, c);
        break;
    case 2:
        sgemm2<<<grid0, block0>>>(a, b, c);
        break;
    case 3:
        sgemm3<<<grid0, block0>>>(a, b, c);
        break;
    case 4:
        sgemm4<<<grid1, block0>>>(a, b, c);
        break;
    case 5:
        sgemm5<<<grid2, block1>>>(a, b, c);
        break;
    case 6:
        sgemm6<<<grid2, block1>>>(a, b, c);
        break;
    case 7:
        sgemm7<<<grid3, block2>>>(a, b, c);
        break;
    case 8:
        sgemm8<<<grid3, block2>>>(a, b, c);
        break;
    default:
        printf("Error: wrong method\n");
        exit(1);
        break;
    }
}

void timing(float *a, float *b, float *c, const int method) {
    float t_avg = 0.f;
    for (int repeat = 0; repeat < NUM_REPEATS; repeat++) {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        cudaEventQuery(start);

        sgemm(a, b, c, method);

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
        t_avg += elapsed_time;

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }
    t_avg /= NUM_REPEATS;
    printf("Average Time = %.6f ms.\n", t_avg);
}

int main() {
    const size_t a_mem = M * K * sizeof(float);
    const size_t b_mem = K * N * sizeof(float);
    const size_t c_mem = M * N * sizeof(float);
    
    float *a_host = (float *)malloc(a_mem);
    float *b_host = (float *)malloc(b_mem);
    float *c_host_cpu = (float *)malloc(c_mem);
    float *c_host_gpu = (float *)malloc(c_mem);
    
    random_matrix(M, K, a_host);
    random_matrix(K, N, b_host);
    memset(c_host_cpu, 0, c_mem);
    memset(c_host_gpu, 0, c_mem);

    float *a_device, *b_device, *c_device;
    CHECK_CUDA(cudaMalloc((void **)&a_device, a_mem));
    CHECK_CUDA(cudaMalloc((void **)&b_device, b_mem));
    CHECK_CUDA(cudaMalloc((void **)&c_device, c_mem));

    CHECK_CUDA(cudaMemcpy(a_device, a_host, a_mem, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b_device, b_host, b_mem, cudaMemcpyHostToDevice));

    printf("\nsGEmm v0 CPU:                               "); // CPU
    timing(a_host, b_host, c_host_cpu, 0);      //    ,              MN, 455.0ms
    printf("\nsGEmm v1 GPU Global Memory:                 "); // GPU
    timing(a_device, b_device, c_device, 1);    // 2MNK            , MN, 47.0ms
    // printf("\nsGEmm v2 GPU Shared Memory with TILE * K    "); // SM with TILE*K
    // timing(a_device, b_device, c_device, 2);    // error: Shared Memory limit exceeded
    printf("\nsGEmm v3 GPU Shared Memory with TILE * TILE "); // SM with TILE*TILE
    timing(a_device, b_device, c_device, 3);    // MNK(1/bn + 1/bm), MN, 47.0ms
    printf("\nsGEmm v4 GPU Increase Work of Per Thread    "); // more work per thread
    timing(a_device, b_device, c_device, 4);    // MNK(1/bn + 1/bm), MN, 47.0ms
    printf("\nsGEmm v5 GPU Using Float4                   "); // more work per thread
    timing(a_device, b_device, c_device, 5);
    printf("\nsGEmm v6 GPU Register and Outer Product     "); // Register cache data and using outer product
    timing(a_device, b_device, c_device, 6);
    printf("\nsGEmm v7 GPU Optimize block based v6        "); // Optimized blockDim=(16,16) and NUM_PER_BLOCK=(64,64) based v6
    timing(a_device, b_device, c_device, 7);
    printf("\nsGEmm v8 GPU Smem Transpose                 "); // Register+Transpose A to impove access
    timing(a_device, b_device, c_device, 8);
    
    CHECK_CUDA(cudaMemcpy(c_host_gpu, c_device, c_mem, cudaMemcpyDeviceToHost));

    float diff = compare_matrices(M, N, c_host_cpu, c_host_gpu);
    if (diff > 0.5f) {
        printf("diff too big!\n");
    } else {
        printf("right!\n");
    }

    free(a_host);
    free(b_host);
    free(c_host_cpu);
    free(c_host_gpu);
    CHECK_CUDA(cudaFree(a_device));
    CHECK_CUDA(cudaFree(b_device));
    CHECK_CUDA(cudaFree(c_device));

    return 0;
}