#include "error.cuh"
#include <cstdio>

const int NUM_REPEAtS = 10;
void timing(const double *x, const int N);
double reduce(const double *x, const int N);

int main() {
    const int N = 1e8;
    const int M = sizeof(double) * N;
    double *x = (double *)malloc(M);
    for (int n = 0; n < N; n++) {
        x[n] = 1.23;
    }

    timing(x, N);

    free(x);
}

void timing(const double *x, const int N) {
    double sum = 0;

    for (int repeat = 0; repeat < NUM_REPEAtS; repeat++) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(x, N);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    printf("sum = %f.\n", sum);
}

double reduce(const double *x, const int N) {
    double sum = 0.0;
    for (int n = 0; n < N; n++) {
        sum += x[n];
    }
    return sum;
}